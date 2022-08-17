import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils
from sklearn.preprocessing import LabelEncoder
from torch_sparse import SparseTensor
from speos.utils.logger import setup_logger
import speos.layers as layers


class LINKX(nn.Module):
    def __init__(self, config, input_dim: tuple):
        super().__init__()
        # based on https://arxiv.org/pdf/2110.14446.pdf
        self.config = config
        self.input_dim_x = input_dim[0]
        self.input_dim_a = input_dim[1]
        self.n_mlp_a = self.n_mlp_x = self.config.model.pre_mp.n_layers
        self.n_mlp_f = self.config.model.post_mp.n_layers
        self.output_dim = 1

        self.pre_dim_hid = config.model.pre_mp.dim

        self.norm = nn.ModuleList()
        pre_mp_dropout = self.config.model.pre_mp.dropout
        pre_mp_act = nn.ELU
        post_mp_dropout = self.config.model.post_mp.dropout
        post_mp_act = nn.ELU

        mlp_a_list = []
        mlp_x_list = []
        mlp_f_list = []

        # Pre Message Passing
        mlp_a_list.append(nn.Linear(self.input_dim_a, self.pre_dim_hid))
        mlp_a_list.append(pre_mp_act())
        mlp_x_list.append(nn.Linear(self.input_dim_x, self.pre_dim_hid))
        mlp_x_list.append(pre_mp_act())

        for i in range(self.n_mlp_a):
            mlp_a_list.append(nn.Linear(self.pre_dim_hid, self.pre_dim_hid))
            mlp_a_list.append(pre_mp_act())
            mlp_x_list.append(nn.Linear(self.pre_dim_hid, self.pre_dim_hid))
            mlp_x_list.append(pre_mp_act())
            if pre_mp_dropout is not None:
                mlp_a_list.append(nn.Dropout(p=pre_mp_dropout))
                mlp_x_list.append(nn.Dropout(p=pre_mp_dropout))

        for i in range(self.n_mlp_f):
            if i == self.n_mlp_f - 1:
                # last layer half the parameters
                mlp_f_list.append(nn.Linear(self.pre_dim_hid, self.pre_dim_hid // 2))
                mlp_f_list.append(post_mp_act())
            else:
                mlp_f_list.append(nn.Linear(self.pre_dim_hid, self.pre_dim_hid))
                mlp_f_list.append(post_mp_act())
            if post_mp_dropout is not None:
                mlp_f_list.append(nn.Dropout(p=post_mp_dropout))

        mlp_f_list.append(nn.Linear(self.pre_dim_hid // 2, self.output_dim))

        self.mlp_a = nn.Sequential(*mlp_a_list)
        self.mlp_x = nn.Sequential(*mlp_x_list)
        self.mlp_f = nn.Sequential(*mlp_f_list)

        self.linkxlayer = LINKXLayer(self.pre_dim_hid)

    def regularization(self):
        l1 = torch.norm(self.mlp_a[0].weight, p=1)
        return l1

    def forward(self, x, edge_index):
        adj = torch.squeeze(pyg_utils.to_dense_adj(edge_index))

        x = self.mlp_x(x)
        adj = self.mlp_a(adj)

        # mat = self.linkxblock(x, adj)
        mat = F.elu(self.linkxlayer(x, adj))
        y_hat = self.mlp_f(mat)

        return y_hat


class LINKXLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weights = torch.nn.Parameter(torch.rand((2 * dim, dim)))
        nn.init.xavier_normal_(self.weights)

    def forward(self, out_x, out_a):

        concat = torch.hstack((out_a, out_x))
        mat = torch.mm(concat, self.weights)
        mat += out_x
        mat += out_a
        return mat


class GeneNetwork(nn.Module):
    def __init__(self, config, input_dim, num_adjacencies=1):
        super(GeneNetwork, self).__init__()
        self.num_adjacencies = num_adjacencies
        self.reset_rgcn_to = "tag"
        self.input_dim = input_dim
        self.config = config
        self.final_conv_acts = None
        self.final_conv_grads = None
        self.input = None
        self.input_grads = None
        self.gcnconv_num_layers = self.config.model.mp.n_layers
        self.gcnconv_parameters = {"type": self.config.model.mp.type,
                                   "dim": self.config.model.mp.dim}
        self.dim_hid = self.gcnconv_parameters["dim"]
        self.out_dim_hid = self.gcnconv_parameters["dim"]
        self.in_dim_hid = 2 * self.dim_hid if self.config.model.skip else self.dim_hid

        self.npremp = self.config.model.pre_mp.n_layers
        self.npostmp = self.config.model.post_mp.n_layers
        self.output_dim = 1
        self.nheads = self.config.model.mp.nheads if self.gcnconv_num_layers > 1 else 1

        if self.config.model.skip_mp:
            self.skip_mp_norm = pyg_nn.LayerNorm(self.gcnconv_parameters["dim"])

        self.pre_mp = nn.ModuleList()
        self.post_mp = nn.ModuleList()

        self.rgcn_kwargs = {"num_bases": 1}
        self.gat_kwargs = {"dropout": 0.1}
        self.cheb_kwargs = {"K": 5}

        # Pre Message Passing
        self.make_pre_mp()

        # Graph Message Passing
        self.make_mp()

        # Post Message Passing
        self.make_post_mp()

    def get_act(self):
        if self.config.model.pre_mp.act.lower() == "elu":
            act = nn.ELU()
        elif self.config.model.pre_mp.act.lower() == "relu":
            act = nn.ReLU()
        else:
            raise ValueError("Only implemented elu and relu")

        return act

    def make_mp(self):
        self.graph_message_passing = self.gcn_message_passing

        mp_list = []
        flow_list = []

        for i in range(self.gcnconv_num_layers):
            mp_list.append(self.get_mp_layer(i))
            flow_list.append('x, edge_index -> x')
            mp_list.append(self.get_act())
            flow_list.append('x -> x')
            mp_list.append(self.get_mp_norm())
            flow_list.append('x -> x')

        if len(mp_list) > 0:
            self.mp = pyg_nn.Sequential('x, edge_index', [(layer, flow) for layer, flow in zip(mp_list, flow_list)])
        else:
            self.mp = None

    def get_mp_layer(self, i):
        kwargs = self.config.model.mp.kwargs
        if self.gcnconv_parameters["type"] == "sage":
            mp_layer = pyg_nn.SAGEConv(self.gcnconv_parameters["dim"], self.gcnconv_parameters["dim"], **kwargs)
        elif self.gcnconv_parameters["type"] == "transformer":
            mp_layer = pyg_nn.TransformerConv(self.gcnconv_parameters["dim"], self.gcnconv_parameters["dim"], **kwargs)
        elif self.gcnconv_parameters["type"] == "fac":
            mp_layer = pyg_nn.FAConv(self.gcnconv_parameters["dim"], self.gcnconv_parameters["dim"], **kwargs)
        elif self.gcnconv_parameters["type"] == "tag":
            mp_layer = pyg_nn.TAGConv(self.gcnconv_parameters["dim"], self.gcnconv_parameters["dim"], K=self.config.model.mp.k, **kwargs)
        elif self.gcnconv_parameters["type"] == "cheb":
            self.cheb_kwargs.update(kwargs)
            kwargs = self.cheb_kwargs
            mp_layer = pyg_nn.ChebConv(self.gcnconv_parameters["dim"], self.gcnconv_parameters["dim"], **kwargs)
        elif self.gcnconv_parameters["type"] == "sgcn":
            mp_layer = pyg_nn.SGConv(self.gcnconv_parameters["dim"], self.gcnconv_parameters["dim"], **kwargs)
        elif self.gcnconv_parameters["type"] == "gcn":
            mp_layer = pyg_nn.GCNConv(self.gcnconv_parameters["dim"], self.gcnconv_parameters["dim"], **kwargs)
        elif self.gcnconv_parameters["type"] == "gin":
            mp_layer = pyg_nn.GINConv(nn=torch.nn.Sequential(
                    pyg_nn.Linear(self.gcnconv_parameters["dim"], self.gcnconv_parameters["dim"]),
                    torch.nn.ReLU(),
                    pyg_nn.Linear(self.gcnconv_parameters["dim"], self.gcnconv_parameters["dim"]),
                    torch.nn.ReLU()
                ), **kwargs)
        elif self.gcnconv_parameters["type"] == "gat":
            self.gat_kwargs.update(kwargs)
            kwargs = self.gat_kwargs

            if self.nheads is None:
                raise ValueError("The number of attention heads defaults to None, please specify the number of heads explicitely in the config.")
            if i == self.gcnconv_num_layers - 1:
                # if its the last layer, use only one head (dim_out = dim_hid) and dont concat
                mp_layer = pyg_nn.GATConv(self.gcnconv_parameters["dim"] * self.nheads, self.gcnconv_parameters["dim"], heads=1, concat=False, **kwargs)
            elif i == 0:
                # if its the first layer, blow dimensionality up (dim_out = dim_hid * nheads)
                mp_layer = pyg_nn.GATConv(self.gcnconv_parameters["dim"], self.gcnconv_parameters["dim"], heads=self.nheads, **kwargs)
            elif i < self.gcnconv_num_layers - 1:
                # if its anything between the first or the last layer, keep dim_in = dim_out = dim_hid * n_heads
                mp_layer = pyg_nn.GATConv(self.gcnconv_parameters["dim"] * self.nheads, self.gcnconv_parameters["dim"] * self.nheads, heads=1, concat=True, **kwargs)
            
        elif self.gcnconv_parameters["type"] == "gcn2":
            mp_layer = pyg_nn.GCN2Conv(self.gcnconv_parameters["dim"],
                                       alpha=self.gcnconv_parameters["alpha"],
                                       theta=self.gcnconv_parameters["theta"],
                                       layer=i + 1,
                                       add_self_loops=False,
                                       **kwargs)

        elif self.gcnconv_parameters["type"] in ["rgcn", "rgat", "film", "filmtag", "rgattag", "rtag"]:

            if self.num_adjacencies == 1 and not self.config.input.force_multigraph:
                logger = setup_logger(self.config, __name__)
                logger.warning("Requested {} even though we have only 1 Adjacency. Resetting to {}".format(self.gcnconv_parameters["type"], self.reset_rgcn_to))
                self.gcnconv_parameters["type"] = self.reset_rgcn_to
                return self.get_mp_layer(i)

            if self.gcnconv_parameters["type"] in ["rgcn", "rgat", "rgattag", "rtag"]:
                self.rgcn_kwargs.update(kwargs)
                kwargs = self.rgcn_kwargs

            if self.gcnconv_parameters["type"] == "rgcn":
                mp_layer = pyg_nn.RGCNConv(self.gcnconv_parameters["dim"], self.gcnconv_parameters["dim"], self.num_adjacencies, **kwargs)
            if self.gcnconv_parameters["type"] == "rtag":
                mp_layer = layers.RTAGConv(self.gcnconv_parameters["dim"], self.gcnconv_parameters["dim"], self.num_adjacencies, **kwargs)
            elif self.gcnconv_parameters["type"] == "rgat":
                mp_layer = layers.RGATConv(self.gcnconv_parameters["dim"], self.gcnconv_parameters["dim"], self.num_adjacencies, **kwargs)
            elif self.gcnconv_parameters["type"] == "rgattag":
                mp_layer = layers.RGATTAGConv(self.gcnconv_parameters["dim"], self.gcnconv_parameters["dim"], self.num_adjacencies, **kwargs)
            elif self.gcnconv_parameters["type"] == "film":
                mp_layer = pyg_nn.FiLMConv(self.gcnconv_parameters["dim"], self.gcnconv_parameters["dim"], self.num_adjacencies, **kwargs)
            elif self.gcnconv_parameters["type"] == "filmtag":
                mp_layer = layers.FiLMTAGConv(self.gcnconv_parameters["dim"], self.gcnconv_parameters["dim"], self.num_adjacencies, **kwargs)

        else:
            raise ValueError("Could not find layer instructions for type {}".format(self.gcnconv_parameters["type"]))

        return mp_layer

    def make_pre_mp(self):
        pre_mp_list = []

        pre_mp_list.append(pyg_nn.Linear(self.input_dim, self.dim_hid))
        pre_mp_list.append(self.get_act())
        for i in range(self.npremp):
            pre_mp_list.append(pyg_nn.Linear(self.dim_hid, self.dim_hid))
            pre_mp_list.append(self.get_act())

        self.pre_mp = nn.Sequential(*pre_mp_list)

    def make_post_mp(self):
        post_mp_list = []

        if self.config.model.concat_after_mp and self.mp is not None:
            start_factor = 2
        else:
            start_factor = 1

        for i in range(self.npostmp):
            post_mp_list.append(pyg_nn.Linear(self.dim_hid * start_factor, self.dim_hid))
            post_mp_list.append(self.get_act())
            start_factor = 1

        post_mp_list.append(pyg_nn.Linear(self.dim_hid, self.dim_hid // 2))
        post_mp_list.append(self.get_act())
        post_mp_list.append(pyg_nn.Linear(self.dim_hid // 2, self.output_dim))

        self.post_mp = nn.Sequential(*post_mp_list)

    def get_mp_norm(self):
        if self.config.model.mp.normalize == "instance":
            mpnorm = pyg_nn.InstanceNorm(self.gcnconv_parameters["dim"])
        elif self.config.model.mp.normalize == "layer":
            mpnorm = pyg_nn.LayerNorm(self.gcnconv_parameters["dim"])
        elif self.config.model.mp.normalize == "graph":
            mpnorm = pyg_nn.GraphNorm(self.gcnconv_parameters["dim"])
        return mpnorm

    def regularization(self):
        return torch.Tensor((0,))

    def forward(self, x, edge_index, edge_weight=None):
        # pre message passing
        x_pre = self.pre_mp(x)

        # apply graph convolutions
        """
        x_0 = x.clone()
        for i in range(len(self.gcnconv)):
            x = self.norm[i](x)
            try:
                x = self.gcnconv[i](x, edge_index)
            except TypeError:
                x = self.gcnconv[i](x, x_0, edge_index)
            x = F.elu(x)
        """
        if self.mp is not None:
            x = self.mp(x_pre, edge_index)
            self.final_conv_acts = x

            if self.config.model.skip_mp:
                x = self.skip_mp_norm(x + x_pre)

            if self.config.model.concat_after_mp:
                x = torch.hstack((x, x_pre))
        else:
            x = x_pre

        # post message passing
        x = self.post_mp(x)

        return x

    def add_norm_layer(self, ndim):
        if self.mpnorm is not None:
            self.norm.append(self.mpnorm(ndim))

    def gcn_message_passing(self, x, edge_index, edge_weight=None):
        pass

    def gcn2_message_passing(self, x, edge_index):

        for i in range(self.gcnconv_num_layers):
            x_0 = x.clone().detach()
            x = self.gcnconv[i](x, x_0, edge_index)
            x = F.relu(x)

        return x

    def conv_activations_hook(self, grad):
        self.final_conv_grads = grad

    def input_activations_hook(self, grad):
        self.input_grads = grad


class RelationalGeneNetwork(GeneNetwork):
    def __init__(self, config, dim, num_adjacencies):
        if config.model.mp.type not in ["rgcn", "rgat", "film", "filmtag", "rgattag", "rtag"]:
            config.model.mp.type = "rgcn"
        super(RelationalGeneNetwork, self).__init__(config, dim, num_adjacencies)
        self.has_cache = False

    def forward(self, x, edge_index, cached=True):
        x = list(x.values())[0]  # we have only one node type anyway, no need for keeping in a dict
        if cached:
            if not self.has_cache:
                edges = torch.cat([edges for edges in edge_index.values()], dim=1)
                types = [[edge_type[1]] * edges.shape[-1] for edge_type, edges in edge_index.items()]
                types = [value for sublist in types for value in sublist]
                types = torch.Tensor(LabelEncoder().fit_transform(types)).to(edges.device)
                self.edge_index = SparseTensor.from_edge_index(edges.long(), types, (x.shape[0], x.shape[0]))
                self.has_cache = True
            edge_index = self.edge_index
        else:
            edges = torch.cat([edges for edges in edge_index.values()], dim=1)
            types = [[edge_type[1]] * edges.shape[-1] for edge_type, edges in edge_index.items()]
            types = [value for sublist in types for value in sublist]
            types = torch.Tensor(LabelEncoder().fit_transform(types)).to(edges.device)
            edge_index = SparseTensor.from_edge_index(edges.long(), types, (x.shape[0], x.shape[0]))

        return super().forward(x, edge_index)


class FCNN(nn.Module):
    def __init__(self, input_dim=4):
        super(FCNN, self).__init__()
        self.name = "FCNN"
        self.nhid = 50
        self.dim_hid = 70
        self.output_dim = 1

        self.list = nn.ModuleList()
        self.list.append(nn.Linear(input_dim, self.dim_hid))
        for i in range(self.nhid):
            if i < self.nhid - 1:
                self.list.append(nn.Linear(self.dim_hid, self.dim_hid))
            else:
                self.list.append(nn.Linear(self.dim_hid, self.dim_hid // 2))

        self.list.append(nn.Linear(self.dim_hid // 2, self.output_dim))

    def forward(self, x, edge_index, edge_weight=None):
        for i in range(len(self.list)):
            x = self.list[i](x)
            if i < len(self.list) - 1:
                x = F.elu(x)
        return x


class SimpleGCN(nn.Module):
    """ A simple GCN architecture for debugging """
    def __init__(self, input_dim):
        super().__init__()
        self.output_dim = 1
        self.conv1 = pyg_nn.GraphConv(input_dim, 32)
        self.conv2 = pyg_nn.GraphConv(32, self.output_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        return x


class SimpleHeteroGCN(nn.Module):
    """ A simple GCN architecture for debugging """
    def __init__(self, input_dim):
        super().__init__()
        self.output_dim = 1
        self.conv1 = pyg_nn.GCNConv(input_dim, 32)
        self.conv2 = pyg_nn.GCNConv(32, self.output_dim)

    def forward(self, x, edge_index):
        x = x[0]
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        return x
