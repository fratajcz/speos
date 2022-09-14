import pandas as pd
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from captum.attr import DeepLift
import torch
from torch import Tensor
import torch_geometric as pyg
from torch_geometric.typing import PairTensor
from torch_sparse import SparseTensor, masked_select_nnz
from sklearn.preprocessing import LabelEncoder
from typing import Optional


class InputExplainer:
    def __init__(self, dataset, model, feature_names, config):
        self.config = config
        self.num_nodes, self.num_features = dataset.x.shape
        self.data = dataset
        self.labels = dataset.y.detach().cpu().numpy().astype(np.bool8)
        self.model = model
        self.dl = DeepLift(self.model.architectures[0])
        self.feature_names = feature_names

    def explain(self, plot=True):
        attribution = self.deepLift(self.data)

        if plot:
            self.plot = self.plot_explanations(
                attribution.abs().numpy(), sd=False)
        else:
            self.plot = None

        return attribution, self.plot

    def explain_positive_unknowns(self, plot=True):

        attribution = self.deepLift(self.data)

        if plot:
            self.plot = self.plot_explanations(
                attribution.abs().numpy(), sd=False)
        else:
            self.plot = None

        mask = torch.ones_like(self.data.test_mask, dtype=torch.bool)

        logits, _ = self.model.step(self.data, mask, eval_flag=True)
        predictions = F.sigmoid(logits) > self.config.inference.cutoff_value
        mask = torch.logical_and(torch.BoolTensor(
            1 - self.labels), predictions.squeeze())
        positive_predicted_unknowns = attribution[mask]

        return positive_predicted_unknowns, self.plot

    def deepLift(self, input, target=0, baselines=None, plot=True):
        input.x.requires_grad = True

        if baselines is None:
            baselines = torch.zeros_like(input.x)
            pvals = torch.BoolTensor([1 if value.startswith(
                "P ") else 0 for value in self.feature_names])
            maxes, _ = torch.max(input.x, axis=0)
            mins, _ = torch.min(input.x, axis=0)
            baselines[:, pvals] = maxes.repeat(
                baselines.shape[0], 1).type(baselines.type())[:, pvals]
            baselines[:, ~pvals] = mins.repeat(
                baselines.shape[0], 1).type(baselines.type())[:, ~pvals]

        attribution = self.dl.attribute(
            input.x, baselines=baselines, target=target, additional_forward_args=input.edge_index)

        return attribution.detach().cpu()

    def saliency_map(self, input_grads):
        saliency_matrix = F.elu(input_grads.t()).cpu().detach().numpy()
        return saliency_matrix

    def plot_explanations(self, saliency_matrix, sd=True):

        total_std = pd.DataFrame(
            np.std(saliency_matrix, axis=0), columns=["Std"])
        total_mean = pd.DataFrame(
            np.mean(saliency_matrix, axis=0), columns=["Mean"])

        pos = saliency_matrix[self.labels]
        pos_std = pd.DataFrame(np.std(pos, axis=0), columns=["Std"])
        pos_mean = pd.DataFrame(np.mean(pos, axis=0), columns=["Mean"])
        neg = saliency_matrix[1 - self.labels]
        neg_std = pd.DataFrame(np.std(neg, axis=0), columns=["Std"])
        neg_mean = pd.DataFrame(np.mean(neg, axis=0), columns=["Mean"])

        figures = []

        for i, (name, std, mean) in enumerate(zip(["Total", "Positives", "Negatives"], [total_std, pos_std, neg_std], [total_mean, pos_mean, neg_mean])):

            joined = std.join(mean)

            fig, ax = plt.subplots(figsize=(5, 20))
            bar_pos = range(len(joined.index))
            if sd:
                ax.barh(bar_pos, joined["Mean"].tolist(), xerr=joined["Std"].tolist(
                ), align='center', alpha=0.5, ecolor='black')
            else:
                ax.barh(bar_pos, joined["Mean"].tolist(),
                        align='center', alpha=0.5)
            ax.set_yticks(bar_pos)
            if i == 0:
                ax.set_yticklabels(self.feature_names)
            else:
                ax.set_yticklabels([])
            ax.set_title(name)
            ax.yaxis.grid(True)
            plt.tight_layout()
            plt.xlim((0, 25))
            fig.canvas.draw()
            w, h = fig.canvas.get_width_height()
            data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8)
            data = data.reshape((h, w, 3))
            data = np.transpose(data, (2, 0, 1))
            plt.close()

            figures.append(data)

        return figures


class MessagePassingExplainer:
    """This class should hold all necessary functions to explain the message passing.
        This includes reasoning on the forward pass (FiLM) or the backpropagation (LRP)"""
    def __init__(self, model, data, adjacency_names, config):
        self.model = model
        self.data = data
        self.adjacency_names = ["Identity"] + adjacency_names
        self.config = config
        self.layer = 0
        self.adjacency = 0

    def __next__(self):
 
        # Stop iteration if limit is reached
        if self.adjacency + 1 == len(self.adjacency_names) and self.layer + 1 == self.config.model.mp.n_layers:
            raise StopIteration

        adjacency_name, null_length, length = self.inspect_film()

        if self.adjacency + 1 < len(self.adjacency_names):
            self.adjacency += 1
        else:
            self.layer += 1
            self.adjacency = 0

        yield adjacency_name, null_length, length
 

    def inspect_film(self, testing=False):
        if not self.config.model.mp.type == "film":
            return

        latent_node_features = self.get_latent_features()

        if isinstance(latent_node_features, Tensor):
            latent_node_features: PairTensor = (latent_node_features, latent_node_features)

        mp_layers = self.model.get_mp_layers()

        edges = torch.cat([edges['edge_index'] for edges in self.data.edge_stores], dim=1)
        types = [[edge_type[1]] * edges['edge_index'].shape[-1] for edge_type, edges in zip(self.data.edge_types, self.data.edge_stores)]
        types = [value for sublist in types for value in sublist]
        edge_types = torch.Tensor(LabelEncoder().fit_transform(types)).to(self.data.x.device)
        if testing:
            edge_types = edge_types[-100:]
            edges = edges[:, -100:]

        adjacency_name = self.adjacency_names[self.adjacency]
        
        if self.adjacency == 0:
            lin = mp_layers[self.layer].lin_skip
        else:
            lin = mp_layers[self.layer].lins[self.adjacency - 1]

        beta, gamma = self.get_beta_gamma(latent_node_features, mp_layers[self.layer], self.adjacency)

        if self.adjacency == 0:
            messages = self.get_identity_messages(lin(latent_node_features[0]), beta, gamma)
            null_messages = self.get_identity_messages(lin(latent_node_features[0]), torch.zeros_like(beta), torch.ones_like(gamma))
        else:
            messages = self.get_messages(lin(latent_node_features[0]), edges, edge_types, beta, gamma, self.adjacency)
            null_messages = self.get_messages(lin(latent_node_features[0]), edges, edge_types, torch.zeros_like(beta), torch.ones_like(gamma), self.adjacency)

        length = messages.norm(p=2, dim=1)
        null_length = null_messages.norm(p=2, dim=1)

        return adjacency_name, null_length, length
                
        # TODO: Do something with messages such as plotting or saving        
        # ideas: get change of direction (cosine sim.) and change of magnitude (norm)

    def get_beta_gamma(self, x, layer, i):
        """extracts beta and gamma for input x in layer for edge type i. 
           This is basically a disambiguation of a part of: 
           https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/conv/film_conv.html#FiLMConv"""

        if isinstance(x, Tensor):
            x: PairTensor = (x, x)

        if i == 0:
            beta, gamma = layer.film_skip(x[1]).split(layer.out_channels, dim=-1)
        else:
            beta, gamma = layer.films[i - 1](x[1]).split(layer.out_channels, dim=-1)

        return beta, gamma

    def get_identity_messages(self, x, beta, gamma):
        edge_index = torch.LongTensor((range(x.shape[0]), range(x.shape[0])))
        return self.calculate_messages(edge_index, x=x, beta=beta, gamma=gamma)


    def get_messages(self, x, edge_index, edge_type, beta, gamma, i):
        """ This method returns all messages for edge type i """
        if isinstance(edge_index, SparseTensor):
            edge_type = edge_index.storage.value()
            assert edge_type is not None
            mask = edge_type == i - 1
            messages = self.calculate_messages(
                masked_select_nnz(edge_index, mask, layout='coo'),
                x=x, beta=beta, gamma=gamma)
        else:
            assert edge_type is not None
            mask = edge_type == i - 1
            messages = self.calculate_messages(edge_index[:, mask], x=x, beta=beta, gamma=gamma)

        return messages

    def calculate_messages(self, edge_index, x, beta, gamma):
        senders = x[edge_index[0, :], :]
        receiver_gamma = gamma[edge_index[1, :], :]
        receiver_beta = beta[edge_index[1, :], :]
        return receiver_gamma * senders + receiver_beta

    def get_latent_features(self):
        return self.model.architectures[0].pre_mp(self.data.x)


class Explainer(pyg.nn.models.Explainer):

    def visualize_subgraph(self, node_idx: Optional[int], edge_index: Tensor,
                           edge_mask: Tensor, y: Optional[Tensor] = None,
                           threshold: Optional[int] = None,
                           edge_y: Optional[Tensor] = None,
                           node_alpha: Optional[Tensor] = None, seed: int = 10,
                           colormap: Optional[str] = "viridis",
                           **kwargs):
        r"""Visualizes the subgraph given an edge mask :attr:`edge_mask`.

        Args:
            node_idx (int): The node id to explain.
                Set to :obj:`None` to explain a graph.
            edge_index (LongTensor): The edge indices.
            edge_mask (Tensor): The edge mask.
            y (Tensor, optional): The ground-truth node-prediction labels used
                as node colorings. All nodes will have the same color
                if :attr:`node_idx` is :obj:`-1`.(default: :obj:`None`).
            threshold (float, optional): Sets a threshold for visualizing
                important edges. If set to :obj:`None`, will visualize all
                edges with transparancy indicating the importance of edges.
                (default: :obj:`None`)
            edge_y (Tensor, optional): The edge labels used as edge colorings.
            node_alpha (Tensor, optional): Tensor of floats (0 - 1) indicating
                transparency of each node.
            seed (int, optional): Random seed of the :obj:`networkx` node
                placement algorithm. (default: :obj:`10`)
            **kwargs (optional): Additional arguments passed to
                :func:`nx.draw`.

        :rtype: :class:`matplotlib.axes.Axes`, :class:`networkx.DiGraph`
        """
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        import networkx as nx
        from torch_geometric.utils import k_hop_subgraph, to_networkx
        from inspect import signature
        from math import sqrt
        from matplotlib.colors import ListedColormap
        from torch_geometric.data import Data

        assert edge_mask.size(0) == edge_index.size(1)

        if node_idx is None or node_idx < 0:
            hard_edge_mask = torch.BoolTensor([True] * edge_index.size(1),
                                              device=edge_mask.device)
            subset = torch.arange(edge_index.max().item() + 1,
                                  device=edge_index.device)
            y = None

        else:
            # Only operate on a k-hop subgraph around `node_idx`.
            subset, edge_index, _, hard_edge_mask = k_hop_subgraph(
                node_idx, self.num_hops, edge_index, relabel_nodes=True,
                num_nodes=None, flow=self._flow())

        edge_mask = edge_mask[hard_edge_mask]

        if threshold is not None:
            edge_mask = (edge_mask >= threshold).to(torch.float)

        if y is None:
            y = torch.zeros(edge_index.max().item() + 1,
                            device=edge_index.device)
        else:
            y = y[subset].to(torch.float) / y.max().item()

        if edge_y is None:
            if colormap is not None:
                cmap = mpl.cm.get_cmap(colormap)
                my_cmap = cmap(np.arange(cmap.N))
                half_the_numbers = int(cmap.N / 2)
                my_cmap[:, -1] = np.concatenate((np.linspace(0.1, 1, half_the_numbers), np.ones((int(cmap.N - half_the_numbers),))))
                my_cmap = ListedColormap(my_cmap)
                norm = mpl.colors.Normalize(vmin=0, vmax=1)
                mapper = mpl.cm.ScalarMappable(norm=norm, cmap=my_cmap)
                edge_color = [mapper.to_rgba(value) for value in edge_mask.detach().cpu().numpy()]
            else:
                edge_color = ['black'] * edge_index.size(1)
        else:
            colors = list(plt.rcParams['axes.prop_cycle'])
            edge_color = [
                colors[i % len(colors)]['color']
                for i in edge_y[hard_edge_mask]
            ]

        data = Data(edge_index=edge_index, att=edge_mask,
                    edge_color=edge_color, y=y, num_nodes=y.size(0)).to('cpu')
        G = to_networkx(data, node_attrs=['y'],
                        edge_attrs=['att', 'edge_color'])
        mapping = {k: i for k, i in enumerate(subset.tolist())}
        G = nx.relabel_nodes(G, mapping)



        node_args = set(signature(nx.draw_networkx_nodes).parameters.keys())
        node_kwargs = {k: v for k, v in kwargs.items() if k in node_args}
        node_kwargs['node_size'] = kwargs.get('node_size') or 800
        node_kwargs['cmap'] = colormap
        
        pos_cmap = mpl.cm.get_cmap("Reds")
        pos_norm = mpl.colors.Normalize(vmin=0, vmax=1)
        pos_mapper = mpl.cm.ScalarMappable(norm=pos_norm, cmap=pos_cmap)

        node_colors = np.asarray([mapper.to_rgba(value) if y == 0 else pos_mapper.to_rgba(value) for y, value in zip(y, node_alpha[subset].detach().cpu().numpy())])

        label_args = set(signature(nx.draw_networkx_labels).parameters.keys())
        label_kwargs = {k: v for k, v in kwargs.items() if k in label_args}
        label_kwargs['font_size'] = kwargs.get('font_size') or 10

        pos = nx.spring_layout(G, seed=seed)
        pos_positives = {key: value for y, (key, value) in zip(y, pos.items()) if y == 1}
        pos_unknowns = {key: value for y, (key, value) in zip(y, pos.items()) if y == 0}
        ax = plt.gca()
        for source, target, data in G.edges(data=True):
            ax.annotate(
                '', xy=pos[target], xycoords='data', xytext=pos[source],
                textcoords='data', arrowprops=dict(
                    arrowstyle="->",
                    #alpha=min(max(data['att'], 0.05) * 2, 1),
                    color=data['edge_color'],
                    shrinkA=sqrt(node_kwargs['node_size']) / 2.0,
                    shrinkB=sqrt(node_kwargs['node_size']) / 2.0,
                    connectionstyle="arc3,rad=0.1",
                ))

        if node_alpha is None:
            nx.draw_networkx_nodes(G, pos[~y.numpy()], node_color=node_colors[~y.numpy()],
                                   **node_kwargs)
            nx.draw_networkx_nodes(G, pos[y.numpy()], node_color=node_colors[y.numpy()],
                                   node_shape="^", **node_kwargs)
        else:
            node_alpha_subset = node_alpha[subset]
            assert ((node_alpha_subset >= 0) & (node_alpha_subset <= 1)).all()
            #node_alpha = np.fmin(np.fmax(node_alpha_subset, 0.1) * 2, 1)
            positives = y.numpy().astype(np.bool8)
            not_positives = ~positives
            nx.draw_networkx_nodes(G, pos_unknowns, node_color=np.asarray(node_colors)[not_positives], nodelist=list(pos_unknowns.keys()), **node_kwargs)
            nx.draw_networkx_nodes(G, pos_positives, node_color=np.asarray(node_colors)[positives], node_shape="^", nodelist=list(pos_positives.keys()), edgecolors="#333333", **node_kwargs)
        
        nx.draw_networkx_labels(G, pos, **label_kwargs)
        plt.rc('axes', labelsize=16)    # fontsize of the x and y labels
        plt.rc('axes', labelsize=16)    # fontsize of the x and y labels
        plt.rc('xtick', labelsize=14)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=14)    # fontsize of the tick labels
        cbar = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=my_cmap),
                            ax=ax,
                            label="Relative Importance (Unknowns)",
                            ticks=[0, 0.5, 1],
                            pad=0,
                            location="top")
        cbar.ax.set_xticklabels(["Never", "Sometimes", "Always"])
        cbar = plt.colorbar(mpl.cm.ScalarMappable(norm=pos_norm, cmap=pos_cmap),
                            ax=ax,
                            label="Relative Importance (Positives)",
                            ticks=[0, 0.5, 1],
                            pad=0,
                            location="top")
        cbar.ax.set_xticklabels(["Never", "Sometimes", "Always"])
        from matplotlib.lines import Line2D
        legend_elements = [Line2D([0], [0], marker='o', color='w', label='Unknowns', markeredgecolor='black', linestyle=None,
                           markerfacecolor='w', markersize=15),
                           Line2D([0], [0], marker='^', color='w', label='Positives', markeredgecolor='black', linestyle=None,
                           markerfacecolor='w', markersize=15)]
        ax.legend(handles=legend_elements)

        return ax, G
