from torch_geometric.nn import RGCNConv
from torch_geometric.typing import Adj, OptTensor
from torch_sparse import SparseTensor
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import zeros
from torch_geometric.nn.conv.gcn_conv import gcn_norm
import torch

from typing import Optional, Tuple, Union
from torch import Tensor


class RTAGConv(RGCNConv):
    """Integrating the receptive field idea from from the`"Topology Adaptive Graph Convolutional Networks"
    <https://arxiv.org/abs/1710.10370> into RGCNs"""

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 num_relations: int,
                 K: int = 3,
                 num_bases: Optional[int] = None,
                 num_blocks: Optional[int] = None,
                 aggr: str = 'mean',
                 aggr_first: bool = True,
                 root_weight: bool = True,
                 bias: bool = True,
                 normalize: bool = False,
                 **kwargs):

        super(RTAGConv, self).__init__(in_channels, out_channels=out_channels, num_relations=num_relations, num_bases=num_bases, num_blocks=num_blocks, aggr=aggr,
                                       root_weight=root_weight, bias=bias, **kwargs)

        self.aggr_first = aggr_first

        self.K = K
        self.normalize = normalize

        self.lins = torch.nn.ModuleList([
            Linear(in_channels, out_channels, bias=False) for _ in range(K + 1)
        ])

        if bias:
            self.tag_bias = torch.nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('tag_bias', None)

        self.reset_new_parameters()

    def reset_new_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        zeros(self.tag_bias)

    def forward(self, x: Union[OptTensor, Tuple[OptTensor, Tensor]],
                edge_index: Adj, edge_type: OptTensor = None, edge_weight: OptTensor = None):

        if self.normalize:
            for i in range(self.num_relations):
                if isinstance(edge_index, Tensor):
                    mask = edge_type == i
                    edge_index[:, mask], edge_weight = gcn_norm(  # yapf: disable
                        edge_index[:, mask], edge_weight, x.size(self.node_dim),
                        improved=False, add_self_loops=False, dtype=x.dtype)
                elif isinstance(edge_index, SparseTensor):
                    edge_type = edge_index.storage.value()
                    assert edge_type is not None
                    mask = edge_type == i
                    edge_index_coo = torch.vstack(edge_index.coo()[:2])
                    edge_index_coo[:, mask], edge_weight = gcn_norm(  # yapf: disable
                        edge_index_coo[:, mask], edge_weight, x.size(self.node_dim),
                        improved=False, add_self_loops=False, dtype=x.dtype)

                    edge_index = SparseTensor.from_edge_index(edge_index_coo, edge_type)

        out = self.lins[0](x)
        for i, lin in enumerate(self.lins[1:]):
            x = super().forward(x, edge_index, edge_type)
            out += x

        if self.tag_bias is not None:
            out += self.tag_bias

        return out
