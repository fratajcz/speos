from speos.layers.rgat import RGATConv
from torch_geometric.typing import Adj, OptTensor
from torch_sparse import SparseTensor
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import zeros
from torch_geometric.nn.conv.gcn_conv import gcn_norm
import torch

from typing import Optional, Tuple, Union
from torch import Tensor


class RGATTAGConv(RGATConv):
    """Integrating the receptive field idea from from the`"Topology Adaptive Graph Convolutional Networks"
    <https://arxiv.org/abs/1710.10370> into RGATs"""

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 num_relations: int,
                 K: int = 3,
                 num_bases: Optional[int] = None,
                 num_blocks: Optional[int] = None,
                 aggr_first: bool = True,
                 mod: Optional[str] = None,
                 attention_mechanism: str = "across-relation",
                 attention_mode: str = "additive-self-attention",
                 heads: int = 1,
                 dim: int = 1,
                 concat: bool = True,
                 negative_slope: float = 0.2,
                 dropout: float = 0.0,
                 edge_dim: Optional[int] = None,
                 bias: bool = True,
                 normalize: bool = False,
                 **kwargs):

        super(RGATTAGConv, self).__init__(in_channels=in_channels,
                                          out_channels=out_channels,
                                          num_relations=num_relations,
                                          num_bases=num_bases,
                                          num_blocks=num_blocks,
                                          mod=mod,
                                          attention_mechanism=attention_mechanism,
                                          attention_mode=attention_mode,
                                          heads=heads,
                                          dim=dim,
                                          concat=concat,
                                          negative_slope=negative_slope,
                                          dropout=dropout,
                                          edge_dim=edge_dim,
                                          bias=bias,
                                          **kwargs)

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
        
        self.reset_parameters()

        def reset_parameters(self):
            for lin in self.lins:
                lin.reset_parameters()
            zeros(self.tag_bias)
            super().reset_parameters()

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
                    edge_index[:, mask] = gcn_norm(  # yapf: disable
                        edge_index[:, mask], edge_weight, x.size(self.node_dim),
                        add_self_loops=False, dtype=x.dtype)

        out = self.lins[0](x)
        for lin in self.lins[1:]:
            # propagate_type: (x: Tensor, edge_weight: OptTensor)
            #x = self.propagate(edge_index, x=x, edge_weight=edge_weight,
            #                size=None)
            x = super().forward(x, edge_index, edge_type)
            out += lin.forward(x)

        if self.tag_bias is not None:
            out += self.tag_bias

        return out
