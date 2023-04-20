"""Hyperbolic layers."""
import math
from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.modules.module import Module
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree
from speos.layers.hlinear import HypLinear
import speos.layers.hyperbolic.manifolds as manifolds
from torch_geometric.typing import PairTensor
from torch import Tensor
from torch.nn import Parameter
import torch_sparse

from torch_geometric.typing import (
    Adj,
    OptPairTensor,
    OptTensor,
    Size,
    SparseTensor,
    torch_sparse,
)
from torch_geometric.utils import (
    add_self_loops,
    is_torch_sparse_tensor,
    remove_self_loops,
    softmax,
)
from torch_geometric.utils.sparse import set_sparse_value

from torch_geometric.nn.inits import glorot, zeros

class HGATConv(MessagePassing):
    """
    Hyperbolic graph attention layer.

    It assumes that the input is already on the manifold and outputs the feature matrix on the manifold.

    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 c,
                 manifold="PoincareBall",
                 dropout=0,
                 heads=1,
                 use_bias=True,
                 edge_dim=None,
                 aggr="add",
                 local_agg=False,
                 concat: bool = True,
                 negative_slope: float = 0.2,
                 fill_value: Union[float, Tensor, str] = 'mean',
                 **kwargs):
        super().__init__(aggr=aggr, node_dim=0, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.c = c
        self.manifold = getattr(manifolds, manifold)()
        self.lin = HypLinear(in_channels, in_channels, c, dropout, manifold, use_bias)
        self.use_bias = use_bias
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        self.local_agg = True
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.edge_dim = edge_dim
        self.fill_value = fill_value
        # In case we are operating in bipartite graphs, we apply separate
        # transformations 'lin_src' and 'lin_dst' to source and target nodes:
        if isinstance(in_channels, int):
            self.lin_src = HypLinear(in_channels, heads * out_channels, use_bias=False, c=c)
            self.lin_dst = self.lin_src
        else:
            self.lin_src = HypLinear(in_channels[0], heads * out_channels, use_bias=False, c=c)
            self.lin_dst = HypLinear(in_channels[1], heads * out_channels, use_bias=False, c=c)

        # The learnable parameters to compute attention coefficients:
        self.att_src = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_dst = Parameter(torch.Tensor(1, heads, out_channels))

        if edge_dim is not None:
            self.lin_edge = HypLinear(in_channels[0], heads * out_channels, use_bias=False)
            self.att_edge = Parameter(torch.Tensor(1, heads, out_channels))
        else:
            self.lin_edge = None
            self.register_parameter('att_edge', None)

        if use_bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif use_bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.lin_src.reset_parameters()
        self.lin_dst.reset_parameters()
        if self.lin_edge is not None:
            self.lin_edge.reset_parameters()
        glorot(self.att_src)
        glorot(self.att_dst)
        glorot(self.att_edge)
        zeros(self.bias)

    def forward(self, x, edge_index: Adj, edge_attr: OptTensor = None, size: Size = None,
                return_attention_weights=None):

        x = self.lin(x)  # need a linear transformation of features first


        H, C = self.heads, self.out_channels

        # We first transform the input node features. If a tuple is passed, we
        # transform source and target node features via separate weights:

        # map the features to tangent space at origin before commputing the attention
        if isinstance(x, Tensor):
            assert x.dim() == 2, "Static graphs not supported in 'GATConv'"
            x_src = x_dst = self.lin_src(self.manifold.logmap0(x, self.c)).view(-1, H, C)
        else:  # Tuple of source and target node features:
            x_src, x_dst = x
            assert x_src.dim() == 2, "Static graphs not supported in 'GATConv'"
            x_src = self.lin_src(self.manifold.logmap0(x_src), self.c).view(-1, H, C)
            if x_dst is not None:
                x_dst = self.lin_dst(self.manifold.logmap0(x_dst), self.c).view(-1, H, C)

        x = (x_src, x_dst)

        # Next, we compute node-level attention coefficients, both for source
        # and target nodes (if present):
        alpha_src = (x_src * self.att_src).sum(dim=-1)
        alpha_dst = None if x_dst is None else (x_dst * self.att_dst).sum(-1)
        alpha = (alpha_src, alpha_dst)

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                # We only want to add self-loops for nodes that appear both as
                # source and target nodes:
                num_nodes = x_src.size(0)
                if x_dst is not None:
                    num_nodes = min(num_nodes, x_dst.size(0))
                num_nodes = min(size) if size is not None else num_nodes
                edge_index, edge_attr = remove_self_loops(
                    edge_index, edge_attr)
                edge_index, edge_attr = add_self_loops(
                    edge_index, edge_attr, fill_value=self.fill_value,
                    num_nodes=num_nodes)
            elif isinstance(edge_index, SparseTensor):
                if self.edge_dim is None:
                    edge_index = torch_sparse.set_diag(edge_index)
                else:
                    raise NotImplementedError(
                        "The usage of 'edge_attr' and 'add_self_loops' "
                        "simultaneously is currently not yet supported for "
                        "'edge_index' in a 'SparseTensor' form")

        # edge_updater_type: (alpha: OptPairTensor, edge_attr: OptTensor)
        alpha = self.edge_updater(edge_index, alpha=alpha, edge_attr=edge_attr)

        # Step 3: Project Feature Matrix into Tangent Space.
        if not self.local_agg:
            x = self.manifold.logmap0(x, c=self.c)

        # Step 4: Start propagating messages.
        out = self.propagate(edge_index, x=x, alpha=alpha)

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        # Step 5: Project Feature Map back in Hyperbolic Space.
        if self.local_agg:
            out = self.manifold.proj(self.manifold.expmap(out, x_dst, c=self.c), c=self.c)
        else:
            out = self.manifold.proj(self.manifold.expmap0(out, c=self.c), c=self.c)

        # Step 6: Apply a final bias vector.
        if self.use_bias:
            bias = self.manifold.proj_tan0(self.bias.view(1, -1), self.c)
            hyp_bias = self.manifold.expmap0(bias, self.c)
            hyp_bias = self.manifold.proj(hyp_bias, self.c)
            out = self.manifold.mobius_add(out, hyp_bias, c=self.c)
            out = self.manifold.proj(out, self.c)

        if isinstance(return_attention_weights, bool):
            if isinstance(edge_index, Tensor):
                if is_torch_sparse_tensor(edge_index):
                    # TODO TorchScript requires to return a tuple
                    adj = set_sparse_value(edge_index, alpha)
                    return out, (adj, alpha)
                else:
                    return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out

        return out

    def message(self, x_i: Tensor,  x_j: Tensor, alpha: Tensor) -> Tensor:
        # x_j has shape [E, out_channels]
        
        if self.local_agg:
            # use features projected into local tangent space of center node x_i
            x_j = self.manifold.proj(self.manifold.logmap(x_j, x_i, c=self.c), c=self.c)

        return alpha.unsqueeze(-1) * x_j

    
    def edge_update(self, alpha_j: Tensor, alpha_i: OptTensor,
                    edge_attr: OptTensor, index: Tensor, ptr: OptTensor,
                    size_i: Optional[int]) -> Tensor:
        # Given edge-level attention coefficients for source and target nodes,
        # we simply need to sum them up to "emulate" concatenation:
        alpha = alpha_j if alpha_i is None else alpha_j + alpha_i
        if index.numel() == 0:
            return alpha
        if edge_attr is not None and self.lin_edge is not None:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.view(-1, 1)
            edge_attr = self.lin_edge(edge_attr)
            edge_attr = edge_attr.view(-1, self.heads, self.out_channels)
            alpha_edge = (edge_attr * self.att_edge).sum(dim=-1)
            alpha = alpha + alpha_edge

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return alpha

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')
