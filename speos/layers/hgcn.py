"""Hyperbolic layers."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.modules.module import Module
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from speos.layers.hlinear import HypLinear
import speos.layers.hyperbolic.manifolds as manifolds
from torch_geometric.typing import PairTensor
from torch import Tensor

class HGCNConv(MessagePassing):
    """
    Hyperbolic graph convolution layer.

    It assumes that the input is already on the manifold and outputs the feature matrix on the manifold.

    Implementation based on https://github.com/HazyResearch/hgcn/blob/master/layers/hyp_layers.py 
    but implemented for the MessagePassing framework using the GCN template from https://pytorch-geometric.readthedocs.io/en/latest/tutorial/create_gnn.html#implementing-the-gcn-layer
    """

    def __init__(self, in_channels, out_channels, c, manifold="PoincareBall", dropout=0, use_bias=True, aggr="add", normalize=False, use_att=False, local_agg=False):
        super(HGCNConv, self).__init__()
        super().__init__(aggr=aggr)
        self.use_att = use_att
        self.c = c
        self.manifold = getattr(manifolds, manifold)()
        self.lin = HypLinear(in_channels, out_channels, c, dropout, manifold, use_bias)
        if self.use_att:
            self.attention_lin = nn.Linear(out_channels * 2, 1)

        self.use_bias = use_bias
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        self.reset_parameters()
        self.normalize = normalize
        self.local_agg = True

    def forward(self, x, edge_index):
        # Step 1: Add self-loops to the adjacency matrix.
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 2: Linearly transform node feature matrix.
        x = self.lin(x)

        # Step 3: Compute normalization.
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Step 3: Project Feature Matrix into Tangent Space.
        if not self.local_agg:
            x = self.manifold.logmap0(x, c=self.c)
        
        if isinstance(x, Tensor):
            x: PairTensor = (x, x)

        # Step 4: Start propagating messages.
        out = self.propagate(edge_index, x=x[0], x_j=x[1], norm=norm)

        # Step 5: Project Feature Map back in Hyperbolic Space.
        if self.local_agg:
            out = self.manifold.proj(self.manifold.expmap(out, x[0], c=self.c), c=self.c)
        else:
            out = self.manifold.proj(self.manifold.expmap0(out, c=self.c), c=self.c)

        # Step 6: Apply a final bias vector.
        if self.use_bias:
            bias = self.manifold.proj_tan0(self.bias.view(1, -1), self.c)
            hyp_bias = self.manifold.expmap0(bias, self.c)
            hyp_bias = self.manifold.proj(hyp_bias, self.c)
            out = self.manifold.mobius_add(out, hyp_bias, c=self.c)
            out = self.manifold.proj(out, self.c)

        return out

    def message(self, x_i, x_j, norm):
        """ If we use local aggregation, x_i and x_j are still on the manifold, else they are in tangent space of origin """
        # x_j has shape [E, out_channels]
        
        if self.local_agg:
            # use features projected into local tangent space of center node x_i
            x_j = self.manifold.proj(self.manifold.logmap(x_j, x_i, c=self.c), c=self.c)
        if self.use_att:
            if self.local_agg:
                x_i_o = self.manifold.logmap0(x_i, c=self.c)
                x_j_o = self.manifold.logmap0(x_j, c=self.c)
                alpha = F.softmax(self.attention_lin(torch.cat((x_i_o, x_j_o), dim=-1)).squeeze(), dim=0)
            else:
                alpha = F.softmax(self.attention_lin(torch.cat((x_i, x_j))))
            x_j = alpha.view(-1, 1) * x_j
        if self.normalize:
            # Normalize node features.
            norm.view(-1, 1) * x_j
        return x_j