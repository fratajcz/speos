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

class HGCNConv(MessagePassing):
    """
    Hyperbolic graph convolution layer.

    Implementation based on https://github.com/HazyResearch/hgcn/blob/master/layers/hyp_layers.py 
    but implemented for the MessagePassing framework using the GCN template from https://pytorch-geometric.readthedocs.io/en/latest/tutorial/create_gnn.html#implementing-the-gcn-layer
    """

    def __init__(self, in_channels, out_channels, c_in, c_out, manifold="PoincareBall", dropout=0, use_bias=True, aggr="add"):
        super(HGCNConv, self).__init__()
        super().__init__(aggr=aggr)
        self.c = c_in
        self.manifold = getattr(manifolds, manifold)()
        self.lin = HypLinear(manifold, in_channels, out_channels, c_in, dropout, use_bias)
        self.use_bias = use_bias
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        self.reset_parameters()

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
        x = self.manifold.logmap0(x, c=self.c)

        # Step 4: Start propagating messages.
        out = self.propagate(edge_index, x=x, norm=norm)

        # Step 5: Project Feature Map back in Hyperbolic Space.
        out = self.manifold.proj(self.manifold.expmap0(out, c=self.c), c=self.c)

        # Step 6: Apply a final bias vector.
        if self.use_bias:
            bias = self.manifold.proj_tan0(self.bias.view(1, -1), self.c)
            hyp_bias = self.manifold.expmap0(bias, self.c)
            hyp_bias = self.manifold.proj(hyp_bias, self.c)
            out = self.manifold.mobius_add(out, hyp_bias, c=self.c)
            out = self.manifold.proj(out, self.c)

        return out

    def message(self, x_j, norm):
        # x_j has shape [E, out_channels]

        # Step 4: Normalize node features.
        return norm.view(-1, 1) * x_j