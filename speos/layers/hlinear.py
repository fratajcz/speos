import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.modules.module import Module
import speos.layers.hyperbolic.manifolds as manifolds
class HypLinear(nn.Module):

    """
    Hyperbolic linear layer.
    """

    def __init__(self, in_channels, out_channels, c, dropout=0, manifold="PoincareBall", use_bias=True):
        super(HypLinear, self).__init__()
        self.manifold = getattr(manifolds, manifold)()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.c = c
        self.dropout = dropout
        self.use_bias = use_bias
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels))
        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform_(self.weight, gain=math.sqrt(2))
        init.constant_(self.bias, 0)

    def forward(self, x):
        drop_weight = F.dropout(self.weight, self.dropout, training=self.training)
        mv = self.manifold.mobius_matvec(drop_weight, x, self.c)
        #mv = self.manifold.expmap0(torch.mm(self.manifold.logmap0(x, self.c), drop_weight.T), self.c)
        res = self.manifold.proj(mv, self.c)
        if self.use_bias:
            bias = self.manifold.proj_tan0(self.bias.view(1, -1), self.c)
            hyp_bias = self.manifold.expmap0(bias, self.c)
            hyp_bias = self.manifold.proj(hyp_bias, self.c)
            res = self.manifold.mobius_add(res, hyp_bias, c=self.c)
            res = self.manifold.proj(res, self.c)
        return res

    def extra_repr(self):
        return 'in_channels={}, out_channels={}, c={}'.format(
            self.in_channels, self.out_channels, self.c
        )
