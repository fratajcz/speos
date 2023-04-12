"""Hyperbolic layers."""
import torch.nn as nn
from torch.nn.modules.module import Module
import speos.layers.hyperbolic.manifolds as manifolds

class HypAct(Module):
    """
    Hyperbolic activation layer.
    """

    def __init__(self, manifold, c_in=None, c_out=None, act, first=False, last=False):
        super(HypAct, self).__init__()
        if not first:
            assert c_in is not None 
        if not last:
            assert c_out is not None 
        self.manifold = getattr(manifolds, manifold)()
        self.first = first
        self.last = last
        self.c_in = c_in
        self.c_out = c_out
        self.act = act if not first else nn.Identity()

    def forward(self, x):
        if not self.first:
            xt = self.act(self.manifold.logmap0(x, c=self.c_in))
        xt = self.manifold.proj_tan0(xt, c=self.c_out)
        if self.last:
            return xt
        else:
            return self.manifold.proj(self.manifold.expmap0(xt, c=self.c_out), c=self.c_out)

    def extra_repr(self):
        return 'c_in={}, c_out={}'.format(
            self.c_in, self.c_out
        )
