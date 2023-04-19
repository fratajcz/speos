"""Hyperbolic layers."""
import torch.nn as nn
from torch.nn.modules.module import Module
import speos.layers.hyperbolic.manifolds as manifolds

class HypAct(Module):
    """
    Hyperbolic activation layer.
    """

    def __init__(self, act, c_in=None, c_out=None, manifold="PoincareBall"):
        super(HypAct, self).__init__()
        self.manifold = getattr(manifolds, manifold)()
        self.act = act
        self.c_in = c_in
        self.c_out = c_out

    def forward(self, x):
        x = self.act(self.manifold.logmap0(x, c=self.c_in))
        x = self.manifold.proj_tan0(x, c=self.c_out)
        return self.manifold.proj(self.manifold.expmap0(x, c=self.c_out), c=self.c_out)

    def extra_repr(self):
        return 'c_in={}, c_out={}'.format(
            self.c_in, self.c_out
        )

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)