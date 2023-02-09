from torch_geometric.nn import FiLMConv
from torch_geometric.typing import Adj, OptTensor, PairTensor
from torch_sparse import SparseTensor, masked_select_nnz
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import zeros
from torch_geometric.nn.conv.gcn_conv import gcn_norm
import torch
from torch.nn import ModuleList, ReLU
from typing import Optional, Tuple, Union, Callable
from torch import Tensor
import copy


class MLPFiLM(FiLMConv):
    """Adapting FiLM Layer so it computes beta and gamma also based on sender features"""
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 num_relations: int,
                 nn: Optional[Callable] = None,
                 act: Optional[Callable] = ReLU(),
                 aggr: str = 'mean',
                 aggr_first: bool = True,
                 bias: bool = True,
                 normalize: bool = False,
                 **kwargs):

        super(MLPFiLM, self).__init__(in_channels, out_channels=out_channels, num_relations=num_relations, nn=nn, act=act, aggr=aggr, **kwargs)

        self.films = ModuleList()
        for _ in range(num_relations):
            if nn is None:
                film = Linear(2 * in_channels, 2 * out_channels)
            else:
                film = copy.deepcopy(nn)
            self.films.append(film)

        if nn is None:
            self.film_skip = Linear(2 * in_channels, 2 * self.out_channels,
                                    bias=False)
        else:
            self.film_skip = copy.deepcopy(nn)

        self.reset_parameters()

    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj,
                edge_type: OptTensor = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: PairTensor = (x, x)

        beta, gamma = self.film_skip(torch.hstack((x[1], x[1]))).split(self.out_channels, dim=-1)
        out = gamma * self.lin_skip(x[1]) + beta
        if self.act is not None:
            out = self.act(out)

        # propagate_type: (x: Tensor, beta: Tensor, gamma: Tensor)
        if self.num_relations <= 1:
            beta, gamma = self.films[0](x[1]).split(self.out_channels, dim=-1)
            out = out + self.propagate(edge_index, x=self.lins[0](x[0]),
                                       beta=beta, gamma=gamma, size=None)
        else:
            for i, (lin, film) in enumerate(zip(self.lins, self.films)):
                if isinstance(edge_index, SparseTensor):
                    edge_type = edge_index.storage.value()
                    assert edge_type is not None
                    mask = edge_type == i
                    out = out + self.propagate(
                        masked_select_nnz(edge_index, mask, layout='coo'),
                        x=lin(x[0]), film=film, x1=lin(x[1]), size=None)
                else:
                    assert edge_type is not None
                    mask = edge_type == i
                    out = out + self.propagate(edge_index[:, mask], x=lin(
                        x[0]), film=film, x_i=lin(x[1]), size=None)

        return out


    def message(self, x_j: Tensor, film, x_i) -> Tensor:
        beta_i, gamma_i = film(torch.hstack((x_j, x_i))).split(self.out_channels, dim=-1)
        out = gamma_i * x_j + beta_i
        if self.act is not None:
            out = self.act(out)
        return out