
import torch
from torch_sparse import SparseTensor
from sklearn.preprocessing import LabelEncoder


def typed_edges_to_sparse_tensor(x, edge_index):
    edges = torch.cat([edges for edges in edge_index.values()], dim=1)
    types = [[edge_type[1]] * edges.shape[-1] for edge_type, edges in edge_index.items()]
    types = [value for sublist in types for value in sublist]
    encoder = LabelEncoder()
    types = torch.Tensor(encoder.fit_transform(types)).to(edges.device)
    return SparseTensor.from_edge_index(edges.long(), types, (x.shape[0], x.shape[0])), encoder