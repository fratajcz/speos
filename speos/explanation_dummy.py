from speos.utils.config import Config
from speos.models import ModelBootstrapper
from speos.datasets import DatasetBootstrapper
from speos.preprocessing.mappers import GWASMapper, AdjacencyMapper

from speos.helpers import CheckPointer

import matplotlib.pyplot as plt
import torch
from captum.attr import IntegratedGradients

from inspect import signature
from math import sqrt
from typing import Optional

from torch import Tensor

from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import k_hop_subgraph, to_networkx
from torch_geometric.nn import Explainer, to_captum

config = Config()
config.parse_yaml("config_cardiovascular_tag.yaml")
config.input.save_dir = "./data/"
config.logging.dir =  "./logs/"

mappings = GWASMapper(config.input.gene_sets, config.input.gwas).get_mappings(
            config.input.tag, fields=config.input.field)

tag = "" if config.input.adjacency == "all" else config.input.adjacency
adjacencies = AdjacencyMapper(config.input.adjacency_mappings).get_mappings(tag, fields=config.input.adjacency_field)

dataset = DatasetBootstrapper(
            mappings, adjacencies, holdout_size=config.input.holdout_size, name=config.name, config=config).get_dataset()

data = dataset.data
input_dim = data.x.shape[1]

model = ModelBootstrapper(
            config, input_dim, len(adjacencies)).get_model()

config.name = "cardiovascular_tag_outer_1_fold_0"
config.model.save_dir = "./models/"


checkpointer = CheckPointer(
                model, config.model.save_dir + config.name, mode=config.es.mode)
checkpointer.restore()

from torch_geometric.nn import Sequential
"""
latent_features = model.architectures[0].pre_mp(data.x.float())

flow_list = []
flow_list.append('x, edge_index -> x')
flow_list.append('x -> x')
flow_list.append('x -> x')
flow_list.append('x, edge_index -> x')
flow_list.append('x -> x')
flow_list.append('x -> x')
layer_list = []

messagepassing = model.architectures[0].mp

for i, layer in enumerate(messagepassing._modules.values()):
    layer_list.append(layer)

for layer in model.architectures[0].post_mp:
    flow_list.append('x -> x')
    layer_list.append(layer)

surrogate_model = Sequential('x, edge_index', [(layer, flow) for layer, flow in zip(layer_list, flow_list)])
"""

surrogate_model = model.architectures[0].repackage_into_one_sequential()
device = "cpu"
output_idx = 14

edge_mask = torch.ones(data.num_edges, requires_grad=True, device=device)

captum_model = to_captum(surrogate_model, mask_type='node_and_edge',
                         output_idx=output_idx)

ig = IntegratedGradients(captum_model)

ig_attr_node, ig_attr_edge = ig.attribute(
    (data.x.float().unsqueeze(0), edge_mask.unsqueeze(0)),
    additional_forward_args=(data.edge_index), internal_batch_size=1)

# Scale attributions to [0, 1]:
ig_attr_node = ig_attr_node.squeeze(0).abs().sum(dim=1)
ig_attr_node /= ig_attr_node.max()
ig_attr_edge = ig_attr_edge.squeeze(0).abs()
ig_attr_edge /= ig_attr_edge.max()

fig, ax = plt.subplots(figsize=(15, 15))
# Visualize absolute values of attributions:
explainer = Explainer(surrogate_model)
ax, G = explainer.visualize_subgraph(output_idx, data.edge_index, ig_attr_edge, y=data.y,
                                     node_alpha=ig_attr_node)

fig.tight_layout()
plt.savefig("explanation3.png")
