from speos.utils.config import Config
from speos.models import ModelBootstrapper
from speos.datasets import DatasetBootstrapper
from speos.preprocessing.mappers import GWASMapper, AdjacencyMapper

from speos.helpers import CheckPointer

import matplotlib.pyplot as plt
import torch
from captum.attr import IntegratedGradients

from torch_geometric.nn import to_captum
from speos.explanation import Explainer

config = Config()
config.parse_yaml("/home/florin.ratajczak/ppi-core-genes/configs/config_cardiovascular_tag.yaml")
#print(config)
config.input.save_dir = "./data/"
config.logging.dir =  "./logs/"

mappings = GWASMapper().get_mappings(
            config.input.tag, fields=config.input.field)


adjacencies = AdjacencyMapper(blacklist=config.input.adjacency_blacklist).get_mappings(config.input.adjacency, fields=config.input.adjacency_field)


#print(adjacencies)

dataset = DatasetBootstrapper(
            mappings, adjacencies, holdout_size=config.input.holdout_size, name=config.name, config=config).get_dataset()

data = dataset.data
input_dim = data.x.shape[1]
model = ModelBootstrapper(
            config, input_dim, len(adjacencies)).get_model()

ig_attr_edge_ = None
ig_attr_node_ = None
#print(model)

num_inner = 10
num_outer = 11

for outer_fold in range(0, num_outer):
    for inner_fold in range(0, num_inner + 1):
        if inner_fold == outer_fold:
            continue
        config.name = "cardiovascular_tag_outer_{}_fold_{}".format(outer_fold, inner_fold)
        #config.model.save_dir = "./models/"
        print("Loading model from {}".format(config.model.save_dir + config.name + ".pt"))

        checkpointer = CheckPointer(
                    model, config.model.save_dir + config.name, mode=config.es.mode)
        checkpointer.restore()

        # from torch_geometric.nn import Sequential


        surrogate_model = model.architectures[0].repackage_into_one_sequential()
        device = "cpu"
        output_idx = 6637 

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

        if ig_attr_edge_ is None:
            ig_attr_edge_ = ig_attr_edge
        else:
            ig_attr_edge_ += ig_attr_edge

        if ig_attr_node_ is None:
            ig_attr_node_ = ig_attr_node
        else:
            ig_attr_node_ += ig_attr_node


ig_attr_edge_ /= inner_fold * outer_fold
ig_attr_node_ /= 10

fig, ax = plt.subplots(figsize=(15, 15))
# Visualize absolute values of attributions:

explainer = Explainer(surrogate_model)

ax, G = explainer.visualize_subgraph(output_idx, data.edge_index, ig_attr_edge_, y=data.y,
                                     node_alpha=ig_attr_node_)

id2hgnc = dataset.preprocessor.id2hgnc

for textbox in ax.texts:
    index = textbox._text
    try:
        hgnc = id2hgnc[int(index)]
        textbox._text = hgnc
    except (ValueError, KeyError):
        continue

fig.tight_layout()
plt.savefig("explanation_new.png")