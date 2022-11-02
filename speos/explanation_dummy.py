from speos.utils.config import Config
from speos.models import ModelBootstrapper
from speos.datasets import DatasetBootstrapper
from speos.preprocessing.mappers import GWASMapper, AdjacencyMapper

from speos.helpers import CheckPointer
import os
import matplotlib.pyplot as plt
import torch
from captum.attr import IntegratedGradients

from torch_geometric.nn import to_captum
from speos.explanation import Explainer

import argparse

parser = argparse.ArgumentParser(description='Get Promising Drug Development Candidates from Postprocessing Table')

parser.add_argument('--gene', "-g", type=str, default="",
                    help='Path to the Postprocessing Table.')

args = parser.parse_args()

config = Config()
config.parse_yaml("/home/florin.ratajczak/ppi-core-genes/configs/config_immune_dysregulation_tag.yaml")
#print(config)
config.input.save_dir = "./data/"
config.logging.dir =  "./logs/"

pre_mappings = GWASMapper().get_mappings(
            config.input.tag, fields=config.input.field)

mappings = []

for mapping in pre_mappings:
    if not "AMD" in mapping["name"]:
        mappings.append(mapping)


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

import json

outer_results_file = "/lustre/groups/epigenereg01/projects/ppi-florin/results/immune_dysregulation_tagouter_results.json"

with open(outer_results_file, "r") as file:
            outer_results = json.load(file)[0]

if args.gene == "":
    candidates = [dataset.preprocessor.hgnc2id[key] for key, value in outer_results.items() if value == 11]
else:
    candidates = [dataset.preprocessor.hgnc2id[args.gene]]


for i, output_idx in enumerate(candidates):
    #if dataset.preprocessor.id2hgnc[output_idx] == "ABI1":
    #    continue
    #if output_idx not in dataset.data.edge_index:
    #    continue
    #if dataset.preprocessor.G.degree[output_idx] == 0:
    #    print("Skipping Candidate Gene #{} out of {} ({}) because it has no incident edges.".format(i, len(candidates), dataset.preprocessor.id2hgnc[output_idx]))
    path = "/lustre/groups/epigenereg01/projects/ppi-florin/explanation_tag_immune_{}.png".format(dataset.preprocessor.id2hgnc[output_idx])
    if os.path.exists(path):
        continue
    print("Processing Candidate Gene #{} out of {}: {}".format(i, len(candidates), dataset.preprocessor.id2hgnc[output_idx]))
    ig_attr_edge_ = None
    ig_attr_node_ = None
    for outer_fold in range(0, num_outer):
        for inner_fold in range(0, num_inner + 1):
            if inner_fold == outer_fold:
                continue
            config.name = "immune_dysregulation_tag_outer_{}_fold_{}".format(outer_fold, inner_fold)
            #config.model.save_dir = "./models/"
            print("Loading model from {}".format(config.model.save_dir + config.name + ".pt"))

            checkpointer = CheckPointer(
                    model, config.model.save_dir + config.name, mode=config.es.mode)
            checkpointer.restore()

            surrogate_model = model.architectures[0].repackage_into_one_sequential()
            device = "cpu"
        

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


    ig_attr_edge_ /= num_inner * num_outer
    ig_attr_node_ /= num_inner * num_outer

    fig, ax = plt.subplots(figsize=(12, 12))
    # Visualize absolute values of attributions:

    explainer = Explainer(surrogate_model)

    ax, G = explainer.visualize_subgraph(output_idx, data.edge_index, ig_attr_edge_, y=data.y,
                                     node_alpha=ig_attr_node_)

    id2hgnc = {value: key for key, value in dataset.preprocessor.hgnc2id.items()}

    for textbox in ax.texts:
        index = textbox._text
        try:
            hgnc = id2hgnc[int(index)]
            textbox._text = hgnc
        except (ValueError, KeyError):
            continue

    fig.tight_layout()
    plt.savefig(path)
    print("Saved Explanation for {}".format(id2hgnc[output_idx]))