from speos.utils.config import Config
from speos.models import ModelBootstrapper
from speos.datasets import DatasetBootstrapper
from speos.preprocessing.mappers import GWASMapper, AdjacencyMapper
import speos.utils.nn_utils as nn_utils
from speos.helpers import CheckPointer
import os
import matplotlib.pyplot as plt
import torch
from captum.attr import IntegratedGradients
import torch_geometric.nn as pyg_nn
from torch_geometric.nn import to_captum
from speos.explanation import Explainer

import argparse

parser = argparse.ArgumentParser(description='Get Promising Drug Development Candidates from Postprocessing Table')

parser.add_argument('--gene', "-g", type=str, default="",
                    help='Path to the Postprocessing Table.')

args = parser.parse_args()

ig_attr_edge = torch.load("ig_attr_edge_outer{}_inner{}.pt".format(0, 1))
print(ig_attr_edge.shape)

config = Config()
config.parse_yaml("/home/florin.ratajczak/ppi-core-genes/configs/config_immune_dysregulation_film_forreal.yaml")
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
num_outer = 10
data.x = list(data.x_dict.values())[0]
data.edge_index, edge_encoder = nn_utils.typed_edges_to_sparse_tensor(data.x, data.edge_index_dict)

#print(edge_encoder.inverse_transform([0]))
#print(data.edge_index)
#print(dir(data.edge_index))
#edge_masks = [None] * len(data.edge_index_dict.keys())
#for key, value in data.edge_index_dict.items():
#    index = edge_encoder.transform([key[1]])[0]
#    type_ = torch.Tensor((index,)).repeat(value.size(1))
#    mask = torch.ones(value.size(1), requires_grad=True, device="cpu")
#    mask = torch.stack((mask, type_), 1)
#    #print(mask)
#    edge_masks[index] = mask
#    #print(mask.shape)
#edge_masks = torch.cat(edge_masks, 0)
#print(edge_masks.shape)
#print(edge_masks[:,1].max())

edge_mask = torch.ones_like(data.edge_index.storage.value(), requires_grad=True)
edge_types = data.edge_index.storage.value()
del data.x_dict
del data.edge_index_dict

surrogate_model = None
import json

outer_results_file = "/lustre/groups/epigenereg01/projects/ppi-florin/results/immune_dysregulation_film_forrealouter_results.json"

with open(outer_results_file, "r") as file:
    outer_results = json.load(file)[0]

#if args.gene == "":
#    candidates = [dataset.preprocessor.hgnc2id[key] for key, value in outer_results.items() if value == 11]
#else: 
#    candidates = [dataset.preprocessor.hgnc2id[args.gene]]


#print("Candidates: {}".format(candidates))
candidates = [6690]
for i, output_idx in enumerate(candidates):
    #if dataset.preprocessor.id2hgnc[output_idx] == "ABI1":
    #    continue
    #if dataset.preprocessor.G.degree[output_idx] == 0:
    #    print("Skipping Candidate Gene #{} out of {} ({}) because it has no incident edges.".format(i, len(candidates), dataset.preprocessor.id2hgnc[output_idx]))
    path = "/lustre/groups/epigenereg01/projects/ppi-florin/explanation_film_{}.png".format(args.gene)
    #if os.path.exists(path):
    #    continue
    #print("Processing Candidate Gene #{} out of {}: {}".format(i, len(candidates), dataset.preprocessor.id2hgnc[output_idx]))
    ig_attr_edge_ = None
    ig_attr_node_ = None
    num_processed = 0
    for outer_fold in range(0, num_outer):
        for inner_fold in range(0, num_inner + 1):
            if inner_fold == outer_fold:
                continue
            try:
                ig_attr_node = torch.load("ig_attr_node_outer{}_inner{}.pt".format(outer_fold, inner_fold))
                ig_attr_edge = torch.load("ig_attr_edge_outer{}_inner{}.pt".format(outer_fold, inner_fold))
                print("Loaded edge and node attributes for outer {} inner {}".format(outer_fold, inner_fold))
                ig_attr_node.requires_grad = False
                ig_attr_edge.requires_grad = False
            except FileNotFoundError:
                
                config.name = "cardiovascular_film_outer_{}_fold_{}".format(outer_fold, inner_fold)
                #config.model.save_dir = "./models/"
                print("Loading model from {}".format(config.model.save_dir + config.name + ".pt"))

                checkpointer = CheckPointer(
                        model, config.model.save_dir + config.name, mode=config.es.mode)
                checkpointer.restore()

                surrogate_model = model.architectures[0].repackage_into_one_sequential()
                for module in surrogate_model.modules():
                    #print(module)
                    if isinstance(module, pyg_nn.FiLMConv):
                        module.add_types(edge_types)
                #print(surrogate_model)
                device = "cpu"
            

                #edge_mask = torch.ones(data.num_edges, requires_grad=True, device=device)
                #edge_mask = edge_masks
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
                
                torch.save(ig_attr_node, "ig_attr_node_outer{}_inner{}.pt".format(outer_fold, inner_fold))
                torch.save(ig_attr_edge, "ig_attr_edge_outer{}_inner{}.pt".format(outer_fold, inner_fold))
            
            num_processed += 1

            if ig_attr_edge_ is None:
                ig_attr_edge_ = ig_attr_edge
            else:
                ig_attr_edge_ += ig_attr_edge

            if ig_attr_node_ is None:
                ig_attr_node_ = ig_attr_node
            else:
                ig_attr_node_ += ig_attr_node


    ig_attr_edge_ /= num_processed
    ig_attr_node_ /= num_processed
    
    torch.save(ig_attr_edge, 'ig_attr_edge_film.pt')
    torch.save(ig_attr_node, 'ig_attr_node_film.pt')

    fig, ax = plt.subplots(figsize=(12, 12))
    # Visualize absolute values of attributions:
    if surrogate_model is None:
        surrogate_model = model.architectures[0].repackage_into_one_sequential()

    explainer = Explainer(surrogate_model)
    
    row, col, value = data.edge_index.coo()
    edge_index = torch.vstack((row,col))
    print(edge_index)
    print(edge_index.shape)
    #ig_attr_edge_, types = torch.tensor_split(ig_attr_edge_, 2, dim=1)
    
    threshold = 0.1
    print(torch.sum(ig_attr_edge_ > threshold))
    print(edge_index[:, ig_attr_edge_.squeeze() > threshold])
    print(edge_types[ig_attr_edge_.squeeze() > threshold])
    print(ig_attr_edge_[ig_attr_edge_.squeeze() > threshold])
    #ax, G = explainer.visualize_subgraph(output_idx, edge_index, ig_attr_edge_.squeeze(), threshold=threshold,
    #                                 node_alpha=ig_attr_node_)

    #id2hgnc = dataset.preprocessor.id2hgnc

    for textbox in ax.texts:
        index = textbox._text
        try:
            hgnc = id2hgnc[int(index)]
            textbox._text = hgnc
        except (ValueError, KeyError):
            continue

    fig.tight_layout()
    plt.savefig(path)
    print("Saved Explanation for {}".format(args.gene))