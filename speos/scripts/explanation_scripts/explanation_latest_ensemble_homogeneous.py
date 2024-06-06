from speos.utils.config import Config
from speos.models import ModelBootstrapper
from speos.preprocessing.datasets import DatasetBootstrapper
from speos.preprocessing.mappers import GWASMapper, AdjacencyMapper

from speos.helpers import CheckPointer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch_geometric.explain import Explainer, CaptumExplainer

import os
import argparse
import json

parser = argparse.ArgumentParser(description='Run model interpretation for a selected gene and model')

parser.add_argument('--config', "-c", type=str,
                    help='path to config of the run that should be examined')
parser.add_argument('--gene', "-g", type=str, default="",
                    help='HGNC gene symbol which should be interpreted')
parser.add_argument('--index', "-i", type=int, default=-1,
                    help='index of gene to examine')
parser.add_argument('--threshold', "-t", type=float, default=0,
                    help='minimum importance of nodes and edges required to be plotted')
parser.add_argument('--mincs', "-m", type=int, default=-1,
                    help='minimal cs of candidates to examine')
parser.add_argument('--readonly', "-r", action='store_true',
                    default=False,
                    help='if run should be readonly.')
parser.add_argument('--device', "-d", type=str,
                    default="cpu",
                    help='The device on which the calculations should be ru on (i.e. "cpu", or "cuda:0" etc.)')




args = parser.parse_args()

torch.set_default_dtype(torch.float64)

assert args.gene != "" or args.index != -1 or args.mincs != -1, ("At least a gene name, index or minimal cs has to be chosen")

config = Config()
#config.parse_yaml("/home/florin.ratajczak/ppi-core-genes/configs/config_immune_dysregulation_tag.yaml")
config.parse_yaml(args.config)
old_config_name = config.name[:]

config.input.save_dir = "./data/"
config.logging.dir =  "./logs/"

dataset = DatasetBootstrapper(holdout_size=config.input.holdout_size, name=config.name, config=config).get_dataset()

if args.index == -1:
    dataset.preprocessor.build_graph(features=True)

data = dataset.data
data.to(args.device)
input_dim = data.x.shape[1]
model = ModelBootstrapper(
            config, input_dim, dataset.num_relations).get_model()

ig_attr_edge_ = None
ig_attr_node_ = None

if args.gene != "":
    candidates = [dataset.preprocessor.hgnc2id[args.gene]]
    genes = [args.gene]
    print("Candidates: {}, {}".format(args.gene, candidates))
elif args.index != -1:
    candidates = [args.index]
    genes = candidates[:]
    print("Candidates: {}".format(candidates))

outer_results_file = os.path.join(config.pp.save_dir, "{}outer_results.json".format(config.name))

with open(outer_results_file, "r") as file:
    outer_results = json.load(file)[0]
    
if args.gene == "" and args.index == -1:
    genes = [key for key, value in outer_results.items() if value >= args.mincs]
    candidates = [dataset.preprocessor.hgnc2id[gene] for gene in genes]
    print("Candidates: {} / {}".format(genes, candidates))

normalized_data = torch.empty(dataset.data.x.shape, dtype=torch.double).to(args.device)
normalized_data_positives = dataset.data.x / dataset.data.x.topk(dim=0, k =int(dataset.data.x.shape[0]/100), sorted=True).values[-1,:]
normalized_data_negatives = (dataset.data.x / dataset.data.x.topk(dim=0, k=int(dataset.data.x.shape[0]/100), sorted=True, largest=False).values[-1,:]) * -1
normalized_data[dataset.data.x >= 0] = normalized_data_positives[dataset.data.x >= 0]
normalized_data[dataset.data.x < 0] = normalized_data_negatives[dataset.data.x < 0]
features = dataset.preprocessor.get_feature_names()


num_inner = 10
num_outer = 11

for i, (output_idx, gene) in enumerate(zip(candidates, genes)):
    
    #if os.path.exists(path):
    #    continue
    try:
        print("Processing Candidate Gene #{} out of {}: {}".format(i + 1, len(candidates), dataset.preprocessor.id2hgnc[output_idx]))
    except AttributeError:
        print("Processing Candidate Gene #{} out of {}: {}".format(i + 1, len(candidates), output_idx))
    ig_attr_edge_all = None
    ig_attr_node_all = None
    ig_attr_self_all = None
    ig_attr_self_abs_all = None
    num_processed = 0
    for outer_fold in range(0, num_outer):
        for inner_fold in range(0, num_inner):
            if inner_fold == outer_fold:
                continue
            try:
                ig_attr_node = torch.load(os.path.join(config.pp.save_dir, "{}_ig_attr_node_outer{}_inner{}_{}.pt".format(old_config_name, outer_fold, inner_fold, gene)), map_location=torch.device(args.device))
                ig_attr_edge = torch.load(os.path.join(config.pp.save_dir, "{}_ig_attr_edge_outer{}_inner{}_{}.pt".format(old_config_name, outer_fold, inner_fold, gene)), map_location=torch.device(args.device))
                ig_attr_self = torch.load(os.path.join(config.pp.save_dir, "{}_ig_attr_self_outer{}_inner{}_{}.pt".format(old_config_name, outer_fold, inner_fold, gene)), map_location=torch.device(args.device))
                ig_attr_self_abs = torch.load(os.path.join(config.pp.save_dir, "{}_ig_attr_self_abs_outer{}_inner{}_{}.pt".format(old_config_name, outer_fold, inner_fold, gene)), map_location=torch.device(args.device))
                print("Loaded edge and node attributes for outer {} inner {}".format(outer_fold, inner_fold))
                ig_attr_node.requires_grad = False
                ig_attr_edge.requires_grad = False
                ig_attr_self.requires_grad = False
                ig_attr_self_abs.requires_grad = False

            except FileNotFoundError:
                if args.readonly:
                    continue 
                config.name = "{}_outer_{}_fold_{}".format(old_config_name, outer_fold, inner_fold)
                #config.model.save_dir = "./models/"
                print("Loading model from {}".format(config.model.save_dir + config.name + ".pt"))

                checkpointer = CheckPointer(
                        model, config.model.save_dir + config.name, mode=config.es.mode)
                checkpointer.restore()

                surrogate_model = model.architectures[0].repackage_into_one_sequential()
                surrogate_model.to(args.device)
                
                explainer = Explainer(
                surrogate_model,  # It is assumed that model outputs a single tensor.
                algorithm=CaptumExplainer('IntegratedGradients'),
                explanation_type='model',
                node_mask_type='attributes',
                edge_mask_type='object',
                model_config = dict(
                    mode='binary_classification',
                    task_level="node",
                    return_type='raw',  # Model returns probabilities.
                    ),
                )
                explanation = explainer(data.x, data.edge_index, index=output_idx)

                # Scale attributions to [0, 1]:
                ig_attr_self = explanation.node_mask.squeeze(0)[output_idx]
                ig_attr_self_abs = explanation.node_mask.squeeze(0)[output_idx].abs()
                ig_attr_node = explanation.node_mask.squeeze(0).abs().sum(dim=1)
                
                ig_attr_self /= ig_attr_self.abs().max().detach()
                ig_attr_self_abs /= ig_attr_self_abs.max().detach()
                ig_attr_node /= ig_attr_node.max().detach()

                ig_attr_edge = explanation.edge_mask.squeeze(0).abs().detach()
                ig_attr_edge /= ig_attr_edge.max().detach()
                
                torch.save(ig_attr_self, os.path.join(config.pp.save_dir, "{}_ig_attr_self_outer{}_inner{}_{}.pt".format(old_config_name, outer_fold, inner_fold, gene)))
                torch.save(ig_attr_self_abs, os.path.join(config.pp.save_dir, "{}_ig_attr_self_abs_outer{}_inner{}_{}.pt".format(old_config_name, outer_fold, inner_fold, args.gene)))
                torch.save(ig_attr_node, os.path.join(config.pp.save_dir, "{}_ig_attr_node_outer{}_inner{}_{}.pt".format(old_config_name, outer_fold, inner_fold, gene)))
                torch.save(ig_attr_edge, os.path.join(config.pp.save_dir, "{}_ig_attr_edge_outer{}_inner{}_{}.pt".format(old_config_name, outer_fold, inner_fold, gene)))
            
            num_processed += 1

            if ig_attr_self_abs_all is None:
                ig_attr_self_abs_all = ig_attr_self_abs
            else:
                ig_attr_self_abs_all += ig_attr_self_abs

            if ig_attr_self_all is None:
                ig_attr_self_all = ig_attr_self
            else:
                ig_attr_self_all += ig_attr_self

            if ig_attr_edge_all is None:
                ig_attr_edge_all = ig_attr_edge
            else:
                ig_attr_edge_all += ig_attr_edge

            if ig_attr_node_all is None:
                ig_attr_node_all = ig_attr_node
            else:
                ig_attr_node_all += ig_attr_node

    ig_attr_self_all /= num_processed
    ig_attr_self_abs_all /= num_processed
    ig_attr_edge_all /= num_processed
    ig_attr_node_all /= num_processed

    torch.save(ig_attr_self_all, os.path.join(config.pp.save_dir,  '{}_ig_attr_self_abs_{}.pt'.format(old_config_name, gene)))
    torch.save(ig_attr_self_abs_all, os.path.join(config.pp.save_dir, '{}_ig_attr_self_{}.pt'.format(old_config_name, gene)))
    torch.save(ig_attr_edge_all, os.path.join(config.pp.save_dir, '{}_ig_attr_edge_{}.pt'.format(old_config_name, gene)))
    torch.save(ig_attr_node_all, os.path.join(config.pp.save_dir, '{}_ig_attr_node_{}.pt'.format(old_config_name, gene)))
    
    # write node attributions to csv
    input_attributions = ig_attr_self_all.cpu().numpy().tolist()
    input_values = normalized_data[dataset.preprocessor.hgnc2id[gene]].tolist()
    list1, list2, list3 = zip(*sorted(zip(input_attributions, features, input_values)))

    df = pd.DataFrame(columns=["Label", "Input Value"])
    labels = [" ".join([word[0].upper() + word[1:] for word in label.split(" ")]) for label in list2]
    labels = ["-".join([word[0].upper() + word[1:] for word in label.split("-")]) for label in labels]
    labels = ["Z-Score " + label[6:]if label.startswith("ZSTAT") else label for label in labels]
    labels = ["P-Value " + label[2:] if label.startswith("P ") else label for label in labels]
    labels = ["n SNPs " + label[6:] if label.startswith("NSNPS") else label for label in labels]
    labels = [label + "\n(GWAS)" if label.startswith(("Z-Score", "P-Value", "n SNPs")) else label + "\n(Gene Expr.)" for label in labels]
    labels = [label[8:] if label.startswith("Cells") else label for label in labels]
    df["Label"] = ["".join(label.split("\n")) for label in labels]
    df["Input Value"] = list3
    df["Importance"] = list1 / np.max(list1)
    df.to_csv(os.path.join(config.pp.save_dir, "Feature_Importance_{}_{}.tsv".format(old_config_name, gene)), index=False, sep="\t")

    # write 100 most important edge attributions to csv
    k = 100
    id2hgnc = dataset.preprocessor.id2hgnc
    top_indices = torch.topk(ig_attr_edge, k).indices
    top_edges = [(id2hgnc[key1.item()], dataset.data.y[key1.item()].item(), id2hgnc[key2.item()],  dataset.data.y[key2.item()].item(), value.item()) for key1, key2, value in zip(data.edge_index[0, top_indices], data.edge_index[1, top_indices], ig_attr_edge_all[top_indices])]
    df = pd.DataFrame(data=top_edges, columns=["Sender", "Sender Label", "Receiver", "Receiver Label", "Importance"])
    for column in df.columns:
        if column.endswith("Label"):
            df[column] = df[column].astype(int)
    df.to_csv(os.path.join(config.pp.save_dir, "Edge_Importance_{}_{}.tsv".format(old_config_name, gene)), sep="\t", index=False)
