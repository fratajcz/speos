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

import argparse

parser = argparse.ArgumentParser(description='Run model interpretation for a selected gene and model')

parser.add_argument('--config', "-c", type=str,
                    help='path to config of the run that should be examined')
parser.add_argument('--gene', "-g", type=str, default="",
                    help='HGNC gene symbol which should be interpreted')
parser.add_argument('--index', "-i", type=int, default=-1,
                    help='index of gene to examine')
parser.add_argument('--threshold', "-t", type=float, default=0.0,
                    help='minimum importance of nodes and edges required to be plotted')

torch.set_default_dtype(torch.float64)


args = parser.parse_args()

config = Config()
#config.parse_yaml("/home/florin.ratajczak/ppi-core-genes/configs/config_immune_dysregulation_tag.yaml")
config.parse_yaml(args.config)

config.input.save_dir = "./data/"
config.logging.dir =  "./logs/"

dataset = DatasetBootstrapper(holdout_size=config.input.holdout_size, name=config.name, config=config).get_dataset()

if args.index == -1:
    dataset.preprocessor.build_graph(features=True)

data = dataset.data
input_dim = data.x.shape[1]
model = ModelBootstrapper(config, input_dim, dataset.num_relations).get_model()

#config.model.save_dir = "./models/"
print("Loading model from {}".format(config.model.save_dir + config.name + ".pt"))

checkpointer = CheckPointer(model, config.model.save_dir + config.name, mode=config.es.mode)
checkpointer.restore()

surrogate_model = model.architectures[0].repackage_into_one_sequential()

explainer = Explainer(
    surrogate_model,  # It is assumed that model outputs a single tensor.
    algorithm=CaptumExplainer('IntegratedGradients'),
    explanation_type='model',
    node_mask_type='attributes',
    edge_mask_type='object',
    model_config = dict(
        mode='regression',
        task_level="node",
        return_type='raw',  # Model returns probabilities.
    ),
)

if args.gene != "":
    candidates = [dataset.preprocessor.hgnc2id[args.gene]]
    genes = [args.gene]
    print("Candidates: {}, {}".format(args.gene, candidates))
elif args.index != -1:
    candidates = [args.index]
    genes = candidates[:]
    print("Candidates: {}".format(candidates))

normalized_data = torch.empty(dataset.data.x.shape, dtype=torch.double)
normalized_data_positives = dataset.data.x / dataset.data.x.topk(dim=0, k =int(dataset.data.x.shape[0]/100), sorted=True).values[-1,:]
normalized_data_negatives = (dataset.data.x / dataset.data.x.topk(dim=0, k=int(dataset.data.x.shape[0]/100), sorted=True, largest=False).values[-1,:]) * -1
normalized_data[dataset.data.x >= 0] = normalized_data_positives[dataset.data.x >= 0]
normalized_data[dataset.data.x < 0] = normalized_data_negatives[dataset.data.x < 0]
features = dataset.preprocessor.get_feature_names()


for i, (output_idx, gene) in enumerate(zip(candidates, genes)):
    
    #if os.path.exists(path):
    #    continue
    try:
        print("Processing Candidate Gene #{} out of {}: {}".format(i, len(candidates), dataset.preprocessor.id2hgnc[output_idx]))
    except AttributeError:
        print("Processing Candidate Gene #{} out of {}: {}".format(i, len(candidates), output_idx))
    ig_attr_edge_all = None
    ig_attr_node_all = None
    ig_attr_self_all = None

    explanation = explainer(data.x, data.edge_index, index=output_idx)

    # Scale attributions to [0, 1]:
    ig_attr_self = explanation.node_mask.squeeze(0)[output_idx]
    ig_attr_self_abs = explanation.node_mask.squeeze(0)[output_idx].abs()
    ig_attr_node = explanation.node_mask.squeeze(0).abs().sum(dim=1)
                
    ig_attr_self /= ig_attr_self.abs().max()
    ig_attr_self_abs /= ig_attr_self_abs.max()
    ig_attr_node /= ig_attr_node.max()

    ig_attr_edge = explanation.edge_mask.squeeze(0).abs()
    ig_attr_edge /= ig_attr_edge.max()

    torch.save(ig_attr_self, '{}/{}_ig_attr_self_abs_{}.pt'.format(config.inference.save_dir, config.name, gene))
    torch.save(ig_attr_self_abs, '{}/{}_ig_attr_self_{}.pt'.format(config.inference.save_dir, config.name, gene))
    torch.save(ig_attr_edge, '{}/{}_ig_attr_edge_{}.pt'.format(config.inference.save_dir, config.name, gene))
    torch.save(ig_attr_node, '{}/{}_ig_attr_node_{}.pt'.format(config.inference.save_dir, config.name, gene))

    # write node attributions to csv
    input_attributions = ig_attr_self.numpy().tolist()
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
    df.to_csv("{}/Feature_Importance_{}_{}.tsv".format(config.inference.save_dir, config.name, output_idx), index=False, sep="\t")

    # write 100 most important edge attributions to csv
    k = 100
    id2hgnc = dataset.preprocessor.id2hgnc
    top_indices = torch.topk(ig_attr_edge, k).indices
    top_edges = [(id2hgnc[key1.item()], dataset.data.y[key1.item()].item(), id2hgnc[key2.item()],  dataset.data.y[key2.item()].item(), value.item()) for key1, key2, value in zip(data.edge_index[0, top_indices], data.edge_index[1, top_indices], ig_attr_edge[top_indices])]
    df = pd.DataFrame(data=top_edges, columns=["Sender", "Sender Label", "Receiver", "Receiver Label", "Importance"])
    for column in df.columns:
        if column.endswith("Label"):
            df[column] = df[column].astype(int)
    df.to_csv("{}/Edge_Importance_{}_{}.tsv".format(config.inference.save_dir, config.name, output_idx), sep="\t", index=False)