from speos.benchmark import TestBench
from speos.datasets import DatasetBootstrapper
from speos.preprocessing.mappers import GWASMapper, AdjacencyMapper
import numpy as np

import argparse

parser = argparse.ArgumentParser(description='Split a finished testbench run into performance on connected vs disconnected')

parser.add_argument('--config', "-c", type=str, default="testbench_config_adj_mr_immu_tag_skip.yaml",
                    help='Path to the config that should be used for the run.')
parser.add_argument('--parameters', "-p", type=str, default="testbench_adjacencies_parameters_tag.yaml",
                    help='Path to the parameters that should be used for the run.')

args = parser.parse_args()


connected_pos_only_masks = []
disconnected_pos_only_masks = []

tb = TestBench(args.parameters, args.config)
tb.compile_configs()
tb.compile_resultshandlers()

# first get connected and disconnected positive nodes for every testbench setting

for i, (config, resultshandler_path) in enumerate(zip(tb.configs, tb.resultshandlers)):
    if i % 4 != 0:
        continue
    mappings = GWASMapper(config.input.gene_sets, config.input.gwas).get_mappings(
            config.input.tag, fields=config.input.field)

    tag = "" if config.input.adjacency == "all" else config.input.adjacency
    adjacencies = AdjacencyMapper(config.input.adjacency_mappings).get_mappings(tag)

    config.input.save_data = False

    dataset = DatasetBootstrapper(
            mappings, adjacencies, holdout_size=config.input.holdout_size, name=config.name, config=config).get_dataset()
    connected, disconnected = dataset.preprocessor.get_connected_components()
    disconnected_positives = np.array(disconnected[1])
    connected_positives = np.array(connected[0][1])

    connected_pos_only_mask = np.ones_like(dataset.data.y.cpu().detach().numpy())
    if len(disconnected_positives) > 0:
        connected_pos_only_mask[disconnected_positives] = 0
    connected_pos_only_masks.append(connected_pos_only_mask.astype(np.bool8))

    disconnected_pos_only_mask = np.ones_like(dataset.data.y.cpu().detach().numpy())
    if len(connected_positives) > 0:
        disconnected_pos_only_mask[connected_positives] = 0
    disconnected_pos_only_masks.append(disconnected_pos_only_mask.astype(np.bool8))

# now evaluate either only on connected or on disconnected or on all

tb.metrics = ["mrr_filtered", "mean_rank_filtered"]
common_save_path = tb.config.name + "_" + tb.name + "_{}.tsv"

df_connected = tb.compare(save=False, additional_masks=np.array(connected_pos_only_masks).repeat(tb.repeats, axis=0).tolist())
df_connected.to_csv(common_save_path.format("connected"), sep="\t")

df_disconnected = tb.compare(save=False, additional_masks=np.array(disconnected_pos_only_masks).repeat(tb.repeats, axis=0).tolist())
df_disconnected.to_csv(common_save_path.format("disconnected"), sep="\t")

df_all = tb.compare(save=False)
df_all.to_csv(common_save_path.format("all"), sep="\t")


# Now the same masks but only the predictions from the mlp (to check if its the adjacency)
# for this to work, the MLP must be the first settings that have been checked!

tb.resultshandlers = tb.resultshandlers[:tb.repeats] * int(len(tb.resultshandlers) / tb.repeats)

common_save_path = tb.config.name + "_" + tb.name + "_{}_mlp.tsv"

df_connected_noadj = tb.compare(save=False, additional_masks=np.array(connected_pos_only_masks).repeat(tb.repeats, axis=0).tolist())
df_connected_noadj.to_csv(common_save_path.format("connected"), sep="\t")

df_disconnected_noadj = tb.compare(save=False, additional_masks=np.array(disconnected_pos_only_masks).repeat(tb.repeats, axis=0).tolist())
df_connected_noadj.to_csv(common_save_path.format("disconnected"), sep="\t")

df_all_noadj = tb.compare(save=False)
df_all_noadj.to_csv(common_save_path.format("all"), sep="\t")
