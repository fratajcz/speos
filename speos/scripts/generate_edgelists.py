import os

from speos.preprocessing.mappers import GWASMapper, AdjacencyMapper
from speos.preprocessing.preprocessor import PreProcessor
from speos.utils.config import Config

config = Config()
gwasmapper = GWASMapper(config.input.gene_sets, config.input.gwas, config.input.gwas_mappings)
adjacencymapper = AdjacencyMapper(mapping_file=config.input.adjacency_mappings)
gwasmappings = gwasmapper.get_mappings(tags="Immune_Dysregulation", fields="ground_truth")
adjacencies = adjacencymapper.get_mappings(tags="", fields="name")
preprocessor = PreProcessor(config, gwasmappings, adjacencies)
preprocessor.build_graph()
preprocessor.dump_edgelist(os.path.join("data/misc/all_edgelist.tsv.gz"))
