import unittest

from speos.preprocessing.mappers import GWASMapper, AdjacencyMapper
from speos.preprocessing.preprocessor import PreProcessor
from speos.utils.config import Config


class AdjacencyTest(unittest.TestCase):

    def test_intact_direct(self):
        config = Config()
        config.input.adjacency = "IntAct_Direct"
        gwasmapper = GWASMapper(config.input.gene_sets, config.input.gwas)
        gwasmappings = gwasmapper.get_mappings(tags="immune_dysregulation", fields="name")
        adjacencymapper = AdjacencyMapper()
        adjacencies = adjacencymapper.get_mappings(tags=config.input.adjacency, fields="name")

        preprocessor = PreProcessor(config, gwasmappings, adjacencies)
        X, y, adj = preprocessor.get_data()
        self.assertEqual(adj[adjacencies[0]["name"]].shape[1], 14411)

    def test_intact_pa(self):
        config = Config()
        config.input.adjacency = "IntAct_PA"
        gwasmapper = GWASMapper(config.input.gene_sets, config.input.gwas)
        gwasmappings = gwasmapper.get_mappings(tags="immune_dysregulation", fields="name")
        adjacencymapper = AdjacencyMapper()
        adjacencies = adjacencymapper.get_mappings(tags=config.input.adjacency, fields="name")

        preprocessor = PreProcessor(config, gwasmappings, adjacencies)
        X, y, adj = preprocessor.get_data()
        self.assertEqual(adj[adjacencies[0]["name"]].shape[1], 207761)

if __name__ == '__main__':
    unittest.main(warnings='ignore')
