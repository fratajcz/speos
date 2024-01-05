import unittest
from speos.utils.config import Config
from speos.preprocessing.mappers import GWASMapper, AdjacencyMapper
from speos.preprocessing.datasets import MultiGeneDataset, GeneDataset, DatasetBootstrapper
import shutil
from utils import TestSetup


class MultiGeneDatasetTest(TestSetup):

    def setUp(self):
        super().setUp()

        self.config.input.save_data = True
        self.config.name = "MultiGeneDatasetTest"

    def test_one_adjacency(self):
        
        config = self.config.copy()
        config.input.adjacency = "DummyUndirectedGraph"

        self.dataset = MultiGeneDataset(holdout_size=self.config.input.holdout_size, name=self.config.name, config=config)

        self.assertEqual(self.dataset.num_relations, 1)

    def test_multiple_adjacencies(self):
        config = self.config.copy()
        config.input.adjacency = "irectedgraph"

        self.dataset = MultiGeneDataset(holdout_size=self.config.input.holdout_size, name=self.config.name, config=config)

        self.assertEqual(self.dataset.num_relations, 2)

class GeneDatasetTest(TestSetup):

    def setUp(self):
        super().setUp()

        self.config.name = "GeneDatasetTest"
        self.config.input.save_data = True

    def test_one_adjacency(self):
        config = self.config.copy()
        config.input.adjacency = "DummyUndirectedGraph"

        self.dataset = GeneDataset(holdout_size=self.config.input.holdout_size, name=self.config.name, config=config)

        self.assertEqual(self.dataset.num_relations, 1)
        self.assertTrue(isinstance(self.dataset, GeneDataset))

    def test_multiple_adjacencies(self):
        config = self.config.copy()
        config.input.adjacency = "irectedgraph"

        self.dataset = GeneDataset(holdout_size=self.config.input.holdout_size, name=self.config.name, config=config)

        # this results in 2 relations by definition, but they are not typed in the data, for this usecase select MultiGeneDataset
        self.assertEqual(self.dataset.num_relations, 2)
        self.assertTrue(isinstance(self.dataset, GeneDataset))


class DatasetBootstrapperTest(TestSetup):
    def setUp(self):
        super().setUp()

        self.config.name = "DatasetBootstrapperTest"
        self.config.input.save_data = True

    def test_one_adjacency(self):
        config = self.config.copy()
        config.input.adjacency = "DummyUndirectedGraph"

        self.dataset = DatasetBootstrapper(name=config.name, holdout_size=config.input.holdout_size, config=config).get_dataset()

        self.assertEqual(self.dataset.num_relations, 1)
        self.assertTrue(isinstance(self.dataset, GeneDataset))

    def test_multiple_adjacencies(self):
        config = self.config.copy()
        config.input.adjacency = "irectedgraph"

        self.dataset = DatasetBootstrapper(name=config.name, holdout_size=config.input.holdout_size, config=config).get_dataset()

        # the bootstrapper automatically selects the MultiGeneDataset
        self.assertEqual(self.dataset.num_relations, 2)
        self.assertTrue(isinstance(self.dataset, MultiGeneDataset))
    

if __name__ == '__main__':
    unittest.main(warnings='ignore')
    
