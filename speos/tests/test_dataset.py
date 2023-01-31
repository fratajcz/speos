import unittest
from speos.utils.config import Config
from speos.preprocessing.mappers import GWASMapper, AdjacencyMapper
from speos.preprocessing.datasets import MultiGeneDataset, GeneDataset
import shutil


class MultiGeneDatasetTest(unittest.TestCase):

    def setUp(self):
        self.config = Config()

        self.config.logging.file = "/dev/null"
        self.config.input.save_data = True
        self.config.input.save_dir = "speos/tests/data"

        self.config.name = "MultiGeneDatasetTest"

        self.config.model.save_dir = "tests/"
        self.config.inference.save_dir = "tests/results"

    def tearDown(self):
        shutil.rmtree(self.config.input.save_dir, ignore_errors=True)
        pass

    def test_one_adjacency(self):
        
        config = self.config.copy()
        config.input.adjacency = "BioPlex 3.0 293T"

        self.dataset = MultiGeneDataset(holdout_size=self.config.input.holdout_size, name=self.config.name, config=config)

        self.assertEqual(self.dataset.num_relations, 1)

    def test_multiple_adjacencies(self):
        config = self.config.copy()
        config.input.adjacency = "BioPlex"

        self.dataset = MultiGeneDataset(holdout_size=self.config.input.holdout_size, name=self.config.name, config=config)

        self.assertEqual(self.dataset.num_relations, 2)

class GeneDatasetTest(unittest.TestCase):

    def setUp(self):
        self.config = Config()

        self.config.name = "GeneDatasetTest"
        self.config.input.save_data = True
        self.config.input.save_dir = "coregenes/tests/data"

        self.config.logging.file = "/dev/null"

        self.config.model.save_dir = "tests/"
        self.config.inference.save_dir = "tests/results"

    def tearDown(self):
        shutil.rmtree(self.config.input.save_dir, ignore_errors=True)
        pass

    def test_one_adjacency(self):
        config = self.config.copy()
        config.input.adjacency = "BioPlex 3.0 293T"

        self.dataset = GeneDataset(holdout_size=self.config.input.holdout_size, name=self.config.name, config=config)

        self.assertEqual(self.dataset.num_relations, 1)

    def test_multiple_adjacencies(self):
        config = self.config.copy()
        config.input.adjacency = "BioPlex"

        self.dataset = GeneDataset(holdout_size=self.config.input.holdout_size, name=self.config.name, config=config)

        # this results in 2 relations by definition, but they are not distinguishable in the data, for this use MultiGeneDataset
        self.assertEqual(self.dataset.num_relations, 2)


if __name__ == '__main__':
    unittest.main(warnings='ignore')
