import unittest
from speos.utils.config import Config
from speos.preprocessing.mappers import GWASMapper, AdjacencyMapper
from speos.datasets import MultiGeneDataset, GeneDataset
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
        mappings = GWASMapper(self.config.input.gene_sets, self.config.input.gwas).get_mappings(
            self.config.input.tag, fields=self.config.input.field)

        tag = "BioPlex 3.0 293T"
        adjacencies = AdjacencyMapper(self.config.input.adjacency_mappings).get_mappings(tag)

        self.dataset = MultiGeneDataset(
            mappings, adjacencies, holdout_size=self.config.input.holdout_size, name=self.config.name, config=self.config)

    def test_multiple_adjacencies(self):
        mappings = GWASMapper(self.config.input.gene_sets, self.config.input.gwas).get_mappings(
            self.config.input.tag, fields=self.config.input.field)

        tag = "BioPlex"
        adjacencies = AdjacencyMapper(self.config.input.adjacency_mappings).get_mappings(tag)

        self.dataset = MultiGeneDataset(
            mappings, adjacencies, holdout_size=self.config.input.holdout_size, name=self.config.name, config=self.config)


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
        mappings = GWASMapper(self.config.input.gene_sets, self.config.input.gwas).get_mappings(
            self.config.input.tag, fields=self.config.input.field)

        tag = "BioPlex 3.0 293T"
        adjacencies = AdjacencyMapper(self.config.input.adjacency_mappings).get_mappings(tag)

        self.dataset = GeneDataset(
            mappings, adjacencies, holdout_size=self.config.input.holdout_size, name=self.config.name, config=self.config)

    def test_multiple_adjacencies(self):
        mappings = GWASMapper(self.config.input.gene_sets, self.config.input.gwas).get_mappings(
            self.config.input.tag, fields=self.config.input.field)

        tag = "BioPlex"
        adjacencies = AdjacencyMapper(self.config.input.adjacency_mappings).get_mappings(tag)

        self.dataset = GeneDataset(
            mappings, adjacencies, holdout_size=self.config.input.holdout_size, name=self.config.name, config=self.config)


if __name__ == '__main__':
    unittest.main(warnings='ignore')
