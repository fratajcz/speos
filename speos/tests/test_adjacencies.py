import unittest
from utils import TestSetup
from speos.preprocessing.handler import InputHandler
from speos.utils.config import Config


class AdjacencyTest(TestSetup):

    def test_dummy_directed(self):
        self.config.input.adjacency = "DummyDirectedGraph"

        preprocessor = InputHandler(self.config).get_preprocessor()
        X, y, adj = preprocessor.get_data(features=False)
        self.assertEqual(adj["DummyDirectedGraph"].shape[1], 6)

    def test_dummy_undirected(self):
        self.config.input.adjacency = "DummyUndirectedGraph"

        preprocessor = InputHandler(self.config).get_preprocessor()
        X, y, adj = preprocessor.get_data(features=False)
        self.assertEqual(adj["DummyUndirectedGraph"].shape[1], 12)

if __name__ == '__main__':
    unittest.main(warnings='ignore')
