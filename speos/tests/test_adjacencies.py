import unittest

from speos.preprocessing.handler import InputHandler
from speos.utils.config import Config


class AdjacencyTest(unittest.TestCase):

    def test_intact_direct(self):
        config = Config()
        config.input.adjacency = "IntAct_Direct"

        preprocessor = InputHandler(config).get_preprocessor()
        X, y, adj = preprocessor.get_data()
        self.assertEqual(adj["IntActDirect"].shape[1], 14411)

    def test_intact_pa(self):
        config = Config()
        config.input.adjacency = "IntAct_PA"

        preprocessor = InputHandler(config).get_preprocessor()
        X, y, adj = preprocessor.get_data()
        self.assertEqual(adj["IntActPA"].shape[1], 207761)

if __name__ == '__main__':
    unittest.main(warnings='ignore')
