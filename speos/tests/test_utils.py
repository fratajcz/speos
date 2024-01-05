import unittest
import torch

class NNUtilsTest(unittest.TestCase):
        
    def test_typed_to_sparse_single_adjacency(self):
        from speos.utils.nn_utils import typed_edges_to_sparse_tensor

        x = torch.rand((5, 5))

        row_indices = [0, 2, 3]
        col_indices = [1, 3, 4]

        edge_dict = {"test": torch.LongTensor([row_indices,
                                               col_indices])}
        edge_index_sparse, encoder = typed_edges_to_sparse_tensor(x, edge_dict)

        self.assertEqual(edge_index_sparse.storage.row().tolist(), row_indices)
        self.assertEqual(edge_index_sparse.storage.col().tolist(), col_indices)

if __name__ == '__main__':
    unittest.main(warnings='ignore')