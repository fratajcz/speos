import unittest
from speos.utils.metrics import MetricsHelper
import numpy as np


class MetricsTest(unittest.TestCase):

    def setUp(self) -> None:

        y_truth = np.array((1, 0, 1, 0, 1, 0))
        masks = {"train": np.array((1, 1, 0, 0, 0, 0), dtype=np.bool8),
                 "val": np.array((0, 0, 1, 1, 0, 0), dtype=np.bool8),
                 "test": np.array((0, 0, 0, 0, 1, 1), dtype=np.bool8)}
        self.metrics = MetricsHelper(y_truth, 0.5, masks)

    def test_update(self):
        self.metrics.update(np.zeros((6,)), "train")
        self.assertEquals(self.metrics.mask_key, "train")
        self.assertTrue(np.equal(self.metrics.output, np.zeros((6,))).all())
        self.metrics.update(np.ones((6,)), "val")
        self.assertEqual(self.metrics.mask_key, "val")
        self.assertTrue(np.equal(self.metrics.output, np.ones((6,))).all())

    def test_mrr_raw(self):
        output = np.array((1, 0, 0, 0, 0, 0))
        self.metrics.update(output, "train")
        self.assertEqual(self.metrics.get_metrics("mrr_raw")[0], 1)

        output = np.array((0.5, 1, 0, 0, 0, 0))
        self.metrics.update(output, "train")
        self.assertEqual(self.metrics.get_metrics("mrr_raw")[0], 0.5)

        output = np.array((0.5, 1, 0.7, 0, 0, 0))
        self.metrics.update(output, "train")
        self.assertEqual(self.metrics.get_metrics("mrr_raw")[0], 1 / 3)

        output = np.array((0, 1, 1, 1, 1, 1))
        self.metrics.update(output, "train")
        self.assertEqual(self.metrics.get_metrics("mrr_raw")[0], 1 / 6)

    def test_mrr_filtered(self):
        output = np.array((1, 0, 0, 0, 0, 0))
        self.metrics.update(output, "train")
        self.assertEqual(self.metrics.get_metrics("mrr_filtered")[0], 1)

        output = np.array((0.5, 1, 0, 0, 0, 0))
        self.metrics.update(output, "train")
        self.assertEqual(self.metrics.get_metrics("mrr_filtered")[0], 0.5)

        output = np.array((0.5, 1, 0.7, 0, 0, 0))
        self.metrics.update(output, "train")
        self.assertEqual(self.metrics.get_metrics("mrr_filtered")[0], 0.5)

        output = np.array((0, 1, 1, 1, 1, 1))
        self.metrics.update(output, "train")
        self.assertEqual(self.metrics.get_metrics("mrr_filtered")[0], 1 / 4)

    def test_mean_rank_raw(self):
        output = np.array((1, 0, 0, 0, 0, 0))
        self.metrics.update(output, "train")
        self.assertEqual(self.metrics.get_metrics("mean_rank_raw")[0], 1)

        output = np.array((0.5, 1, 0, 0, 0, 0))
        self.metrics.update(output, "train")
        self.assertEqual(self.metrics.get_metrics("mean_rank_raw")[0], 2)

        output = np.array((0.5, 1, 0.7, 0, 0, 0))
        self.metrics.update(output, "train")
        self.assertEqual(self.metrics.get_metrics("mean_rank_raw")[0], 3)

        output = np.array((0, 1, 1, 1, 1, 1))
        self.metrics.update(output, "train")
        self.assertEqual(self.metrics.get_metrics("mean_rank_raw")[0], 6)

    def test_mean_rank_filtered(self):
        output = np.array((1, 0, 0, 0, 0, 0))
        self.metrics.update(output, "train")
        self.assertEqual(self.metrics.get_metrics("mean_rank_filtered")[0], 1)

        output = np.array((0.5, 1, 0, 0, 0, 0))
        self.metrics.update(output, "train")
        self.assertEqual(self.metrics.get_metrics("mean_rank_filtered")[0], 2)

        output = np.array((0.5, 1, 0.7, 0, 0, 0))
        self.metrics.update(output, "train")
        self.assertEqual(self.metrics.get_metrics("mean_rank_filtered")[0], 2)

        output = np.array((0, 1, 1, 1, 1, 1))
        self.metrics.update(output, "train")
        self.assertEqual(self.metrics.get_metrics("mean_rank_filtered")[0], 4)

    def test_hits_at_k_raw(self):
        output = np.array((1, 0, 0, 0, 0, 0))
        self.metrics.update(output, "train")
        self.assertEqual(self.metrics.hits_at_k_raw(1), 1)
        self.assertEqual(self.metrics.hits_at_k_raw(2), 1)

        output = np.array((0.5, 1, 0, 0, 0, 0))
        self.metrics.update(output, "train")
        self.assertEqual(self.metrics.hits_at_k_raw(1), 0)
        self.assertEqual(self.metrics.hits_at_k_raw(2), 1)

        output = np.array((0.5, 1, 0.7, 0, 0, 0))
        self.metrics.update(output, "train")
        self.assertEqual(self.metrics.hits_at_k_raw(1), 0)
        self.assertEqual(self.metrics.hits_at_k_raw(2), 0)

        output = np.array((0, 1, 1, 1, 1, 1))
        self.metrics.update(output, "train")
        self.assertEqual(self.metrics.hits_at_k_raw(1), 0)
        self.assertEqual(self.metrics.hits_at_k_raw(2), 0)

    def test_hits_at_k_filtered(self):
        output = np.array((1, 0, 0, 0, 0, 0))
        self.metrics.update(output, "train")
        self.assertEqual(self.metrics.hits_at_k_filtered(1), 1)
        self.assertEqual(self.metrics.hits_at_k_filtered(2), 1)

        output = np.array((0.5, 1, 0, 0, 0, 0))
        self.metrics.update(output, "train")
        self.assertEqual(self.metrics.hits_at_k_filtered(1), 0)
        self.assertEqual(self.metrics.hits_at_k_filtered(2), 1)

        output = np.array((0.5, 1, 0.7, 0, 0, 0))
        self.metrics.update(output, "train")
        self.assertEqual(self.metrics.hits_at_k_filtered(1), 0)
        self.assertEqual(self.metrics.hits_at_k_filtered(2), 1)

        output = np.array((0, 1, 1, 1, 1, 1))
        self.metrics.update(output, "train")
        self.assertEqual(self.metrics.hits_at_k_filtered(1), 0)
        self.assertEqual(self.metrics.hits_at_k_filtered(2), 0)

    def test_au_rank_cdf(self):
        output = np.array((0.7, 0, 0.8, 0, 0.9, 0))
        self.metrics.update(output, "train")
        self.assertEqual(self.metrics.au_rank_cdf, 1)
        self.metrics.update(output, "all")
        self.assertEqual(self.metrics.au_rank_cdf, 1)

        output = np.array((1, 0, 0.9, 0.1, 0.8, 0.2))
        self.metrics.update(output, "train")
        self.assertEqual(self.metrics.au_rank_cdf, 1)
        self.metrics.update(output, "all")
        self.assertEqual(self.metrics.au_rank_cdf, 1)

        output = np.array((0, 1, 0, 1, 0, 1))
        self.metrics.update(output, "train")
        self.assertEqual(self.metrics.au_rank_cdf, 0)
        self.metrics.update(output, "all")
        self.assertEqual(self.metrics.au_rank_cdf, 0)

        # we need an easily divisible numver of examples for easy benchmarking
        y_truth = np.array((1, 1, 0,   0,   1,   0,    1,    0))
        output = np.array(( 0, 1, 0.2, 0.8, 0.9, 0.15, 0.05, 0.1))
        masks = {"train": np.array((1, 1, 1, 1, 0, 0, 0, 0), dtype=np.bool8),
                 "val": np.array((0, 0, 0, 0, 1, 1, 0, 0), dtype=np.bool8),
                 "test": np.array((0, 0, 0, 0, 0, 0, 1, 1), dtype=np.bool8)}
        self.metrics = MetricsHelper(y_truth, 0.5, masks)

        # half of train false, val completely correct, test completely false, half of all correct
        """
        ranks: [0,  1,   2,    3,   4,   5,    6,      7]
        out:   [1,  0.9, 0.8, 0.2, 0.15, 0.1,  0.05,   0]
        truth: [1,  1,   0,    0,   0,   0,    1,      1]
        train: [1,  0,   1,    1,   0,   0,    0,      1]
        val:   [0,  1,   0,    0,   1,   0,    0,      0]
        test:  [0,  0,   0,    0,   0,   1,    1,      0]

        for train this selects:
        ranks: [0,  x,   1,    2,   3,   4,    x,      5]
        out:   [1,  x, 0.8, 0.2, 0.15, 0.1,  0.05,     0]
        truth: [1,  x,   0,    0,   0,   0,    x,      1]
        cutoff at: 4
        cdf:   [0.5, 0.5, 0.5, 0.5] = 0.5

        for val this selects:
        ranks: [x,  1,   2,    3,   4,   5,    x,      x]
        out:   [x,  0.9, 0.8, 0.2, 0.15, 0.1,  x,      x]
        truth: [x,  1,   0,    0,   0,   0,    x,      x]
        cutoff at: 2
        cdf:   [1, 1] = 1

        for test this selects:
        ranks: [x,  x,   0,    1,   2,   3,    4,     5]
        out:   [x,  x, 0.8,  0.2, 0.15, 0.1,  0.05,   0]
        truth: [x,  x,   0,    0,   0,   0,    1,     1]
        cutoff at: 2
        cdf:   [0, 0] = 0

        for all this selects:
        ranks: [0,  1,   2,    3,   4,   5,    6,      7]
        out:   [1,  0.9, 0.8, 0.2, 0.15, 0.1,  0.05,   0]
        truth: [1,  1,   0,    0,   0,   0,    1,      1]
        cutoff at: 4
        cdf:   [0.5, 0.5, 0.5, 0.5] = 0.5 (since both positives tie for rank 0, we have tpr=0.5 from the first rank on)
        """
        self.metrics.update(output, "train")
        self.assertEqual(self.metrics.au_rank_cdf, 0.5)
        self.metrics.update(output, "val")
        self.assertAlmostEqual(self.metrics.au_rank_cdf, 1)
        self.metrics.update(output, "test")
        self.assertEqual(self.metrics.au_rank_cdf, 0)
        self.metrics.update(output, "all")
        self.assertEqual(self.metrics.au_rank_cdf, 0.5)
  
    def test_process_function(self):
        output = np.array((1, 0, 0, 0, 0, 0))
        self.metrics.update(output, "train")

        self.assertEqual(self.metrics.get_metrics("hits_at_1_filtered")[0], 1)


if __name__ == '__main__':
    unittest.main(warnings='ignore')
