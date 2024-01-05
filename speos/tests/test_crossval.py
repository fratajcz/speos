from speos.wrappers import OuterCVWrapper, CVWrapper
from speos.utils.config import Config
import unittest
import shutil
import numpy as np
import os
from utils import TestSetup

class OuterCrossvalTest(TestSetup):
    def setUp(self):
        super().setUp()
        self.config.training.max_epochs = 1
        self.config.crossval.n_folds = 2
        self.outer_cv = OuterCVWrapper(self.config)

    def test_correct_holdout_assignment(self):
        import numpy as np
        outer_test_masks = []
        for outer_fold_nr in range(self.outer_cv.n_folds):

            self.outer_cv.inner_cv.test_split = outer_fold_nr
            outer_test_mask = self.outer_cv.inner_cv.test_mask
            outer_test_masks.append(outer_test_mask)

            inner_test_masks = []
            for inner_fold_nr in range(self.outer_cv.inner_cv.n_folds + 1):
                if inner_fold_nr == outer_fold_nr:
                    continue
                self.outer_cv.inner_cv.test_split = inner_fold_nr
                inner_test_masks.append(self.outer_cv.inner_cv.test_mask)
            inner_test_masks.append(outer_test_mask)

            # make sure that each positive value ends up in one inner holdout set or the current outer holdout set without overlap
            self.assertEqual(np.sum(np.logical_or.reduce(np.array(inner_test_masks))), self.outer_cv.inner_cv.data.y.shape[0])

        # make sure that each positive value ends up in one outer holdout set without overlap
        self.assertEqual(np.sum(np.logical_or.reduce(np.array(outer_test_masks))), self.outer_cv.inner_cv.data.y.shape[0])

    def test_outercv_runs(self):
        self.outer_cv.run()


class InnerCrossvalTest(TestSetup):

    def setUp(self):
        super().setUp()

        self.config.training.max_epochs = 1
        self.config.crossval.n_folds = 4
        self.config.crossval.positive_only = False

        self.cv = CVWrapper(self.config)

    def test_no_overlap(self):
        all_samples = np.zeros_like(self.cv.test_mask)
        for indices in self.cv.indices:
            all_samples[indices] = 1

        self.assertEqual(all_samples.max(), 1)
        self.assertEqual(all_samples.min(), 1)
        self.assertEqual(all_samples.mean(), 1)

    def test_same_splits(self):
        another_cv = CVWrapper(self.config)

        for indices, other_indices in zip(self.cv.indices, another_cv.indices):
            self.assertTrue((indices == other_indices).all())

    def test_no_holdout(self):
        new_config = self.config.copy()
        new_config.crossval.hold_out_test = False
        another_cv = CVWrapper(new_config)
        no_test = another_cv.indices
        with_test = self.cv.indices

        #check that we have one less split
        self.assertEqual(len(with_test), len(no_test) + 1)

        #check that all splits together have equal length (i.e. there is nothing missing)
        self.assertEqual(np.sum([len(split) for split in with_test]), np.sum([len(split) for split in no_test]))

if __name__ == '__main__':
    unittest.main(warnings='ignore')
