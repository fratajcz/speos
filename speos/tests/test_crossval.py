from speos.wrappers import OuterCVWrapper, CVWrapper
from speos.utils.config import Config
import unittest
import shutil
import numpy as np


class OuterCrossvalTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:

        cls.config = Config()
        cls.config.logging.dir = "speos/tests/logs/"

        cls.config.name = "ExperimentTest"

        cls.config.model.save_dir = "tests/models/"
        cls.config.inference.save_dir = "tests/results"
        cls.config.input.save_dir = "tests/data"

        cls.config.training.max_epochs = 1
        cls.config.crossval.n_folds = 2

    def setUp(self):
        self.outer_cv = OuterCVWrapper(self.config)

    def tearDown(self):
        shutil.rmtree(self.config.model.save_dir, ignore_errors=True)
        shutil.rmtree(self.config.inference.save_dir, ignore_errors=True)

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
            self.assertEqual(np.sum(np.logical_or.reduce(np.array(inner_test_masks))), self.outer_cv.inner_cv.data.y.sum())

        # make sure that each positive value ends up in one outer holdout set without overlap
        self.assertEqual(np.sum(np.logical_or.reduce(np.array(outer_test_masks))), self.outer_cv.inner_cv.data.y.sum())

    def test_outercv_runs(self):
        self.outer_cv.run()


class InnerCrossvalTest(unittest.TestCase):

    def setUp(self):
        self.config = Config()
        self.config.logging.dir = "speos/tests/logs/"

        self.config.name = "InnerCVTest"

        self.config.model.save_dir = "tests/models/"
        self.config.inference.save_dir = "tests/results"
        self.config.input.save_dir = "tests/data"

        self.config.training.max_epochs = 1
        self.config.crossval.n_folds = 4
        self.config.crossval.positive_only = False

        self.cv = CVWrapper(self.config)

    def tearDown(self):
        shutil.rmtree(self.config.model.save_dir, ignore_errors=True)
        shutil.rmtree(self.config.inference.save_dir, ignore_errors=True)
        shutil.rmtree(self.config.input.save_dir, ignore_errors=True)

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


if __name__ == '__main__':
    unittest.main(warnings='ignore')
