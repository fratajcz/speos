import unittest
from speos.utils.config import Config
from speos.experiment import Experiment
import shutil
import os


class CompanionTest(unittest.TestCase):

    def setUp(self) -> None:

        self.config = Config()
        self.config.logging.dir = "speos/tests/logs/companiontest"

        self.config.name = "CompanionTest"
        self.config.crossval.n_folds = 1

        self.config.model.save_dir = "speos/tests/models/companiontest"
        self.config.inference.save_dir = "speos/tests/resultscompaniontest"

        self.config.training.max_epochs = 15

        self.config.es.mode = "max"  # erroneously set to max so that we enforce stopping after patience
        self.config.es.patience = 2

        self.config.scheduler.mode = "max"
        self.config.scheduler.patience = 1
        self.config.scheduler.limit = 1e-10
        self.config.scheduler.factor = 0.1

        for directory in [self.config.model.save_dir, self.config.inference.save_dir, self.config.logging.dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)

        self.experiment = Experiment(self.config)

    def tearDown(self):
        shutil.rmtree(self.config.model.save_dir, ignore_errors=True)
        shutil.rmtree(self.config.inference.save_dir, ignore_errors=True)
        shutil.rmtree(self.config.logging.dir, ignore_errors=True)

    def test_step_functional(self):

        _ = self.experiment.earlystopper.patience = 5
        for param_group in self.experiment.model.optimizers[0].param_groups:
            lr_before = float(param_group['lr'])

        self.experiment.run()

        _ = self.experiment.earlystopper.patience

        for param_group in self.experiment.model.optimizers[0].param_groups:
            lr_after = float(param_group['lr'])

        self.assertGreater(lr_before, lr_after)

    def test_step_earlystopper(self):

        patience_before = self.experiment.earlystopper.patience = 2

        stop_training = self.experiment.earlystopper.step(1e10)
        self.assertFalse(stop_training)
        patience_after = self.experiment.earlystopper.patience
        self.assertEqual(patience_after, patience_before)

        for i in range(patience_before):
            stop_training = self.experiment.earlystopper.step(-2 * i)
            if i < patience_before - 1:
                self.assertFalse(stop_training)
            else:
                self.assertTrue(stop_training)

    def test_step_scheduler(self):

        patience_before = self.experiment.scheduler.patience = 2

        for param_group in self.experiment.model.optimizers[0].param_groups:
            lr_before = float(param_group['lr'])

        self.experiment.scheduler.step(1e+10)
        self.experiment.scheduler.step(1e-10)

        patience_after = self.experiment.scheduler.patience

        for param_group in self.experiment.model.optimizers[0].param_groups:
            lr_after = float(param_group['lr'])

        self.assertEqual(lr_before * self.config.scheduler.factor, lr_after)
        self.assertEqual(patience_before - 1, patience_after)


if __name__ == '__main__':
    unittest.main(warnings='ignore')
