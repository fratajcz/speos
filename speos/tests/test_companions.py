import unittest
from speos.utils.config import Config
from speos.experiment import Experiment
import shutil
import os
from utils import TestSetup


class CompanionTest(TestSetup):

    def setUp(self) -> None:
        super().setUp()

        self.config.name = "CompanionTest"
        self.config.crossval.n_folds = 1

        self.config.training.max_epochs = 15


        self.config.es.mode = "min"  # erroneously set to max so that we enforce stopping after patience
        self.config.es.metric = "auroc"
        self.config.es.patience = 5

        self.config.scheduler.mode = "min"
        self.config.scheduler.patience = 2
        self.config.scheduler.limit = 1e-10
        self.config.scheduler.factor = 0.1

        self.experiment = Experiment(self.config)


    def test_step_functional(self):

        for param_group in self.experiment.model.optimizers[0].param_groups:
            lr_before = float(param_group['lr'])

        self.experiment.run()

        for param_group in self.experiment.model.optimizers[0].param_groups:
            lr_after = float(param_group['lr'])

        self.assertGreater(lr_before, lr_after)

    def test_step_earlystopper(self):

        patience_before = self.experiment.earlystopper.patience

        # we pass a very small value while settings is in "min" so the patience should not change
        stop_training = self.experiment.earlystopper.step(1e-10)
        self.assertFalse(stop_training)
        patience_after = self.experiment.earlystopper.patience
        self.assertEqual(patience_after, patience_before)

        for i in range(patience_before):
            stop_training = self.experiment.earlystopper.step(1 + (2 * i)) # starts at 1 and steadily increases
            if i < patience_before - 1:
                # for the first n-1 ticks we don't stop training
                self.assertFalse(stop_training)
            else:
                # for the nth tick we have exhausted the patience and stop training
                self.assertTrue(stop_training)

    def test_step_scheduler(self):

        patience_before = self.experiment.scheduler.patience

        for param_group in self.experiment.model.optimizers[0].param_groups:
            lr_before = float(param_group['lr'])

        self.experiment.scheduler.step(1e+10)
    
        patience_after = self.experiment.scheduler.patience

        # after one step, only the patience should have decreased
        self.assertEqual(patience_before - 1, patience_after)

        for param_group in self.experiment.model.optimizers[0].param_groups:
            lr_after = float(param_group['lr'])

        # but the learning rate should be stable
        self.assertEqual(lr_before, lr_after)

        self.experiment.scheduler.step(1e+10)

        for param_group in self.experiment.model.optimizers[0].param_groups:
            lr_after = float(param_group['lr'])

        # after the second step we have exhausted the patience and the learning rate has decreesed by the factor
        self.assertEqual(lr_before * self.config.scheduler.factor, lr_after)
        


if __name__ == '__main__':
    unittest.main(warnings='ignore')
