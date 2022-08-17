import unittest
from speos.experiment import Experiment, InferenceEngine
from speos.utils.config import Config
from speos.postprocessing.postprocessor import PostProcessor
import shutil
import torch
import numpy as np
from speos.utils.logger import setup_logger


class ExperimentTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:

        cls.config = Config()
        cls.config.logging.dir = "speos/tests/logs/"

        cls.config.name = "ExperimentTest"

        cls.config.model.save_dir = "speos/tests/models/"
        cls.config.inference.save_dir = "speos/tests/results"
        cls.config.optim.lr = 0.01

        cls.config.input.adjacency = "BioPlex 3.0 293T"
        #cls.config.input.adjacency = "BioPlex"
        cls.config.input.save_data = True
        cls.config.input.save_dir = "speos/tests/data"

        cls.config.training.max_epochs = 10

        cls.experiment = Experiment(cls.config)
        resultshandler = cls.experiment.resultshandler
        cls.inference_engine = InferenceEngine(cls.config, resultshandler=resultshandler)
        cls.postprocessor = PostProcessor(cls.config)
        cls.params_old = [{key: values.clone() for parameters in cls.experiment.model.state_dict() for key, values in parameters.items()}]
        cls.experiment.run()
        cls.inference_engine.restore_model()

    def tearDown(self):
        shutil.rmtree(self.config.model.save_dir, ignore_errors=True)
        shutil.rmtree(self.config.inference.save_dir, ignore_errors=True)
        shutil.rmtree(self.config.logging.dir, ignore_errors=True)
        shutil.rmtree(self.config.input.save_dir, ignore_errors=True)

    def test_inference_engine_restores_parameters_from_experiment(self):

        params_exp = self.experiment.model.state_dict()
        params_ie = self.inference_engine.model.state_dict()

        for single_arch_params_exp, single_arch_params_ie in zip(params_exp, params_ie):
            for exp_values, ie_values in zip(single_arch_params_exp.values(), single_arch_params_ie.values()):
                self.assertTrue(torch.equal(exp_values, ie_values))

    def test_inference_engine_predicts_the_same_as_experiment(self):

        _, pred_exp, prob_exp = self.experiment.eval(target="all")
        _, pred_ie, prob_ie = self.inference_engine.eval(target="all")

        self.assertTrue(np.allclose(pred_exp, pred_ie))
        self.assertTrue(np.allclose(prob_exp, prob_ie))

    def test_model_updates_parameters(self):

        params_new = self.experiment.model.state_dict()

        fail = False

        for old, new in zip(self.params_old, params_new):
            for old_key, new_key, in zip(old.keys(), new.keys()):
                old_values = old[old_key]
                new_values = new[new_key]
                try:
                    self.assertFalse(torch.equal(old_values, new_values), msg='{} not updated: old: {}, new: {}'.format(old_key, old_values, new_values))
                except AssertionError:
                    fail = True
                    percentage_not_updated = torch.eq(old_values, new_values).sum() * 100 / old_values.reshape(-1).shape[0]
                    logger = setup_logger(self.config, __name__)
                    logger.warning("Did not update {}% of parameters in {}".format(percentage_not_updated, old_key))

        if fail:
            raise ValueError("At least one layer failed to update weights. See logs for warnings.")


if __name__ == '__main__':
    unittest.main(warnings='ignore')
