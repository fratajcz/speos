import unittest
from speos.experiment import InferenceEngine
from speos.utils.config import Config
import shutil


class InferenceEngineTest(unittest.TestCase):

    def setUp(self):
        self.config = Config()
        self.config.parse_yaml("speos/tests/files/inference_test_config.yaml")
        self.config.logging.dir = "speos/tests/logs/"

        self.config.model.save_dir = "speos/tests/files/"
        self.config.inference.save_dir = "speos/tests/results"

        self.ie = InferenceEngine(self.config)

    def tearDown(self):
        shutil.rmtree(self.config.inference.save_dir, ignore_errors=True)
        pass

    def test_infer(self):
        self.ie.infer()
        self.ie.resultshandler.close()


if __name__ == '__main__':
    unittest.main(warnings='ignore')
