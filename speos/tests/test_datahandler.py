import unittest
from speos.utils.config import Config
from speos.utils.datahandlers import ResultsHandler
import shutil


class DataHandlerReadTest(unittest.TestCase):

    def setUp(self):
        self.config = Config()
        self.config.parse_yaml("speos/tests/files/inference_test_config.yaml")
        self.config.logging.file = "/dev/null"

        self.config.model.save_dir = "tests/"
        self.config.inference.save_dir = "tests/results"

        # self.resultshandler = ResultsHandler('speos/tests/files/inference_test.h5',read_only=True)
        self.resultshandler = ResultsHandler('results/bm_disorder3.h5', read_only=True)

    def tearDown(self):
        self.resultshandler.close()
        # shutil.rmtree(self.config.inference.save_dir, ignore_errors=True)
        pass

    def test_get_results_for_gene(self):
        self.resultshandler.get_results_for_gene("A1CF")

    def test_get_explanation_for_gene(self):
        self.resultshandler.get_explanation_for_gene("A1CF")


class DataHandlerWriteTest(unittest.TestCase):

    def setUp(self):
        self.config = Config()
        self.config.parse_yaml("speos/tests/files/inference_test_config.yaml")

        self.config.model.save_dir = "speos/tests/models/"
        self.config.inference.save_dir = "speos/tests/results"

        self.config.logging.dir = "speos/tests/logs/"
        self.config.name = "DataHandlerWriteTest"
        self.n_folds = 2

        self.index = ["gene1", "gene2", "gene3"]
        self.resultshandler = ResultsHandler(self.config.inference.save_dir + self.config.name + "h5",
                                             n_folds=self.n_folds,
                                             shape=(3, 3),
                                             explanation=True,
                                             index=self.index)

    def tearDown(self):
        shutil.rmtree(self.config.inference.save_dir, ignore_errors=True)
        shutil.rmtree(self.config.model.save_dir, ignore_errors=True)
        shutil.rmtree(self.config.logging.dir, ignore_errors=True)
        pass

    def test_write_results(self):
        import pandas as pd
        import numpy as np
        results = pd.DataFrame(np.random.rand(3, 6), index=self.index)
        for i in range(self.n_folds):
            self.resultshandler.add_results(results, "fold_{}".format(i))
            read_results = self.resultshandler.get_results_for_fold(i)
            self.assertTrue(np.allclose(results.values, read_results))

        read_gene1 = self.resultshandler.get_results_for_gene("gene1")
        gene1 = results.values[0, :]
        self.assertTrue(np.allclose(gene1, read_gene1, atol=1e-05, rtol=0.01))

    def test_write_explanation(self):
        import pandas as pd
        import numpy as np
        explanations = pd.DataFrame(np.random.rand(3, 3).astype(np.float32), index=self.index)
        for i in range(self.n_folds):
            self.resultshandler.add_explanations(explanations, "fold_{}".format(i))
            read_explanations = self.resultshandler.get_explanation_for_fold(i)
            self.assertTrue(np.allclose(explanations.values, read_explanations))

        read_gene1 = self.resultshandler.get_explanation_for_gene("gene1")
        gene1 = explanations.values[0, :]
        self.assertTrue(np.allclose(gene1, read_gene1, atol=1e-05, rtol=0.01))


if __name__ == '__main__':
    unittest.main(warnings='ignore')
