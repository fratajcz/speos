import unittest
import json
import shutil

from speos.postprocessing.postprocessor import PostProcessor
from speos.utils.config import Config


class PostProcessorTest(unittest.TestCase):

    def setUp(self):
        self.config = Config()
        self.config.name = "TestPostProcessor"

        self.config.logging.dir = "speos/tests/logs/"
        self.config.pp.save_dir = "speos/tests/results"
        self.config.pp.plot_dir = "speos/tests/plots"
        self.config.model.save_dir = "speos/tests/models/"
        self.config.inference.save_dir = "speos/tests/results"

        self.pp = PostProcessor(self.config)

        self.test_outer_results = "speos/tests/files/cardiovascular_filmouter_results.json"
        self.results_file = "speos/tests/files/cardiovascular_film_outer_0_fold_1.tsv"
        #self.test_outer_results = "/home/icb/florin.ratajczak/ppi-core-genes/results/c7e39douter_results.json"
        #self.results_file = "/home/icb/florin.ratajczak/ppi-core-genes/results/c7e39d_outer_0_fold_1_sorted.tsv"
        # self.test_outer_results = "/home/icb/florin.ratajczak/ppi-core-genes/results/962982outer_results.json"
        # self.results_file = "/home/icb/florin.ratajczak/ppi-core-genes/results/962982_outer_0_fold_1.tsv"

    def tearDown(self):
        shutil.rmtree(self.config.model.save_dir, ignore_errors=True)
        shutil.rmtree(self.config.inference.save_dir, ignore_errors=True)

    def test_random_overlap_descriptive_algorithm(self):
        import numpy as np

        config = self.config.copy()
        config.crossval.n_folds = 5

        pp = PostProcessor(config)
        pp.num_runs_for_random_experiments = 100

        eligible_genes = np.asarray([str(x) for x in range(0, 100)])
        kept_genes = np.asarray([str(x) for x in range(0, 100, 10)])

        eligible_genes = [eligible_genes] * config.crossval.n_folds
        kept_genes = [kept_genes] * config.crossval.n_folds

        mean_counter, sd_counter = pp.get_random_overlap(eligible_genes, kept_genes, algorithm="descriptive")


    def test_random_overlap_fast_algorithm(self):
        import numpy as np

        config = self.config.copy()
        config.crossval.n_folds = 5

        pp = PostProcessor(config)
        pp.num_runs_for_random_experiments = 100

        eligible_genes = np.asarray([str(x) for x in range(0, 100)])
        kept_genes = np.asarray([str(x) for x in range(0, 100, 10)])

        eligible_genes = [eligible_genes] * config.crossval.n_folds
        kept_genes = [kept_genes] * config.crossval.n_folds

        mean_counter, sd_counter = pp.get_random_overlap(eligible_genes, kept_genes, algorithm="fast")


    def test_random_overlap_both_algorithms_identical_results(self):
        import numpy as np

        config = self.config.copy()
        config.crossval.n_folds = 5

        pp = PostProcessor(config)
        pp.num_runs_for_random_experiments = 100

        eligible_genes = np.asarray([str(x) for x in range(0, 100)])
        kept_genes = np.asarray([str(x) for x in range(0, 100, 10)])

        eligible_genes = [eligible_genes] * config.crossval.n_folds
        kept_genes = [kept_genes] * config.crossval.n_folds

        mean_counter_fast, sd_counter_fast = pp.get_random_overlap(eligible_genes, kept_genes, algorithm="fast")
        mean_counter_descriptive, sd_counter_descriptive = pp.get_random_overlap(eligible_genes, kept_genes, algorithm="descriptive")

        for fast, descriptive in zip((mean_counter_fast, sd_counter_fast), (mean_counter_descriptive, sd_counter_descriptive)):
            fast_values = list(fast.values())
            descriptive_values = list(descriptive.values())
            self.assertTrue(np.allclose(list(fast.values()), list(descriptive.values()), atol=0.5, rtol=0.1))
        
    def test_random_overlap_fast_faster_than_descriptive(self):
        import timeit
        import numpy as np

        config = self.config.copy()
        config.crossval.n_folds = 10

        pp = PostProcessor(config)
        pp.num_runs_for_random_experiments = 100

        eligible_genes = np.asarray([str(x) for x in range(0, 1000)])
        kept_genes = np.asarray([str(x) for x in range(0, 1000, 10)])

        eligible_genes = [eligible_genes] * config.crossval.n_folds
        kept_genes = [kept_genes] * config.crossval.n_folds

        fast = timeit.timeit(lambda: pp.get_random_overlap(eligible_genes, kept_genes, algorithm="fast"), number=10)
        descriptive = timeit.timeit(lambda: pp.get_random_overlap(eligible_genes, kept_genes, algorithm="descriptive"), number=10)
        self.assertLess(fast, descriptive)

    def test_drugtarget(self):

        with open(self.test_outer_results, "r") as file:
            outer_results = json.load(file)

        self.pp.outer_result = outer_results

        print(self.pp.drugtarget(self.results_file))

    def test_druggable(self):

        with open(self.test_outer_results, "r") as file:
            outer_results = json.load(file)

        self.pp.outer_result = outer_results

        print(self.pp.druggable(self.results_file))

    def test_mouseKO(self):

        with open(self.test_outer_results, "r") as file:
            outer_results = json.load(file)

        self.pp.outer_result = outer_results

        print(self.pp.mouseKO(self.results_file))

    def test_mouseKO_missing_phenotype(self):

        config = self.config.copy()
        config.input.tag = "autism"

        pp = PostProcessor(config)

        with open(self.test_outer_results, "r") as file:
            outer_results = json.load(file)

        pp.outer_result = outer_results

        self.assertIsNone(self.pp.mouseKO())

    def test_lof(self):

        with open(self.test_outer_results, "r") as file:
            outer_results = json.load(file)

        self.pp.outer_result = outer_results
        
        lof, tukey = self.pp.lof_intolerance(self.results_file)
        print(tukey)

    def test_pathwayea(self):

        with open(self.test_outer_results, "r") as file:
            outer_results = json.load(file)

        self.pp.outer_result = outer_results

        print(self.pp.pathway(self.results_file))

    def test_hpoea(self):

        with open(self.test_outer_results, "r") as file:
            outer_results = json.load(file)

        self.pp.outer_result = outer_results

        print(self.pp.hpo_enrichment(self.results_file))

    def test_goea(self):

        with open(self.test_outer_results, "r") as file:
            outer_results = json.load(file)

        self.pp.outer_result = outer_results

        print(self.pp.go_enrichment(self.results_file))

    def test_dge_cad(self):

        with open(self.test_outer_results, "r") as file:
            outer_results = json.load(file)

        self.pp.config.input.tag = "Cardiovascular_Disease"
        self.pp.outer_result = outer_results

        print(self.pp.dge(self.results_file))

    def test_contingency_table(self):
        import numpy as np
        full_set = set((1, 2, 3, 4, 5, 6))
        A = set((1, 2))
        B = set((2, 3, 4))

        testarray = np.array([[1, 2], [1, 2]])

        array = self.pp.make_contingency_table(full_set, A, B)

        self.assertTrue(np.equal(array, testarray).all())


if __name__ == '__main__':
    unittest.main(warnings='ignore')
