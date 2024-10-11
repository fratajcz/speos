import unittest
import json

from speos.postprocessing.postprocessor import PostProcessor
from speos.utils.config import Config
from speos.tests.utils import TestSetup
import os

NODATA = int(os.getenv('NODATA', '1'))

class PostProcessorTestNoData(TestSetup):
    def setUp(self):
        super().setUp()
        self.config.name = "TestPostProcessor"

        self.pp = PostProcessor(self.config, translation_table=self.translation_table_path)

        self.test_outer_results = "speos/tests/files/dummy_outer_results.json"
        self.results_file = "speos/tests/files/dummy_inner_results.tsv"

    def test_random_overlap_descriptive_algorithm(self):
        import numpy as np

        config = self.config.copy()
        config.crossval.n_folds = 5

        pp = PostProcessor(config, translation_table=self.translation_table_path)
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

        pp = PostProcessor(config, translation_table=self.translation_table_path)
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

        pp = PostProcessor(config, translation_table=self.translation_table_path)
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
            self.assertTrue(np.allclose(list(fast.values()), list(descriptive.values()), atol=0.5, rtol=0.5))
        
    def test_random_overlap_fast_faster_than_descriptive(self):
        import timeit
        import numpy as np

        config = self.config.copy()
        config.crossval.n_folds = 10

        pp = PostProcessor(config, translation_table=self.translation_table_path)
        pp.num_runs_for_random_experiments = 100

        eligible_genes = np.asarray([str(x) for x in range(0, 1000)])
        kept_genes = np.asarray([str(x) for x in range(0, 1000, 10)])

        eligible_genes = [eligible_genes] * config.crossval.n_folds
        kept_genes = [kept_genes] * config.crossval.n_folds

        fast = timeit.timeit(lambda: pp.get_random_overlap(eligible_genes, kept_genes, algorithm="fast"), number=3)
        descriptive = timeit.timeit(lambda: pp.get_random_overlap(eligible_genes, kept_genes, algorithm="descriptive"), number=3)
        self.assertLess(fast, descriptive)


    def test_contingency_table(self):
        import numpy as np
        full_set = set((1, 2, 3, 4, 5, 6))
        A = set((1, 2))
        B = set((2, 3, 4))

        testarray = np.array([[1, 2], [1, 2]])

        array = self.pp.make_contingency_table(full_set, A, B)

        self.assertTrue(np.equal(array, testarray).all())


@unittest.skipIf(NODATA, "nodata")
class PostProcessorTest(TestSetup):

    def setUp(self):
        super().setUp()
        self.config.name = "TestPostProcessor"

        self.pp = PostProcessor(self.config, translation_table=self.translation_table_path)

        self.test_outer_results = "speos/tests/files/dummy_outer_results.json"
        self.results_file = "speos/tests/files/dummy_inner_results.tsv"

    def test_drugtarget(self):
        # cant do without data
        with open(self.test_outer_results, "r") as file:
            outer_results = json.load(file)

        self.pp.outer_result = outer_results

        self.pp.drugtarget(self.results_file)

    def test_druggable(self):
        # cant do without data
        with open(self.test_outer_results, "r") as file:
            outer_results = json.load(file)

        self.pp.outer_result = outer_results

        self.pp.druggable(self.results_file)

    def test_mouseKO(self):
        # cant do without data
        with open(self.test_outer_results, "r") as file:
            outer_results = json.load(file)

        self.pp.outer_result = outer_results

        self.pp.mouseKO(self.results_file)

    def test_mouseKO_missing_phenotype(self):
        # cant do without data
        config = self.config.copy()
        config.input.tag = "autism"

        pp = PostProcessor(config)

        with open(self.test_outer_results, "r") as file:
            outer_results = json.load(file)

        pp.outer_result = outer_results

        self.assertIsNone(pp.mouseKO(self.results_file))

    def test_lof(self):
        # cant do without data
        with open(self.test_outer_results, "r") as file:
            outer_results = json.load(file)

        self.pp.outer_result = outer_results
        
        lof, tukey = self.pp.lof_intolerance(self.results_file)

    def test_pathwayea(self):
        # cant do without data
        with open(self.test_outer_results, "r") as file:
            outer_results = json.load(file)

        self.pp.outer_result = outer_results

        self.pp.pathway(self.results_file)

    def test_hpoea(self):
        # cant do without data
        with open(self.test_outer_results, "r") as file:
            outer_results = json.load(file)

        self.pp.outer_result = outer_results

        self.pp.hpo_enrichment(self.results_file)

    def test_goea(self):
        # cant do without data
        with open(self.test_outer_results, "r") as file:
            outer_results = json.load(file)

        self.pp.outer_result = outer_results

        self.pp.go_enrichment(self.results_file)

    
    def test_dge_missing_phenotype(self):

        config = self.config.copy()
        config.input.tag = "autism"

        pp = PostProcessor(config, translation_table=self.translation_table_path)

        with open(self.test_outer_results, "r") as file:
            outer_results = json.load(file)
        
        pp.outer_result = outer_results

        self.assertIsNone(pp.dge(self.results_file))

   


if __name__ == '__main__':
    suite = unittest.TestSuite()
    suite.addTest(PostProcessorTest('test_random_overlap_both_algorithms_identical_results'))
    suite.run("result")
