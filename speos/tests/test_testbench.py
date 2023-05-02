import unittest
from speos.benchmark import TestBench


class TestBenchTest(unittest.TestCase):

    def test_read_parameter_list(self):
        tb = TestBench("speos/tests/files/parameter_list.yaml")
        self.assertEqual(len(tb.parameter_list), 2)
        self.assertEqual(len(tb.parameter_list[0]), 2)
        self.assertEqual(tb.parameter_list[0]["name"], "bar")

    def test_adapt_config(self):
        tb = TestBench("speos/tests/files/parameter_list.yaml")
        config = tb.adapt_config(tb.parameter_list[0])
        self.assertEqual(config.name, "bar")

    def test_compile_resultshandlers(self):
        tb = TestBench("speos/tests/files/benchmark_parameters.yaml", config_path="speos/tests/files/benchmark_config.yaml")
        tb.compile_resultshandlers()
        self.assertEqual(tb.resultshandlers[0], "./results/benchconfig_test_benchmark_gcnrep0.h5")

    def test_compile_configs(self):
        tb = TestBench("speos/tests/files/benchmark_parameters.yaml", config_path="speos/tests/files/benchmark_config.yaml")
        tb.compile_configs()
        self.assertEqual(tb.configs[0].model.mp.n_layers, 1)
        self.assertEqual(tb.configs[tb.repeats].model.mp.n_layers, 0)


if __name__ == '__main__':
    unittest.main(warnings='ignore')
