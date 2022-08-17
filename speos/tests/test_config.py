from speos.utils.config import Config
import unittest
import shutil


class ConfigTest(unittest.TestCase):

    def tearDown(self) -> None:
        shutil.rmtree("testconfig.yaml", ignore_errors=True)

    def test_config_can_be_restored(self):
        config = Config()
        config.save(path="testconfig.yaml")

        config2 = Config()
        config2.parse_yaml("testconfig.yaml")

        self.recursiveEqual(config, config2)

    def recursiveEqual(self, dict1, dict2):
        for key in dict1.keys():
            if type(dict1[key]) == type(dict1):
                self.recursiveEqual(dict1[key], dict2[key])
            else:
                self.assertEqual(dict1[key], dict2[key])

    def test_copy_is_equal(self):
        config = Config()
        config.name = "SomeName"
        config.model.mp.n_layers = 10
        config2 = config.deepcopy()

        self.recursiveEqual(config, config2)

    def test_copy_is_independent(self):
        config = Config()
        config2 = config.deepcopy()
        config2.name = "HasChanged"
        config2.model.mp.n_layers = 10

        self.assertNotEqual(config.name, config2.name)
        self.assertNotEqual(config.model.mp.n_layers, config2.model.mp.n_layers)

    def test_model_kwargs_parsing(self):

        config_str = "model:\n  kwargs: {foo: 'bar'}"

        with open("testconfig.yaml", "w") as file:
            file.writelines(config_str)

        # parse kwargs into an empty kwargs dict
        config = Config()
        config.parse_yaml("testconfig.yaml")
        self.assertIn("foo", config.model.kwargs.keys())
        self.assertIn("bar", config.model.kwargs.values())

        # parse kwargs into a non-empty kwargs dict
        config2 = Config()
        config2.model.kwargs.update({"yaml": "rules"})
        config2.parse_yaml("testconfig.yaml")
        self.assertIn("foo", config2.model.kwargs.keys())
        self.assertIn("bar", config2.model.kwargs.values())
        self.assertIn("yaml", config2.model.kwargs.keys())
        self.assertIn("rules", config2.model.kwargs.values())



if __name__ == '__main__':
    unittest.main(warnings='ignore')
