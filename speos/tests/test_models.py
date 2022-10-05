import unittest
from speos.utils.config import Config
from speos.models import ModelBootstrapper
from speos.architectures import RelationalGeneNetwork
from speos.datasets import DatasetBootstrapper, MultiGeneDataset, GeneDataset
from speos.experiment import Experiment, InferenceEngine
from speos.preprocessing.mappers import GWASMapper, AdjacencyMapper
from torch_geometric.nn import GCNConv
import torch
import torch.nn as nn

torch.set_default_tensor_type(torch.DoubleTensor)


class SimpleModelTest(unittest.TestCase):

    def setUp(self):
        self.config = Config()

        self.config.logging.file = "/dev/null"

        self.config.name = "SimpleModelTest"
        self.config.model.model = "SimpleModel"

        self.config.model.save_dir = "tests/"
        self.config.inference.save_dir = "tests/results"
        self.config.training.pos_weight = 2
        self.config.training.dilution = 2
        self.config.model.mp.type = "gcn"

        self.model = ModelBootstrapper(self.config, 90, 1).get_model()

    def test_bootstrap(self):
        self.assertEqual(self.config.model.model, str(self.model.__class__.__name__))

    def test_returns_mp_layers(self):
        layers = self.model.get_mp_layers()
        self.assertEqual(len(layers), self.config.model.mp.n_layers)
        for layer in layers:
            self.assertTrue(isinstance(layer, GCNConv))

    def test_assigns_gcn(self):
        model = ModelBootstrapper(self.config, 90).get_model()
        layers = [module for module in model.architectures[0].mp.modules() if not isinstance(module, nn.Sequential)]
        self.assertEqual("GCNConv", str(layers[1].__class__.__name__))

    def test_repackage_into_one_sequential(self):
        model = ModelBootstrapper(self.config, 90, 1).get_model()

        pre_mp_layers = [module for module in model.architectures[0].pre_mp.modules() if not isinstance(module, nn.Sequential)]
        mp_layers = [module for module in model.architectures[0].mp.modules() if not isinstance(module, nn.Sequential)]
        post_mp_layers = [module for module in model.architectures[0].post_mp.modules() if not isinstance(module, nn.Sequential)]

        repackaged = model.architectures[0].repackage_into_one_sequential()

        repackaged_layers = [module for module in repackaged.modules() if not isinstance(module, nn.Sequential)]

        # see if we have gotten all layers or if some were lost
        self.assertEqual(len(pre_mp_layers) + len(mp_layers) + len(post_mp_layers), len(repackaged_layers))

        old_layers = torch.nn.Sequential(*pre_mp_layers, *mp_layers, *post_mp_layers)
        # see if their weights are identical
        for old_param, new_param in zip(old_layers.parameters(), repackaged.parameters()):
            self.assertTrue(torch.eq(old_param, new_param).all())

    def test_forward(self):
        mappings = GWASMapper(self.config.input.gene_sets, self.config.input.gwas).get_mappings(
            self.config.input.tag, fields=self.config.input.field)

        tag = "" if self.config.input.adjacency == "all" else self.config.input.adjacency
        adjacencies = AdjacencyMapper(self.config.input.adjacency_mappings).get_mappings(tag)

        dataset = DatasetBootstrapper(
            mappings, adjacencies, holdout_size=self.config.input.holdout_size, name=self.config.name, config=self.config).get_dataset()

        model = ModelBootstrapper(self.config, dataset.data.x.shape[1], 1).get_model()

        train_out, loss = model.step(dataset.data, dataset.data.train_mask)

    def test_forward_concat(self):
        mappings = GWASMapper(self.config.input.gene_sets, self.config.input.gwas).get_mappings(
            self.config.input.tag, fields=self.config.input.field)

        tag = "" if self.config.input.adjacency == "all" else self.config.input.adjacency
        adjacencies = AdjacencyMapper(self.config.input.adjacency_mappings).get_mappings(tag)

        dataset = DatasetBootstrapper(
            mappings, adjacencies, holdout_size=self.config.input.holdout_size, name=self.config.name, config=self.config).get_dataset()

        config = self.config.deepcopy()
        config.model.concat_after_mp = True

        model = ModelBootstrapper(config, dataset.data.x.shape[1], 1).get_model()

        train_out, loss = model.step(dataset.data, dataset.data.train_mask)

    def test_forward_skip(self):
        mappings = GWASMapper(self.config.input.gene_sets, self.config.input.gwas).get_mappings(
            self.config.input.tag, fields=self.config.input.field)

        tag = "" if self.config.input.adjacency == "all" else self.config.input.adjacency
        adjacencies = AdjacencyMapper(self.config.input.adjacency_mappings).get_mappings(tag)

        dataset = DatasetBootstrapper(
            mappings, adjacencies, holdout_size=self.config.input.holdout_size, name=self.config.name, config=self.config).get_dataset()

        config = self.config.deepcopy()
        config.model.skip_mp = True

        model = ModelBootstrapper(config, dataset.data.x.shape[1], 1).get_model()

        train_out, loss = model.step(dataset.data, dataset.data.train_mask)

    def test_balance_losses(self):
        losses = torch.Tensor((5, 1, 1))
        truth = torch.LongTensor((1, 0, 0))
        balanced_loss = self.model.balance_losses(losses, truth)
        self.assertEqual(balanced_loss, torch.mean(torch.cat((losses[truth.bool()] * self.config.training.pos_weight, losses[~truth.bool()] / self.config.training.dilution), 0)))

    def test_balance_classes(self):
        losses = torch.Tensor((1, 1, 1, 1))
        truth = torch.Tensor((1, 0, 0, 0))
        positive_losses, negative_losses, dilution = self.model.balance_classes(losses, truth)
        self.assertEqual(positive_losses.shape[0] * dilution, (negative_losses.shape[0]))

    def test_forward_random_input_features(self):

        config = self.config.deepcopy()
        config.input.adjacency = ["BioPlex 3.0 293T"]
        config.input.use_gwas = False
        config.input.use_expression = False


        mappings = GWASMapper(self.config.input.gene_sets, self.config.input.gwas).get_mappings(
            self.config.input.tag, fields=self.config.input.field)

        tag = "" if self.config.input.adjacency == "all" else self.config.input.adjacency
        adjacencies = AdjacencyMapper(self.config.input.adjacency_mappings).get_mappings(tag)

        dataset = DatasetBootstrapper(
            mappings, adjacencies, holdout_size=config.input.holdout_size, name=config.name, config=config).get_dataset()

        self.model = ModelBootstrapper(config, dataset.data.x.shape[1], 1).get_model()

        train_out, loss = self.model.step(dataset.data, dataset.data.train_mask)

    def test_forward_mlp_random_input_features(self):

        config = self.config.deepcopy()
        config.input.adjacency = ["BioPlex 3.0 293T"]
        config.input.use_gwas = False
        config.input.use_expression = False
        config.model.mp.n_layers = 0

        mappings = GWASMapper(self.config.input.gene_sets, self.config.input.gwas).get_mappings(
            self.config.input.tag, fields=self.config.input.field)

        tag = "" if self.config.input.adjacency == "all" else self.config.input.adjacency
        adjacencies = AdjacencyMapper(self.config.input.adjacency_mappings).get_mappings(tag)

        dataset = DatasetBootstrapper(
            mappings, adjacencies, holdout_size=config.input.holdout_size, name=config.name, config=config).get_dataset()

        self.model = ModelBootstrapper(config, dataset.data.x.shape[1], 1).get_model()

        train_out, loss = self.model.step(dataset.data, dataset.data.train_mask)


class RelationalGeneNetworkTest(unittest.TestCase):

    def setUp(self):
        self.config = Config()

        self.config.logging.file = "/dev/null"

        self.config.name = "RelationalGeneNetworkTest"
        self.config.model.model = "SimpleModel"

        self.config.model.save_dir = "tests/"
        self.config.inference.save_dir = "tests/results"
        self.config.training.pos_weight = 2
        self.config.training.dilution = 2
        self.config.input.save_data = False

        self.config.input.adjacency = ["BioPlex"]

    def test_bootstrap(self):
        model = ModelBootstrapper(self.config, 90, 2).get_model()
        self.assertEqual("RelationalGeneNetwork", str(model.architectures[0].__class__.__name__))

    def test_assigns_rgcn(self):
        # if there are 2 adjacencies but a homogenoues conv is specified, return RGCN instead
        model = ModelBootstrapper(self.config, 90, 2).get_model()
        layers = [module for module in model.architectures[0].mp.modules() if not isinstance(module, nn.Sequential)]
        self.assertEqual("RGCNConv", str(layers[1].__class__.__name__))

    def test_hands_through_kwargs_from_config(self):
        # kwarg assignment in config is handed through to the layer definition
        config = self.config.deepcopy()
        config.model.mp.kwargs.update({"num_bases": 5})
        model = ModelBootstrapper(config, 90, 2).get_model()
        layers = [module for module in model.architectures[0].mp.modules() if not isinstance(module, nn.Sequential)]
        self.assertEqual(5, layers[1].num_bases)

    def test_bootstrap_rgcn(self):
        config = self.config.deepcopy()
        config.model.mp.type = "rgcn"
        model = ModelBootstrapper(config, 90, 2).get_model()
        layers = [module for module in model.architectures[0].mp.modules() if not isinstance(module, nn.Sequential)]
        self.assertEqual("RGCNConv", str(layers[1].__class__.__name__))

    def test_bootstrap_film(self):
        config = self.config.deepcopy()
        config.model.mp.type = "film"
        model = ModelBootstrapper(config, 90, 2).get_model()
        layers = [module for module in model.architectures[0].mp.modules() if not isinstance(module, nn.Sequential)]
        self.assertEqual("FiLMConv", str(layers[1].__class__.__name__))

    def test_bootstrap_filmtag(self):
        config = self.config.deepcopy()
        config.model.mp.type = "filmtag"
        model = ModelBootstrapper(config, 90, 2).get_model()
        layers = [module for module in model.architectures[0].mp.modules() if not isinstance(module, nn.Sequential)]
        self.assertEqual("FiLMConv", str(layers[1].__class__.__name__))

    def test_bootstrap_rtag(self):
        config = self.config.deepcopy()
        config.model.mp.type = "rtag"
        model = ModelBootstrapper(config, 90, 2).get_model()
        layers = [module for module in model.architectures[0].mp.modules() if not isinstance(module, nn.Sequential)]
        self.assertEqual("FiLMConv", str(layers[1].__class__.__name__))

    def test_bootstrap_gat(self):
        config = self.config.deepcopy()
        config.model.mp.type = "rgat"
        model = ModelBootstrapper(config, 90, 2).get_model()
        layers = [module for module in model.architectures[0].mp.modules() if not isinstance(module, nn.Sequential)]
        self.assertEqual("RGATConv", str(layers[1].__class__.__name__))

    def test_forward_rgcn(self):

        mappings = GWASMapper().get_mappings(
            self.config.input.tag, fields=self.config.input.field)

        adjacencies = AdjacencyMapper(self.config.input.adjacency_mappings).get_mappings("BioPlex")

        dataset = DatasetBootstrapper(
            mappings, adjacencies, holdout_size=self.config.input.holdout_size, name=self.config.name, config=self.config).get_dataset()

        self.model = ModelBootstrapper(self.config, dataset.data.x.shape[1], 2).get_model()

        train_out, loss = self.model.step(dataset.data, dataset.data.train_mask)

    def test_forward_force_multigraph(self):

        config = self.config.deepcopy()
        config.input.adjacency = ["BioPlex 3.0 293T"]
        config.input.force_multigraph = True
        config.model.mp.type = "film"

        mappings = GWASMapper(self.config.input.gene_sets, self.config.input.gwas).get_mappings(
            self.config.input.tag, fields=self.config.input.field)

        tag = "" if self.config.input.adjacency == "all" else self.config.input.adjacency
        adjacencies = AdjacencyMapper(self.config.input.adjacency_mappings).get_mappings(tag)

        dataset = DatasetBootstrapper(
            mappings, adjacencies, holdout_size=config.input.holdout_size, name=config.name, config=config).get_dataset()

        self.model = ModelBootstrapper(config, dataset.data.x.shape[1], 1).get_model()

        train_out, loss = self.model.step(dataset.data, dataset.data.train_mask)

    def test_forward_rtag(self):
        import numpy as np

        mappings = GWASMapper(self.config.input.gene_sets, self.config.input.gwas).get_mappings(
            self.config.input.tag, fields=self.config.input.field)

        tag = "" if self.config.input.adjacency == "all" else self.config.input.adjacency
        adjacencies = AdjacencyMapper(self.config.input.adjacency_mappings).get_mappings(tag)
        dataset = DatasetBootstrapper(
            mappings, adjacencies, holdout_size=self.config.input.holdout_size, name=self.config.name, config=self.config).get_dataset()

        config = self.config.deepcopy()
        config.model.mp.type = "rtag"
        failed_runs = 0
        failed_run_list = []
        for i in range(10):
            model = ModelBootstrapper(config, dataset.data.x.shape[1], 2).get_model()
            params = model.state_dict()[0]
            for name, param in params.items():
                self.assertTrue(torch.isfinite(param).all(), "params for {} are non-finite BEFORE step".format(name))
            train_out, loss = model.step(dataset.data, dataset.data.train_mask)
            params = model.state_dict()[0]
            for name, param in params.items():
                self.assertTrue(torch.isfinite(param).all(), "params for {} are non-finite AFTER step".format(name))
            try:
                self.assertFalse(np.isnan(loss))
            except AssertionError:
                failed_runs += 1
                failed_run_list.append(i)

        self.assertTrue(failed_runs == 0, "Had {} failed runs (loss is np.nan): {}".format(failed_runs, failed_run_list))

    def test_forward_filmtag(self):
        import numpy as np

        mappings = GWASMapper(self.config.input.gene_sets, self.config.input.gwas).get_mappings(
            self.config.input.tag, fields=self.config.input.field)

        tag = "" if self.config.input.adjacency == "all" else self.config.input.adjacency
        adjacencies = AdjacencyMapper(self.config.input.adjacency_mappings).get_mappings(tag)
        dataset = DatasetBootstrapper(
            mappings, adjacencies, holdout_size=self.config.input.holdout_size, name=self.config.name, config=self.config).get_dataset()

        config = self.config.deepcopy()
        config.model.mp.type = "filmtag"
        failed_runs = 0
        failed_run_list = []
        for i in range(10):
            model = ModelBootstrapper(config, dataset.data.x.shape[1], 2).get_model()
            params = model.state_dict()[0]
            for name, param in params.items():
                self.assertTrue(torch.isfinite(param).all(), "params for {} are non-finite BEFORE step".format(name))
            train_out, loss = model.step(dataset.data, dataset.data.train_mask)
            params = model.state_dict()[0]
            for name, param in params.items():
                self.assertTrue(torch.isfinite(param).all(), "params for {} are non-finite AFTER step".format(name))
            try:
                self.assertFalse(np.isnan(loss))
            except AssertionError:
                failed_runs += 1
                failed_run_list.append(i)

        self.assertTrue(failed_runs == 0, "Had {} failed runs (loss is np.nan): {}".format(failed_runs, failed_run_list))


class SKLearnModelTest(unittest.TestCase):

    def setUp(self):
        self.config = Config()

        self.config.logging.file = "/dev/null"

        self.config.name = "SKLearnModelTest"

        self.config.model.save_dir = "tests/"
        self.config.inference.save_dir = "tests/results"
        self.config.training.pos_weight = 2
        self.config.training.max_epochs = 1

    def test_logistic_regression_bootstrap(self):
        self.config.model.model = "LogisticRegressionModel"
        model = ModelBootstrapper(self.config, 90, 1).get_model()
        self.assertEqual(self.config.model.model, str(model.__class__.__name__))

    def test_logistic_regression_forward(self):
        from torch_geometric.data import Data
        self.config.model.model = "LogisticRegressionModel"
        model = ModelBootstrapper(self.config, 90, 1).get_model()
        x = torch.rand((10, 90))
        y = torch.LongTensor([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        mask = torch.LongTensor([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
        data = Data(x=x, y=y)
        model.step(data, mask)
        self.assertEqual(self.config.model.model, str(model.__class__.__name__))

    def test_run_save_load_regression(self):
        import numpy as np
        self.config.model.model = "LogisticRegressionModel"
        self.experiment = Experiment(self.config)
        resultshandler = self.experiment.resultshandler
        self.inference_engine = InferenceEngine(self.config, resultshandler=resultshandler)
        self.experiment.run()
        self.inference_engine.restore_model()
        _, pred_exp, prob_exp = self.experiment.eval(target="all")
        _, pred_ie, prob_ie = self.inference_engine.eval(target="all")

        self.assertTrue(np.allclose(pred_exp, pred_ie))
        self.assertTrue(np.allclose(prob_exp, prob_ie))

    def test_random_forest_bootstrap(self):
        self.config.model.model = "RandomForestModel"
        model = ModelBootstrapper(self.config, 90, 1).get_model()
        self.assertEqual(self.config.model.model, str(model.__class__.__name__))

    def test_random_forest_forward(self):
        from torch_geometric.data import Data
        self.config.model.model = "RandomForestModel"
        model = ModelBootstrapper(self.config, 90, 1).get_model()
        x = torch.rand((10, 90))
        y = torch.LongTensor([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        mask = torch.LongTensor([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
        data = Data(x=x, y=y)
        model.step(data, mask)
        self.assertEqual(self.config.model.model, str(model.__class__.__name__))

    def test_run_save_load_randomforest(self):
        import numpy as np
        self.config.model.model = "RandomForestModel"
        self.experiment = Experiment(self.config)
        resultshandler = self.experiment.resultshandler
        self.inference_engine = InferenceEngine(self.config, resultshandler=resultshandler)
        self.experiment.run()
        self.inference_engine.restore_model()
        _, pred_exp, prob_exp = self.experiment.eval(target="all")
        _, pred_ie, prob_ie = self.inference_engine.eval(target="all")

        self.assertTrue(np.allclose(pred_exp, pred_ie))
        self.assertTrue(np.allclose(prob_exp, prob_ie))

    def test_svm_bootstrap(self):
        self.config.model.model = "SupportVectorModel"
        model = ModelBootstrapper(self.config, 90, 1).get_model()
        self.assertEqual(self.config.model.model, str(model.__class__.__name__))

    def test_svm_forward(self):
        from torch_geometric.data import Data
        self.config.model.model = "SupportVectorModel"
        model = ModelBootstrapper(self.config, 90, 1).get_model()
        x = torch.rand((10, 90))
        y = torch.LongTensor([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        mask = torch.LongTensor([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
        data = Data(x=x, y=y)
        model.step(data, mask)
        self.assertEqual(self.config.model.model, str(model.__class__.__name__))

    def test_run_save_load_svm(self):
        import numpy as np
        self.config.model.model = "SupportVectorModel"
        self.experiment = Experiment(self.config)
        resultshandler = self.experiment.resultshandler
        self.inference_engine = InferenceEngine(self.config, resultshandler=resultshandler)
        self.experiment.run()
        self.inference_engine.restore_model()
        _, pred_exp, prob_exp = self.experiment.eval(target="all")
        _, pred_ie, prob_ie = self.inference_engine.eval(target="all")

        self.assertTrue(np.allclose(pred_exp, pred_ie))
        self.assertTrue(np.allclose(prob_exp, prob_ie))


class RWRMTest(unittest.TestCase):

    def setUp(self):
        self.config = Config()

        self.config.logging.file = "/dev/null"

        self.config.name = "RWRMTest"

        self.config.model.save_dir = "speos/tests/models"
        self.config.input.save_dir = "speos/tests/data/"
        self.config.inference.save_dir = "speos/tests/results"
        self.config.training.pos_weight = 2
        self.config.training.max_epochs = 1

    def test_rwrm_bootstrap(self):
        self.config.model.model = "RWRM"
        model = ModelBootstrapper(self.config, 90, 1).get_model()
        self.assertEqual(self.config.model.model, str(model.__class__.__name__))

    def test_rwrm_forward(self):
        from torch_geometric.data import Data
        self.config.model.model = "RWRM"
        model = ModelBootstrapper(self.config, 90, 1).get_model()
        x = torch.rand((10, 90))
        y = torch.LongTensor([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        mask = torch.LongTensor([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
        data = Data(x=x, y=y)
        model.step(data, mask)
        self.assertEqual(self.config.model.model, str(model.__class__.__name__))

if __name__ == '__main__':
    unittest.main(warnings='ignore')


