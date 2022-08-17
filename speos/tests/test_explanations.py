import unittest
from speos.experiment import Experiment
from speos.utils.config import Config
from speos.explanation import MessagePassingExplainer
import shutil
import torch
import numpy as np
from speos.utils.logger import setup_logger


class MPExplanationTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:

        cls.config = Config()
        cls.config.logging.dir = "speos/tests/logs/"

        cls.config.name = "MPExplanationTest"

        cls.config.model.save_dir = "speos/tests/models/"
        cls.config.inference.save_dir = "speos/tests/results"
        cls.config.optim.lr = 0.01

        cls.config.input.adjacency = "BioPlex"
        cls.config.input.save_data = True
        cls.config.input.save_dir = "speos/tests/data"

        cls.config.training.max_epochs = 10
        cls.config.model.mp.type = "film"

        cls.experiment = Experiment(cls.config)
        cls.explainer = MessagePassingExplainer(cls.experiment.model, cls.experiment.data, ["0", "1"], cls.config)

    def test_get_beta_gamma(self):

        latent_node_features = self.explainer.get_latent_features()
        mp_layer = self.explainer.model.get_mp_layers()[0]

        beta, gamma = self.explainer.get_beta_gamma(latent_node_features, mp_layer, 0)

        print("oki")

    def test_calculate_messages(self):
        edges = torch.LongTensor([[0, 1], [0, 2]]).T.long()
        x = torch.rand((3, 10))
        gammas = torch.rand((3, 10))
        betas = torch.rand((3, 10))

        messages = self.explainer.calculate_messages(edges, x, betas, gammas)

        # even though the sender is identical (node 0), the messages that are sent should differ
        self.assertFalse(torch.eq(messages[0,:], messages[1,:]).all())

        gammas = torch.zeros((3, 10))
        betas = torch.zeros((3, 10))

        messages = self.explainer.calculate_messages(edges, x, betas, gammas)

        # with beta and gamma of zero, the message should be zero
        self.assertTrue(torch.eq(messages, torch.zeros((2, 10))).all())

        gammas = torch.ones((3, 10))
        betas = torch.zeros((3, 10))

        messages = self.explainer.calculate_messages(edges, x, betas, gammas)

        # with gamma of one and beta of zero, the message should be identical to the sender features
        self.assertTrue(torch.eq(messages, x[edges[0, :]]).all())

        gammas = torch.zeros((3, 10))
        betas = torch.rand((3, 10))

        messages = self.explainer.calculate_messages(edges, x, betas, gammas)

        # with gamma of zero the message should be identical to the betas of the receivers
        self.assertTrue(torch.eq(messages, betas[edges[1, :]]).all())

    def test_get_messages(self):
        edges = torch.LongTensor([[0, 1], [0, 2], [2, 3], [2, 4]]).T.long()
        edge_types = torch.LongTensor([0, 0, 1, 1])
        num_adjacencies = 2
        x = torch.rand((5, 10))
        gammas = torch.rand((num_adjacencies, 5, 10))
        betas = torch.rand((num_adjacencies, 5, 10))
        for i in range(num_adjacencies):
            messages = self.explainer.get_messages(x, edges, edge_types, betas[i].squeeze(), gammas[i].squeeze(), i)
        
        print("oki")

    def test_inspect_film(self):

        amplification = self.explainer.inspect_film(testing=True)
        
        print("oki")