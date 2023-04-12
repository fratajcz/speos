from speos.layers import *
from torch.nn.functional import binary_cross_entropy_with_logits as loss_function
from speos.utils.config import Config
from speos.architectures import RelationalGeneNetwork
import unittest
import torch
import numpy as np


class HypLinearTest(unittest.TestCase):

    def test_init(self):
        hlin = HypLinear(in_channels=10, out_channels=10, c=1.5)

    def test_forward(self):
        x_input = torch.rand((5, 10))

        hlin = HypLinear(in_channels=10, out_channels=10, c=1.5)
        x = hlin.forward(x_input)
        self.assertTrue(not torch.allclose(x, x_input))

         # check if implicitely calling forward works too
        x_direct = hlin(x_input)
        self.assertTrue(torch.allclose(x_direct, x))

class HypActTest(unittest.TestCase):

    def test_init(self):
        hact = HypAct(act=torch.nn.ELU(), c_in=1.5, c_out=1.5)

    def test_forward(self):
        x_input = torch.rand((5, 10))

        hact = HypAct(act=torch.nn.ELU(), c_in=1.5, c_out=1.5, first=True)
        x = hact.forward(x_input)
        self.assertTrue(not torch.allclose(x, x_input))

         # check if implicitely calling forward works too
        x_direct = hact(x_input)
        self.assertTrue(torch.allclose(x_direct, x))

    def test_project_and_back_again(self):
        x_input = torch.rand((5, 10))

        hact_exp = HypAct(act=torch.nn.Identity(), c_in=1.5, c_out=1.5, first=True)
        hact_log = HypAct(act=torch.nn.Identity(), c_in=1.5, c_out=1.5, last=True)

        x_exp = hact_exp.forward(x_input)

        # check if features in hyperbolic space are different
        self.assertTrue(not torch.allclose(x_input, x_exp))

        x_tangent = hact_log.forward(x_exp)

        # check if features mapped back to euclidean space are the same again
        self.assertTrue(torch.allclose(x_input, x_tangent))

    def test_sandwich(self):
        x_input = torch.rand((5, 10))

        hact_exp = HypAct(act=torch.nn.Identity(), c_out=1.5, first=True)
        hact_mid = HypAct(act=torch.nn.Identity(), c_in=1.5, c_out=0.5)
        hact_log = HypAct(act=torch.nn.Identity(), c_in=0.5, last=True)

        x_exp_first = hact_exp.forward(x_input)

        # check if features in hyperbolic space are different
        self.assertTrue(not torch.allclose(x_input, x_exp_first))

        x_exp_second = hact_mid.forward(x_exp_first)

        # check if features in hyperbolic space with different curvature are different
        self.assertTrue(not torch.allclose(x_exp_second, x_exp_first))

        x_tangent = hact_log.forward(x_exp_second)

        # check if features mapped back to euclidean space are the same again
        self.assertTrue(torch.allclose(x_input, x_tangent))


class HGCNConvTest(unittest.TestCase):

    def test_init(self):
        hgcn = HGCNConv(in_channels=10, out_channels=10, c=1.5)

    def test_forward(self):
        edges = torch.LongTensor([[0, 1], [0, 2], [2, 3], [2, 4]])
        x_input = torch.rand((5, 10))

        hgcn = HGCNConv(in_channels=10, out_channels=10, c=1.5)
        x = hgcn.forward(x_input, edges.T.long())
        self.assertTrue(not torch.allclose(x, x_input))

         # check if implicitely calling forward works too
        x_direct = hgcn(x_input, edges.T.long())
        self.assertTrue(torch.allclose(x_direct, x))

class RTAGConvTest(unittest.TestCase):

    def test_init(self):
        rgat = RTAGConv(in_channels=10, out_channels=10, K=3, num_relations=3)

    def test_forward(self):
        edges = torch.LongTensor([[0, 1], [0, 2], [2, 3], [2, 4]])
        edge_types = torch.LongTensor([0, 1, 0, 1])
        x_input = torch.rand((5, 10))

        rtag = RTAGConv(in_channels=10, out_channels=10, K=3, num_relations=2)
        x = rtag.forward(x_input, edges.T.long(), edge_types.long())
        self.assertTrue(not torch.allclose(x, x_input))

         # check if implicitely calling forward works too
        x_direct = rtag(x_input, edges.T.long(), edge_types.long())
        self.assertTrue(torch.allclose(x_direct, x))

    def test_backward(self):
        for _ in range(10):
            edges = torch.LongTensor([[0, 1], [0, 2], [2, 3], [2, 4]])
            edge_types = torch.LongTensor([0, 1, 0, 1])
            x_input = torch.rand((5, 10))

            rtag = RTAGConv(in_channels=10, out_channels=10, K=3, num_relations=2)
            x = rtag.forward(x_input, edges.T.long(), edge_types.long())
            self.assertTrue(rtag.weight.grad is None)
            loss = loss_function(x.sum(dim=1), torch.Tensor([0, 1, 0, 1, 0]))
            loss.backward()
            self.assertTrue(rtag.weight.grad is not None)
            self.assertFalse(np.isnan(rtag.weight.grad).any())

    def test_architecture(self):
        edge_index = torch.LongTensor([[0, 0, 2, 2],
                                       [1, 2, 3, 4]])
        edge_types = torch.LongTensor([0, 1, 0, 1])
        x_input = torch.rand((5, 10))

        num_relations = 2
        edges = {}
        for i in range(num_relations):
            mask = edge_types == i
            edges.update({("gene", str(i)): edge_index[:, mask]})

        config = Config()
        config.model.mp.type = "rtag"

        # need to rewrite the data structure

        model = RelationalGeneNetwork(config, 10, num_relations)
        x = model({"gene": x_input}, edges)
        self.assertTrue(not torch.allclose(x, x_input))


class FiLMTAGConvTest(unittest.TestCase):

    def test_init(self):
        filmtag = FiLMTAGConv(in_channels=10, out_channels=10, K=3, num_relations=3)

    def test_forward(self):
        edges = torch.LongTensor([[0, 1], [0, 2], [2, 3], [2, 4]])
        edge_types = torch.LongTensor([0, 1, 0, 1])
        x_input = torch.rand((5, 10))

        rtag = FiLMTAGConv(in_channels=10, out_channels=10, K=3, num_relations=2)
        x = rtag.forward(x_input, edges.T.long(), edge_types.long())
        self.assertTrue(not torch.allclose(x, x_input))


class MLPFiLMTest(unittest.TestCase):

    def test_init(self):
        filmtag = MLPFiLM(in_channels=10, out_channels=10, num_relations=3)

    def test_forward(self):
        edges = torch.LongTensor([[0, 1], [0, 2], [2, 3], [2, 4]])
        edge_types = torch.LongTensor([0, 1, 0, 1])
        x_input = torch.rand((5, 10))

        mlpfilm = MLPFiLM(in_channels=10, out_channels=10, num_relations=2)
        x = mlpfilm.forward(x_input, edges.T.long(), edge_types.long())
        self.assertTrue(not torch.allclose(x, x_input))

class FiLMFiLMTest(unittest.TestCase):

    def test_init(self):
        filmfilm = FiLMFiLM(in_channels=10, out_channels=10, num_relations=3)

    def test_forward(self):
        edges = torch.LongTensor([[0, 1], [0, 2], [2, 3], [2, 4]])
        edge_types = torch.LongTensor([0, 1, 0, 1])
        x_input = torch.rand((5, 10))

        filmfilm = FiLMFiLM(in_channels=10, out_channels=10, num_relations=2)
        x = filmfilm.forward(x_input, edges.T.long(), edge_types.long())
        self.assertTrue(not torch.allclose(x, x_input))


class RGATConvTest(unittest.TestCase):

    def test_init(self):
        rgat = RGATConv(in_channels=10, out_channels=10, num_relations=3)

    def test_forward(self):
        edges = torch.LongTensor([[0, 1], [0, 2], [2, 3], [2, 4]])
        edge_types = torch.LongTensor([0, 1, 0, 1])
        x_input = torch.rand((5, 10))

        rtag = RGATConv(in_channels=10, out_channels=10, num_relations=2)
        x = rtag.forward(x_input, edges.T.long(), edge_types.long())
        self.assertTrue(not torch.allclose(x, x_input))


    def test_architecture(self):
        edge_index = torch.LongTensor([[0, 0, 2, 2],
                                       [1, 2, 3, 4]])
        edge_types = torch.LongTensor([0, 1, 0, 1])
        x_input = torch.rand((5, 10))

        num_relations = 2
        edges = {}
        for i in range(num_relations):
            mask = edge_types == i
            edges.update({("gene", str(i)): edge_index[:, mask]})

        config = Config()
        config.model.mp.type = "rgat"

        # TODO: need to rewrite the data structure

        model = RelationalGeneNetwork(config, 10, num_relations)
        x = model({"gene": x_input}, edges)
        self.assertTrue(not torch.allclose(x, x_input))


class RGATTAGConvTest(unittest.TestCase):

    def test_init(self):
        rgattag = RGATTAGConv(in_channels=10, out_channels=10, K=3, num_relations=3)

    def test_forward(self):
        edges = torch.LongTensor([[0, 1], [0, 2], [2, 3], [2, 4]])
        edge_types = torch.LongTensor([0, 1, 0, 1])
        x_input = torch.rand((5, 10))

        rtag = RGATTAGConv(in_channels=10, out_channels=10, K=3, num_relations=2)
        x = rtag.forward(x_input, edges.T.long(), edge_types.long())
        self.assertTrue(not torch.allclose(x, x_input))


if __name__ == '__main__':
    unittest.main(warnings='ignore')
