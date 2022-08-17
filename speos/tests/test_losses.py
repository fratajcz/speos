from speos.losses.lambdaloss import lambdaLoss
from speos.losses.approxndcg import approxNDCGLoss
from speos.losses.neuralndcg import neuralNDCGLoss
from speos.losses.unbiased import upu, nnpu
import torch

import unittest


class LossesTest(unittest.TestCase):

    def test_lambdarank(self):
        y_true = torch.Tensor((0, 0, 0, 0))
        y_pred = torch.Tensor((0, 0, 0, 0))
        minimal_loss = lambdaLoss(y_pred, y_true)

        self.assertEqual(minimal_loss, 0)

        y_true = torch.Tensor((1, 0, 0, 0))
        y_pred = torch.Tensor((1, 0, 0, 0))
        one_correct = lambdaLoss(y_pred, y_true)

        y_true = torch.Tensor((1, 0, 0, 0))
        y_pred = torch.Tensor((1, 1, 0, 0))
        false_positive = lambdaLoss(y_pred, y_true)

        y_true = torch.Tensor((1, 0, 0, 0))
        y_pred = torch.Tensor((0, 0, 0, 0))
        false_negative = lambdaLoss(y_pred, y_true)

        self.assertLess(one_correct, false_positive)
        self.assertLess(one_correct, false_negative)
        self.assertLess(false_positive, false_negative)

    def test_approxNDCG(self):
        y_true = torch.Tensor((0, 0, 0, 0))
        y_pred = torch.Tensor((0, 0, 0, 0))
        minimal_loss = approxNDCGLoss(y_pred, y_true)

        self.assertEqual(minimal_loss, 0)

        y_true = torch.Tensor((1, 0, 0, 0))
        y_pred = torch.Tensor((1, 0, 0, 0))
        one_correct = approxNDCGLoss(y_pred, y_true)

        y_true = torch.Tensor((1, 0, 0, 0))
        y_pred = torch.Tensor((1, 1, 0, 0))
        false_positive = approxNDCGLoss(y_pred, y_true)

        y_true = torch.Tensor((1, 0, 0, 0))
        y_pred = torch.Tensor((0, 0, 0, 0))
        false_negative = approxNDCGLoss(y_pred, y_true)

        self.assertLess(one_correct, false_positive)
        self.assertLess(one_correct, false_negative)
        self.assertLess(false_positive, false_negative)

    def test_neuralNDCG(self):
        y_true = torch.DoubleTensor((0, 0, 0, 0))
        y_pred = torch.DoubleTensor((0, 0, 0, 0))
        minimal_loss = neuralNDCGLoss(y_pred, y_true)

        self.assertEqual(minimal_loss, 0)

        y_true = torch.DoubleTensor((1, 0, 0, 0))
        y_pred = torch.DoubleTensor((1, 0, 0, 0))
        one_correct = neuralNDCGLoss(y_pred, y_true)

        y_true = torch.DoubleTensor((1, 0, 0, 0))
        y_pred = torch.DoubleTensor((1, 1, 0, 0))
        false_positive = neuralNDCGLoss(y_pred, y_true)

        y_true = torch.DoubleTensor((1, 0, 0, 0))
        y_pred = torch.DoubleTensor((0, 0, 0, 0))
        false_negative = neuralNDCGLoss(y_pred, y_true)

        self.assertLess(one_correct, false_positive)
        self.assertLess(one_correct, false_negative)
        self.assertLess(false_positive, false_negative)

    def test_upu(self):
        y_true = torch.DoubleTensor((1, 0, 0, 0))
        y_pred = torch.DoubleTensor((1, 0, 0, 0))
        one_correct = upu(y_pred, y_true)

        y_true = torch.DoubleTensor((1, 0, 0, 0))
        y_pred = torch.DoubleTensor((1, 1, 0, 0))
        false_positive = upu(y_pred, y_true)

        y_true = torch.DoubleTensor((1, 0, 0, 0))
        y_pred = torch.DoubleTensor((0, 1, 0, 0))
        one_wrong = upu(y_pred, y_true)

        self.assertLess(one_correct, false_positive)
        self.assertLess(one_correct, one_wrong)
        self.assertLess(false_positive, one_wrong)

    def test_nnpu(self):
        y_true = torch.DoubleTensor((1, 0, 0, 0))
        y_pred = torch.DoubleTensor((1, 0, 0, 0))
        one_correct = nnpu(y_pred, y_true)

        y_true = torch.DoubleTensor((1, 0, 0, 0))
        y_pred = torch.DoubleTensor((1, 1, 0, 0))
        false_positive = nnpu(y_pred, y_true)

        y_true = torch.DoubleTensor((1, 0, 0, 0))
        y_pred = torch.DoubleTensor((0, 1, 0, 0))
        one_wrong = nnpu(y_pred, y_true)

        self.assertLessEqual(one_correct, false_positive)
        self.assertLess(one_correct, one_wrong)
        self.assertLess(false_positive, one_wrong)


if __name__ == '__main__':
    unittest.main(warnings='ignore')
