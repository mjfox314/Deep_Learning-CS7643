import torch
import torch.nn.functional as F
import numpy as np
import unittest

from losses.focal_loss import FocalLoss, reweight

def logit(p: torch.Tensor) -> torch.Tensor:
    return torch.log(p / (1 - p))

class TestFocalLoss(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        np.random.seed(42)
        self.class_size_list = [645, 387, 232, 139]
        self.per_class_weight = reweight(self.class_size_list)

    def test_reweighting(self) -> None:
        self.assertTrue(torch.is_tensor(self.per_class_weight))
        self.assertEqual(self.per_class_weight.shape[0], len(self.class_size_list))
        expected = np.array([0.4043, 0.6652, 1.1011, 1.8294])
        np.testing.assert_array_almost_equal(self.per_class_weight.numpy(), expected, 4)

    def test_focal_loss_equals_ce_loss(self) -> None:
        inputs = logit(
            torch.tensor(
                [
                    [
                        [0.95, 0.55, 0.12, 0.05],
                        [0.09, 0.95, 0.36, 0.11],
                        [0.06, 0.12, 0.56, 0.07],
                        [0.09, 0.15, 0.25, 0.45],
                    ]
                ],
                dtype=torch.float32,
            ).squeeze(0)
        )
        targets = torch.tensor([1, 3, 0, 2], dtype=torch.long).squeeze(0)

        focalloss = FocalLoss(weight=self.per_class_weight, gamma=0.)
        focal_loss = focalloss.forward(input=inputs, target=targets)
        ce_loss = F.cross_entropy(inputs, targets, weight=self.per_class_weight.float())

        self.assertAlmostEquals(ce_loss.item(), focal_loss.item(), places=6)