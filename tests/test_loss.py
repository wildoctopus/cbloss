import unittest

import torch

from cb_loss.loss import ClassBalancedLoss, FocalLoss


class TestLoss(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_BCEfocal_loss(self):
        # Define the loss function with default parameters
        loss_fn = FocalLoss(num_classes=2)

        # Test Focal Loss with binary classification
        outputs = torch.tensor([[2.1], [0.5], [2.1], [0.5]]).to(self.device)
        targets = torch.tensor([[0.0], [1.0], [1.0], [0.0]]).to(self.device)
        loss = loss_fn(outputs, targets)
        expected_loss = 0.5512
        self.assertAlmostEqual(loss.item(), expected_loss, delta=0.01)

    def test_CEfocal_loss(self):
        # Define the loss function with default parameters
        loss_fn = FocalLoss(num_classes=3)

        # Test multi-class classification case with balanced classes
        outputs = torch.tensor([[0.5, -0.5, 1], [-1, 2, -1], [0, 0, 0]], dtype=torch.float32).to(self.device)
        targets = torch.tensor([2, 1, 0], dtype=torch.long).to(self.device)
        loss = loss_fn(outputs, targets)
        expected_loss = 0.2044
        self.assertAlmostEqual(loss.item(), expected_loss, delta=0.01)

    def test_CB_loss(self):
        # Define the loss function with default parameters
        num_classes = 3
        loss_fn = FocalLoss(num_classes=num_classes, reduction="none")

        # number of samples per class in the training dataset
        samples_per_class = [30, 100, 25]  # 30, 100, 25 samples for labels 0, 1 and 2, respectively

        criterian = ClassBalancedLoss(
            samples_per_cls=samples_per_class, beta=0.2, num_classes=num_classes, loss_func=loss_fn
        )

        # Test multi-class classification case with balanced classes
        outputs = torch.tensor([[0.5, -0.5, 1], [-1, 2, -1], [0, 0, 0]], dtype=torch.float32).to(self.device)
        targets = torch.tensor([2, 1, 0], dtype=torch.long).to(self.device)
        loss = criterian(outputs, targets)
        expected_loss = 0.2044
        self.assertAlmostEqual(loss.item(), expected_loss, delta=0.01)


if __name__ == "__main__":
    unittest.main()
