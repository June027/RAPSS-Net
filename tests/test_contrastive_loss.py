import unittest

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - depends on local environment
    torch = None

if torch is not None:
    from losses.contrastive_loss import SupervisedContrastiveLoss


@unittest.skipIf(torch is None, "torch is not installed")
class TestContrastiveLoss(unittest.TestCase):
    def test_returns_finite_scalar_for_valid_batch(self):
        loss_fn = SupervisedContrastiveLoss(temperature=0.2)
        features = torch.tensor(
            [
                [1.0, 0.0],
                [0.9, 0.1],
                [0.0, 1.0],
                [0.1, 0.9],
            ],
            dtype=torch.float32,
        )
        labels = torch.tensor([0, 0, 1, 1], dtype=torch.long)
        loss = loss_fn(features, labels)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(loss), 0.0)

    def test_returns_zero_when_no_positive_pairs_exist(self):
        loss_fn = SupervisedContrastiveLoss(temperature=0.2)
        features = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        labels = torch.tensor([0, 1], dtype=torch.long)
        loss = loss_fn(features, labels)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(float(loss), 0.0)


if __name__ == "__main__":
    unittest.main()
