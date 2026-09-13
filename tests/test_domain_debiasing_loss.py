import unittest

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - depends on local environment
    torch = None

if torch is not None:
    from losses.domain_debiasing_loss import DomainDebiasingLoss


@unittest.skipIf(torch is None, "torch is not installed")
class TestDomainDebiasingLoss(unittest.TestCase):
    def test_returns_zero_when_no_cross_center_positive_pairs_exist(self):
        loss_fn = DomainDebiasingLoss()
        features = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        center_ids = torch.tensor([0, 1], dtype=torch.long)
        labels = torch.tensor([0, 1], dtype=torch.long)

        loss = loss_fn(features, center_ids, labels)

        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(float(loss), 0.0)


if __name__ == "__main__":
    unittest.main()
