import unittest

from utils.loss_schedule import compute_aux_loss_warmup_factor


class TestLossSchedule(unittest.TestCase):
    def test_aux_loss_warmup_factor_caps_at_one(self):
        self.assertAlmostEqual(compute_aux_loss_warmup_factor(20, 10), 1.0)

    def test_aux_loss_warmup_factor_scales_during_warmup(self):
        self.assertAlmostEqual(compute_aux_loss_warmup_factor(3, 10), 0.3)

    def test_non_positive_warmup_epochs_disable_scaling(self):
        self.assertAlmostEqual(compute_aux_loss_warmup_factor(1, 0), 1.0)


if __name__ == "__main__":
    unittest.main()
