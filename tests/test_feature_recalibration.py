import unittest

from utils.config_manager import ConfigManager

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - test environment dependent
    torch = None


class TestFeatureRecalibration(unittest.TestCase):
    @unittest.skipIf(torch is None, "torch is not installed in this interpreter")
    def test_recalibration_preserves_shape_and_finite_values(self):
        from models.blocks.cnn_blocks import GlobalFeatureRecalibration

        module = GlobalFeatureRecalibration(dim=32, reduction=4, dropout=0.1)
        x = torch.randn(5, 32)
        y = module(x)
        self.assertEqual(tuple(y.shape), (5, 32))
        self.assertTrue(torch.isfinite(y).all().item())

    def test_active_configs_do_not_enable_feature_recalibration_variant(self):
        config = ConfigManager(
            "configs/server/random/train_config_RAPSS_Net.yaml"
        ).config
        self.assertFalse(config["model"].get("use_feature_recalibration", False))


if __name__ == "__main__":
    unittest.main()
