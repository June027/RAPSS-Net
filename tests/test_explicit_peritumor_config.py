import unittest

from utils.config_manager import ConfigManager


class TestExplicitPeritumorConfig(unittest.TestCase):
    def test_final_config_uses_explicit_peritumor_ring(self):
        config = ConfigManager(
            "configs/server/random/train_config_RAPSS_Net.yaml"
        ).config
        self.assertTrue(config["model"]["use_peritumor"])
        self.assertTrue(config["model"].get("use_explicit_peritumor_ring", False))
        self.assertEqual(int(config["model"].get("peritumor_ring_kernel_size")), 5)
        self.assertTrue(
            config["model"].get("use_multiscale_feature_aggregation", False)
        )


if __name__ == "__main__":
    unittest.main()
