import pathlib
import tempfile
import unittest

import yaml

from utils.config_manager import ConfigManager


class TestContrastiveConfig(unittest.TestCase):
    def test_active_configs_do_not_enable_contrastive_loss(self):
        root = pathlib.Path(__file__).resolve().parents[1]
        for name in [
            "train_config_Proposed_3D_Base.yaml",
            "train_config_RAPSS_Net.yaml",
        ]:
            config_path = root / "configs" / "server" / "random" / name
            with self.subTest(config=name):
                config = ConfigManager(str(config_path)).config
                self.assertEqual(float(config["loss_weights"].get("contrastive", 0.0)), 0.0)

    def test_non_positive_contrastive_temperature_is_rejected(self):
        root = pathlib.Path(__file__).resolve().parents[1]
        base_config = ConfigManager(
            str(
                root
                / "configs"
                / "server"
                / "random"
                / "train_config_RAPSS_Net.yaml"
            )
        ).config
        bad_config = dict(base_config)
        bad_config["train"] = dict(base_config["train"])
        bad_config["loss_weights"] = dict(base_config.get("loss_weights", {}))
        bad_config["loss_weights"]["contrastive"] = 0.02
        bad_config["train"]["contrastive_temperature"] = 0.0

        with tempfile.TemporaryDirectory() as tmpdir:
            temp_path = pathlib.Path(tmpdir) / "bad_contrastive.yaml"
            temp_path.write_text(
                yaml.safe_dump(bad_config, sort_keys=False), encoding="utf-8"
            )
            with self.assertRaisesRegex(
                ValueError, "train.contrastive_temperature"
            ):
                ConfigManager(str(temp_path))


if __name__ == "__main__":
    unittest.main()
