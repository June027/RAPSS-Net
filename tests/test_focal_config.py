import pathlib
import tempfile
import unittest

import yaml

from utils.config_manager import ConfigManager


class TestFocalConfig(unittest.TestCase):
    def test_active_configs_use_cross_entropy_by_default(self):
        root = pathlib.Path(__file__).resolve().parents[1]
        for name in [
            "train_config_Proposed_3D_Base.yaml",
            "train_config_RAPSS_Net.yaml",
        ]:
            config_path = root / "configs" / "server" / "random" / name
            with self.subTest(config=name):
                config = ConfigManager(str(config_path)).config
                self.assertEqual(
                    str(config["train"].get("classification_loss", "ce")).lower(), "ce"
                )

    def test_invalid_classification_loss_is_rejected(self):
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
        bad_config["train"]["classification_loss"] = "bad_loss"

        with tempfile.TemporaryDirectory() as tmpdir:
            temp_path = pathlib.Path(tmpdir) / "bad_focal.yaml"
            temp_path.write_text(
                yaml.safe_dump(bad_config, sort_keys=False), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "train.classification_loss"):
                ConfigManager(str(temp_path))


if __name__ == "__main__":
    unittest.main()
