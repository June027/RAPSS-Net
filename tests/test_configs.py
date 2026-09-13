import pathlib
import tempfile
import unittest

import yaml

from utils.config_manager import ConfigManager


class TestConfigs(unittest.TestCase):
    def test_all_training_configs_load(self):
        root = pathlib.Path(__file__).resolve().parents[1]
        config_paths = [
            *sorted((root / "configs" / "server" / "random").glob("*.yaml")),
            *sorted((root / "configs" / "server" / "cv").glob("*.yaml")),
        ]

        for config_path in config_paths:
            with self.subTest(config=str(config_path)):
                config = ConfigManager(str(config_path)).config
                self.assertIn("model", config)
                self.assertIn("train", config)
                self.assertIn(
                    config["train"]["validation_mode"], {"global_cv", "random_split"}
                )
                if config["train"]["validation_mode"] == "global_cv":
                    self.assertEqual(int(config["train"]["n_splits"]), 5)

    def test_single_run_validation_mode_is_rejected(self):
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
        bad_config["train"]["validation_mode"] = "single_run"

        with tempfile.TemporaryDirectory() as tmpdir:
            temp_path = pathlib.Path(tmpdir) / "bad_config.yaml"
            temp_path.write_text(
                yaml.safe_dump(bad_config, sort_keys=False), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "Unsupported train.validation_mode"):
                ConfigManager(str(temp_path))

    def test_invalid_intensity_gamma_is_rejected(self):
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
        bad_config["data"] = dict(base_config["data"])
        bad_config["data"]["augmentation"] = dict(base_config["data"]["augmentation"])
        bad_config["data"]["augmentation"]["intensity_gamma"] = [1.5, 0.7]

        with tempfile.TemporaryDirectory() as tmpdir:
            temp_path = pathlib.Path(tmpdir) / "bad_gamma.yaml"
            temp_path.write_text(
                yaml.safe_dump(bad_config, sort_keys=False), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "data.augmentation.intensity_gamma"):
                ConfigManager(str(temp_path))

    def test_even_peritumor_ring_kernel_is_rejected(self):
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
        bad_config["model"] = dict(base_config["model"])
        bad_config["model"]["peritumor_ring_kernel_size"] = 4

        with tempfile.TemporaryDirectory() as tmpdir:
            temp_path = pathlib.Path(tmpdir) / "bad_ring_kernel.yaml"
            temp_path.write_text(
                yaml.safe_dump(bad_config, sort_keys=False), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "peritumor_ring_kernel_size"):
                ConfigManager(str(temp_path))

    def test_final_configs_match_rapss_training_contract(self):
        root = pathlib.Path(__file__).resolve().parents[1]
        config_paths = [
            root
            / "configs"
            / "server"
            / "random"
            / "train_config_RAPSS_Net.yaml",
            root
            / "configs"
            / "server"
            / "cv"
            / "train_config_RAPSS_Net.yaml",
        ]
        for config_path in config_paths:
            with self.subTest(config=str(config_path)):
                config = ConfigManager(str(config_path)).config
                aug_cfg = config["data"]["augmentation"]
                model_cfg = config["model"]
                train_cfg = config["train"]

                self.assertNotIn("intensity_gamma", aug_cfg)
                self.assertEqual(float(aug_cfg["intensity_brightness"]), 0.0)
                self.assertEqual(
                    model_cfg["intensity_norm_mode"],
                    "global_temporal_zscore",
                )
                self.assertTrue(model_cfg["use_explicit_peritumor_ring"])
                self.assertEqual(int(model_cfg["peritumor_ring_kernel_size"]), 5)
                self.assertTrue(model_cfg["use_multiscale_feature_aggregation"])
                self.assertEqual(float(model_cfg["multiscale_dropout"]), 0.0)
                self.assertEqual(train_cfg["classification_loss"], "ce")
                self.assertEqual(float(train_cfg["label_smoothing"]), 0.02)
                self.assertTrue(train_cfg["use_class_weights"])
                self.assertEqual(int(train_cfg["epochs"]), 120)
                self.assertEqual(int(train_cfg["patience"]), 20)

    def test_server_preprocessing_keeps_raw_phase_intensity(self):
        root = pathlib.Path(__file__).resolve().parents[1]
        prep_config_path = root / "configs" / "server" / "prep_config_public_server.yaml"
        with prep_config_path.open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        processing_params = config["data_prep"]["processing_params"]
        self.assertEqual(processing_params["standardization_mode"], "none")

    def test_training_configs_read_preprocessing_output(self):
        root = pathlib.Path(__file__).resolve().parents[1]
        prep_config_path = root / "configs" / "server" / "prep_config_public_server.yaml"
        with prep_config_path.open("r", encoding="utf-8") as f:
            prep_config = yaml.safe_load(f)
        expected_data_dir = prep_config["data_prep"]["unified_pool_dir"]
        expected_metadata_path = f"{expected_data_dir}/dataset_metadata.csv"

        config_paths = [
            *sorted((root / "configs" / "server" / "random").glob("*.yaml")),
            *sorted((root / "configs" / "server" / "cv").glob("*.yaml")),
        ]
        for config_path in config_paths:
            with self.subTest(config=str(config_path)):
                with config_path.open("r", encoding="utf-8") as f:
                    config = yaml.safe_load(f)
                self.assertEqual(config["paths"]["data_dir"], expected_data_dir)
                self.assertEqual(
                    config["paths"]["metadata_path"], expected_metadata_path
                )


if __name__ == "__main__":
    unittest.main()
