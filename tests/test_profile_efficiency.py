import pathlib
import unittest

from tools import profile_efficiency


class TestProfileEfficiencyHelpers(unittest.TestCase):
    def test_loads_server_input_shape_from_prep_config(self):
        root = pathlib.Path(__file__).resolve().parents[1]
        prep_config = root / "configs" / "server" / "prep_config_public_server.yaml"

        self.assertEqual(
            profile_efficiency._load_prep_target_shape(prep_config),
            (32, 128, 128),
        )

    def test_rapss_module_summary_distinguishes_base_and_final(self):
        root = pathlib.Path(__file__).resolve().parents[1]
        base_config = profile_efficiency._load_yaml(
            root / "configs" / "server" / "cv" / "train_config_Proposed_3D_Base.yaml"
        )
        final_config = profile_efficiency._load_yaml(
            root / "configs" / "server" / "cv" / "train_config_RAPSS_Net.yaml"
        )

        base_summary = profile_efficiency.summarize_rapss_modules(base_config)
        final_summary = profile_efficiency.summarize_rapss_modules(final_config)

        self.assertEqual(base_summary["dsta_mamba_blocks"], 0)
        self.assertFalse(base_summary["peritumor_dual_stream"])
        self.assertEqual(final_summary["dsta_mamba_blocks"], 1)
        self.assertTrue(final_summary["peritumor_dual_stream"])
        self.assertTrue(final_summary["dynamic_axial_routing"])
        self.assertTrue(final_summary["explicit_peritumor_ring"])
        self.assertTrue(final_summary["kinetics_channel"])
        self.assertTrue(final_summary["multiscale_feature_aggregation"])


if __name__ == "__main__":
    unittest.main()
