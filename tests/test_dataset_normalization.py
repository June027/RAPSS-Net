import importlib.util
import unittest


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None

if TORCH_AVAILABLE:
    import torch

    from datasets.mri_dataset import MRIDataset


@unittest.skipUnless(TORCH_AVAILABLE, "torch is required for normalization tests")
class TestDatasetNormalization(unittest.TestCase):
    def setUp(self):
        self.dataset = MRIDataset.__new__(MRIDataset)
        self.dataset.intensity_norm_mode = "global_temporal_zscore"

    def test_global_temporal_zscore_preserves_zero_background(self):
        image = torch.zeros(2, 2, 2, 2, dtype=torch.float32)
        image[0, 0, 0, 0] = 1.0
        image[1, 1, 1, 1] = 3.0

        normalized = self.dataset._normalize_image_tensor(image)

        self.assertEqual(float(normalized[0, 0, 0, 1].item()), 0.0)
        self.assertEqual(float(normalized[1, 0, 0, 0].item()), 0.0)
        foreground = normalized[image != 0]
        self.assertAlmostEqual(float(foreground.mean().item()), 0.0, places=6)

    def test_kinetics_normalization_preserves_zero_background(self):
        kinetics = torch.zeros(2, 2, 2, dtype=torch.float32)
        kinetics[0, 0, 0] = 2.0
        kinetics[1, 1, 1] = 6.0

        normalized = self.dataset._normalize_single_volume(kinetics)

        self.assertEqual(float(normalized[0, 0, 1].item()), 0.0)
        self.assertEqual(float(normalized[1, 0, 0].item()), 0.0)
        foreground = normalized[kinetics != 0]
        self.assertAlmostEqual(float(foreground.mean().item()), 0.0, places=6)

    def test_single_foreground_voxel_remains_finite(self):
        image = torch.zeros(2, 2, 2, 2, dtype=torch.float32)
        image[0, 0, 0, 0] = 5.0

        normalized = self.dataset._normalize_image_tensor(image)

        self.assertTrue(torch.isfinite(normalized).all().item())
        self.assertEqual(float(normalized[0, 0, 0, 0].item()), 0.0)


if __name__ == "__main__":
    unittest.main()
