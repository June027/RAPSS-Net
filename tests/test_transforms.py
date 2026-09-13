import importlib.util
import unittest
from unittest import mock


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None

if TORCH_AVAILABLE:
    import torch

    from datasets.transforms import MedicalTransforms


@unittest.skipUnless(TORCH_AVAILABLE, "torch is required for transform tests")
class TestMedicalTransforms(unittest.TestCase):
    def test_spatial_flip_keeps_kinetics_aligned_with_image_and_mask(self):
        transform = MedicalTransforms(
            {
                "flip_prob": 1.0,
                "affine_prob": 0.0,
                "intensity_prob": 0.0,
            }
        )
        image = torch.arange(2 * 3 * 4 * 5, dtype=torch.float32).reshape(2, 3, 4, 5)
        mask = torch.arange(1 * 3 * 4 * 5, dtype=torch.float32).reshape(1, 3, 4, 5)
        kinetics = torch.arange(3 * 4 * 5, dtype=torch.float32).reshape(3, 4, 5)

        with (
            mock.patch("datasets.transforms.random.random", side_effect=[0.0, 1.0, 1.0]),
            mock.patch("datasets.transforms.random.choice", return_value=2),
        ):
            aug_image, aug_mask, aug_kinetics = transform(image, mask, kinetics)

        self.assertTrue(torch.equal(aug_image, torch.flip(image, dims=[2])))
        self.assertTrue(torch.equal(aug_mask, torch.flip(mask, dims=[2])))
        self.assertTrue(torch.equal(aug_kinetics, torch.flip(kinetics, dims=[1])))

    def test_intensity_gamma_is_applied_without_implicit_brightness(self):
        transform = MedicalTransforms(
            {
                "flip_prob": 0.0,
                "affine_prob": 0.0,
                "intensity_gamma": [2.0, 2.0],
                "intensity_contrast": [1.0, 1.0],
            }
        )
        image = torch.tensor([[[[0.0, 1.0, 2.0]]]], dtype=torch.float32)

        aug_image = transform.random_intensity(image)

        expected = torch.tensor([[[[0.0, 0.5, 2.0]]]], dtype=torch.float32)
        self.assertTrue(torch.allclose(aug_image, expected))
        self.assertEqual(transform.intensity_brightness, 0.0)


if __name__ == "__main__":
    unittest.main()
