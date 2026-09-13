import importlib.util
import unittest


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None

if TORCH_AVAILABLE:
    import torch
    import torch.nn as nn


@unittest.skipUnless(TORCH_AVAILABLE, "torch is required for model tests")
class TestExplicitPeritumorModel(unittest.TestCase):
    def test_mask_downsampling_preserves_a_single_voxel_lesion(self):
        from models.lesion_centric_3d_net import TriDual3D

        model = TriDual3D(
            in_channels=2,
            base_dim=1,
            spatial_mamba_layers=0,
            use_peritumor=True,
            use_deep_supervision=False,
            use_explicit_peritumor_ring=True,
        )
        mask = torch.zeros(1, 1, 9, 17, 17)
        mask[:, :, 4, 8, 8] = 1.0

        resized = model._resize_binary_mask(mask, (2, 4, 4))

        self.assertGreaterEqual(float(resized.sum().item()), 1.0)

    def test_explicit_ring_enhances_regions_without_zeroing_shared_features(self):
        from models.lesion_centric_3d_net import TriDual3D

        class CaptureBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.core_input = None
                self.peri_input = None

            def forward(self, x_core, x_peri, use_cross_gating=True):
                self.core_input = x_core.detach().clone()
                self.peri_input = x_peri.detach().clone()
                return (
                    x_core,
                    x_peri,
                    x_core.flatten(2).mean(dim=-1),
                    x_peri.flatten(2).mean(dim=-1),
                )

        torch.manual_seed(0)
        model = TriDual3D(
            in_channels=2,
            base_dim=1,
            spatial_mamba_layers=0,
            use_peritumor=True,
            use_deep_supervision=False,
            use_explicit_peritumor_ring=True,
            peritumor_ring_kernel_size=3,
        )
        capture_block = CaptureBlock()
        model.spatial_blocks = nn.ModuleList([capture_block])
        model.eval()

        image = torch.randn(1, 2, 8, 16, 16)
        mask = torch.zeros(1, 1, 8, 16, 16)
        mask[:, :, 3:5, 7:9, 7:9] = 1.0

        with torch.no_grad():
            model(image, lesion_mask=mask)

        self.assertIsNotNone(capture_block.core_input)
        self.assertIsNotNone(capture_block.peri_input)
        self.assertGreater(float(capture_block.core_input.abs().sum().item()), 0.0)
        self.assertGreater(float(capture_block.peri_input.abs().sum().item()), 0.0)
        self.assertGreater(
            float((capture_block.core_input != 0).float().mean().item()),
            0.5,
        )
        self.assertGreater(
            float((capture_block.peri_input != 0).float().mean().item()),
            0.5,
        )


if __name__ == "__main__":
    unittest.main()
