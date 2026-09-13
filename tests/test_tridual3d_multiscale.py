import unittest

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - test environment dependent
    torch = None


class TestTriDual3DMultiScale(unittest.TestCase):
    @unittest.skipIf(torch is None, "torch is not installed in this interpreter")
    def test_multiscale_aggregation_forward_shapes(self):
        from models.lesion_centric_3d_net import TriDual3D

        model = TriDual3D(
            in_channels=2,
            base_dim=4,
            spatial_mamba_layers=0,
            use_peritumor=True,
            use_deep_supervision=False,
            use_kinetics_channel=True,
            use_multiscale_feature_aggregation=True,
        )
        model.eval()

        x = torch.randn(2, 2, 8, 32, 32)
        kinetics = torch.randn(2, 8, 32, 32)
        mask = torch.ones(2, 1, 8, 32, 32)
        with torch.no_grad():
            logits, seg_out, core_feat, peri_feat, global_feat = model(
                x,
                kinetics=kinetics,
                lesion_mask=mask,
            )

        self.assertEqual(tuple(logits.shape), (2, 2))
        self.assertEqual(len(seg_out), 1)
        self.assertEqual(tuple(core_feat.shape), (2, 32))
        self.assertEqual(tuple(peri_feat.shape), (2, 32))
        self.assertEqual(tuple(global_feat.shape), (2, 96))
        self.assertTrue(torch.isfinite(logits).all().item())
        self.assertTrue(torch.isfinite(global_feat).all().item())


if __name__ == "__main__":
    unittest.main()
