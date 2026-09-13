import importlib.util
import sys
import types
import unittest


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None

if TORCH_AVAILABLE:
    import torch
    import torch.nn as nn


@unittest.skipUnless(TORCH_AVAILABLE, "torch is required for model gradient tests")
class TestTriDual3DMambaGradients(unittest.TestCase):
    def test_classifier_loss_reaches_mamba_gates_and_output_projections(self):
        class DummyMamba(nn.Module):
            def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
                super().__init__()
                self.proj = nn.Linear(d_model, d_model)

            def forward(self, x):
                return self.proj(x)

        mamba_stub = types.ModuleType("mamba_ssm")
        mamba_stub.Mamba = DummyMamba
        sys.modules["mamba_ssm"] = mamba_stub
        sys.modules.pop("models.blocks.contextual_mamba_block", None)

        from models.lesion_centric_3d_net import TriDual3D

        torch.manual_seed(0)
        model = TriDual3D(
            in_channels=2,
            base_dim=1,
            spatial_mamba_layers=1,
            use_peritumor=True,
            use_deep_supervision=False,
            use_dynamic_scan_router=True,
            drop_path_rate=0.0,
            layer_scale_init_value=1.0,
        )
        model.eval()

        images = torch.randn(2, 2, 8, 16, 16)
        logits, _, _, _, _ = model(images)
        loss = logits[:, 1].sum()
        loss.backward()

        block = model.spatial_blocks[0]
        checked_params = [
            block.gate_peri2core[0].weight,
            block.gate_core2peri[0].weight,
            block.out_proj_core[0].weight,
            block.out_proj_peri[0].weight,
            block.layer_scale,
        ]
        for param in checked_params:
            self.assertIsNotNone(param.grad)
            self.assertGreater(float(param.grad.abs().sum().item()), 0.0)


if __name__ == "__main__":
    unittest.main()
