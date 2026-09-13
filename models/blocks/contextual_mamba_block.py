import torch
import torch.nn as nn
from typing import Tuple
from mamba_ssm import Mamba


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob <= 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        binary_tensor = random_tensor.floor()
        return x.div(keep_prob) * binary_tensor


class MorphAdaptiveLayer(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dwconv = nn.Conv3d(
            dim, dim, kernel_size=3, padding=1, groups=dim, bias=False
        )
        self.norm = nn.InstanceNorm3d(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.norm(self.dwconv(x))


class EfficientContextualMambaBlock(nn.Module):
    """
    Stepwise tri-axial dual-stream Mamba block for RAPSS-Net/TriDual3D.

    Each stream (Core / Peri) is processed independently through:
      (1) MorphAdaptive preprocessing
      (2) Dynamic stepwise tri-axial SSM propagation (Z, Y, X)
      (3) Cross-habitat gating (Peri-to-Core and Core-to-Peri)
      (4) Independent per-stream output projection (dim -> dim)
      (5) Layer Scale + Drop Path + Residual

    Design note: each stream has its own out_proj_core / out_proj_peri.
    This guarantees both projections receive gradients via their respective
    residual paths, eliminating the dead-node issue of a single merged projection.
    The single-stream ablation (use_peritumor=False) uses a plain out_proj (dim -> dim).
    """

    def __init__(
        self,
        dim: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        use_peritumor: bool = True,
        use_dynamic_scan_router: bool = True,
        drop_path_rate: float = 0.0,
        layer_scale_init_value: float = 1.0e-4,
    ):
        super().__init__()
        self.dim = dim
        self.use_peritumor = use_peritumor
        self.use_dynamic_scan_router = use_dynamic_scan_router
        self.drop_path = DropPath(drop_path_rate)

        # --- Core stream components ---
        self.norm_core = nn.LayerNorm(dim)
        self.core_morph = MorphAdaptiveLayer(dim)
        self.core_mamba_z = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.core_mamba_y = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.core_mamba_x = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)

        # --- Peri stream components (conditional) ---
        if self.use_peritumor:
            self.norm_peri = nn.LayerNorm(dim)
            self.peri_dilate = nn.Conv3d(
                dim, dim, kernel_size=3, padding=2, dilation=2, groups=dim, bias=False
            )
            self.peri_norm = nn.InstanceNorm3d(dim, affine=True)
            self.peri_mamba_z = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
            self.peri_mamba_y = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
            self.peri_mamba_x = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)

            # Cross-habitat interaction gates
            self.gate_peri2core = nn.Sequential(nn.Linear(dim, dim), nn.Sigmoid())
            self.gate_core2peri = nn.Sequential(nn.Linear(dim, dim), nn.Sigmoid())

        # --- Dynamic Scan Router (optional) ---
        if self.use_dynamic_scan_router:
            router_dim = max(dim // 4, 8)
            self.scan_router = nn.Sequential(
                nn.Linear(dim * 3, router_dim), nn.GELU(), nn.Linear(router_dim, 3)
            )

        # --- Output projections ---
        # Dual-stream: independent per-stream projections (dim -> dim each).
        # Both receive gradients through their own residual path; no dead nodes.
        # Single-stream: a single out_proj (dim -> dim).
        if self.use_peritumor:
            self.out_proj_core = nn.Sequential(nn.Linear(dim, dim), nn.GELU())
            self.out_proj_peri = nn.Sequential(nn.Linear(dim, dim), nn.GELU())
        else:
            self.out_proj = nn.Sequential(nn.Linear(dim, dim), nn.GELU())

        self.layer_scale = (
            nn.Parameter(layer_scale_init_value * torch.ones(dim))
            if layer_scale_init_value > 0
            else None
        )

    def _axis_context_descriptor(self, x_3d: torch.Tensor, axis: str) -> torch.Tensor:
        if axis == "z":
            axis_profile = x_3d.mean(dim=(3, 4))
        elif axis == "y":
            axis_profile = x_3d.mean(dim=(2, 4))
        elif axis == "x":
            axis_profile = x_3d.mean(dim=(2, 3))
        else:
            raise ValueError(f"Unsupported axis for routing: {axis}")
        return axis_profile.mean(dim=-1) + axis_profile.amax(dim=-1)

    def _route_axis_weights(self, x_3d: torch.Tensor) -> torch.Tensor:
        if not self.use_dynamic_scan_router:
            return x_3d.new_ones((x_3d.shape[0], 3))
        pooled = torch.cat(
            [
                self._axis_context_descriptor(x_3d, "z"),
                self._axis_context_descriptor(x_3d, "y"),
                self._axis_context_descriptor(x_3d, "x"),
            ],
            dim=-1,
        )
        return torch.softmax(self.scan_router(pooled), dim=-1)

    def _axis_scan(
        self,
        x_3d: torch.Tensor,
        axis: str,
        mamba_layer: Mamba,
        norm_layer: nn.Module,
    ) -> torch.Tensor:
        B, C, D, H, W = x_3d.shape
        if axis == "z":
            seq = x_3d.permute(0, 3, 4, 2, 1).contiguous().view(B * H * W, D, C)
            return (
                mamba_layer(norm_layer(seq))
                .view(B, H, W, D, C)
                .permute(0, 4, 3, 1, 2)
                .contiguous()
            )
        if axis == "y":
            seq = x_3d.permute(0, 2, 4, 3, 1).contiguous().view(B * D * W, H, C)
            return (
                mamba_layer(norm_layer(seq))
                .view(B, D, W, H, C)
                .permute(0, 4, 1, 3, 2)
                .contiguous()
            )
        if axis == "x":
            seq = x_3d.permute(0, 2, 3, 4, 1).contiguous().view(B * D * H, W, C)
            return (
                mamba_layer(norm_layer(seq))
                .view(B, D, H, W, C)
                .permute(0, 4, 1, 2, 3)
                .contiguous()
            )
        raise ValueError(f"Unsupported axis for scan: {axis}")

    def _patient_axis_orders(self, scan_weights: torch.Tensor) -> torch.Tensor:
        if self.use_dynamic_scan_router:
            return torch.argsort(scan_weights, dim=1, descending=True)
        default_order = torch.tensor([0, 1, 2], device=scan_weights.device, dtype=torch.long)
        return default_order.view(1, 3).expand(scan_weights.shape[0], -1)

    def _tri_oriented_scan(
        self,
        x_3d: torch.Tensor,
        m_z: Mamba,
        m_y: Mamba,
        m_x: Mamba,
        norm_layer: nn.Module,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Propagate 3D features in patient-specific router-prioritized axis steps.

        Axis-length serialization walkthrough (B=batch, C=channels, D/H/W=spatial dims):
          Z-scan : (B,C,D,H,W) -> flatten DHW -> (B,N,C) -> Mamba -> (B,C,D,H,W)
          Y-scan : transpose so Y-tokens are contiguous, then use the same reshape
          X-scan : transpose so X-tokens are contiguous, then use the same reshape
        The router predicts per-sample axis weights and scan order. Samples with
        the same route order are grouped to avoid per-sample loops.
        """
        axis_modules = {
            0: ("z", m_z),
            1: ("y", m_y),
            2: ("x", m_x),
        }
        scan_weights = self._route_axis_weights(x_3d)
        axis_orders = self._patient_axis_orders(scan_weights)
        unique_orders = torch.unique(axis_orders, dim=0)

        routed_states = []
        routed_indices = []
        for order in unique_orders:
            group_mask = (axis_orders == order.view(1, 3)).all(dim=1)
            group_idx = torch.nonzero(group_mask, as_tuple=False).flatten()
            if group_idx.numel() == 0:
                continue
            group_state = x_3d.index_select(0, group_idx)
            group_weights = scan_weights.index_select(0, group_idx)
            for axis_idx in order.tolist():
                axis_name, mamba_layer = axis_modules[int(axis_idx)]
                axis_out = self._axis_scan(group_state, axis_name, mamba_layer, norm_layer)
                axis_weight = group_weights[:, int(axis_idx)].view(-1, 1, 1, 1, 1)
                group_state = group_state + axis_weight * axis_out
            routed_states.append(group_state)
            routed_indices.append(group_idx)

        all_indices = torch.cat(routed_indices, dim=0)
        all_states = torch.cat(routed_states, dim=0)
        state = x_3d.new_zeros(x_3d.shape).index_copy(0, all_indices, all_states)

        fused_seq = state.reshape(state.shape[0], state.shape[1], -1).transpose(1, 2).contiguous()
        return state, fused_seq

    def forward(
        self,
        x_core: torch.Tensor,
        x_peri: torch.Tensor | None,
        use_cross_gating: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x_core: Core (tumor interior) feature map   (B, C, D, H, W)
            x_peri: Peri (peritumoral) feature map      (B, C, D, H, W) or None
            use_cross_gating: Enable bi-directional cross-stream gating

        Returns:
            feat_core_out : residual-updated core feature map   (B, C, D, H, W)
            feat_peri_out : residual-updated peri feature map   (B, C, D, H, W)
                            [equals feat_core_out when use_peritumor=False]
            global_core   : pooled core descriptor              (B, C)
            global_peri   : pooled peri descriptor              (B, C)
        """
        B, C, D, H, W = x_core.shape

        # --- Core tri-axial scan ---
        feat_c = self.core_morph(x_core)
        _, mamba_c = self._tri_oriented_scan(
            feat_c, self.core_mamba_z, self.core_mamba_y, self.core_mamba_x, self.norm_core
        )
        # Global descriptor: mean + max pooling over the flattened spatial sequence
        global_core = mamba_c.mean(dim=1, keepdim=True) + mamba_c.amax(dim=1, keepdim=True)

        if self.use_peritumor and x_peri is not None:
            # --- Peri tri-axial scan (dilated conv for expanded habitat receptive field) ---
            feat_p = x_peri + self.peri_norm(self.peri_dilate(x_peri))
            _, mamba_p = self._tri_oriented_scan(
                feat_p, self.peri_mamba_z, self.peri_mamba_y, self.peri_mamba_x, self.norm_peri
            )
            global_peri = mamba_p.mean(dim=1, keepdim=True) + mamba_p.amax(dim=1, keepdim=True)

            if use_cross_gating:
                # Bi-directional cross-habitat modulation:
                #   peri microenvironment signals modulate core tumor features, and vice versa.
                core_out = mamba_c * (1 + self.gate_peri2core(mamba_p))
                peri_out = mamba_p * (1 + self.gate_core2peri(mamba_c))
            else:
                core_out, peri_out = mamba_c, mamba_p

            # Independent per-stream output projections (dim -> dim each).
            # Both out_proj_core and out_proj_peri receive full gradients through
            # their own residual paths, avoiding dead nodes in the computation graph.
            core_out_3d = (
                self.out_proj_core(core_out).transpose(1, 2).contiguous().reshape(B, C, D, H, W)
            )
            peri_out_3d = (
                self.out_proj_peri(peri_out).transpose(1, 2).contiguous().reshape(B, C, D, H, W)
            )

            if self.layer_scale is not None:
                core_out_3d = core_out_3d * self.layer_scale.view(1, -1, 1, 1, 1)
                peri_out_3d = peri_out_3d * self.layer_scale.view(1, -1, 1, 1, 1)

            return (
                x_core + self.drop_path(core_out_3d),
                x_peri + self.drop_path(peri_out_3d),
                global_core.squeeze(1),
                global_peri.squeeze(1),
            )

        # --- Single-stream path (use_peritumor=False, ablation mode) ---
        global_peri = torch.zeros_like(global_core)
        out_3d = self.out_proj(mamba_c).transpose(1, 2).contiguous().reshape(B, C, D, H, W)
        if self.layer_scale is not None:
            out_3d = out_3d * self.layer_scale.view(1, -1, 1, 1, 1)
        feat_out = x_core + self.drop_path(out_3d)
        return (
            feat_out,
            feat_out,   # Return identical tensor when Peri stream is disabled
            global_core.squeeze(1),
            global_peri.squeeze(1),
        )
