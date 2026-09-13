import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

from .blocks.cnn_blocks import (
    ConvBlock3D,
    GlobalFeatureRecalibration,
    UpConvBlock3D,
)


class MultiScaleFeatureAggregation(nn.Module):
    def __init__(
        self,
        stage_dims: List[int],
        out_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.stage_projs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv3d(stage_dim, out_dim, kernel_size=1, bias=False),
                    nn.InstanceNorm3d(out_dim, affine=True),
                    nn.GELU(),
                )
                for stage_dim in stage_dims
            ]
        )
        self.fuse = nn.Sequential(
            nn.Conv3d(out_dim * len(stage_dims), out_dim, kernel_size=1, bias=False),
            nn.InstanceNorm3d(out_dim, affine=True),
            nn.GELU(),
            nn.Dropout3d(float(dropout)),
        )

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        if len(features) != len(self.stage_projs):
            raise ValueError(
                f"Expected {len(self.stage_projs)} feature maps, got {len(features)}."
            )
        target_size = tuple(features[-1].shape[2:])
        aligned = []
        for feat, proj in zip(features, self.stage_projs):
            feat = proj(feat)
            if tuple(feat.shape[2:]) != target_size:
                feat = F.interpolate(
                    feat,
                    size=target_size,
                    mode="trilinear",
                    align_corners=False,
                )
            aligned.append(feat)
        fused = self.fuse(torch.cat(aligned, dim=1))
        seq = fused.flatten(2)
        return seq.mean(dim=-1) + seq.amax(dim=-1)


class TriDual3D(nn.Module):
    """
    Lesion-centric 3D classifier used by the RAPSS-Net final configuration.

    Design goal:
      - keep lesion-centric dual-stream spatial modeling
      - keep spatial Mamba blocks
      - remove explicit temporal modeling entirely
      - treat multi-phase DCE input as multi-channel 3D volume
    """

    def __init__(
        self,
        in_channels: int = 2,
        base_dim: int = 32,
        spatial_mamba_layers: int = 1,
        use_peritumor: bool = True,
        use_deep_supervision: bool = True,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        use_dynamic_scan_router: bool = True,
        drop_path_rate: float = 0.1,
        layer_scale_init_value: float = 1.0e-4,
        use_kinetics_channel: bool = False,
        kinetics_fusion_scale: float = 1.0,
        use_feature_recalibration: bool = False,
        feature_recalibration_reduction: int = 4,
        feature_recalibration_dropout: float = 0.0,
        use_explicit_peritumor_ring: bool = False,
        peritumor_ring_kernel_size: int = 5,
        use_clinical: bool = False,
        clinical_input_dim: int = 0,
        clinical_hidden_dim: int = 32,
        clinical_dropout: float = 0.1,
        use_temporal_order_aux: bool = False,
        temporal_order_hidden_dim: int | None = None,
        use_multiscale_feature_aggregation: bool = False,
        multiscale_dropout: float = 0.0,
    ):
        super().__init__()
        self.use_peritumor = use_peritumor
        self.use_deep_supervision = use_deep_supervision
        self.use_kinetics_channel = use_kinetics_channel
        self.kinetics_fusion_scale = float(kinetics_fusion_scale)
        self.use_explicit_peritumor_ring = bool(use_explicit_peritumor_ring)
        self.peritumor_ring_kernel_size = int(max(1, peritumor_ring_kernel_size))
        self.use_clinical = bool(use_clinical) and int(clinical_input_dim) > 0
        self.use_temporal_order_aux = bool(use_temporal_order_aux)
        self.use_multiscale_feature_aggregation = bool(
            use_multiscale_feature_aggregation
        )

        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, base_dim, 3, 1, 1),
            nn.InstanceNorm3d(base_dim, affine=True),
            nn.LeakyReLU(0.2),
        )
        self.kinetics_stem = (
            nn.Sequential(
                nn.Conv3d(1, base_dim, 3, 1, 1),
                nn.InstanceNorm3d(base_dim, affine=True),
                nn.LeakyReLU(0.2),
            )
            if self.use_kinetics_channel
            else None
        )

        self.enc1 = ConvBlock3D(base_dim, base_dim * 2, stride=2)
        self.enc2 = ConvBlock3D(base_dim * 2, base_dim * 4, stride=2)
        self.enc3 = ConvBlock3D(base_dim * 4, base_dim * 8, stride=2)

        hidden_dim = base_dim * 8
        self.core_gate = nn.Conv3d(hidden_dim, 1, kernel_size=3, padding=1)

        if spatial_mamba_layers > 0:
            from .blocks.contextual_mamba_block import EfficientContextualMambaBlock

            if spatial_mamba_layers <= 1:
                spatial_drop_path_rates = [float(drop_path_rate)]
            else:
                spatial_drop_path_rates = torch.linspace(
                    0.0, float(drop_path_rate), spatial_mamba_layers
                ).tolist()
            self.spatial_blocks = nn.ModuleList(
                [
                    EfficientContextualMambaBlock(
                        hidden_dim,
                        d_state=d_state,
                        d_conv=d_conv,
                        expand=expand,
                        use_peritumor=use_peritumor,
                        use_dynamic_scan_router=use_dynamic_scan_router,
                        drop_path_rate=spatial_drop_path_rates[block_idx],
                        layer_scale_init_value=layer_scale_init_value,
                    )
                    for block_idx in range(spatial_mamba_layers)
                ]
            )
        else:
            self.spatial_blocks = nn.ModuleList()

        if self.use_peritumor:
            self.peri_gate = nn.Conv3d(hidden_dim, 1, kernel_size=3, padding=1)
            fusion_dim = hidden_dim * 2
        else:
            fusion_dim = hidden_dim
        self.multiscale_aggregation = (
            MultiScaleFeatureAggregation(
                [base_dim, base_dim * 2, base_dim * 4, hidden_dim],
                hidden_dim,
                dropout=multiscale_dropout,
            )
            if self.use_multiscale_feature_aggregation
            else None
        )
        if self.multiscale_aggregation is not None:
            fusion_dim += hidden_dim
        self.feature_recalibration = (
            GlobalFeatureRecalibration(
                fusion_dim,
                reduction=feature_recalibration_reduction,
                dropout=feature_recalibration_dropout,
            )
            if use_feature_recalibration
            else nn.Identity()
        )
        classifier_dim = fusion_dim
        if self.use_clinical:
            clinical_hidden_dim = int(max(1, clinical_hidden_dim))
            self.clinical_encoder = nn.Sequential(
                nn.LayerNorm(int(clinical_input_dim)),
                nn.Linear(int(clinical_input_dim), clinical_hidden_dim),
                nn.GELU(),
                nn.Dropout(float(clinical_dropout)),
                nn.Linear(clinical_hidden_dim, clinical_hidden_dim),
                nn.GELU(),
            )
            classifier_dim = fusion_dim + clinical_hidden_dim
        else:
            self.clinical_encoder = None

        self.dec3 = UpConvBlock3D(hidden_dim, base_dim * 4)
        self.dec2 = UpConvBlock3D(base_dim * 4, base_dim * 2)
        self.dec1 = UpConvBlock3D(base_dim * 2, base_dim)

        self.out_seg_d3 = nn.Conv3d(base_dim * 4, 1, kernel_size=1)
        self.out_seg_d2 = nn.Conv3d(base_dim * 2, 1, kernel_size=1)
        self.out_seg_d1 = nn.Conv3d(base_dim, 1, kernel_size=3, padding=1)

        self.classifier = nn.Sequential(
            nn.LayerNorm(classifier_dim),
            nn.Linear(classifier_dim, classifier_dim // 2),
            nn.LayerNorm(classifier_dim // 2),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(classifier_dim // 2, 2),
        )
        order_hidden_dim = int(temporal_order_hidden_dim or max(16, classifier_dim // 4))
        self.temporal_order_head = (
            nn.Sequential(
                nn.LayerNorm(classifier_dim),
                nn.Linear(classifier_dim, order_hidden_dim),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(order_hidden_dim, 2),
            )
            if self.use_temporal_order_aux
            else None
        )

    def _binary_dilate(self, mask: torch.Tensor, kernel_size: int) -> torch.Tensor:
        padding = kernel_size // 2
        return torch.nn.functional.max_pool3d(
            mask, kernel_size=kernel_size, stride=1, padding=padding
        )

    def _resize_binary_mask(
        self, mask: torch.Tensor, spatial_size: Tuple[int, int, int]
    ) -> torch.Tensor:
        source_size = tuple(mask.shape[2:])
        if all(target <= source for target, source in zip(spatial_size, source_size)):
            # Preserve small lesions when reducing the mask to the encoder bottleneck.
            # Nearest-neighbor interpolation can otherwise miss every positive voxel.
            mask = torch.nn.functional.adaptive_max_pool3d(
                mask.float(), output_size=spatial_size
            )
        else:
            mask = torch.nn.functional.interpolate(
                mask.float(), size=spatial_size, mode="nearest"
            )
        return (mask > 0.5).float()

    def _masked_mean_max_pool(
        self, feat: torch.Tensor, region_mask: torch.Tensor
    ) -> torch.Tensor:
        weighted_feat = feat * region_mask
        flat_feat = weighted_feat.flatten(2)
        flat_mask = region_mask.flatten(2)
        valid_counts = flat_mask.sum(dim=-1).clamp_min(1.0)
        mean_feat = flat_feat.sum(dim=-1) / valid_counts

        masked_max_input = flat_feat.masked_fill(flat_mask <= 0, float("-inf"))
        max_feat = masked_max_input.amax(dim=-1)
        fallback_max = flat_feat.amax(dim=-1)
        max_feat = torch.where(torch.isfinite(max_feat), max_feat, fallback_max)
        return mean_feat + max_feat

    def _pool_spatial_feature(
        self, feat: torch.Tensor, region_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if region_mask is not None:
            return self._masked_mean_max_pool(feat, region_mask)
        flat_feat = feat.flatten(2)
        return flat_feat.mean(dim=-1) + flat_feat.amax(dim=-1)

    def forward(
        self,
        x: torch.Tensor,
        kinetics: torch.Tensor | None = None,
        lesion_mask: torch.Tensor | None = None,
        clinical: torch.Tensor | None = None,
        return_attention_maps: bool = False,
        return_temporal_order_logits: bool = False,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.dim() == 4:
            x = x.unsqueeze(0)
        if x.dim() != 5:
            raise ValueError(f"Input must be 4D or 5D, but got {x.dim()}D.")

        x0 = self.stem(x)
        if self.kinetics_stem is not None and kinetics is not None:
            if kinetics.dim() == 3:
                kinetics = kinetics.unsqueeze(0).unsqueeze(1)
            elif kinetics.dim() == 4:
                kinetics = kinetics.unsqueeze(1)
            if kinetics.dim() != 5:
                raise ValueError(
                    f"Kinetics tensor must be 4D or 5D, got shape={tuple(kinetics.shape)}"
                )
            x0 = x0 + self.kinetics_fusion_scale * self.kinetics_stem(kinetics)

        x1 = self.enc1(x0)
        x2 = self.enc2(x1)
        cnn_feat = self.enc3(x2)

        core_region_mask = None
        peri_region_mask = None
        if self.use_explicit_peritumor_ring and lesion_mask is not None:
            if lesion_mask.dim() == 4:
                lesion_mask = lesion_mask.unsqueeze(0)
            if lesion_mask.dim() != 5:
                raise ValueError(
                    f"Lesion mask must be 4D or 5D, got shape={tuple(lesion_mask.shape)}"
                )
            if lesion_mask.shape[1] != 1:
                lesion_mask = lesion_mask[:, :1]
            lesion_mask = (lesion_mask > 0.5).float()
            target_size = tuple(cnn_feat.shape[2:])
            core_region_mask = self._resize_binary_mask(lesion_mask, target_size)
            dilated_mask = self._binary_dilate(
                core_region_mask, self.peritumor_ring_kernel_size
            )
            peri_region_mask = torch.clamp(
                dilated_mask - core_region_mask, min=0.0, max=1.0
            )

        core_attn_raw = torch.sigmoid(self.core_gate(cnn_feat))
        if core_region_mask is not None:
            core_attn = core_attn_raw * core_region_mask
        else:
            core_attn = core_attn_raw
        x_core = cnn_feat * (1 + core_attn)

        x_peri, peri_attn = None, None
        if self.use_peritumor:
            peri_attn_raw = torch.sigmoid(self.peri_gate(cnn_feat))
            if peri_region_mask is not None:
                peri_attn = peri_attn_raw * peri_region_mask
            else:
                peri_attn = peri_attn_raw
            x_peri = cnn_feat * (1 + peri_attn)

        feat_bottle = cnn_feat
        for block in self.spatial_blocks:
            feat_core_out, feat_peri_out, _, _ = block(
                x_core, x_peri, use_cross_gating=self.use_peritumor
            )
            feat_bottle = feat_core_out
            x_core = feat_core_out * (1 + core_attn)
            if self.use_peritumor:
                x_peri = feat_peri_out * (1 + peri_attn)

        final_g_core = self._pool_spatial_feature(x_core, core_region_mask)

        if self.use_peritumor and x_peri is not None:
            final_g_peri = self._pool_spatial_feature(x_peri, peri_region_mask)
            global_img_feat = torch.cat([final_g_core, final_g_peri], dim=-1)
        else:
            final_g_peri = torch.zeros_like(final_g_core)
            global_img_feat = final_g_core
        if self.multiscale_aggregation is not None:
            multiscale_feat = self.multiscale_aggregation([x0, x1, x2, feat_bottle])
            global_img_feat = torch.cat([global_img_feat, multiscale_feat], dim=-1)
        global_img_feat = self.feature_recalibration(global_img_feat)
        classifier_feat = global_img_feat
        if self.clinical_encoder is not None:
            if clinical is None:
                raise ValueError("Clinical input is required when use_clinical=True.")
            if clinical.dim() == 1:
                clinical = clinical.unsqueeze(0)
            clinical_feat = self.clinical_encoder(clinical.to(dtype=global_img_feat.dtype))
            classifier_feat = torch.cat([global_img_feat, clinical_feat], dim=-1)

        d3 = self.dec3(feat_bottle) + x2
        out_seg3 = self.out_seg_d3(d3)
        d2 = self.dec2(d3) + x1
        out_seg2 = self.out_seg_d2(d2)
        d1 = self.dec1(d2) + x0
        out_seg1 = self.out_seg_d1(d1)

        seg_out = [out_seg1]
        if self.use_deep_supervision:
            seg_out.extend([out_seg2, out_seg3])

        logits = self.classifier(classifier_feat)
        order_logits = (
            self.temporal_order_head(classifier_feat)
            if return_temporal_order_logits and self.temporal_order_head is not None
            else None
        )

        if return_attention_maps:
            attention_maps = {
                "core": core_attn,
                "peri": peri_attn if peri_attn is not None else torch.zeros_like(core_attn),
            }
            if order_logits is not None:
                return (
                    logits,
                    seg_out,
                    final_g_core,
                    final_g_peri,
                    global_img_feat,
                    attention_maps,
                    order_logits,
                )
            return logits, seg_out, final_g_core, final_g_peri, global_img_feat, attention_maps

        if order_logits is not None:
            return logits, seg_out, final_g_core, final_g_peri, global_img_feat, order_logits
        return logits, seg_out, final_g_core, final_g_peri, global_img_feat
