import torch
import torch.nn as nn
import torch.nn.functional as F


class HabitatAlignmentLoss(nn.Module):
    def __init__(
        self,
        erosion_kernel_size: int = 3,
        dilation_kernel_size: int = 5,
    ):
        super().__init__()
        self.erosion_kernel_size = int(max(1, erosion_kernel_size))
        self.dilation_kernel_size = int(max(1, dilation_kernel_size))

    def _binary_dilate(self, mask: torch.Tensor, kernel_size: int) -> torch.Tensor:
        padding = kernel_size // 2
        return F.max_pool3d(mask, kernel_size=kernel_size, stride=1, padding=padding)

    def _binary_erode(self, mask: torch.Tensor, kernel_size: int) -> torch.Tensor:
        padding = kernel_size // 2
        return 1.0 - F.max_pool3d(
            1.0 - mask, kernel_size=kernel_size, stride=1, padding=padding
        )

    def _repeat_mask_if_needed(
        self, target_mask: torch.Tensor, target_batch: int
    ) -> torch.Tensor:
        if target_mask.shape[0] == target_batch:
            return target_mask
        if target_batch % target_mask.shape[0] != 0:
            raise ValueError(
                f"Cannot align mask batch={target_mask.shape[0]} to target batch={target_batch}."
            )
        repeat_factor = target_batch // target_mask.shape[0]
        return target_mask.repeat_interleave(repeat_factor, dim=0)

    def forward(
        self,
        core_gate: torch.Tensor,
        peri_gate: torch.Tensor,
        masks: torch.Tensor,
        has_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if core_gate is None or peri_gate is None:
            device = masks.device if isinstance(masks, torch.Tensor) else None
            return torch.zeros((), device=device)

        if core_gate.dim() == 6:
            core_gate = core_gate.reshape(-1, *core_gate.shape[2:])
        if peri_gate.dim() == 6:
            peri_gate = peri_gate.reshape(-1, *peri_gate.shape[2:])
        if masks.dim() == 6:
            masks = masks.reshape(-1, *masks.shape[2:])
        if masks.dim() != 5:
            raise ValueError(f"Masks must be 5D after reshape, got shape={tuple(masks.shape)}")

        masks = self._repeat_mask_if_needed(masks.float(), core_gate.shape[0])

        if has_mask is not None:
            has_mask = has_mask.view(-1).float()
            if core_gate.shape[0] % has_mask.numel() != 0:
                raise ValueError(
                    f"Cannot align has_mask batch={has_mask.numel()} to target batch={core_gate.shape[0]}."
                )
            repeat_factor = core_gate.shape[0] // has_mask.numel()
            valid_mask = has_mask.repeat_interleave(repeat_factor) > 0.5
        else:
            valid_mask = torch.ones(core_gate.shape[0], device=core_gate.device, dtype=torch.bool)

        if not valid_mask.any():
            return (core_gate.sum() * 0.0) + (peri_gate.sum() * 0.0)

        masks = masks[valid_mask]
        core_gate = core_gate[valid_mask]
        peri_gate = peri_gate[valid_mask]

        # Critical Fix: Perform morphological operations on high-resolution masks
        # to prevent "scale collapse" at 1/8 original resolution.
        core_target_highres = self._binary_erode(masks, self.erosion_kernel_size)
        peri_target_highres = torch.clamp(
            self._binary_dilate(masks, self.dilation_kernel_size) - masks,
            min=0.0,
            max=1.0,
        )

        # Downsample labels AFTER precise morphology to match bottleneck feature size
        core_target = F.interpolate(core_target_highres, size=core_gate.shape[2:], mode="nearest")
        peri_target = F.interpolate(peri_target_highres, size=peri_gate.shape[2:], mode="nearest")

        # Convert sigmoid outputs back to logits so the loss stays AMP-safe.
        core_gate_c = core_gate.clamp(1e-6, 1.0 - 1e-6)
        peri_gate_c = peri_gate.clamp(1e-6, 1.0 - 1e-6)
        core_logits = torch.logit(core_gate_c.float(), eps=1e-6)
        peri_logits = torch.logit(peri_gate_c.float(), eps=1e-6)
        core_target = core_target.float()
        peri_target = peri_target.float()
        loss_core = F.binary_cross_entropy_with_logits(core_logits, core_target)
        loss_peri = F.binary_cross_entropy_with_logits(peri_logits, peri_target)
        return 0.5 * (loss_core + loss_peri)
