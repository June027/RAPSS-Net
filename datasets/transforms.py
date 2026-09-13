import random

import numpy as np
import torch
import torch.nn.functional as F


class MedicalTransforms:
    """3D spatial and intensity augmentation for MRI tensors."""

    def __init__(self, aug_cfg=None):
        self.aug_cfg = aug_cfg or {}
        self.flip_prob = self.aug_cfg.get("flip_prob", 0.5)

        self.affine_prob = self.aug_cfg.get("affine_prob", 0.5)
        self.affine_degrees = self.aug_cfg.get("affine_degrees", 15)
        self.affine_translate = self.aug_cfg.get("affine_translate", 0.1)
        self.affine_scale = self.aug_cfg.get("affine_scale", (0.9, 1.1))

        self.intensity_prob = self.aug_cfg.get("intensity_prob", 0.5)
        self.intensity_brightness = float(
            self.aug_cfg.get("intensity_brightness", 0.0)
        )
        self.intensity_contrast = self._resolve_positive_range(
            self.aug_cfg.get("intensity_contrast", (0.75, 1.25)),
            "intensity_contrast",
        )
        self.intensity_gamma = self._resolve_positive_range(
            self.aug_cfg.get("intensity_gamma", None),
            "intensity_gamma",
        )

    @staticmethod
    def _resolve_positive_range(value, name):
        if value is None:
            return None
        if isinstance(value, (list, tuple)):
            if len(value) != 2:
                raise ValueError(f"{name} must be a scalar or a two-value range.")
            low, high = float(value[0]), float(value[1])
        else:
            low = high = float(value)
        if low <= 0 or high <= 0 or high < low:
            raise ValueError(f"{name} must be positive and ordered as [min, max].")
        return low, high

    @staticmethod
    def _flip_like_image_axis(tensor, image_axis):
        if tensor is None:
            return None
        if tensor.dim() == 4:
            return torch.flip(tensor, dims=[image_axis])
        if tensor.dim() == 3:
            return torch.flip(tensor, dims=[image_axis - 1])
        raise ValueError(
            f"Expected a 3D or 4D spatial tensor, got shape={tuple(tensor.shape)}"
        )

    def __call__(self, img_tensor, mask_tensor, kinetics_tensor=None):
        has_kinetics = kinetics_tensor is not None

        if random.random() < self.flip_prob:
            axis = random.choice([1, 2, 3])  # image tensor axes: C, D, H, W
            img_tensor = torch.flip(img_tensor, dims=[axis])
            mask_tensor = torch.flip(mask_tensor, dims=[axis])
            kinetics_tensor = self._flip_like_image_axis(kinetics_tensor, axis)

        if random.random() < self.affine_prob:
            img_tensor, mask_tensor, kinetics_tensor = self.random_affine(
                img_tensor, mask_tensor, kinetics_tensor
            )

        if random.random() < self.intensity_prob:
            img_tensor = self.random_intensity(img_tensor)

        if has_kinetics:
            return img_tensor, mask_tensor, kinetics_tensor
        return img_tensor, mask_tensor

    def random_affine(self, img, mask, kinetics=None):
        device = img.device
        C, D, H, W = img.shape

        angle_z = np.random.uniform(-self.affine_degrees, self.affine_degrees)
        angle_y = np.random.uniform(-self.affine_degrees, self.affine_degrees)
        angle_x = np.random.uniform(-self.affine_degrees, self.affine_degrees)

        angle_z = angle_z * np.pi / 180.0
        angle_y = angle_y * np.pi / 180.0
        angle_x = angle_x * np.pi / 180.0

        rot_z = torch.tensor(
            [
                [np.cos(angle_z), -np.sin(angle_z), 0],
                [np.sin(angle_z), np.cos(angle_z), 0],
                [0, 0, 1],
            ],
            dtype=torch.float32,
        )
        rot_y = torch.tensor(
            [
                [np.cos(angle_y), 0, np.sin(angle_y)],
                [0, 1, 0],
                [-np.sin(angle_y), 0, np.cos(angle_y)],
            ],
            dtype=torch.float32,
        )
        rot_x = torch.tensor(
            [
                [1, 0, 0],
                [0, np.cos(angle_x), -np.sin(angle_x)],
                [0, np.sin(angle_x), np.cos(angle_x)],
            ],
            dtype=torch.float32,
        )
        rotation_matrix = rot_z @ rot_y @ rot_x

        scale = np.random.uniform(self.affine_scale[0], self.affine_scale[1])
        scale_matrix = torch.diag(
            torch.tensor([scale, scale, scale], dtype=torch.float32)
        )

        affine_matrix = torch.cat(
            [rotation_matrix @ scale_matrix, torch.zeros(3, 1)], dim=1
        ).to(device)

        trans_x = np.random.uniform(-self.affine_translate, self.affine_translate)
        trans_y = np.random.uniform(-self.affine_translate, self.affine_translate)
        trans_z = np.random.uniform(-self.affine_translate, self.affine_translate)
        affine_matrix[:, 3] = torch.tensor(
            [trans_x, trans_y, trans_z], dtype=torch.float32, device=device
        )

        grid = F.affine_grid(
            affine_matrix.unsqueeze(0), (1, C, D, H, W), align_corners=False
        ).to(device)

        img = F.grid_sample(
            img.unsqueeze(0),
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        ).squeeze(0)
        mask = F.grid_sample(
            mask.unsqueeze(0),
            grid,
            mode="nearest",
            padding_mode="zeros",
            align_corners=False,
        ).squeeze(0)

        if kinetics is not None:
            kinetics = self._apply_affine_to_kinetics(
                kinetics, affine_matrix, D, H, W, device
            )

        return img, mask, kinetics

    def _apply_affine_to_kinetics(self, kinetics, affine_matrix, D, H, W, device):
        if kinetics.dim() == 3:
            kinetics_in = kinetics.unsqueeze(0)
            squeeze_channel = True
        elif kinetics.dim() == 4:
            kinetics_in = kinetics
            squeeze_channel = False
        else:
            raise ValueError(
                f"Expected kinetics to be 3D or 4D, got shape={tuple(kinetics.shape)}"
            )
        kinetics_in = kinetics_in.to(device)
        grid = F.affine_grid(
            affine_matrix.unsqueeze(0),
            (1, kinetics_in.shape[0], D, H, W),
            align_corners=False,
        ).to(device)
        kinetics_out = F.grid_sample(
            kinetics_in.unsqueeze(0),
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        ).squeeze(0)
        if squeeze_channel:
            kinetics_out = kinetics_out.squeeze(0)
        return kinetics_out

    def random_intensity(self, img):
        if self.intensity_gamma is not None:
            gamma = np.random.uniform(self.intensity_gamma[0], self.intensity_gamma[1])
            img = self._apply_gamma(img, gamma)

        if self.intensity_brightness > 0:
            brightness = np.random.uniform(
                -self.intensity_brightness, self.intensity_brightness
            )
            img = img + brightness

        if self.intensity_contrast is not None:
            contrast = np.random.uniform(
                self.intensity_contrast[0], self.intensity_contrast[1]
            )
            mean = torch.mean(img)
            img = (img - mean) * contrast + mean

        return img

    def _apply_gamma(self, img, gamma):
        if img.dim() != 4:
            raise ValueError(
                f"Expected image tensor shape (C, D, H, W), got {tuple(img.shape)}"
            )
        out = img.clone()
        for channel_idx in range(img.shape[0]):
            channel = img[channel_idx]
            finite_mask = torch.isfinite(channel)
            if not finite_mask.any():
                continue
            values = channel[finite_mask]
            value_min = values.min()
            value_max = values.max()
            denom = (value_max - value_min).clamp_min(1.0e-6)
            normalized = ((channel - value_min) / denom).clamp(0.0, 1.0)
            gamma_channel = torch.pow(normalized, float(gamma)) * denom + value_min
            out[channel_idx] = torch.where(finite_mask, gamma_channel, channel)
        return out
