import os
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from .transforms import MedicalTransforms
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


def _is_valid_scalar(value):
    return value is not None and not pd.isna(value)


class MRIDataset(Dataset):
    """PyTorch Dataset for loading 3D MRI data。"""

    def __init__(
        self,
        data_list,
        data_dir,
        model_config,
        is_train=False,
        aug_cfg=None,
        use_mask=True,
        mask_drop_prob=0.0,
        strict_file_check=True,
        cache_in_memory=False,
        center_to_id=None,
        split_name="train",
        clinical_keys=None,
        clinical_stats=None,
        clinical_missing_indicator=True,
    ):
        self.data_list = data_list
        self.data_dir = data_dir
        self.is_train = is_train
        self.split_name = split_name
        self.max_phases = model_config.get("in_channels", 4)  # Treated as max T
        raw_phase_indices = model_config.get("phase_indices")
        self.phase_indices = (
            [int(idx) for idx in raw_phase_indices]
            if raw_phase_indices is not None
            else None
        )
        if self.phase_indices is not None and len(self.phase_indices) != self.max_phases:
            raise ValueError(
                "model.phase_indices length must match model.in_channels: "
                f"{len(self.phase_indices)} != {self.max_phases}"
            )
        self.use_kinetics_channel = bool(
            model_config.get("use_kinetics_channel", False)
        )
        self.intensity_norm_mode = str(
            model_config.get("intensity_norm_mode", "global_temporal_zscore")
        ).lower()
        self.mask_input_mode = str(model_config.get("mask_input_mode", "none")).lower()
        if self.mask_input_mode not in {"none", "multiply"}:
            raise ValueError(
                "model.mask_input_mode must be either 'none' or 'multiply'."
            )
        _aug_cfg = aug_cfg or {}
        self.transform = (
            MedicalTransforms(_aug_cfg)
            if is_train and _aug_cfg.get("enabled", False)
            else None
        )
        self.use_mask = use_mask
        self.mask_drop_prob = mask_drop_prob
        self.strict_file_check = strict_file_check
        self.clinical_keys = [str(key) for key in (clinical_keys or [])]
        self.clinical_stats = clinical_stats or {}
        self.clinical_missing_indicator = bool(clinical_missing_indicator)
        if center_to_id is None:
            all_centers = sorted(list(set(p["center"] for p in self.data_list)))
            self.center_to_id = {name: i for i, name in enumerate(all_centers)}
        else:
            self.center_to_id = dict(center_to_id)
        self.cache_in_memory = cache_in_memory
        self.image_cache = {}
        self.mask_cache = {}
        if self.cache_in_memory:
            logger.info(
                f"Caching dataset in memory for {split_name} ({len(self.data_list)} samples)..."
            )
            for p_info in tqdm(self.data_list, desc=f"Caching-{split_name}"):
                img_path = self._resolve_sample_path(p_info, "images")
                mask_path = self._resolve_sample_path(p_info, "masks")
                try:
                    if img_path is not None:
                        raw_img = np.load(img_path, mmap_mode="c").astype(np.float32)
                        raw_img = self._resize_temporal_phases(raw_img)
                        self.image_cache[p_info["id"]] = raw_img
                    if self.use_mask and mask_path is not None:
                        self.mask_cache[p_info["id"]] = np.load(
                            mask_path, mmap_mode="c"
                        ).astype(np.float32)
                except Exception as e:
                    logger.error(f"Failed to cache data for {p_info['id']}: {e}")
                    if self.strict_file_check:
                        raise

    def __len__(self):
        return len(self.data_list)

    def _resolve_sample_path(self, p_info, subdir):
        sample_id = str(p_info["id"])
        direct_path = os.path.join(self.data_dir, subdir, f"{sample_id}.npy")
        if os.path.exists(direct_path):
            return direct_path

        center_name = str(p_info.get("center", "")).strip()
        if center_name:
            center_path = os.path.join(
                self.data_dir, center_name, subdir, f"{sample_id}.npy"
            )
            if os.path.exists(center_path):
                return center_path
        return None

    def _resize_temporal_phases(self, raw_img: np.ndarray) -> np.ndarray:
        actual_phases = raw_img.shape[0]
        if self.phase_indices is not None:
            selected = []
            for phase_idx in self.phase_indices:
                if phase_idx < actual_phases:
                    selected.append(raw_img[phase_idx])
                else:
                    raise ValueError(
                        "Requested model.phase_indices contains an unavailable "
                        f"phase index {phase_idx}, but image only has {actual_phases} phase(s)."
                    )
            return np.stack(selected, axis=0)
        if actual_phases > self.max_phases:
            indices = np.round(
                np.linspace(0, actual_phases - 1, self.max_phases)
            ).astype(np.int64)
            return raw_img[indices, :, :, :]
        if actual_phases < self.max_phases:
            pad_width = (
                (0, self.max_phases - actual_phases),
                (0, 0),
                (0, 0),
                (0, 0),
            )
            return np.pad(raw_img, pad_width, mode="constant", constant_values=0)
        return raw_img

    def _normalize_image_tensor(self, img_tensor: torch.Tensor) -> torch.Tensor:
        if self.intensity_norm_mode == "none":
            return img_tensor
        if self.intensity_norm_mode == "global_temporal_zscore":
            valid_mask = torch.isfinite(img_tensor) & (img_tensor != 0)
            if valid_mask.any():
                valid_values = img_tensor[valid_mask]
                mean = valid_values.mean()
                std = valid_values.std(unbiased=False).clamp_min(1e-6)
                normalized = torch.zeros_like(img_tensor)
                normalized[valid_mask] = (valid_values - mean) / std
                return normalized
            return torch.zeros_like(img_tensor)
        raise ValueError(
            f"Unsupported model.intensity_norm_mode: {self.intensity_norm_mode}"
        )

    def _normalize_single_volume(self, vol_tensor: torch.Tensor) -> torch.Tensor:
        valid_mask = torch.isfinite(vol_tensor) & (vol_tensor != 0)
        if valid_mask.any():
            valid_values = vol_tensor[valid_mask]
            mean = valid_values.mean()
            std = valid_values.std(unbiased=False).clamp_min(1e-6)
            normalized = torch.zeros_like(vol_tensor)
            normalized[valid_mask] = (valid_values - mean) / std
            return normalized
        return torch.zeros_like(vol_tensor)

    def _compute_kinetics_channel(self, img_tensor: torch.Tensor) -> torch.Tensor:
        if img_tensor.ndim != 4 or img_tensor.shape[0] < 2:
            return torch.zeros(img_tensor.shape[1:], dtype=torch.float32)
        phase_energy = img_tensor.abs().flatten(1).sum(dim=1)
        valid_indices = torch.nonzero(phase_energy > 0, as_tuple=False).view(-1)
        if valid_indices.numel() == 0:
            early_idx, late_idx = 0, min(1, img_tensor.shape[0] - 1)
        else:
            early_idx = int(valid_indices[0].item())
            late_idx = int(valid_indices[-1].item())
            if late_idx == early_idx and img_tensor.shape[0] > 1:
                late_idx = min(early_idx + 1, img_tensor.shape[0] - 1)
        early = img_tensor[early_idx]
        late = img_tensor[late_idx]
        delta = torch.clamp(late - early, min=0.0)
        foreground = early[early > 0]
        if foreground.numel() > 0:
            tissue_threshold = torch.quantile(foreground, 0.05)
        else:
            tissue_threshold = early.new_tensor(0.01)
        eps = torch.maximum(early * 0.05, tissue_threshold) + 1e-8
        kinetics = delta / (early + eps)
        kinetics = torch.nan_to_num(kinetics, nan=0.0, posinf=0.0, neginf=0.0)
        return self._normalize_single_volume(kinetics)

    def __getitem__(self, idx):
        p_info = self.data_list[idx]
        sample_id = str(p_info["id"])
        try:
            if self.cache_in_memory:
                if sample_id not in self.image_cache:
                    raise FileNotFoundError(f"Image not in cache: {sample_id}")
                raw_img = self.image_cache[sample_id]
            else:
                img_path = self._resolve_sample_path(p_info, "images")
                if img_path is None:
                    raise FileNotFoundError(
                        f"Image file not found for sample '{sample_id}' in "
                        f"'{os.path.join(self.data_dir, 'images')}' or "
                        f"'{os.path.join(self.data_dir, str(p_info.get('center', '')), 'images')}'"
                    )
                raw_img = np.load(img_path).astype(np.float32)
                raw_img = self._resize_temporal_phases(raw_img)

            clean_img = np.nan_to_num(raw_img, nan=0.0, posinf=1.0, neginf=-1.0)
            img_tensor = torch.from_numpy(clean_img).float()
            if img_tensor.ndim != 4 or img_tensor.shape[0] != self.max_phases:
                raise ValueError(
                    f"Image tensor for ID {sample_id} has incorrect dimensions: {tuple(img_tensor.shape)}"
                )

            # Compute kinetic metrics via lazy execution to bypass redundant CPU overhead
            if self.use_kinetics_channel:
                kinetics_tensor = self._compute_kinetics_channel(img_tensor)
            else:
                kinetics_tensor = torch.zeros(img_tensor.shape[1:], dtype=torch.float32)
            img_tensor = self._normalize_image_tensor(img_tensor)

            has_mask = 0.0
            mask_tensor = torch.zeros((1, *img_tensor.shape[1:]), dtype=torch.float32)
            simulate_missing = self.is_train and (random.random() < self.mask_drop_prob)
            if self.use_mask and not simulate_missing:
                raw_mask = None
                if self.cache_in_memory:
                    raw_mask = self.mask_cache.get(sample_id)
                else:
                    mask_path = self._resolve_sample_path(p_info, "masks")
                    if mask_path is not None:
                        raw_mask = np.load(mask_path).astype(np.float32)
                if raw_mask is not None:
                    has_mask = 1.0
                    clean_mask = np.nan_to_num(raw_mask, nan=0.0)
                    mask_tensor = torch.from_numpy(clean_mask).float()
                    # Strict mask dimensionality formulation to satisfy downstream DiceCELoss alignment
                    if mask_tensor.ndim != 4 or mask_tensor.shape[0] != 1:
                        raise ValueError(
                            f"Mask for {sample_id} must be shape (1, D, H, W), "
                            f"got {tuple(mask_tensor.shape)}"
                        )

            if self.transform:
                img_tensor, mask_tensor, kinetics_tensor = self.transform(
                    img_tensor, mask_tensor, kinetics_tensor
                )
            if (
                img_tensor is None
                or img_tensor.nelement() == 0
                or mask_tensor is None
                or mask_tensor.nelement() == 0
            ):
                raise ValueError("Empty tensor after transform")
            if self.mask_input_mode == "multiply" and has_mask > 0:
                img_tensor = img_tensor * mask_tensor

            center_id = self.center_to_id.get(p_info["center"], -1)
            if center_id < 0:
                raise ValueError(
                    f"Unknown center '{p_info['center']}' for sample {sample_id}"
                )

            # IHC Targets processing (CD8, CD34, Ki67, CD163)
            has_ihc = 0.0
            ihc_values = np.array([-1.0, -1.0, -1.0, -1.0], dtype=np.float32)

            keys = ["ihc_cd8", "ihc_cd34", "ihc_ki67", "ihc_cd163"]
            if any(_is_valid_scalar(p_info.get(k)) for k in keys):
                has_ihc = 1.0
                for i, key in enumerate(keys):
                    val = p_info.get(key)
                    if _is_valid_scalar(val):
                        ihc_values[i] = float(val)

            ihc_tensor = torch.from_numpy(ihc_values)
            clinical_values = []
            clinical_missing = []
            for key in self.clinical_keys:
                raw_value = p_info.get(key)
                parsed_value = None
                if _is_valid_scalar(raw_value):
                    try:
                        parsed_value = float(raw_value)
                    except (TypeError, ValueError):
                        parsed_value = None
                is_missing = parsed_value is None
                stats = self.clinical_stats.get(key, {})
                mean = float(stats.get("mean", 0.0))
                std = max(float(stats.get("std", 1.0)), 1.0e-6)
                normalized_value = 0.0 if is_missing else (float(parsed_value) - mean) / std
                clinical_values.append(normalized_value)
                clinical_missing.append(1.0 if is_missing else 0.0)
            if self.clinical_missing_indicator:
                clinical_values.extend(clinical_missing)
            clinical_tensor = torch.tensor(clinical_values, dtype=torch.float32)

            return {
                "id": sample_id,
                "image": img_tensor,
                "kinetics": kinetics_tensor,
                "mask": mask_tensor,
                "label": torch.tensor(int(p_info["label"]), dtype=torch.long),
                "has_mask": torch.tensor(has_mask, dtype=torch.float32),
                "center_id": torch.tensor(center_id, dtype=torch.long),
                "ihc": ihc_tensor,
                "has_ihc": torch.tensor(has_ihc, dtype=torch.float32),
                "clinical": clinical_tensor,
            }
        except Exception as e:
            msg = f"[{self.split_name}] sample {sample_id} failed: {e}"
            if self.strict_file_check or not self.is_train:
                logger.error(msg)
                raise
            logger.warning(msg)
            return None


def safe_collate(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if not batch:
        return {
            "id": [],
            "image": torch.empty(0),
            "kinetics": torch.empty(0),
            "mask": torch.empty(0),
            "label": torch.empty(0, dtype=torch.long),
            "has_mask": torch.empty(0, dtype=torch.float32),
            "center_id": torch.empty(0, dtype=torch.long),
            "ihc": torch.empty(0, 4, dtype=torch.float32),
            "has_ihc": torch.empty(0, dtype=torch.float32),
            "clinical": torch.empty(0),
        }
    ids = [b["id"] for b in batch]
    return {
        "id": ids,
        "image": torch.utils.data.dataloader.default_collate(
            [b["image"] for b in batch]
        ),
        "kinetics": torch.utils.data.dataloader.default_collate(
            [b["kinetics"] for b in batch]
        ),
        "mask": torch.utils.data.dataloader.default_collate([b["mask"] for b in batch]),
        "label": torch.utils.data.dataloader.default_collate(
            [b["label"] for b in batch]
        ),
        "has_mask": torch.utils.data.dataloader.default_collate(
            [b["has_mask"] for b in batch]
        ),
        "center_id": torch.utils.data.dataloader.default_collate(
            [b["center_id"] for b in batch]
        ),
        "ihc": torch.utils.data.dataloader.default_collate([b["ihc"] for b in batch]),
        "has_ihc": torch.utils.data.dataloader.default_collate(
            [b["has_ihc"] for b in batch]
        ),
        "clinical": torch.utils.data.dataloader.default_collate(
            [b["clinical"] for b in batch]
        ),
    }
