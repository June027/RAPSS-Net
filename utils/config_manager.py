import yaml
import os
import re
from typing import Dict, Any, Optional


class ConfigManager:
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.config = self._load_config()
        self._resolve_values(self.config)
        self._validate_and_resolve_paths()
        self._validate_schema()

    def _load_config(self) -> Dict[str, Any]:
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Missing: {self.config_path}")
        with open(self.config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _resolve_values(self, config_dict: Dict[str, Any]):
        env_var_pattern = re.compile(r"\$\{(.*?)(?::-(.*?))?\}")
        for key, value in config_dict.items():
            if isinstance(value, dict):
                self._resolve_values(value)
            elif isinstance(value, str):
                match = env_var_pattern.match(value)
                if match:
                    env_var, default_val = match.groups()
                    config_dict[key] = os.getenv(env_var, default_val)
                elif value.startswith(("./", "../")):
                    config_dict[key] = os.path.normpath(
                        os.path.join(self.project_root, value)
                    )

    def _validate_and_resolve_paths(self):
        if "paths" in self.config:
            for key in ["data_dir", "metadata_path", "log_dir"]:
                val = self.config["paths"].get(key, "")
                if not val:
                    continue
                is_unix_abs_on_windows = (
                    os.name == "nt" and isinstance(val, str) and val.startswith("/")
                )
                if (val and not os.path.isabs(val)) or is_unix_abs_on_windows:
                    self.config["paths"][key] = os.path.join(self.project_root, val)

    @staticmethod
    def _validate_positive_range(value, key_name):
        if value is None:
            return
        if isinstance(value, (list, tuple)):
            if len(value) != 2:
                raise ValueError(f"{key_name} must be a scalar or a two-value range.")
            low, high = float(value[0]), float(value[1])
        else:
            low = high = float(value)
        if low <= 0 or high <= 0 or high < low:
            raise ValueError(
                f"{key_name} must be positive and ordered as [min, max]."
            )

    def _validate_schema(self):
        if not isinstance(self.config, dict):
            raise ValueError("Configuration root must be a dictionary.")
        if "model" not in self.config or "train" not in self.config:
            return
        model_cfg = self.config.get("model", {})
        train_cfg = self.config.get("train", {})
        loss_cfg = self.config.get("loss_weights", {})
        test_cfg = self.config.get("test", {})
        aug_cfg = self.config.get("data", {}).get("augmentation", {}) or {}

        allowed_model_types = {"tridual_3d"}
        model_type = str(model_cfg.get("type", "tridual_3d")).lower()

        if model_type not in allowed_model_types:
            raise ValueError(f"Unsupported model.type: {model_type}")

        in_channels = int(model_cfg.get("in_channels", 0))
        if in_channels <= 0:
            raise ValueError("model.in_channels must be a positive integer.")

        base_dim = int(model_cfg.get("base_dim", 0))
        if base_dim <= 0:
            raise ValueError("model.base_dim must be a positive integer.")
        if "drop_path_rate" in model_cfg:
            drop_path_rate = float(model_cfg.get("drop_path_rate", 0.0))
            if drop_path_rate < 0 or drop_path_rate >= 1:
                raise ValueError("model.drop_path_rate must be in [0, 1).")
        if "use_feature_recalibration" in model_cfg and not isinstance(
            model_cfg.get("use_feature_recalibration"), bool
        ):
            raise ValueError("model.use_feature_recalibration must be a boolean value.")
        if "feature_recalibration_reduction" in model_cfg and int(
            model_cfg.get("feature_recalibration_reduction", 0)
        ) <= 0:
            raise ValueError(
                "model.feature_recalibration_reduction must be a positive integer."
            )
        if "feature_recalibration_dropout" in model_cfg:
            recalib_dropout = float(model_cfg.get("feature_recalibration_dropout", 0.0))
            if recalib_dropout < 0 or recalib_dropout >= 1:
                raise ValueError(
                    "model.feature_recalibration_dropout must be in [0, 1)."
                )
        if "use_explicit_peritumor_ring" in model_cfg and not isinstance(
            model_cfg.get("use_explicit_peritumor_ring"), bool
        ):
            raise ValueError("model.use_explicit_peritumor_ring must be a boolean value.")
        if "peritumor_ring_kernel_size" in model_cfg and int(
            model_cfg.get("peritumor_ring_kernel_size", 0)
        ) <= 0:
            raise ValueError(
                "model.peritumor_ring_kernel_size must be a positive integer."
            )
        if (
            "peritumor_ring_kernel_size" in model_cfg
            and int(model_cfg.get("peritumor_ring_kernel_size", 0)) % 2 == 0
        ):
            raise ValueError(
                "model.peritumor_ring_kernel_size must be odd to preserve spatial size."
            )
        if "layer_scale_init_value" in model_cfg:
            layer_scale_init_value = float(model_cfg.get("layer_scale_init_value", 0.0))
            if layer_scale_init_value < 0:
                raise ValueError("model.layer_scale_init_value must be non-negative.")
        if int(model_cfg.get("mamba_layers", 0)) < 0:
            raise ValueError("model.mamba_layers must be non-negative.")
        intensity_norm_mode = str(
            model_cfg.get("intensity_norm_mode", "global_temporal_zscore")
        ).lower()
        if intensity_norm_mode not in {"global_temporal_zscore", "none"}:
            raise ValueError(
                "model.intensity_norm_mode must be either 'global_temporal_zscore' or 'none'."
            )
        for key in ("intensity_gamma", "intensity_contrast"):
            if key in aug_cfg:
                self._validate_positive_range(
                    aug_cfg.get(key), f"data.augmentation.{key}"
                )
        if "intensity_brightness" in aug_cfg and float(
            aug_cfg.get("intensity_brightness", 0.0)
        ) < 0:
            raise ValueError("data.augmentation.intensity_brightness must be non-negative.")

        for key in [
            "epochs",
            "batch_size",
            "accumulation_steps",
            "patience",
            "loss_warmup_epochs",
            "aux_anneal_epochs",
        ]:
            value = int(train_cfg.get(key, 0))
            if value <= 0:
                raise ValueError(f"train.{key} must be a positive integer.")

        validation_mode = str(train_cfg.get("validation_mode", "global_cv")).lower()
        if validation_mode not in {"global_cv", "random_split"}:
            raise ValueError(f"Unsupported train.validation_mode: {validation_mode}")
        if validation_mode == "global_cv" and int(train_cfg.get("n_splits", 0)) < 2:
            raise ValueError(
                "train.n_splits must be at least 2 when validation_mode=global_cv."
            )

        for key in [
            "lr",
            "weight_decay",
            "label_smoothing",
            "clip_grad",
            "mask_drop_prob",
            "class_balance_power",
            "sampler_num_samples_multiplier",
            "seg_pos_weight",
        ]:
            value = float(train_cfg.get(key, 0.0))
            if value < 0:
                raise ValueError(f"train.{key} must be non-negative.")
        if float(train_cfg.get("sampler_num_samples_multiplier", 1.0)) <= 0:
            raise ValueError("train.sampler_num_samples_multiplier must be positive.")
        if "compile" in train_cfg and not isinstance(train_cfg.get("compile"), bool):
            raise ValueError("train.compile must be a boolean value.")
        if "use_weighted_sampler" in train_cfg and not isinstance(
            train_cfg.get("use_weighted_sampler"), bool
        ):
            raise ValueError("train.use_weighted_sampler must be a boolean value.")
        hard_example_cfg = train_cfg.get("hard_example_mining", {})
        if hard_example_cfg:
            if not isinstance(hard_example_cfg, dict):
                raise ValueError("train.hard_example_mining must be a dictionary.")
            if "enabled" in hard_example_cfg and not isinstance(
                hard_example_cfg.get("enabled"), bool
            ):
                raise ValueError("train.hard_example_mining.enabled must be a boolean value.")
            if float(hard_example_cfg.get("max_weight", 2.0)) < 1.0:
                raise ValueError("train.hard_example_mining.max_weight must be at least 1.")
            if float(hard_example_cfg.get("difficulty_power", 1.0)) <= 0:
                raise ValueError("train.hard_example_mining.difficulty_power must be positive.")
            if int(hard_example_cfg.get("warmup_epochs", 0)) < 0:
                raise ValueError("train.hard_example_mining.warmup_epochs must be non-negative.")
            if "use_cluster_weights" in hard_example_cfg and not isinstance(
                hard_example_cfg.get("use_cluster_weights"), bool
            ):
                raise ValueError(
                    "train.hard_example_mining.use_cluster_weights must be a boolean value."
                )
            cluster_blend = float(hard_example_cfg.get("cluster_blend", 0.5))
            if cluster_blend < 0 or cluster_blend > 1:
                raise ValueError("train.hard_example_mining.cluster_blend must be in [0, 1].")
        patient_cluster_cfg = train_cfg.get("patient_clustering", {})
        if patient_cluster_cfg:
            if not isinstance(patient_cluster_cfg, dict):
                raise ValueError("train.patient_clustering must be a dictionary.")
            if "enabled" in patient_cluster_cfg and not isinstance(
                patient_cluster_cfg.get("enabled"), bool
            ):
                raise ValueError("train.patient_clustering.enabled must be a boolean value.")
            if int(patient_cluster_cfg.get("n_clusters", 4)) <= 0:
                raise ValueError("train.patient_clustering.n_clusters must be positive.")
            if "numeric_keys" in patient_cluster_cfg and not isinstance(
                patient_cluster_cfg.get("numeric_keys"), list
            ):
                raise ValueError("train.patient_clustering.numeric_keys must be a list.")
        classification_loss = str(train_cfg.get("classification_loss", "ce")).lower()
        if classification_loss not in {"ce", "focal"}:
            raise ValueError("train.classification_loss must be either 'ce' or 'focal'.")
        if "focal_gamma" in train_cfg and float(train_cfg.get("focal_gamma", 0.0)) <= 0:
            raise ValueError("train.focal_gamma must be positive.")
        if "contrastive_temperature" in train_cfg and float(
            train_cfg.get("contrastive_temperature", 0.0)
        ) <= 0:
            raise ValueError("train.contrastive_temperature must be positive.")

        supported_loss_keys = {
            "pcr",
            "seg",
            "domain",
            "contrastive",
            "habitat",
            "temporal_order",
        }
        retired_loss_keys = {"proto", "gate"}
        unknown_loss_keys = set(loss_cfg.keys()) - supported_loss_keys
        if unknown_loss_keys:
            retired_requested = sorted(unknown_loss_keys & retired_loss_keys)
            if retired_requested:
                raise ValueError(
                    f"Retired loss_weights keys are not supported by the current training pipeline: {retired_requested}. "
                    f"Supported keys: {sorted(supported_loss_keys)}"
                )
            raise ValueError(
                f"Unsupported loss_weights keys: {sorted(unknown_loss_keys)}"
            )
        for key in supported_loss_keys:
            if float(loss_cfg.get(key, 0.0)) < 0:
                raise ValueError(f"loss_weights.{key} must be non-negative.")

        deep_supervision_weights = self.config.get("deep_supervision_weights", None)
        use_deep_supervision = bool(model_cfg.get("use_deep_supervision", True))
        if use_deep_supervision:
            if deep_supervision_weights is None:
                raise ValueError(
                    "deep_supervision_weights must be provided when model.use_deep_supervision=true."
                )
            if (
                not isinstance(deep_supervision_weights, list)
                or len(deep_supervision_weights) < 3
            ):
                raise ValueError(
                    "deep_supervision_weights must contain at least 3 positive values."
                )
            if any(float(w) <= 0 for w in deep_supervision_weights):
                raise ValueError(
                    "deep_supervision_weights must contain only positive values."
                )
        elif deep_supervision_weights:
            raise ValueError(
                "deep_supervision_weights should be omitted when model.use_deep_supervision=false."
            )

        if bool(test_cfg.get("tta_enabled", False)):
            raw_views = test_cfg.get("tta_flip_axes", [])
            if not isinstance(raw_views, list):
                raise ValueError("test.tta_flip_axes must be a list.")
            allowed_axes = {"D", "H", "W"}
            for view in raw_views:
                axis_names = [view] if isinstance(view, str) else list(view)
                if not axis_names:
                    raise ValueError("Each TTA view must contain at least one axis.")
                norm_axes = [str(axis).upper() for axis in axis_names]
                if len(set(norm_axes)) != len(norm_axes):
                    raise ValueError("Each TTA view must not repeat axes.")
                if any(axis not in allowed_axes for axis in norm_axes):
                    raise ValueError("test.tta_flip_axes only supports D, H, and W.")

    def get(self, key: str, default: Optional[Any] = None) -> Any:
        keys = key.split(".")
        value = self.config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value
