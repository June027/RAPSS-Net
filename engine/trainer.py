import os
import csv
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from utils.experiment import config_fingerprint, save_json, split_fingerprint
from utils.loss_schedule import compute_aux_loss_warmup_factor
from utils.metrics import calculate_fast_metrics
from utils.amp_compat import autocast_context


AXIS_NAME_TO_DIM = {"D": 2, "H": 3, "W": 4}


class KFoldTrainer:
    def __init__(
        self,
        fold,
        model,
        ema_model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        scaler,
        criterions,
        config,
        device,
        logger,
        writer,
        target_loader=None,
        train_ids=None,
        val_ids=None,
        split_label=None,
    ):
        self.fold = fold
        self.split_label = split_label or f"Fold {fold}"
        self.model = model
        self.ema_model = ema_model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.target_loader = target_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = scaler
        self.criterions = criterions
        self.config = config
        self.device = device
        self.logger = logger
        self.writer = writer
        self.train_ids = list(train_ids or [])
        self.val_ids = list(val_ids or [])

        self.epochs = config["train"].get("epochs", 100)
        self.patience = config["train"].get("patience", 15)
        self.accumulation_steps = config["train"].get("accumulation_steps", 1)
        self.loss_warmup_epochs = config["train"].get("loss_warmup_epochs", 10)
        self.aux_anneal_epochs = max(
            1, int(config["train"].get("aux_anneal_epochs", min(self.epochs, 150)))
        )
        self.deep_supervision_weights = self._resolve_deep_supervision_weights(config)
        self.w_seg = float(config.get("loss_weights", {}).get("seg", 0.5))
        self.w_domain = float(config.get("loss_weights", {}).get("domain", 0.1))
        self.w_contrastive = float(
            config.get("loss_weights", {}).get("contrastive", 0.0)
        )
        self.w_habitat = float(config.get("loss_weights", {}).get("habitat", 0.0))
        self.w_temporal_order = float(
            config.get("loss_weights", {}).get("temporal_order", 0.0)
        )
        self.use_domain_debias = bool(config.get("train", {}).get("use_domain_debias", False))
        self.domain_adaptation_cfg = config.get("domain_adaptation", {}) or {}
        self.use_target_domain_adaptation = (
            bool(self.domain_adaptation_cfg.get("enabled", False))
            and self.target_loader is not None
        )
        self.target_domain_weight = float(self.domain_adaptation_cfg.get("weight", 0.0))
        self.target_domain_warmup_epochs = max(
            0,
            int(self.domain_adaptation_cfg.get("warmup_epochs", self.loss_warmup_epochs)),
        )
        self.target_domain_method = str(
            self.domain_adaptation_cfg.get("method", "coral")
        ).lower()
        self._target_iter = None
        self._domain_adaptation_warning_logged = False

        compile_requested = bool(self.config.get("train", {}).get("compile", False))
        if compile_requested:
            try:
                self.model = torch.compile(self.model)
                self.logger.info("torch.compile enabled for training acceleration.")
            except Exception as e:
                self.logger.warning(f"Failed to compile model: {e}")

        self.resume_training = bool(config["train"].get("resume_training", False))
        self.show_batch_progress = bool(
            config["train"].get("show_batch_progress", False)
        )

        self.amp_dtype = (
            torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else torch.float16
        )
        self.fold_dir = os.path.join(config["paths"]["log_dir"], f"fold_{fold}")
        os.makedirs(self.fold_dir, exist_ok=True)
        self.csv_path = os.path.join(self.fold_dir, f"history_fold_{fold}.csv")
        self.manifest_path = os.path.join(self.fold_dir, "fold_manifest.json")
        self._configure_diagnostics()
        self.current_manifest = {
            "fold": str(fold),
            "config_hash": config_fingerprint(config),
            "split_hash": split_fingerprint(self.train_ids, self.val_ids),
            "train_ids": self.train_ids,
            "val_ids": self.val_ids,
            "resume_training": self.resume_training,
        }

        self.best_auc = 0.0
        self.patience_counter = 0
        self.start_epoch = 1
        self.val_threshold_fixed = 0.5
        self._last_train_epoch_rows = []
        self._last_val_epoch_rows = []
        self.eval_tta_views = self._resolve_eval_tta_views()
        hard_cfg = self.config.get("train", {}).get("hard_example_mining", {})
        self.hard_example_enabled = bool(hard_cfg.get("enabled", False))
        self.hard_example_max_weight = max(1.0, float(hard_cfg.get("max_weight", 2.0)))
        self.hard_example_power = max(0.1, float(hard_cfg.get("difficulty_power", 1.0)))
        self.hard_example_warmup_epochs = max(0, int(hard_cfg.get("warmup_epochs", 0)))
        self.hard_example_use_clusters = bool(hard_cfg.get("use_cluster_weights", False))
        self.hard_example_cluster_blend = min(
            max(float(hard_cfg.get("cluster_blend", 0.5)), 0.0), 1.0
        )
        self.hard_example_weights = {}
        self.hard_cluster_weights = {}
        self.sample_cluster_by_id = self._resolve_sample_clusters()
        self._hard_example_warning_logged = False
        self.current_epoch = 0

        save_json(self.manifest_path, self.current_manifest)
        self._resume_checkpoint()
        if self.start_epoch == 1:
            self._init_csv()
            self.logger.info(f"AMP enabled with dtype: {self.amp_dtype}")
            if len(self.eval_tta_views) > 1:
                self.logger.info(
                    f"Internal validation TTA enabled with {len(self.eval_tta_views)} view(s)."
                )
            if self.hard_example_enabled:
                self.logger.info(
                    "Training hard-example mining enabled; weights are updated from training predictions only."
                )
                if self.hard_example_use_clusters:
                    self.logger.info(
                        f"Cluster-aware hard mining enabled for {len(set(self.sample_cluster_by_id.values()))} training cluster(s)."
                    )
            if self.use_target_domain_adaptation:
                self.logger.info(
                    "Unsupervised target-domain adaptation enabled | "
                    f"method={self.target_domain_method}, weight={self.target_domain_weight}, "
                    f"warmup_epochs={self.target_domain_warmup_epochs}, "
                    f"target_batches={len(self.target_loader)}"
                )

    def _raw_model(self):
        raw_model = self.model
        if hasattr(raw_model, "_orig_mod"):
            raw_model = raw_model._orig_mod
        if hasattr(raw_model, "module"):
            raw_model = raw_model.module
        return raw_model

    def _configure_diagnostics(self):
        diag_cfg = self.config.get("train", {}).get(
            "diagnostics", self.config.get("diagnostics", {})
        )
        self.diagnostics_enabled = bool(diag_cfg.get("enabled", False))
        self.diagnostics_interval_batches = max(
            1, int(diag_cfg.get("log_interval_batches", 50))
        )
        self.diagnostics_to_csv = bool(diag_cfg.get("log_to_csv", True))
        self.diagnostics_to_tensorboard = bool(
            diag_cfg.get("log_to_tensorboard", True)
        )
        default_layers = [
            "stem",
            "enc1",
            "enc2",
            "enc3",
            "core_gate",
            "feature_recalibration",
            "classifier",
        ]
        raw_layers = diag_cfg.get("layers", default_layers)
        self.diagnostics_layers = [str(layer) for layer in raw_layers]
        self._diagnostics_capture_active = False
        self._diagnostics_epoch = 0
        self._diagnostics_batch_idx = -1
        self._diagnostics_global_step = 0
        self._diagnostics_feature_stats = {}
        self._diagnostics_loss_stats = {}
        self._diagnostic_hook_handles = []
        self.diagnostics_csv_path = os.path.join(
            self.fold_dir, f"diagnostics_train_fold_{self.fold}.csv"
        )
        if not self.diagnostics_enabled:
            return
        if self.diagnostics_to_csv:
            self._init_diagnostics_csv()
        self._register_diagnostic_hooks()
        self.logger.info(
            f"Training diagnostics enabled every {self.diagnostics_interval_batches} batch(es): "
            f"{', '.join(self.diagnostics_layers)}"
        )

    def _init_diagnostics_csv(self):
        with open(self.diagnostics_csv_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                [
                    "epoch",
                    "batch_idx",
                    "global_step",
                    "kind",
                    "name",
                    "value",
                    "mean",
                    "std",
                    "min",
                    "max",
                    "absmax",
                    "l2",
                    "count",
                ]
            )

    def _register_diagnostic_hooks(self):
        modules = dict(self._raw_model().named_modules())
        for layer_name in self.diagnostics_layers:
            module = modules.get(layer_name)
            if module is None:
                self.logger.warning(
                    f"Diagnostics layer '{layer_name}' was not found in the model; skipping."
                )
                continue
            self._diagnostic_hook_handles.append(
                module.register_forward_hook(
                    self._make_diagnostic_activation_hook(layer_name)
                )
            )

    def _make_diagnostic_activation_hook(self, layer_name):
        def hook(_module, _inputs, output):
            if not self._diagnostics_capture_active:
                return
            tensor = self._first_tensor(output)
            if tensor is not None:
                self._record_tensor_diagnostic("feature", layer_name, tensor)

        return hook

    def _first_tensor(self, value):
        if torch.is_tensor(value):
            return value
        if isinstance(value, (list, tuple)):
            for item in value:
                tensor = self._first_tensor(item)
                if tensor is not None:
                    return tensor
        if isinstance(value, dict):
            for item in value.values():
                tensor = self._first_tensor(item)
                if tensor is not None:
                    return tensor
        return None

    def _should_capture_diagnostics(self, batch_idx):
        if not self.diagnostics_enabled:
            return False
        is_update_step = (batch_idx + 1) % self.accumulation_steps == 0 or (
            batch_idx + 1
        ) == len(self.train_loader)
        if not is_update_step:
            return False
        is_first_update = batch_idx + 1 <= self.accumulation_steps
        if is_first_update:
            return True
        return (batch_idx + 1) % self.diagnostics_interval_batches == 0

    def _begin_diagnostics_capture(self, epoch, batch_idx):
        should_capture = self._should_capture_diagnostics(batch_idx)
        self._diagnostics_capture_active = should_capture
        if not should_capture:
            return
        self._diagnostics_epoch = int(epoch)
        self._diagnostics_batch_idx = int(batch_idx)
        self._diagnostics_global_step = int(
            (epoch - 1) * len(self.train_loader) + batch_idx + 1
        )
        self._diagnostics_feature_stats = {}
        self._diagnostics_loss_stats = {}

    def _end_diagnostics_capture(self):
        self._diagnostics_capture_active = False
        self._diagnostics_feature_stats = {}
        self._diagnostics_loss_stats = {}

    def _tensor_stats(self, tensor):
        with torch.no_grad():
            values = tensor.detach()
            if values.is_sparse:
                values = values.coalesce().values()
            values = values.float()
            finite_mask = torch.isfinite(values)
            if not finite_mask.any():
                return {
                    "mean": float("nan"),
                    "std": float("nan"),
                    "min": float("nan"),
                    "max": float("nan"),
                    "absmax": float("nan"),
                    "l2": float("nan"),
                    "count": 0,
                }
            values = values[finite_mask]
            return {
                "mean": float(values.mean().item()),
                "std": float(values.std(unbiased=False).item()),
                "min": float(values.min().item()),
                "max": float(values.max().item()),
                "absmax": float(values.abs().max().item()),
                "l2": float(torch.linalg.vector_norm(values, ord=2).item()),
                "count": int(values.numel()),
            }

    def _record_tensor_diagnostic(self, kind, name, tensor):
        if not self._diagnostics_capture_active:
            return
        self._diagnostics_feature_stats[f"{kind}/{name}"] = self._tensor_stats(tensor)

    def _record_scalar_diagnostic(self, name, value):
        if not self._diagnostics_capture_active:
            return
        if torch.is_tensor(value):
            value = value.detach().float().item()
        self._diagnostics_loss_stats[str(name)] = {"value": float(value)}

    def _module_grad_stats(self, module):
        total_sq = 0.0
        max_abs = 0.0
        count = 0
        for param in module.parameters(recurse=True):
            if param.grad is None:
                continue
            grad = param.grad.detach()
            if grad.is_sparse:
                grad = grad.coalesce().values()
            grad = grad.float()
            if grad.numel() == 0:
                continue
            finite_mask = torch.isfinite(grad)
            if not finite_mask.any():
                continue
            grad = grad[finite_mask]
            norm = float(torch.linalg.vector_norm(grad, ord=2).item())
            total_sq += norm * norm
            max_abs = max(max_abs, float(grad.abs().max().item()))
            count += int(grad.numel())
        return {
            "l2": float(total_sq**0.5),
            "absmax": float(max_abs),
            "count": int(count),
        }

    def _collect_gradient_diagnostics(self):
        if not self._diagnostics_capture_active:
            return {}
        modules = dict(self._raw_model().named_modules())
        grad_stats = {}
        for layer_name in self.diagnostics_layers:
            module = modules.get(layer_name)
            if module is None:
                continue
            grad_stats[layer_name] = self._module_grad_stats(module)
        return grad_stats

    def _write_diagnostic_row(self, kind, name, stats):
        if not self.diagnostics_to_csv:
            return
        with open(self.diagnostics_csv_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                [
                    self._diagnostics_epoch,
                    self._diagnostics_batch_idx,
                    self._diagnostics_global_step,
                    kind,
                    name,
                    stats.get("value", ""),
                    stats.get("mean", ""),
                    stats.get("std", ""),
                    stats.get("min", ""),
                    stats.get("max", ""),
                    stats.get("absmax", ""),
                    stats.get("l2", ""),
                    stats.get("count", ""),
                ]
            )

    def _write_diagnostic_tensorboard(self, kind, name, stats):
        if (
            not self.diagnostics_to_tensorboard
            or self.writer is None
            or not self._diagnostics_capture_active
        ):
            return
        safe_name = name.replace("/", "_")
        for metric_name, metric_value in stats.items():
            if metric_name == "count" or metric_value == "":
                continue
            self.writer.add_scalar(
                f"diagnostics/{kind}/{safe_name}/{metric_name}",
                float(metric_value),
                self._diagnostics_global_step,
            )

    def _flush_diagnostics(self, grad_stats, total_grad_norm):
        if not self._diagnostics_capture_active:
            return
        if total_grad_norm is not None:
            self._diagnostics_loss_stats["grad_total_pre_clip"] = {
                "value": float(total_grad_norm)
            }
            self._diagnostics_loss_stats["clip_grad_max_norm"] = {
                "value": float(self.config.get("train", {}).get("clip_grad", 1.0))
            }
        for key, stats in self._diagnostics_feature_stats.items():
            kind, name = key.split("/", 1)
            self._write_diagnostic_row(kind, name, stats)
            self._write_diagnostic_tensorboard(kind, name, stats)
        for name, stats in self._diagnostics_loss_stats.items():
            self._write_diagnostic_row("loss", name, stats)
            self._write_diagnostic_tensorboard("loss", name, stats)
        for name, stats in grad_stats.items():
            self._write_diagnostic_row("grad", name, stats)
            self._write_diagnostic_tensorboard("grad", name, stats)
        stem_grad = grad_stats.get("stem", {}).get("l2", 0.0)
        enc1_grad = grad_stats.get("enc1", {}).get("l2", 0.0)
        enc3_grad = grad_stats.get("enc3", {}).get("l2", 0.0)
        cls_grad = grad_stats.get("classifier", {}).get("l2", 0.0)
        stem_std = self._diagnostics_feature_stats.get("feature/stem", {}).get(
            "std", float("nan")
        )
        enc3_std = self._diagnostics_feature_stats.get("feature/enc3", {}).get(
            "std", float("nan")
        )
        self.logger.info(
            f"[{self.split_label}] Diagnostics ep={self._diagnostics_epoch} "
            f"batch={self._diagnostics_batch_idx} | "
            f"grad_l2 stem/enc1/enc3/cls={stem_grad:.3e}/{enc1_grad:.3e}/{enc3_grad:.3e}/{cls_grad:.3e} | "
            f"feat_std stem/enc3={stem_std:.3f}/{enc3_std:.3f}"
        )

    def _resolve_sample_clusters(self):
        dataset = getattr(self.train_loader, "dataset", None)
        data_list = getattr(dataset, "data_list", None)
        if not data_list:
            return {}
        clusters = {}
        for row in data_list:
            sample_id = str(row.get("id", ""))
            if not sample_id:
                continue
            cluster_id = row.get("patient_cluster", row.get("cluster", None))
            if cluster_id is not None and str(cluster_id) != "":
                clusters[sample_id] = str(cluster_id)
        return clusters

    def _resolve_eval_tta_views(self):
        train_cfg = self.config.get("train", {})
        enabled = bool(
            train_cfg.get(
                "internal_val_tta_enabled",
                train_cfg.get("val_tta_enabled", False),
            )
        )
        if not enabled:
            return [()]

        raw_views = train_cfg.get(
            "internal_val_tta_flip_axes",
            train_cfg.get(
                "val_tta_flip_axes",
                self.config.get("test", {}).get("tta_flip_axes", []),
            ),
        )
        views = [()]
        for raw_view in raw_views:
            if isinstance(raw_view, str):
                raw_view = [raw_view]
            dims = []
            for axis_name in raw_view:
                norm_name = str(axis_name).upper()
                if norm_name not in AXIS_NAME_TO_DIM:
                    raise ValueError(
                        f'Unsupported internal validation TTA axis "{axis_name}". '
                        "Expected one of D/H/W."
                    )
                dims.append(AXIS_NAME_TO_DIM[norm_name])
            dims_tuple = tuple(sorted(set(dims)))
            if dims_tuple not in views:
                views.append(dims_tuple)
        return views

    @staticmethod
    def _apply_tta_view(tensor, flip_dims):
        if tensor is None or not flip_dims:
            return tensor
        return torch.flip(tensor, dims=list(flip_dims))

    @staticmethod
    def _apply_tta_volume(tensor, flip_dims):
        if tensor is None or not flip_dims:
            return tensor
        return torch.flip(tensor, dims=[dim - 1 for dim in flip_dims])

    def _predict_batch_tta(self, eval_model, images, kinetics, masks, clinical):
        probs_tta = []
        global_tta = []
        core_tta = []
        peri_tta = []
        for flip_dims in self.eval_tta_views:
            aug_images = self._apply_tta_view(images, flip_dims)
            aug_masks = self._apply_tta_view(masks, flip_dims)
            aug_kinetics = self._apply_tta_volume(kinetics, flip_dims)
            outputs = eval_model(
                aug_images,
                kinetics=aug_kinetics,
                lesion_mask=aug_masks,
                clinical=clinical,
            )
            logits = outputs[0] if isinstance(outputs, tuple) else outputs
            probs_tta.append(F.softmax(logits, dim=1)[:, 1])
            if isinstance(outputs, tuple):
                if len(outputs) > 4 and isinstance(outputs[4], torch.Tensor):
                    global_tta.append(outputs[4])
                if len(outputs) > 2 and isinstance(outputs[2], torch.Tensor):
                    core_tta.append(outputs[2])
                if len(outputs) > 3 and isinstance(outputs[3], torch.Tensor):
                    peri_tta.append(outputs[3])

        probs = torch.stack(probs_tta, dim=0).mean(dim=0)
        global_feats = torch.stack(global_tta, dim=0).mean(dim=0) if global_tta else None
        core_feats = torch.stack(core_tta, dim=0).mean(dim=0) if core_tta else None
        peri_feats = torch.stack(peri_tta, dim=0).mean(dim=0) if peri_tta else None
        return probs, global_feats, core_feats, peri_feats

    def _pcr_loss_per_sample(self, logits, labels):
        criterion = self.criterions["pcr"]
        criterion_name = criterion.__class__.__name__
        if criterion_name == "CrossEntropyLoss":
            return F.cross_entropy(
                logits,
                labels,
                weight=criterion.weight,
                label_smoothing=float(getattr(criterion, "label_smoothing", 0.0)),
                reduction="none",
            )
        if criterion_name == "FocalLoss":
            ce_loss = F.cross_entropy(
                logits,
                labels,
                weight=criterion.weight,
                reduction="none",
            )
            pt = torch.exp(-ce_loss)
            return ((1.0 - pt) ** float(criterion.gamma)) * ce_loss
        if not self._hard_example_warning_logged:
            self.logger.warning(
                f"Hard-example mining does not support {criterion_name}; using the configured PCR loss without sample reweighting."
            )
            self._hard_example_warning_logged = True
        return None

    def _hard_weight_tensor(self, sample_ids, device):
        if (
            not self.hard_example_enabled
            or self.current_epoch <= self.hard_example_warmup_epochs
            or not self.hard_example_weights
        ):
            return None
        weights = [
            float(self._hard_weight_for_sample(str(sample_id)))
            for sample_id in sample_ids
        ]
        return torch.as_tensor(weights, dtype=torch.float32, device=device)

    def _hard_weight_for_sample(self, sample_id):
        sample_weight = float(self.hard_example_weights.get(sample_id, 1.0))
        if not self.hard_example_use_clusters:
            return sample_weight
        cluster_id = self.sample_cluster_by_id.get(sample_id)
        if cluster_id is None:
            return sample_weight
        cluster_weight = float(self.hard_cluster_weights.get(cluster_id, 1.0))
        blend = self.hard_example_cluster_blend
        return (1.0 - blend) * sample_weight + blend * cluster_weight

    def _compute_pcr_loss(self, logits, labels, sample_ids):
        hard_weights = self._hard_weight_tensor(sample_ids, labels.device)
        if hard_weights is None:
            return self.criterions["pcr"](logits, labels)

        per_sample_loss = self._pcr_loss_per_sample(logits, labels)
        if per_sample_loss is None:
            return self.criterions["pcr"](logits, labels)
        return (per_sample_loss * hard_weights).sum() / hard_weights.sum().clamp_min(1e-8)

    def _update_hard_example_weights(self, sample_ids, labels, probs, epoch):
        if not self.hard_example_enabled or epoch < self.hard_example_warmup_epochs:
            return
        if not sample_ids:
            return

        next_weights = {}
        cluster_values = {}
        for sample_id, label, prob in zip(sample_ids, labels, probs):
            label = int(label)
            prob = float(prob)
            difficulty = (1.0 - prob) if label == 1 else prob
            difficulty = min(max(difficulty, 0.0), 1.0)
            weight = 1.0 + (self.hard_example_max_weight - 1.0) * (
                difficulty ** self.hard_example_power
            )
            sample_key = str(sample_id)
            next_weights[sample_key] = max(float(weight), next_weights.get(sample_key, 1.0))
            cluster_id = self.sample_cluster_by_id.get(sample_key)
            if cluster_id is not None:
                cluster_values.setdefault(cluster_id, []).append(float(weight))
        self.hard_example_weights = next_weights
        self.hard_cluster_weights = {
            cluster_id: float(np.mean(values))
            for cluster_id, values in cluster_values.items()
            if values
        }

    def _resolve_deep_supervision_weights(self, config):
        weights = config.get("deep_supervision_weights", None)
        if weights is None:
            return None
        if not isinstance(weights, (list, tuple)) or len(weights) == 0:
            raise ValueError(
                "deep_supervision_weights must be a non-empty list of positive numbers."
            )
        parsed = [float(w) for w in weights]
        if any(w <= 0 for w in parsed):
            raise ValueError(
                "deep_supervision_weights must contain only positive values."
            )
        return parsed

    def _init_csv(self):
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "Epoch",
                    "Train_Loss",
                    "Train_PCR_Loss",
                    "Train_AUC",
                    "Train_PRAUC",
                    "Train_Precision",
                    "Train_Recall",
                    "Train_F1",
                    "Train_ACC",
                    "Train_MicroACC",
                    "Train_NPV",
                    "Train_Kappa",
                    "Train_Sens",
                    "Train_Spec",
                    "InternalVal_PCR_Loss",
                    "InternalVal_AUC",
                    "InternalVal_PRAUC",
                    "InternalVal_Precision_Dyn",
                    "InternalVal_Recall_Dyn",
                    "InternalVal_F1_Dyn",
                    "InternalVal_ACC_Dyn",
                    "InternalVal_MicroACC_Dyn",
                    "InternalVal_NPV_Dyn",
                    "InternalVal_Kappa_Dyn",
                    "InternalVal_Sens_Dyn",
                    "InternalVal_Spec_Dyn",
                    "InternalVal_Precision_Fix",
                    "InternalVal_Recall_Fix",
                    "InternalVal_F1_Fix",
                    "InternalVal_ACC_Fix",
                    "InternalVal_MicroACC_Fix",
                    "InternalVal_NPV_Fix",
                    "InternalVal_Kappa_Fix",
                    "InternalVal_Sens_Fix",
                    "InternalVal_Spec_Fix",
                    "LR",
                ]
            )

    def _log_epoch_scalars(self, epoch, train_loss, train_metrics, val_loss, val_dyn, val_fix):
        if self.writer is None:
            return
        self.writer.add_scalar("loss/train_total", float(train_loss), epoch)
        self.writer.add_scalar("loss/train_pcr", float(train_metrics["PCR_Loss"]), epoch)
        self.writer.add_scalar("loss/internal_val_pcr", float(val_loss), epoch)

        self.writer.add_scalar("auc/train", float(train_metrics["AUC"]), epoch)
        self.writer.add_scalar("auc/internal_val", float(val_dyn["AUC"]), epoch)
        self.writer.add_scalar("prauc/train", float(train_metrics["PR-AUC"]), epoch)
        self.writer.add_scalar("prauc/internal_val", float(val_dyn["PR-AUC"]), epoch)
        self.writer.add_scalar("lr/main", float(self.optimizer.param_groups[0]["lr"]), epoch)
        self.writer.add_scalar("precision/train", float(train_metrics["Precision"]), epoch)
        self.writer.add_scalar("precision/internal_val_dyn", float(val_dyn["Precision"]), epoch)
        self.writer.add_scalar("precision/internal_val_fix", float(val_fix["Precision"]), epoch)
        self.writer.add_scalar("recall/train", float(train_metrics["Recall"]), epoch)
        self.writer.add_scalar("recall/internal_val_dyn", float(val_dyn["Recall"]), epoch)
        self.writer.add_scalar("recall/internal_val_fix", float(val_fix["Recall"]), epoch)
        self.writer.add_scalar("f1/train", float(train_metrics["F1"]), epoch)
        self.writer.add_scalar("f1/internal_val_dyn", float(val_dyn["F1"]), epoch)
        self.writer.add_scalar("f1/internal_val_fix", float(val_fix["F1"]), epoch)
        self.writer.add_scalar("acc/train", float(train_metrics["ACC"]), epoch)
        self.writer.add_scalar("acc/internal_val_dyn", float(val_dyn["ACC"]), epoch)
        self.writer.add_scalar("acc/internal_val_fix", float(val_fix["ACC"]), epoch)
        self.writer.add_scalar("npv/train", float(train_metrics["NPV"]), epoch)
        self.writer.add_scalar("npv/internal_val_dyn", float(val_dyn["NPV"]), epoch)
        self.writer.add_scalar("npv/internal_val_fix", float(val_fix["NPV"]), epoch)
        self.writer.add_scalar("kappa/train", float(train_metrics["Kappa"]), epoch)
        self.writer.add_scalar("kappa/internal_val_dyn", float(val_dyn["Kappa"]), epoch)
        self.writer.add_scalar("kappa/internal_val_fix", float(val_fix["Kappa"]), epoch)

        self.writer.add_scalar("sens/train", float(train_metrics["Sens"]), epoch)
        self.writer.add_scalar("spec/train", float(train_metrics["Spec"]), epoch)
        self.writer.add_scalar("sens/internal_val_dyn", float(val_dyn["Sens"]), epoch)
        self.writer.add_scalar("spec/internal_val_dyn", float(val_dyn["Spec"]), epoch)
        self.writer.add_scalar("sens/internal_val_fix", float(val_fix["Sens"]), epoch)
        self.writer.add_scalar("spec/internal_val_fix", float(val_fix["Spec"]), epoch)
        self.writer.add_scalar("threshold/internal_val_dyn", float(val_dyn["Thresh"]), epoch)

    def _checkpoint_state(self, epoch):
        raw_model = self.model._orig_mod if hasattr(self.model, "_orig_mod") else self.model
        state = {
            "epoch": epoch,
            "model_state_dict": raw_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "best_auc": self.best_auc,
            "patience_counter": self.patience_counter,
            "val_threshold_fixed": self.val_threshold_fixed,
            "clinical_stats": self.config.get("data", {}).get("clinical_stats", {}),
            "config_hash": self.current_manifest["config_hash"],
            "split_hash": self.current_manifest["split_hash"],
            "torch_rng_state": torch.get_rng_state(),
            "numpy_rng_state": np.random.get_state(),
        }
        if torch.cuda.is_available():
            state["cuda_rng_state"] = torch.cuda.get_rng_state_all()
        if self.ema_model:
            state["ema_model_state_dict"] = self.ema_model.module.state_dict()
        return state

    def _save_checkpoint(self, epoch, is_best=False):
        state = self._checkpoint_state(epoch)
        last_path = os.path.join(self.fold_dir, f"last_model_fold_{self.fold}.pth")
        torch.save(state, last_path)
        if is_best:
            best_path = os.path.join(self.fold_dir, f"best_model_fold_{self.fold}.pth")
            torch.save(state, best_path)

    def _write_epoch_prediction_rows(self, path, rows):
        if not rows:
            return
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["id", "label", "prob", "epoch", "fold", "split"],
            )
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def _serializable_metrics(metrics):
        payload = {}
        for key, value in metrics.items():
            if isinstance(value, (np.integer,)):
                payload[key] = int(value)
            elif isinstance(value, (np.floating,)):
                payload[key] = float(value)
            elif value is None:
                payload[key] = None
            else:
                try:
                    payload[key] = float(value)
                except (TypeError, ValueError):
                    payload[key] = value
        return payload

    def _save_best_epoch_prediction_artifacts(self, epoch, train_metrics, val_metrics):
        train_path = os.path.join(self.fold_dir, "best_epoch_train_predictions.csv")
        val_path = os.path.join(self.fold_dir, "best_epoch_val_predictions.csv")
        metrics_path = os.path.join(self.fold_dir, "best_epoch_prediction_metrics.json")

        self._write_epoch_prediction_rows(train_path, self._last_train_epoch_rows)
        self._write_epoch_prediction_rows(val_path, self._last_val_epoch_rows)
        save_json(
            metrics_path,
            {
                "epoch": int(epoch),
                "fold": str(self.fold),
                "selection_metric": "InternalVal_AUC",
                "selection_value": float(val_metrics.get("AUC", 0.0)),
                "train_metrics": self._serializable_metrics(train_metrics),
                "val_metrics": self._serializable_metrics(val_metrics),
                "train_predictions_path": train_path,
                "val_predictions_path": val_path,
                "note": (
                    "These predictions are the exact per-sample probabilities accumulated "
                    "during the epoch that achieved the best internal validation AUC. "
                    "Train rows therefore match the Train_AUC in history_fold_*.csv, "
                    "instead of the later best-checkpoint train re-evaluation metric."
                ),
            },
        )
        self.logger.info(
            f"[{self.split_label}] Saved best-epoch prediction artifacts: "
            f"train={train_path}, val={val_path}"
        )

    def _resume_checkpoint(self):
        last_path = os.path.join(self.fold_dir, f"last_model_fold_{self.fold}.pth")
        if not os.path.exists(last_path):
            return
        if not self.resume_training:
            self.logger.info(
                f"Checkpoint exists for fold {self.fold}, but resume_training=false; starting a fresh run."
            )
            return
        self.logger.info(f"Found checkpoint candidate: {last_path}")
        try:
            checkpoint = torch.load(
                last_path, map_location=self.device, weights_only=True
            )
        except Exception as e:
            self.logger.warning(
                f"weights_only=True load failed ({e}); retrying with weights_only=False. "
                "Ensure this checkpoint originates from a trusted source."
            )
            checkpoint = torch.load(last_path, map_location=self.device)

        if (
            checkpoint.get("config_hash") != self.current_manifest["config_hash"]
            or checkpoint.get("split_hash") != self.current_manifest["split_hash"]
        ):
            self.logger.warning(
                "Checkpoint hash mismatch detected; refusing unsafe resume and starting fresh."
            )
            return
        self.start_epoch = checkpoint["epoch"] + 1
        raw_model = self.model._orig_mod if hasattr(self.model, "_orig_mod") else self.model
        raw_model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.val_threshold_fixed = checkpoint.get("val_threshold_fixed", 0.5)
        if "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        self.best_auc = checkpoint["best_auc"]
        self.patience_counter = checkpoint["patience_counter"]
        if self.ema_model and "ema_model_state_dict" in checkpoint:
            self.ema_model.module.load_state_dict(checkpoint["ema_model_state_dict"])
        if "torch_rng_state" in checkpoint:
            torch.set_rng_state(checkpoint["torch_rng_state"])
            np.random.set_state(checkpoint["numpy_rng_state"])
        if "cuda_rng_state" in checkpoint and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])
        self.logger.info(
            f"Resumed successfully from epoch {checkpoint['epoch']} (best AUC={self.best_auc:.4f})"
        )

    def _compute_seg_loss(self, seg_logits_list, masks, has_mask_flags, epoch):
        w_seg = self.w_seg
        if w_seg <= 0 or not seg_logits_list:
            # Return dummy computational graph node to satisfy DDP integrity
            return seg_logits_list[0].sum() * 0.0 if seg_logits_list else torch.zeros(1, device=self.device, requires_grad=True).squeeze()
        loss_seg = torch.zeros(1, device=self.device).squeeze()
        if self.deep_supervision_weights is not None and len(seg_logits_list) > len(
            self.deep_supervision_weights
        ):
            raise ValueError(
                f"Model returned {len(seg_logits_list)} segmentation outputs, but only "
                f"{len(self.deep_supervision_weights)} deep_supervision_weights were provided."
            )
        valid_mask = has_mask_flags.bool()
        if valid_mask.sum() == 0:
            # Maintain node connection to gracefully handle batches devoid of validated masks.
            dummy_loss = sum((seg.sum() * 0.0) for seg in seg_logits_list)
            return dummy_loss
        batch_size = int(masks.shape[0])
        for i, seg_pred in enumerate(seg_logits_list):
            if seg_pred.dim() != 5:
                raise ValueError(
                    f"Segmentation tensor must be 5D, got shape={tuple(seg_pred.shape)}"
                )
            if seg_pred.shape[0] % batch_size != 0:
                raise ValueError(
                    f"Segmentation batch mismatch: seg_pred batch={seg_pred.shape[0]}, mask batch={batch_size}"
                )
            _, c_s, d_s, h_s, w_s = seg_pred.shape
            seg_pred = seg_pred.reshape(
                batch_size, -1, c_s, d_s, h_s, w_s
            ).mean(dim=1)
            seg_pred = seg_pred[valid_mask]
            target_scaled = F.interpolate(
                masks[valid_mask].float(), size=seg_pred.shape[2:], mode="nearest"
            )
            l_s = self.criterions["seg"](seg_pred, target_scaled)
            if torch.isnan(l_s):
                self.logger.warning(
                    f"NaN seg loss at scale {i} (epoch={epoch}, batch estimated); skipping this scale."
                )
                continue
            if self.deep_supervision_weights is not None:
                base_weight = self.deep_supervision_weights[i]
            else:
                base_weight = 1.0 / (2**i)
            if i == 0:
                dynamic_weight = base_weight
            else:
                anneal_epoch = min(epoch, self.aux_anneal_epochs)
                decay_factor = (1.0 - anneal_epoch / self.aux_anneal_epochs) ** 2
                dynamic_weight = base_weight * decay_factor
            loss_seg += dynamic_weight * l_s
        return loss_seg * w_seg

    def _unpack_batch(self, batch):
        return (
            batch["id"],
            batch["image"],
            batch.get("kinetics"),
            batch["mask"],
            batch["label"],
            batch["has_mask"],
            batch["center_id"],
            batch.get("ihc"),
            batch.get("has_ihc"),
            batch.get("clinical"),
        )

    def _step_optimizer_if_needed(self, batch_idx):
        is_update_step = (batch_idx + 1) % self.accumulation_steps == 0 or (
            batch_idx + 1
        ) == len(self.train_loader)
        if not is_update_step:
            return
        self.scaler.unscale_(self.optimizer)
        grad_stats = self._collect_gradient_diagnostics()
        total_grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            max_norm=self.config.get("train", {}).get("clip_grad", 1.0),
        )
        self._flush_diagnostics(grad_stats, total_grad_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        if self.ema_model:
            self.ema_model.update(self.model)

    def _extract_domain_features_from_outputs(self, outputs):
        if not isinstance(outputs, tuple):
            return None
        core_feats = outputs[2] if len(outputs) > 2 else None
        peri_feats = outputs[3] if len(outputs) > 3 else None
        pooled_feats = (
            outputs[4]
            if len(outputs) > 4 and isinstance(outputs[4], torch.Tensor)
            else None
        )
        if pooled_feats is not None:
            return pooled_feats
        if core_feats is not None and peri_feats is not None:
            return torch.cat([core_feats, peri_feats], dim=-1)
        return core_feats if core_feats is not None else peri_feats

    def _flatten_domain_features(self, features):
        if features is None:
            return None
        if features.dim() > 2:
            features = F.adaptive_avg_pool3d(features, 1).view(features.size(0), -1)
        if features.dim() != 2 or features.size(0) < 2:
            return None
        return features.float()

    def _next_target_batch(self):
        if not self.use_target_domain_adaptation or self.target_loader is None:
            return None
        if self._target_iter is None:
            self._target_iter = iter(self.target_loader)
        for _ in range(2):
            try:
                batch = next(self._target_iter)
            except StopIteration:
                self._target_iter = iter(self.target_loader)
                batch = next(self._target_iter)
            if batch["image"].nelement() > 0:
                return batch
        return None

    def _forward_target_domain_features(self, target_batch):
        if target_batch is None:
            return None
        images = target_batch["image"]
        if images.nelement() == 0:
            return None
        images = images.to(self.device)
        kinetics = target_batch.get("kinetics")
        kinetics = (
            kinetics.to(self.device)
            if kinetics is not None and kinetics.nelement() > 0
            else None
        )
        masks = target_batch["mask"].to(self.device)
        clinical = target_batch.get("clinical")
        clinical = (
            clinical.to(self.device)
            if clinical is not None and clinical.nelement() > 0
            else None
        )
        outputs = self.model(
            images,
            kinetics=kinetics,
            lesion_mask=masks,
            clinical=clinical,
            return_attention_maps=False,
        )
        return self._extract_domain_features_from_outputs(outputs)

    def _compute_target_domain_loss(self, source_features, target_features):
        source = self._flatten_domain_features(source_features)
        target = self._flatten_domain_features(target_features)
        if source is None or target is None:
            return torch.zeros((), device=self.device)
        if source.shape[1] != target.shape[1]:
            if not self._domain_adaptation_warning_logged:
                self.logger.warning(
                    "Skipping target-domain adaptation because feature dimensions differ: "
                    f"source={tuple(source.shape)}, target={tuple(target.shape)}"
                )
                self._domain_adaptation_warning_logged = True
            return source.new_zeros(())

        source_mean = source.mean(dim=0)
        target_mean = target.mean(dim=0)
        mean_loss = F.mse_loss(source_mean, target_mean)
        if self.target_domain_method in {"mean", "mean_only"}:
            return mean_loss
        if self.target_domain_method not in {"coral", "mean_coral"}:
            raise ValueError(
                f"Unsupported domain_adaptation.method: {self.target_domain_method}"
            )
        source_centered = source - source_mean
        target_centered = target - target_mean
        source_cov = source_centered.T.matmul(source_centered) / max(source.size(0) - 1, 1)
        target_cov = target_centered.T.matmul(target_centered) / max(target.size(0) - 1, 1)
        coral_loss = F.mse_loss(source_cov, target_cov)
        mean_weight = float(self.domain_adaptation_cfg.get("mean_weight", 0.25))
        return coral_loss + mean_weight * mean_loss

    def _make_temporal_order_batch(self, images):
        if images.dim() != 5 or images.shape[1] < 3:
            return None, None
        batch_size, num_phases = images.shape[:2]
        order_labels = torch.ones(batch_size, dtype=torch.long, device=images.device)
        shuffled_images = images.clone()
        permute_mask = torch.rand(batch_size, device=images.device) < 0.5
        if not permute_mask.any():
            permute_mask[0] = True
        for sample_idx in torch.nonzero(permute_mask, as_tuple=False).flatten():
            perm = torch.randperm(num_phases, device=images.device)
            if torch.equal(perm, torch.arange(num_phases, device=images.device)):
                perm = torch.roll(perm, shifts=1, dims=0)
            shuffled_images[sample_idx] = images[sample_idx, perm]
            order_labels[sample_idx] = 0
        return shuffled_images, order_labels

    def _model_has_temporal_order_head(self):
        raw_model = self.model
        if hasattr(raw_model, "_orig_mod"):
            raw_model = raw_model._orig_mod
        if hasattr(raw_model, "module"):
            raw_model = raw_model.module
        return getattr(raw_model, "temporal_order_head", None) is not None

    def train_epoch(self, epoch: int):
        self.model.train()
        self.current_epoch = int(epoch)
        # Add epoch-driven stochasticity for complex samplers
        if hasattr(self.train_loader, "batch_sampler") and hasattr(
            self.train_loader.batch_sampler, "set_epoch"
        ):
            self.train_loader.batch_sampler.set_epoch(epoch)
        elif hasattr(self.train_loader, "sampler") and hasattr(
            self.train_loader.sampler, "set_epoch"
        ):
            self.train_loader.sampler.set_epoch(epoch)

        total_loss, total_pcr_loss, valid_batches, all_ids, all_labels, all_probs = (
            0,
            0,
            0,
            [],
            [],
            [],
        )
        self.optimizer.zero_grad(set_to_none=True)
        self._target_iter = iter(self.target_loader) if self.use_target_domain_adaptation else None

        train_iter = (
            tqdm(
                self.train_loader,
                desc=f"{self.split_label} Train Ep {epoch}",
                leave=False,
                dynamic_ncols=True,
                mininterval=1.0,
            )
            if self.show_batch_progress
            else self.train_loader
        )
        for batch_idx, batch in enumerate(train_iter):
            self._begin_diagnostics_capture(epoch, batch_idx)
            (
                sample_ids,
                images,
                kinetics,
                masks,
                labels,
                has_mask_flags,
                center_ids,
                ihc_targets,
                has_ihc_flags,
                clinical,
            ) = self._unpack_batch(batch)
            if images.nelement() == 0:
                self._end_diagnostics_capture()
                continue
            images = images.to(self.device)
            kinetics = kinetics.to(self.device) if kinetics is not None and kinetics.nelement() > 0 else None
            masks = masks.to(self.device)
            labels = labels.to(self.device)
            has_mask_flags = has_mask_flags.to(self.device)
            center_ids = center_ids.to(self.device)
            clinical = clinical.to(self.device) if clinical is not None and clinical.nelement() > 0 else None
            self._record_tensor_diagnostic("feature", "input_image", images)

            with autocast_context(
                enabled=self.config["train"].get("amp", True),
                dtype=self.amp_dtype,
            ):
                need_gate_maps = self.w_habitat > 0 and "habitat" in self.criterions
                outputs = self.model(
                    images,
                    kinetics=kinetics,
                    lesion_mask=masks,
                    clinical=clinical,
                    return_attention_maps=need_gate_maps,
                )
                # Model returns: logits, seg_out, spatial_features, temporal_features
                if isinstance(outputs, tuple):
                    pcr_logits = outputs[0]
                    seg_logits_list = outputs[1] if len(outputs) > 1 else []
                    core_feats = outputs[2] if len(outputs) > 2 else None
                    peri_feats = outputs[3] if len(outputs) > 3 else None
                    pooled_feats = (
                        outputs[4]
                        if len(outputs) > 4 and isinstance(outputs[4], torch.Tensor)
                        else None
                    )
                    if pooled_feats is not None:
                        domain_feats = pooled_feats
                    elif core_feats is not None and peri_feats is not None:
                        domain_feats = torch.cat([core_feats, peri_feats], dim=-1)
                    elif core_feats is not None:
                        domain_feats = core_feats
                    else:
                        domain_feats = peri_feats
                    attention_maps = (
                        outputs[5]
                        if len(outputs) > 5 and isinstance(outputs[5], dict)
                        else None
                    )
                else:
                    pcr_logits = outputs
                    seg_logits_list = []
                    domain_feats = None
                    attention_maps = None

                if pcr_logits is None:
                    self._end_diagnostics_capture()
                    continue
                loss_pcr = self._compute_pcr_loss(pcr_logits, labels, sample_ids)
                loss_seg = self._compute_seg_loss(
                    seg_logits_list, masks, has_mask_flags, epoch
                )
                loss_contrastive = torch.tensor(0.0, device=self.device)
                if (
                    self.w_contrastive > 0
                    and "contrastive" in self.criterions
                    and domain_feats is not None
                ):
                    loss_contrastive = self.criterions["contrastive"](
                        domain_feats, labels
                    )
                loss_habitat = torch.tensor(0.0, device=self.device)
                if (
                    self.w_habitat > 0
                    and "habitat" in self.criterions
                    and attention_maps is not None
                ):
                    # Habitat alignment dynamically regularizes the attention distribution without introducing background noise.
                    loss_habitat = self.criterions["habitat"](
                        attention_maps.get("core"),
                        attention_maps.get("peri"),
                        masks,
                        has_mask_flags,
                    )
                loss_temporal_order = torch.tensor(0.0, device=self.device)
                if (
                    self.w_temporal_order > 0
                    and images.shape[1] >= 3
                    and self._model_has_temporal_order_head()
                ):
                    order_images, order_labels = self._make_temporal_order_batch(images)
                    if order_images is not None:
                        order_outputs = self.model(
                            order_images,
                            kinetics=None,
                            lesion_mask=masks,
                            clinical=clinical,
                            return_temporal_order_logits=True,
                        )
                        if isinstance(order_outputs, tuple) and len(order_outputs) > 5:
                            order_logits = order_outputs[-1]
                            if torch.is_tensor(order_logits):
                                loss_temporal_order = F.cross_entropy(
                                    order_logits, order_labels
                                )

                # Domain Generalization Loss (Domain Debiasing)
                loss_domain = torch.tensor(0.0, device=self.device)
                if self.use_domain_debias and "domain" in self.criterions and domain_feats is not None:
                    # Use the temporal features as global representation for domain debiasing
                    unique_centers = torch.unique(center_ids)
                    if len(unique_centers) > 1:
                        loss_domain = self.criterions["domain"](
                            domain_feats, center_ids, labels
                        )
                loss_target_domain = torch.tensor(0.0, device=self.device)
                if (
                    self.use_target_domain_adaptation
                    and self.target_domain_weight > 0
                    and domain_feats is not None
                ):
                    target_batch = self._next_target_batch()
                    target_domain_feats = self._forward_target_domain_features(target_batch)
                    loss_target_domain = self._compute_target_domain_loss(
                        domain_feats,
                        target_domain_feats,
                    )
                    loss_target_domain = torch.nan_to_num(
                        loss_target_domain,
                        nan=0.0,
                        posinf=0.0,
                        neginf=0.0,
                    )

                # Object function aggregation scaling
                weighted_domain = self.w_domain * loss_domain
                target_domain_warmup = compute_aux_loss_warmup_factor(
                    epoch,
                    self.target_domain_warmup_epochs,
                )
                weighted_target_domain = (
                    self.target_domain_weight
                    * target_domain_warmup
                    * loss_target_domain
                )
                weighted_contrastive = self.w_contrastive * loss_contrastive
                weighted_habitat = self.w_habitat * loss_habitat
                weighted_temporal_order = (
                    self.w_temporal_order * loss_temporal_order
                )
                aux_warmup = compute_aux_loss_warmup_factor(
                    epoch, self.loss_warmup_epochs
                )
                total_combined_loss = (
                    loss_pcr
                    + aux_warmup
                    * (
                        loss_seg
                        + weighted_domain
                        + weighted_contrastive
                        + weighted_habitat
                        + weighted_temporal_order
                    )
                    + weighted_target_domain
                )
                self._record_scalar_diagnostic("total", total_combined_loss)
                self._record_scalar_diagnostic("pcr", loss_pcr)
                self._record_scalar_diagnostic("seg", loss_seg)
                self._record_scalar_diagnostic("domain_weighted", weighted_domain)
                self._record_scalar_diagnostic(
                    "target_domain_weighted", weighted_target_domain
                )
                self._record_scalar_diagnostic(
                    "contrastive_weighted", weighted_contrastive
                )
                self._record_scalar_diagnostic("habitat_weighted", weighted_habitat)
                self._record_scalar_diagnostic(
                    "temporal_order_weighted", weighted_temporal_order
                )
                self._record_scalar_diagnostic("aux_warmup", aux_warmup)
                self._record_scalar_diagnostic(
                    "target_domain_warmup", target_domain_warmup
                )
                if torch.isnan(total_combined_loss):
                    self.logger.warning(
                        f"NaN total loss detected at batch {batch_idx} (epoch={epoch}); skipping backward."
                    )
                    self._end_diagnostics_capture()
                    continue

                current_pcr_loss_val = loss_pcr.item()
                # Scale by accumulation_steps for correct gradient accumulation
                total_loss_for_backward = total_combined_loss / self.accumulation_steps

            self.scaler.scale(total_loss_for_backward).backward()
            self._step_optimizer_if_needed(batch_idx)
            self._end_diagnostics_capture()

            total_loss += total_combined_loss.item()
            total_pcr_loss += current_pcr_loss_val
            valid_batches += 1
            all_ids.extend([str(sample_id) for sample_id in sample_ids])
            all_labels.extend(labels.detach().cpu().numpy())
            all_probs.extend(
                F.softmax(pcr_logits.detach(), dim=1)[:, 1].float().cpu().numpy()
            )

        if self.show_batch_progress:
            train_iter.close()
        if valid_batches == 0:
            self._last_train_epoch_rows = []
            return 0.0, {
                "PCR_Loss": 0.0,
                "AUC": 0.5,
                "PR-AUC": 0.0,
                "Precision": 0.0,
                "Recall": 0.0,
                "F1": 0.0,
                "ACC": 0.0,
                "Micro-ACC": 0.0,
                "NPV": 0.0,
                "Kappa": 0.0,
                "Sens": 0.0,
                "Spec": 0.0,
            }
        m = calculate_fast_metrics(all_labels, all_probs)
        self._update_hard_example_weights(all_ids, all_labels, all_probs, epoch)
        avg_loss = total_loss / valid_batches
        avg_pcr_loss = total_pcr_loss / valid_batches
        m["PCR_Loss"] = avg_pcr_loss
        self._last_train_epoch_rows = [
            {
                "id": str(sample_id),
                "label": int(label),
                "prob": float(prob),
                "epoch": int(epoch),
                "fold": str(self.fold),
                "split": "train_epoch",
            }
            for sample_id, label, prob in zip(all_ids, all_labels, all_probs)
        ]
        self.logger.info(
            f"[{self.split_label}] Epoch [{epoch}] Train | Loss(total): {avg_loss:.4f} | Loss(pcr): {avg_pcr_loss:.4f} | "
            f"AUC: {m['AUC']:.3f} | F1: {m['F1']:.3f} | ACC: {m['ACC']:.3f} | "
            f"Kappa: {m['Kappa']:.3f} | Sens: {m['Sens']:.3f} | Spec: {m['Spec']:.3f}"
        )
        return avg_loss, m

    def validate(self, epoch):
        eval_model = self.ema_model.module if self.ema_model else self.model
        eval_model.eval()
        val_loss, valid_batches, all_ids, all_labels, all_probs = 0, 0, [], [], []
        with torch.no_grad():
            val_iter = (
                tqdm(
                    self.val_loader,
                    desc=f"{self.split_label} Internal Val Ep {epoch}",
                    leave=False,
                    dynamic_ncols=True,
                    mininterval=1.0,
                )
                if self.show_batch_progress
                else self.val_loader
            )
            for batch in val_iter:
                sample_ids, images, kinetics, masks, labels, _, _, _, _, clinical = self._unpack_batch(batch)
                if images.nelement() == 0:
                    raise RuntimeError(
                        f"Validation batch was empty after collation. Samples around batch: {sample_ids}"
                    )
                images, labels = images.to(self.device), labels.to(self.device)
                kinetics = kinetics.to(self.device) if kinetics is not None and kinetics.nelement() > 0 else None
                masks = masks.to(self.device)
                clinical = clinical.to(self.device) if clinical is not None and clinical.nelement() > 0 else None
                with autocast_context(
                    enabled=self.config["train"].get("amp", True),
                    dtype=self.amp_dtype,
                ):
                    outputs = eval_model(
                        images,
                        kinetics=kinetics,
                        lesion_mask=masks,
                        clinical=clinical,
                    )
                    pcr_logits = outputs[0] if isinstance(outputs, tuple) else outputs
                    if pcr_logits is None:
                        raise RuntimeError(
                            f"Validation model returned no logits for samples: {sample_ids}"
                        )
                    l_p = self.criterions["pcr"](pcr_logits, labels)
                    if len(self.eval_tta_views) > 1:
                        probs, _, _, _ = self._predict_batch_tta(
                            eval_model, images, kinetics, masks, clinical
                        )
                    else:
                        probs = F.softmax(pcr_logits, dim=1)[:, 1]
                    if not torch.isnan(l_p):
                        val_loss += l_p.item()
                        valid_batches += 1
                    # Accumulate predictions regardless of NaN loss status so that
                    # all_labels and all_probs remain length-consistent for metric calculation.
                    all_ids.extend([str(sample_id) for sample_id in sample_ids])
                    all_labels.extend(labels.cpu().numpy())
                    all_probs.extend(probs.float().cpu().numpy())
            if self.show_batch_progress:
                val_iter.close()
        if valid_batches == 0:
            self._last_val_epoch_rows = []
            empty_metrics = {
                "AUC": 0.5,
                "PR-AUC": 0.0,
                "Precision": 0.0,
                "Recall": 0.0,
                "F1": 0.0,
                "ACC": 0.0,
                "Micro-ACC": 0.0,
                "NPV": 0.0,
                "Kappa": 0.0,
                "Sens": 0.0,
                "Spec": 0.0,
                "Thresh": 0.5,
            }
            return empty_metrics, empty_metrics.copy(), 0.0
        m_dynamic = calculate_fast_metrics(all_labels, all_probs)
        m_fixed = calculate_fast_metrics(
            all_labels, all_probs, fixed_thresh=self.val_threshold_fixed
        )
        avg_val_loss = val_loss / valid_batches
        self._last_val_epoch_rows = [
            {
                "id": str(sample_id),
                "label": int(label),
                "prob": float(prob),
                "epoch": int(epoch),
                "fold": str(self.fold),
                "split": "internal_val_epoch",
            }
            for sample_id, label, prob in zip(all_ids, all_labels, all_probs)
        ]
        self.logger.info(
            f"[{self.split_label}] Epoch [{epoch}] Internal Val | Loss(pcr): {avg_val_loss:.4f} | "
            f"AUC: {m_dynamic['AUC']:.3f} | F1(fix): {m_fixed['F1']:.3f} | ACC(fix): {m_fixed['ACC']:.3f} | "
            f"Kappa(fix): {m_fixed['Kappa']:.3f} | "
            f"Sens(fix): {m_fixed['Sens']:.3f} | Spec(fix): {m_fixed['Spec']:.3f} (T={self.val_threshold_fixed:.2f})"
        )
        return m_dynamic, m_fixed, avg_val_loss

    def _predict_with_best_model(self):
        best_model_path = os.path.join(
            self.fold_dir, f"best_model_fold_{self.fold}.pth"
        )
        if not os.path.exists(best_model_path):
            raise FileNotFoundError(f"Best checkpoint not found: {best_model_path}")
        try:
            checkpoint = torch.load(
                best_model_path, map_location=self.device, weights_only=True
            )
        except Exception:
            checkpoint = torch.load(best_model_path, map_location=self.device)
        eval_model = self.ema_model.module if self.ema_model else self.model
        raw_eval_model = eval_model._orig_mod if hasattr(eval_model, "_orig_mod") else eval_model
        state_dict_key = (
            "ema_model_state_dict"
            if self.ema_model and "ema_model_state_dict" in checkpoint
            else "model_state_dict"
        )
        raw_eval_model.load_state_dict(checkpoint[state_dict_key])
        eval_model.eval()
        rows = []
        feature_rows = []
        with torch.no_grad():
            for batch in self.val_loader:
                sample_ids, images, kinetics, masks, labels, _, _, _, _, clinical = self._unpack_batch(batch)
                if images.nelement() == 0:
                    raise RuntimeError(
                        f"Validation batch empty during best-model prediction. IDs: {sample_ids}"
                    )
                images = images.to(self.device)
                kinetics = kinetics.to(self.device) if kinetics is not None and kinetics.nelement() > 0 else None
                masks = masks.to(self.device)
                clinical = clinical.to(self.device) if clinical is not None and clinical.nelement() > 0 else None
                with autocast_context(
                    enabled=self.config["train"].get("amp", True),
                    dtype=self.amp_dtype,
                ):
                    probs, global_feats, core_feats, peri_feats = self._predict_batch_tta(
                        eval_model, images, kinetics, masks, clinical
                    )
                    probs = probs.float().cpu().numpy()
                    global_feats = (
                        global_feats.float().cpu().numpy()
                        if global_feats is not None
                        else None
                    )
                    core_feats = (
                        core_feats.float().cpu().numpy()
                        if core_feats is not None
                        else None
                    )
                    peri_feats = (
                        peri_feats.float().cpu().numpy()
                        if peri_feats is not None
                        else None
                    )
                labels_np = labels.cpu().numpy()
                for idx, (sid, y, p) in enumerate(zip(sample_ids, labels_np, probs)):
                    rows.append(
                        {
                            "id": str(sid),
                            "label": int(y),
                            "prob": float(p),
                            "fold": str(self.fold),
                        }
                    )
                    feature_row = {
                        "id": str(sid),
                        "label": int(y),
                        "prob": float(p),
                        "fold": str(self.fold),
                    }
                    if global_feats is not None:
                        for feat_idx, feat_val in enumerate(global_feats[idx]):
                            feature_row[f"global_feat_{feat_idx:03d}"] = float(feat_val)
                    if core_feats is not None:
                        for feat_idx, feat_val in enumerate(core_feats[idx]):
                            feature_row[f"core_feat_{feat_idx:03d}"] = float(feat_val)
                    if peri_feats is not None:
                        for feat_idx, feat_val in enumerate(peri_feats[idx]):
                            feature_row[f"peri_feat_{feat_idx:03d}"] = float(feat_val)
                    feature_rows.append(feature_row)
        return rows, feature_rows

    def fit(self):
        if self.start_epoch > self.epochs or self.patience_counter >= self.patience:
            self.logger.info(f"Fold {self.fold} has already been trained. Skipping.")
            rows, feature_rows = self._predict_with_best_model()
            return {
                "best_auc": float(self.best_auc),
                "best_threshold": float(self.val_threshold_fixed),
                "oof_rows": rows,
                "feature_rows": feature_rows,
            }
        for epoch in range(self.start_epoch, self.epochs + 1):
            t_loss, t_m = self.train_epoch(epoch)
            v_m_dyn, v_m_fix, v_loss = self.validate(epoch)
            self._log_epoch_scalars(epoch, t_loss, t_m, v_loss, v_m_dyn, v_m_fix)
            self.logger.info(
                f"[{self.split_label}] Epoch [{epoch}] Summary | "
                f"Train Loss={t_loss:.4f}, Train AUC={t_m['AUC']:.3f}, "
                f"Internal Val Loss={v_loss:.4f}, Internal Val AUC={v_m_dyn['AUC']:.3f}, "
                f"Internal Val F1/ACC/Kappa(fix)=({v_m_fix['F1']:.3f}/{v_m_fix['ACC']:.3f}/{v_m_fix['Kappa']:.3f}), "
                f"Internal Val Sens/Spec(dyn)=({v_m_dyn['Sens']:.3f}/{v_m_dyn['Spec']:.3f}), "
                f"Internal Val Sens/Spec(fix)=({v_m_fix['Sens']:.3f}/{v_m_fix['Spec']:.3f}), "
                f"Thresh={v_m_dyn['Thresh']:.4f}"
            )
            curr_auc = np.nan_to_num(v_m_dyn["AUC"], nan=0.5)
            is_best = curr_auc > self.best_auc
            if is_best:
                self.best_auc = curr_auc
                self.patience_counter = 0
                self.val_threshold_fixed = float(v_m_dyn["Thresh"])
                self.logger.info(
                    f"[{self.split_label}] New best model (AUC={curr_auc:.4f}); updating fixed threshold to {self.val_threshold_fixed:.4f}"
                )
                self._save_best_epoch_prediction_artifacts(epoch, t_m, v_m_dyn)
            else:
                self.patience_counter += 1
            self.scheduler.step()
            with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(
                    [
                        epoch,
                        t_loss,
                        t_m["PCR_Loss"],
                        t_m["AUC"],
                        t_m["PR-AUC"],
                        t_m["Precision"],
                        t_m["Recall"],
                        t_m["F1"],
                        t_m["ACC"],
                        t_m["Micro-ACC"],
                        t_m["NPV"],
                        t_m["Kappa"],
                        t_m["Sens"],
                        t_m["Spec"],
                        v_loss,
                        v_m_dyn["AUC"],
                        v_m_dyn["PR-AUC"],
                        v_m_dyn["Precision"],
                        v_m_dyn["Recall"],
                        v_m_dyn["F1"],
                        v_m_dyn["ACC"],
                        v_m_dyn["Micro-ACC"],
                        v_m_dyn["NPV"],
                        v_m_dyn["Kappa"],
                        v_m_dyn["Sens"],
                        v_m_dyn["Spec"],
                        v_m_fix["Precision"],
                        v_m_fix["Recall"],
                        v_m_fix["F1"],
                        v_m_fix["ACC"],
                        v_m_fix["Micro-ACC"],
                        v_m_fix["NPV"],
                        v_m_fix["Kappa"],
                        v_m_fix["Sens"],
                        v_m_fix["Spec"],
                        float(self.optimizer.param_groups[0]["lr"]),
                    ]
                )
            self._save_checkpoint(epoch, is_best)
            if self.patience_counter >= self.patience:
                self.logger.info(
                    f"Early stopping triggered after {self.patience} epochs with no improvement."
                )
                break
        rows, feature_rows = self._predict_with_best_model()
        return {
            "best_auc": float(self.best_auc),
            "best_threshold": float(self.val_threshold_fixed),
            "oof_rows": rows,
            "feature_rows": feature_rows,
        }


def ensemble_predict(models, test_loader, device, amp_dtype, tta_views=None, amp_enabled=True):
    for m in models:
        m.eval()
    all_probs, all_labels, all_centers, all_ids = [], [], [], []
    with torch.no_grad():
        for batch in test_loader:
            sample_ids = batch["id"]
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            kinetics = batch.get("kinetics")
            kinetics = kinetics.to(device) if kinetics is not None and kinetics.nelement() > 0 else None
            clinical = batch.get("clinical")
            clinical = clinical.to(device) if clinical is not None and clinical.nelement() > 0 else None
            labels = batch["label"].cpu().numpy()
            center_ids = batch["center_id"].cpu().numpy()
            if images.nelement() == 0:
                continue
            with autocast_context(enabled=amp_enabled, dtype=amp_dtype):
                probs_ensemble = []
                if tta_views is None or len(tta_views) == 0:
                    for model in models:
                        outputs = model(images, kinetics=kinetics, lesion_mask=masks, clinical=clinical)
                        logits = outputs[0] if isinstance(outputs, tuple) else outputs
                        probs_ensemble.append(F.softmax(logits, dim=1)[:, 1])
                else:
                    for model in models:
                        probs_tta = []
                        for flip_dims in tta_views:
                            aug_images = (
                                torch.flip(images, flip_dims) if flip_dims else images
                            )
                            aug_masks = (
                                torch.flip(masks, flip_dims) if flip_dims else masks
                            )
                            aug_kinetics = (
                                torch.flip(kinetics, [d - 1 for d in flip_dims])
                                if kinetics is not None and flip_dims
                                else kinetics
                            )
                            outputs = model(
                                aug_images,
                                kinetics=aug_kinetics,
                                lesion_mask=aug_masks,
                                clinical=clinical,
                            )
                            logits = (
                                outputs[0] if isinstance(outputs, tuple) else outputs
                            )
                            probs_tta.append(F.softmax(logits, dim=1)[:, 1])
                        probs_ensemble.append(torch.stack(probs_tta, dim=0).mean(dim=0))
            avg_probs = (
                torch.stack(probs_ensemble, dim=0).mean(dim=0).float().cpu().numpy()
            )
            all_probs.extend(avg_probs)
            all_labels.extend(labels)
            all_centers.extend(center_ids)
            all_ids.extend(sample_ids)
    return np.array(all_probs), np.array(all_labels), np.array(all_centers), all_ids
