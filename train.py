import os
import random
import argparse
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
try:
    from sklearn.model_selection import StratifiedGroupKFold
except ImportError:  # pragma: no cover - compatibility for old sklearn
    StratifiedGroupKFold = None
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter

from utils.config_manager import ConfigManager
from utils.logger import get_logger
from utils.experiment import save_json, set_deterministic_environment
from utils.metadata import load_metadata_records
from datasets.mri_dataset import MRIDataset, safe_collate
from datasets.multi_center_sampler import MultiCenterBatchSampler
from models.builder import build_model
from losses.builder import build_losses
from engine.trainer import KFoldTrainer
from engine.ema import ModelEMA
from utils.amp_compat import build_grad_scaler


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_loader(
    dataset,
    batch_size,
    num_workers,
    shuffle,
    seed,
    use_multi_center_sampler=False,
    use_weighted_sampler=False,
    class_balance_power=1.0,
    sampler_num_samples_multiplier=1.0,
):
    generator = torch.Generator()
    generator.manual_seed(seed)

    if use_multi_center_sampler and shuffle and len(dataset) > 0:
        sampler = MultiCenterBatchSampler(
            dataset,
            batch_size=batch_size,
            drop_last=False,
            seed=seed,
            balance_classes=use_weighted_sampler,
            class_balance_power=class_balance_power,
            num_samples_multiplier=sampler_num_samples_multiplier,
        )
        return DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=num_workers,
            collate_fn=safe_collate,
            pin_memory=True,
            worker_init_fn=seed_worker,
            generator=generator,
        )
    elif use_weighted_sampler and shuffle and len(dataset) > 0:
        labels = np.array([int(item["label"]) for item in dataset.data_list], dtype=np.int64)
        class_counts = np.bincount(labels, minlength=max(2, labels.max(initial=0) + 1))
        sample_weights = np.zeros(len(labels), dtype=np.float64)
        for class_idx, count in enumerate(class_counts):
            if count > 0:
                sample_weights[labels == class_idx] = 1.0 / float(count) ** float(
                    class_balance_power
                )
        if sample_weights.sum() > 0:
            sample_weights = sample_weights / sample_weights.sum()
        num_samples = max(
            len(dataset),
            int(round(len(dataset) * float(sampler_num_samples_multiplier))),
        )
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=num_samples,
            replacement=True,
            generator=generator,
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            collate_fn=safe_collate,
            pin_memory=True,
            worker_init_fn=seed_worker,
            generator=generator,
        )
    else:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=safe_collate,
            pin_memory=True,
            worker_init_fn=seed_worker,
            generator=generator,
        )


def _resolve_sampling_strategy(config):
    train_cfg = config.get("train", {})
    return {
        "use_weighted_sampler": bool(train_cfg.get("use_weighted_sampler", True)),
        "class_balance_power": float(train_cfg.get("class_balance_power", 1.0)),
        "sampler_num_samples_multiplier": float(
            train_cfg.get("sampler_num_samples_multiplier", 1.0)
        ),
    }


def build_datasets(
    train_list,
    val_list,
    config,
    shared_center_to_id,
    train_split_name="train",
    val_split_name="val",
):
    data_dir = config.get("paths", {}).get("data_dir", "data/processed")
    aug_cfg = config.get("data", {}).get("augmentation", {})
    train_strict = bool(
        config.get("data", {}).get(
            "strict_file_check_train",
            config.get("data", {}).get("strict_file_check", True),
        )
    )
    val_strict = bool(config.get("data", {}).get("strict_file_check_val", True))
    clinical_keys = [str(key) for key in config.get("data", {}).get("clinical_keys", [])]
    clinical_stats = _resolve_clinical_stats(config, train_list, clinical_keys)
    if clinical_keys:
        config.setdefault("data", {})["clinical_stats"] = clinical_stats
    common_kwargs = dict(
        data_dir=data_dir,
        model_config=config.get("model", {}),
        aug_cfg=aug_cfg,
        use_mask=True,
        cache_in_memory=config.get("data", {}).get("cache_in_memory", False),
        center_to_id=shared_center_to_id,
        clinical_keys=clinical_keys,
        clinical_stats=clinical_stats,
        clinical_missing_indicator=config.get("data", {}).get(
            "clinical_missing_indicator", True
        ),
    )
    train_dataset = MRIDataset(
        train_list,
        is_train=True,
        mask_drop_prob=config.get("train", {}).get("mask_drop_prob", 0.0),
        strict_file_check=train_strict,
        split_name=train_split_name,
        **common_kwargs,
    )
    val_dataset = MRIDataset(
        val_list,
        is_train=False,
        mask_drop_prob=0.0,
        strict_file_check=val_strict,
        split_name=val_split_name,
        **common_kwargs,
    )
    return train_dataset, val_dataset


def load_domain_adaptation_records(config):
    da_cfg = config.get("domain_adaptation", {}) or {}
    if not bool(da_cfg.get("enabled", False)):
        return [], []

    records = []
    summaries = []
    targets = da_cfg.get("targets", []) or []
    max_records = da_cfg.get("max_records_per_target", None)
    rng = np.random.default_rng(int(config.get("project", {}).get("seed", 42)))
    for target_cfg in targets:
        if not isinstance(target_cfg, dict):
            continue
        metadata_path = str(target_cfg.get("metadata_path", "")).strip()
        if not metadata_path:
            continue
        target_records = load_metadata_records(
            metadata_path,
            include_ihc=True,
            require_unique_ids=False,
            clinical_keys=config.get("data", {}).get("clinical_keys", []),
        )
        center_name = target_cfg.get("center", None)
        if center_name:
            center_name = str(center_name)
            target_records = [
                row for row in target_records if str(row.get("center", "")) == center_name
            ]
        if max_records is not None and int(max_records) > 0 and len(target_records) > int(max_records):
            indices = rng.choice(len(target_records), size=int(max_records), replace=False)
            target_records = [target_records[int(idx)] for idx in sorted(indices)]
        records.extend(target_records)
        summaries.append(
            {
                "name": str(target_cfg.get("name", center_name or "target")),
                "center": str(center_name or "mixed"),
                "metadata_path": metadata_path,
                "num_records": int(len(target_records)),
                "positive": int(sum(int(row.get("label", 0)) for row in target_records)),
            }
        )
    return records, summaries


def build_domain_adaptation_loader(
    config,
    center_to_id,
    seed,
    source_train_records,
    target_records=None,
):
    da_cfg = config.get("domain_adaptation", {}) or {}
    if not bool(da_cfg.get("enabled", False)):
        return None, []
    target_records = list(target_records or [])
    if not target_records:
        target_records, summaries = load_domain_adaptation_records(config)
    else:
        summaries = []
    if not target_records:
        return None, summaries

    aug_cfg = config.get("data", {}).get("augmentation", {})
    clinical_keys = [str(key) for key in config.get("data", {}).get("clinical_keys", [])]
    clinical_stats = _resolve_clinical_stats(config, source_train_records, clinical_keys)
    target_data_dir = da_cfg.get("data_dir", config.get("paths", {}).get("data_dir", "data/processed"))
    target_dataset = MRIDataset(
        target_records,
        target_data_dir,
        model_config=config.get("model", {}),
        is_train=True,
        aug_cfg=aug_cfg,
        use_mask=True,
        mask_drop_prob=0.0,
        strict_file_check=bool(da_cfg.get("strict_file_check", False)),
        cache_in_memory=bool(da_cfg.get("cache_in_memory", False)),
        center_to_id=center_to_id,
        split_name="domain-target",
        clinical_keys=clinical_keys,
        clinical_stats=clinical_stats,
        clinical_missing_indicator=config.get("data", {}).get(
            "clinical_missing_indicator", True
        ),
    )
    generator = torch.Generator()
    generator.manual_seed(int(seed) + 913)
    return (
        DataLoader(
            target_dataset,
            batch_size=int(da_cfg.get("batch_size", config["train"]["batch_size"])),
            shuffle=True,
            num_workers=int(da_cfg.get("num_workers", config.get("data", {}).get("num_workers", 4))),
            collate_fn=safe_collate,
            pin_memory=True,
            worker_init_fn=seed_worker,
            generator=generator,
            drop_last=False,
        ),
        summaries,
    )


def _parse_float(value):
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(value):
        return None
    return value


def _compute_clinical_stats(records, clinical_keys):
    stats = {}
    for key in clinical_keys:
        values = [_parse_float(row.get(key)) for row in records]
        values = np.asarray([value for value in values if value is not None], dtype=np.float32)
        if values.size == 0:
            stats[key] = {"mean": 0.0, "std": 1.0}
            continue
        std = float(values.std())
        stats[key] = {"mean": float(values.mean()), "std": std if std > 1.0e-6 else 1.0}
    return stats


def _resolve_clinical_stats(config, records, clinical_keys):
    if not clinical_keys:
        return {}
    configured_stats = config.get("data", {}).get("clinical_stats", {})
    if isinstance(configured_stats, dict) and configured_stats:
        normalized = {}
        for key in clinical_keys:
            raw_stats = configured_stats.get(key)
            if not isinstance(raw_stats, dict):
                break
            try:
                mean = float(raw_stats["mean"])
                std = float(raw_stats["std"])
            except (KeyError, TypeError, ValueError):
                break
            normalized[key] = {"mean": mean, "std": std if std > 1.0e-6 else 1.0}
        else:
            return normalized
    return _compute_clinical_stats(records, clinical_keys)


def get_class_counts(data_list):
    labels = [int(d["label"]) for d in data_list]
    unique, counts = np.unique(labels, return_counts=True)
    mapping = {int(k): int(v) for k, v in zip(unique, counts)}
    return [mapping.get(0, 0), mapping.get(1, 0)]


def build_training_components(config, device, class_counts):
    model = build_model(config, device)
    ema_model = (
        ModelEMA(model, decay=config["train"].get("ema_decay", 0.999))
        if config["train"].get("use_ema", False)
        else None
    )
    criterions = build_losses(config, class_counts=class_counts, device=device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["train"]["lr"]),
        weight_decay=float(config["train"]["weight_decay"]),
    )
    train_cfg = config["train"]
    scheduler_name = str(train_cfg.get("scheduler", "cosine")).lower()
    epochs = int(train_cfg["epochs"])
    if scheduler_name in {"cosine", "cosine_annealing"}:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            eta_min=float(train_cfg.get("min_lr", 0.0)),
        )
    elif scheduler_name in {"warmup_cosine", "linear_warmup_cosine"}:
        warmup_epochs = max(1, int(train_cfg.get("warmup_epochs", 5)))
        warmup_start_factor = float(train_cfg.get("warmup_start_factor", 0.1))
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=warmup_start_factor,
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, epochs - warmup_epochs),
            eta_min=float(train_cfg.get("min_lr", 0.0)),
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_epochs],
        )
    elif scheduler_name in {"onecycle", "one_cycle"}:
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=float(train_cfg.get("max_lr", train_cfg["lr"])),
            epochs=epochs,
            steps_per_epoch=1,
            pct_start=float(train_cfg.get("pct_start", 0.1)),
            div_factor=float(train_cfg.get("div_factor", 10.0)),
            final_div_factor=float(train_cfg.get("final_div_factor", 100.0)),
        )
    else:
        raise ValueError(f"Unsupported train.scheduler: {scheduler_name}")
    # Determine if bfloat16 is supported for AMP
    amp_enabled = config["train"].get("amp", True)
    if amp_enabled and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        scaler_enabled = False # bfloat16 does not need GradScaler
    else:
        scaler_enabled = amp_enabled

    scaler = build_grad_scaler(enabled=scaler_enabled)
    return model, ema_model, criterions, optimizer, scheduler, scaler


def _patient_group_id(record):
    return str(record.get("patient_group", record.get("id", ""))).strip()


def validate_cv_feasibility(labels, n_splits, groups=None):
    unique, counts = np.unique(labels, return_counts=True)
    if len(unique) < 2:
        raise ValueError("At least two classes are required for stratified CV.")
    if groups is None:
        min_count = int(counts.min())
    else:
        group_to_label = {}
        for label, group in zip(labels, groups):
            group = str(group)
            label = int(label)
            if group in group_to_label and group_to_label[group] != label:
                raise ValueError(
                    "Patient-level CV is impossible because patient group "
                    f"{group!r} has conflicting labels."
                )
            group_to_label[group] = label
        grouped_labels = np.array(list(group_to_label.values()), dtype=np.int64)
        _, group_counts = np.unique(grouped_labels, return_counts=True)
        min_count = int(group_counts.min())
    if min_count < n_splits:
        raise ValueError(
            f"Stratified CV with n_splits={n_splits} is invalid because the minority class has only {min_count} patient groups."
        )


def log_method_warnings(config, logger):
    return


def _save_fold_split_artifacts(log_dir, fold_idx, train_list, val_list):
    fold_dir = os.path.join(log_dir, f"fold_{fold_idx}")
    os.makedirs(fold_dir, exist_ok=True)
    train_ids = [d["id"] for d in train_list]
    val_ids = [d["id"] for d in val_list]
    with open(os.path.join(fold_dir, "train_ids.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(train_ids))
    with open(os.path.join(fold_dir, "val_ids.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(val_ids))
    center_summary = {"train": {}, "val": {}}
    label_summary = {"train": {}, "val": {}}
    for item in train_list:
        center_summary["train"][item["center"]] = (
            center_summary["train"].get(item["center"], 0) + 1
        )
        label_key = str(item["label"])
        label_summary["train"][label_key] = label_summary["train"].get(label_key, 0) + 1
    for item in val_list:
        center_summary["val"][item["center"]] = (
            center_summary["val"].get(item["center"], 0) + 1
        )
        label_key = str(item["label"])
        label_summary["val"][label_key] = label_summary["val"].get(label_key, 0) + 1
    train_groups = {_patient_group_id(item) for item in train_list}
    val_groups = {_patient_group_id(item) for item in val_list}
    split_summary = dict(center_summary)
    split_summary.update(
        {
            "train_labels": label_summary["train"],
            "val_labels": label_summary["val"],
            "train_patient_groups": int(len(train_groups)),
            "val_patient_groups": int(len(val_groups)),
            "patient_group_overlap": int(len(train_groups & val_groups)),
        }
    )
    save_json(os.path.join(fold_dir, "split_summary.json"), split_summary)


def _pooled_threshold_and_metrics(oof_rows):
    y_true = np.array([r["label"] for r in oof_rows])
    y_prob = np.array([r["prob"] for r in oof_rows])
    auc_val = (
        float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.5
    )
    from utils.metrics import calculate_fast_metrics

    metrics = calculate_fast_metrics(y_true, y_prob)
    metrics["AUC"] = auc_val
    return metrics


def _load_id_file(path):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def _export_best_fold_clinical_artifacts(config, fold_results, oof_rows):
    if not fold_results:
        return
    metadata_path = config.get("paths", {}).get("metadata_path", "")
    if not metadata_path or not os.path.exists(metadata_path):
        return

    best_fold = max(fold_results, key=lambda item: (item["best_auc"], -item["fold"]))
    best_fold_idx = int(best_fold["fold"])
    best_threshold = float(best_fold["best_threshold"])
    fold_dir = os.path.join(config["paths"]["log_dir"], f"fold_{best_fold_idx}")

    metadata_df = pd.read_csv(metadata_path)
    if "id" not in metadata_df.columns:
        return
    metadata_df["id"] = metadata_df["id"].astype(str)
    metadata_df = metadata_df.drop_duplicates(subset=["id"], keep="first")

    train_ids = _load_id_file(os.path.join(fold_dir, "train_ids.txt"))
    val_ids = _load_id_file(os.path.join(fold_dir, "val_ids.txt"))

    train_meta_df = metadata_df[metadata_df["id"].isin(train_ids)].copy()
    val_meta_df = metadata_df[metadata_df["id"].isin(val_ids)].copy()

    best_fold_oof = pd.DataFrame(
        [row for row in oof_rows if int(row["fold"]) == best_fold_idx]
    )
    if not best_fold_oof.empty:
        best_fold_oof["id"] = best_fold_oof["id"].astype(str)
        best_fold_oof["pred_best_fold_thresh"] = (
            best_fold_oof["prob"].astype(float) >= best_threshold
        ).astype(int)
        best_fold_oof["best_fold_threshold"] = best_threshold
        best_fold_oof["best_fold_auc"] = float(best_fold["best_auc"])
        val_pred_df = val_meta_df.merge(best_fold_oof, on="id", how="left")
    else:
        val_pred_df = val_meta_df.copy()

    train_meta_df.to_csv(
        os.path.join(config["paths"]["log_dir"], "best_fold_train_clinical.csv"),
        index=False,
    )
    val_meta_df.to_csv(
        os.path.join(config["paths"]["log_dir"], "best_fold_val_clinical.csv"),
        index=False,
    )
    val_pred_df.to_csv(
        os.path.join(
            config["paths"]["log_dir"], "best_fold_val_predictions_with_clinical.csv"
        ),
        index=False,
    )
    save_json(
        os.path.join(config["paths"]["log_dir"], "best_fold_summary.json"),
        {
            "best_fold": best_fold_idx,
            "best_auc": float(best_fold["best_auc"]),
            "best_threshold": best_threshold,
            "train_size": int(best_fold["train_size"]),
            "val_size": int(best_fold["val_size"]),
        },
    )


def _plot_fold_history_curves(log_dir, fold_idx):
    history_path = os.path.join(log_dir, f"fold_{fold_idx}", f"history_fold_{fold_idx}.csv")
    if not os.path.exists(history_path):
        return
    history_df = pd.read_csv(history_path)
    if history_df.empty or "Epoch" not in history_df.columns:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(history_df["Epoch"], history_df["Train_Loss"], label="Train Loss", linewidth=2)
    axes[0].plot(
        history_df["Epoch"],
        history_df["InternalVal_PCR_Loss"],
        label="Internal Val Loss",
        linewidth=2,
    )
    axes[0].set_title(f"Fold {fold_idx} Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()

    axes[1].plot(history_df["Epoch"], history_df["Train_AUC"], label="Train AUC", linewidth=2)
    axes[1].plot(
        history_df["Epoch"],
        history_df["InternalVal_AUC"],
        label="Internal Val AUC",
        linewidth=2,
    )
    axes[1].set_title(f"Fold {fold_idx} AUC")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("AUC")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(
        os.path.join(log_dir, f"fold_{fold_idx}", f"training_curves_fold_{fold_idx}.png"),
        dpi=200,
    )
    plt.close(fig)


def _export_cv_curves_and_summary_plots(config, fold_results):
    log_dir = config["paths"]["log_dir"]
    for fold_row in fold_results:
        _plot_fold_history_curves(log_dir, int(fold_row["fold"]))

    if not fold_results:
        return

    summary_df = pd.DataFrame(fold_results).sort_values("fold")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].bar(summary_df["fold"].astype(str), summary_df["best_auc"], color="#2a6f97")
    axes[0].axhline(summary_df["best_auc"].mean(), color="#d62828", linestyle="--", linewidth=2)
    axes[0].set_title("Best Internal Val AUC by Fold")
    axes[0].set_xlabel("Fold")
    axes[0].set_ylabel("AUC")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].grid(True, axis="y", alpha=0.25)

    axes[1].plot(
        summary_df["fold"],
        summary_df["train_size"],
        marker="o",
        linewidth=2,
        label="Train Size",
    )
    axes[1].plot(
        summary_df["fold"],
        summary_df["val_size"],
        marker="o",
        linewidth=2,
        label="Internal Val Size",
    )
    axes[1].set_title("Fold Sample Counts")
    axes[1].set_xlabel("Fold")
    axes[1].set_ylabel("Samples")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(os.path.join(log_dir, "cv_summary_plots.png"), dpi=200)
    plt.close(fig)


def _target_min_group_count(targets, groups):
    group_to_target = {}
    for target, group in zip(targets, groups):
        target = str(target)
        group = str(group)
        if group in group_to_target and group_to_target[group] != target:
            return 0
        group_to_target[group] = target
    _, counts = np.unique(list(group_to_target.values()), return_counts=True)
    return int(counts.min()) if len(counts) else 0


def _build_stratify_labels(all_data, n_splits, logger, groups=None):
    labels = np.array([int(d["label"]) for d in all_data])
    centers = np.array([str(d["center"]) for d in all_data])
    joint = np.array([f"{lab}__{ctr}" for lab, ctr in zip(labels, centers)])
    if groups is None:
        values, counts = np.unique(joint, return_counts=True)
        min_count = int(counts.min()) if len(counts) else 0
    else:
        values = np.unique(joint)
        min_count = _target_min_group_count(joint, groups)
    if len(values) > 1 and min_count >= n_splits:
        logger.info("Using joint label-center stratification for internal CV.")
        return joint
    logger.warning(
        "Joint label-center stratification is infeasible because some label-center strata are too small; falling back to label-only stratification."
    )
    return labels


def run_kfold(all_data, config, device, logger, seed):
    n_splits = int(config.get("train", {}).get("n_splits", 5))
    labels = np.array([d["label"] for d in all_data])
    groups = np.array([_patient_group_id(d) for d in all_data])
    use_patient_group_cv = bool(config.get("train", {}).get("group_by_patient", True))
    validate_cv_feasibility(
        labels, n_splits, groups=groups if use_patient_group_cv else None
    )
    stratify_targets = _build_stratify_labels(
        all_data, n_splits, logger, groups=groups if use_patient_group_cv else None
    )
    target_records, target_summaries = load_domain_adaptation_records(config)
    center_names = {d["center"] for d in all_data}
    center_names.update(row["center"] for row in target_records)
    center_to_id = {name: i for i, name in enumerate(sorted(center_names))}
    config.setdefault("data", {})["num_centers"] = len(center_to_id)
    if target_summaries:
        logger.info(
            "Domain adaptation targets: "
            + "; ".join(
                f"{row['name']}({row['num_records']} records, pos={row['positive']})"
                for row in target_summaries
            )
        )
    if use_patient_group_cv and StratifiedGroupKFold is not None:
        logger.info(
            "Running patient-group StratifiedGroupKFold CV with "
            f"n_splits={n_splits}, num_centers={len(center_to_id)}, "
            f"num_patient_groups={len(set(groups))}"
        )
        try:
            splitter = StratifiedGroupKFold(
                n_splits=n_splits, shuffle=True, random_state=seed
            )
        except TypeError:  # pragma: no cover - old sklearn fallback
            splitter = StratifiedGroupKFold(n_splits=n_splits)
        split_iterator = splitter.split(
            np.arange(len(all_data)), stratify_targets, groups
        )
    else:
        if use_patient_group_cv:
            logger.warning(
                "StratifiedGroupKFold is unavailable; falling back to row-level StratifiedKFold."
            )
        logger.info(
            f"Running true StratifiedKFold CV with n_splits={n_splits}, num_centers={len(center_to_id)}"
        )
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        split_iterator = splitter.split(np.arange(len(all_data)), stratify_targets)
    num_workers = config.get("train", {}).get(
        "num_workers", config.get("data", {}).get("num_workers", 4)
    )
    sampling_strategy = _resolve_sampling_strategy(config)
    fold_results = []
    oof_rows = []
    oof_feature_rows = []
    for fold_idx, (train_idx, val_idx) in enumerate(split_iterator, start=1):
        fold_seed = seed + fold_idx
        set_deterministic_environment(fold_seed)
        train_list = [all_data[i] for i in train_idx]
        val_list = [all_data[i] for i in val_idx]
        _save_fold_split_artifacts(
            config["paths"]["log_dir"], fold_idx, train_list, val_list
        )
        train_class_counts = get_class_counts(train_list)
        logger.info(
            f"Fold {fold_idx}/{n_splits}: train={len(train_list)}, val={len(val_list)}, seed={fold_seed}, class_counts={train_class_counts}"
        )
        train_dataset, val_dataset = build_datasets(
            train_list, val_list, config, center_to_id
        )
        use_domain_debias = config.get("train", {}).get("use_domain_debias", False)
        train_loader = make_loader(
            train_dataset,
            config["train"]["batch_size"],
            num_workers,
            shuffle=True,
            seed=fold_seed,
            use_multi_center_sampler=use_domain_debias,
            **sampling_strategy,
        )
        val_loader = make_loader(
            val_dataset,
            config["train"]["batch_size"],
            num_workers,
            shuffle=False,
            seed=fold_seed,
        )
        target_loader, _ = build_domain_adaptation_loader(
            config,
            center_to_id,
            fold_seed,
            train_list,
            target_records=target_records,
        )
        writer = SummaryWriter(
            log_dir=os.path.join(config["paths"]["log_dir"], f"tensorboard", f"fold_{fold_idx}")
        )
        fold_config = deepcopy(config)
        model, ema_model, criterions, optimizer, scheduler, scaler = (
            build_training_components(fold_config, device, train_class_counts)
        )
        trainer = KFoldTrainer(
            fold=fold_idx,
            model=model,
            ema_model=ema_model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            criterions=criterions,
            config=fold_config,
            device=device,
            logger=logger,
            writer=writer,
            target_loader=target_loader,
            train_ids=[d["id"] for d in train_list],
            val_ids=[d["id"] for d in val_list],
        )
        fold_out = trainer.fit()
        writer.close()
        oof_rows.extend(fold_out["oof_rows"])
        oof_feature_rows.extend(fold_out.get("feature_rows", []))
        fold_results.append(
            {
                "fold": int(fold_idx),
                "best_auc": float(fold_out["best_auc"]),
                "best_threshold": float(fold_out["best_threshold"]),
                "seed": int(fold_seed),
                "train_size": int(len(train_list)),
                "val_size": int(len(val_list)),
                "train_class_counts": train_class_counts,
            }
        )
    pooled_metrics = _pooled_threshold_and_metrics(oof_rows)
    pooled_auc = float(pooled_metrics["AUC"])
    pooled_thresh = float(pooled_metrics["Thresh"])
    mean_auc = (
        float(np.mean([r["best_auc"] for r in fold_results])) if fold_results else 0.0
    )
    std_auc = (
        float(np.std([r["best_auc"] for r in fold_results])) if fold_results else 0.0
    )
    best_fold = (
        max(fold_results, key=lambda item: (item["best_auc"], -item["fold"]))
        if fold_results
        else None
    )
    best_fold_auc = float(best_fold["best_auc"]) if best_fold else 0.0
    best_fold_idx = int(best_fold["fold"]) if best_fold else None
    best_fold_threshold = float(best_fold["best_threshold"]) if best_fold else pooled_thresh
    summary = {
        "mode": "global_cv",
        "n_splits": int(n_splits),
        "seed": int(seed),
        "num_centers": int(len(center_to_id)),
        "center_to_id": center_to_id,
        "primary_auc": mean_auc,
        "primary_auc_source": "mean_best_internal_val_auc",
        "best_auc": mean_auc,
        "best_auc_source": "mean_best_internal_val_auc",
        "mean_best_auc": mean_auc,
        "std_best_auc": std_auc,
        "best_fold": best_fold_idx,
        "best_fold_auc": best_fold_auc,
        "best_fold_threshold": best_fold_threshold,
        "pooled_oof_auc": pooled_auc,
        "pooled_oof_pr_auc": float(pooled_metrics["PR-AUC"]),
        "pooled_oof_precision": float(pooled_metrics["Precision"]),
        "pooled_oof_recall": float(pooled_metrics["Recall"]),
        "pooled_oof_f1": float(pooled_metrics["F1"]),
        "pooled_oof_acc": float(pooled_metrics["ACC"]),
        "pooled_oof_micro_acc": float(pooled_metrics["Micro-ACC"]),
        "pooled_oof_npv": float(pooled_metrics["NPV"]),
        "pooled_oof_kappa": float(pooled_metrics["Kappa"]),
        "pooled_oof_sensitivity": float(pooled_metrics["Sens"]),
        "pooled_oof_specificity": float(pooled_metrics["Spec"]),
        "pooled_oof_threshold": pooled_thresh,
        "folds": fold_results,
    }
    save_json(os.path.join(config["paths"]["log_dir"], "cv_results.json"), summary)
    save_json(
        os.path.join(config["paths"]["log_dir"], "best_threshold.json"),
        {
            "best_threshold": pooled_thresh,
            "best_auc": mean_auc,
            "best_auc_source": "mean_best_internal_val_auc",
            "mean_best_auc": mean_auc,
            "std_best_auc": std_auc,
            "best_fold": best_fold_idx,
            "best_fold_auc": best_fold_auc,
            "best_fold_threshold": best_fold_threshold,
            "pooled_oof_auc": pooled_auc,
            "best_pr_auc": float(pooled_metrics["PR-AUC"]),
            "best_f1": float(pooled_metrics["F1"]),
            "best_accuracy": float(pooled_metrics["ACC"]),
            "best_npv": float(pooled_metrics["NPV"]),
            "best_kappa": float(pooled_metrics["Kappa"]),
            "seed": int(seed),
            "mode": "global_cv",
            "n_splits": int(n_splits),
            "threshold_source": "pooled_oof_predictions",
            "note": (
                "best_auc is the mean of each fold's best internal validation AUC. "
                "The fixed threshold is still estimated from pooled OOF predictions "
                "for downstream internal-test and external-test evaluation."
            ),
        },
    )
    with open(
        os.path.join(config["paths"]["log_dir"], "oof_predictions.csv"),
        "w",
        encoding="utf-8",
    ) as f:
        f.write("id,label,prob,fold\n")
        for row in oof_rows:
            f.write(f"{row['id']},{row['label']},{row['prob']:.8f},{row['fold']}\n")
    if oof_feature_rows:
        pd.DataFrame(oof_feature_rows).to_csv(
            os.path.join(config["paths"]["log_dir"], "oof_feature_vectors.csv"),
            index=False,
        )
    _export_best_fold_clinical_artifacts(config, fold_results, oof_rows)
    _export_cv_curves_and_summary_plots(config, fold_results)
    logger.info(
        f"CV summary | primary best-val AUC={mean_auc:.4f} +/- {std_auc:.4f}, "
        f"best fold={best_fold_idx} AUC={best_fold_auc:.4f}, "
        f"pooled OOF AUC={pooled_auc:.4f}, PR-AUC={pooled_metrics['PR-AUC']:.4f}, "
        f"F1={pooled_metrics['F1']:.4f}, ACC={pooled_metrics['ACC']:.4f}, "
        f"NPV={pooled_metrics['NPV']:.4f}, Kappa={pooled_metrics['Kappa']:.4f}, "
        f"Precision={pooled_metrics['Precision']:.4f}, Recall={pooled_metrics['Recall']:.4f}, "
        f"pooled threshold={pooled_thresh:.4f}"
    )
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Training entry for pooled internal 5-fold cross-validation."
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="configs/server/random/train_config_RAPSS_Net.yaml",
        help="Path to the training config.",
    )
    parser.add_argument(
        "--train_csv", type=str, default=None, help="Optional metadata CSV override."
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed override."
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        help="Optional log directory suffix.",
    )
    args = parser.parse_args()
    config_mgr = ConfigManager(args.config)
    config = config_mgr.config
    seed = int(
        args.seed
        if args.seed is not None
        else config.get("project", {}).get("seed", 42)
    )
    set_deterministic_environment(seed)
    if args.experiment_name:
        base_log_dir = config.get("paths", {}).get("log_dir", "logs")
        config["paths"]["log_dir"] = os.path.join(base_log_dir, args.experiment_name)
    os.makedirs(config["paths"]["log_dir"], exist_ok=True)
    logger = get_logger("Train", config["paths"]["log_dir"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    validation_mode = config.get("train", {}).get("validation_mode", "global_cv")
    logger.info(
        f"Using seed={seed}, validation_mode={validation_mode}, device={device}"
    )
    sampling_strategy = _resolve_sampling_strategy(config)
    if sampling_strategy["use_weighted_sampler"]:
        logger.info(
            "Class-balanced sampling enabled | power=%.3f | epoch_multiplier=%.3f",
            sampling_strategy["class_balance_power"],
            sampling_strategy["sampler_num_samples_multiplier"],
        )
    log_method_warnings(config, logger)
    csv_path = (
        args.train_csv
        if args.train_csv
        else config.get("paths", {}).get("metadata_path", None)
    )
    if not csv_path or not os.path.exists(csv_path):
        raise FileNotFoundError(f"Metadata CSV not found: {csv_path}")
    all_data = load_metadata_records(
        csv_path,
        include_ihc=True,
        require_unique_ids=True,
        clinical_keys=config.get("data", {}).get("clinical_keys", []),
    )
    logger.info(f"Loaded {len(all_data)} valid records from metadata.")
    if validation_mode != "global_cv":
        raise ValueError(
            f"train.py only supports pooled 5-fold cross-validation. "
            f"Got validation_mode={validation_mode!r}. Use run_random_split.py for random_split configs."
        )
    run_kfold(all_data, config, device, logger, seed)


if __name__ == "__main__":
    main()
