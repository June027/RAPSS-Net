import argparse
import csv
import math
import os
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from torch.utils.tensorboard import SummaryWriter

from utils.amp_compat import autocast_context
from utils.config_manager import ConfigManager
from utils.experiment import save_json, set_deterministic_environment
from utils.logger import get_logger
from utils.metadata import load_metadata_records
from utils.metrics import bootstrap_auc_ci, calculate_fast_metrics
from train import (
    build_datasets,
    build_domain_adaptation_loader,
    build_training_components,
    get_class_counts,
    load_domain_adaptation_records,
    make_loader,
    _plot_fold_history_curves,
)
from engine.trainer import KFoldTrainer


def _resolve_centers(config, cli_centers):
    if cli_centers:
        return [str(center) for center in cli_centers]
    centers = config.get("train", {}).get("internal_centers", None)
    return [str(center) for center in centers] if centers else []


def _choose_stratify_targets(records, val_size):
    labels = [str(row["label"]) for row in records]
    joint = [f"{row['label']}__{row['center']}" for row in records]
    total = len(records)
    val_count = max(1, int(math.ceil(total * float(val_size))))
    train_count = total - val_count

    for mode, targets in (("label-center", joint), ("label", labels)):
        values = list(set(targets))
        values.sort()
        min_count = min(targets.count(value) for value in values)
        if min_count < 2:
            continue
        if val_count < len(values) or train_count < len(values):
            continue
        return targets, mode
    return None, "random"


def _patient_group(record):
    return str(record.get("patient_group") or record["id"])


def _build_patient_group_records(records):
    grouped = {}
    for row in records:
        grouped.setdefault(_patient_group(row), []).append(row)

    group_records = []
    for group_id, group_rows in sorted(grouped.items()):
        labels = sorted({str(row["label"]) for row in group_rows})
        if len(labels) != 1:
            raise ValueError(
                "Patient-level random split is impossible because patient group "
                f"{group_id!r} has conflicting labels: {labels}"
            )
        centers = sorted({str(row["center"]) for row in group_rows})
        group_records.append(
            {
                "patient_group": group_id,
                "label": labels[0],
                "center": centers[0] if len(centers) == 1 else "mixed",
                "records": group_rows,
                "num_records": len(group_rows),
            }
        )
    return group_records


def _split_patient_group_records(records, val_size, seed):
    group_records = _build_patient_group_records(records)
    if len(group_records) < 2:
        raise ValueError("At least two patient groups are required for a random split.")

    stratify_targets, stratify_mode = _choose_stratify_targets(group_records, val_size)
    train_groups, val_groups = train_test_split(
        group_records,
        test_size=float(val_size),
        random_state=seed,
        shuffle=True,
        stratify=stratify_targets,
    )
    train_group_ids = {row["patient_group"] for row in train_groups}
    val_group_ids = {row["patient_group"] for row in val_groups}
    overlap = sorted(train_group_ids & val_group_ids)
    if overlap:
        raise ValueError(f"Patient groups overlap between train/val: {overlap[:5]}")

    train_records = [
        row for group in train_groups for row in group["records"]
    ]
    val_records = [
        row for group in val_groups for row in group["records"]
    ]
    return (
        sorted(train_records, key=lambda row: str(row["id"])),
        sorted(val_records, key=lambda row: str(row["id"])),
        f"patient_group-{stratify_mode}",
    )


def _safe_tag(value):
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(value))
    return safe.strip("_") or "split"


def _write_records_csv(path, records):
    if not records:
        return
    fieldnames = sorted(set().union(*(row.keys() for row in records)))
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def _read_split_ids(path):
    if str(path).lower().endswith(".txt"):
        ids = []
        with open(path, "r", encoding="utf-8-sig") as handle:
            for line in handle:
                value = line.strip()
                if value and value.lower() != "id":
                    ids.append(value)
        return ids

    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "id" not in reader.fieldnames:
            raise ValueError(f"Split file must contain an 'id' column: {path}")
        return [str(row["id"]) for row in reader if str(row.get("id", "")).strip()]


def _load_predefined_split_records(split_dir, all_records):
    candidate_pairs = [
        (
            os.path.join(split_dir, "train_records.csv"),
            os.path.join(split_dir, "val_records.csv"),
        ),
        (
            os.path.join(split_dir, "train_ids.txt"),
            os.path.join(split_dir, "val_ids.txt"),
        ),
    ]
    train_path, val_path = next(
        (
            (candidate_train, candidate_val)
            for candidate_train, candidate_val in candidate_pairs
            if os.path.exists(candidate_train) and os.path.exists(candidate_val)
        ),
        (None, None),
    )
    if train_path is None or val_path is None:
        raise FileNotFoundError(
            "Predefined split directory must contain train_records.csv/val_records.csv "
            "or train_ids.txt/val_ids.txt."
        )

    record_by_id = {str(row["id"]): row for row in all_records}
    train_ids = _read_split_ids(train_path)
    val_ids = _read_split_ids(val_path)
    train_set = set(train_ids)
    val_set = set(val_ids)
    overlap = sorted(train_set & val_set)
    if overlap:
        raise ValueError(f"Predefined split has overlapping train/val ids: {overlap[:5]}")

    missing = sorted((train_set | val_set) - set(record_by_id))
    if missing:
        raise ValueError(f"Predefined split contains ids absent from metadata: {missing[:5]}")

    train_records = [record_by_id[sample_id] for sample_id in train_ids]
    val_records = [record_by_id[sample_id] for sample_id in val_ids]
    train_groups = {_patient_group(row) for row in train_records}
    val_groups = {_patient_group(row) for row in val_records}
    group_overlap = sorted(train_groups & val_groups)
    if group_overlap:
        raise ValueError(
            f"Predefined split has overlapping patient groups: {group_overlap[:5]}"
        )
    return (
        sorted(train_records, key=lambda row: str(row["id"])),
        sorted(val_records, key=lambda row: str(row["id"])),
    )


def _count_by(records, key):
    counts = {}
    for row in records:
        value = str(row.get(key, ""))
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


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


def _resolve_cluster_numeric_keys(config, records):
    cluster_cfg = config.get("train", {}).get("patient_clustering", {})
    explicit_keys = cluster_cfg.get("numeric_keys", None)
    if explicit_keys:
        return [str(key) for key in explicit_keys]

    keys = [str(key) for key in config.get("data", {}).get("clinical_keys", [])]
    for key in ["ihc_cd8", "ihc_cd34", "ihc_ki67", "ihc_cd163"]:
        if any(row.get(key) not in (None, "") for row in records):
            keys.append(key)
    return list(dict.fromkeys(keys))


def _build_patient_cluster_matrix(records, numeric_keys, center_levels, numeric_stats=None):
    if numeric_stats is None:
        numeric_stats = {}
        for key in numeric_keys:
            values = [_parse_float(row.get(key)) for row in records]
            values = np.asarray([value for value in values if value is not None], dtype=np.float32)
            numeric_stats[key] = float(values.mean()) if values.size else 0.0

    matrix = []
    for row in records:
        features = []
        center = str(row.get("center", ""))
        features.extend([1.0 if center == level else 0.0 for level in center_levels])
        for key in numeric_keys:
            value = _parse_float(row.get(key))
            features.append(float(numeric_stats[key] if value is None else value))
        matrix.append(features)

    return np.asarray(matrix, dtype=np.float32), numeric_stats


def _write_patient_cluster_artifacts(run_dir, train_records, val_records, cluster_info):
    cluster_dir = os.path.join(run_dir, "patient_clusters")
    os.makedirs(cluster_dir, exist_ok=True)
    all_rows = []
    for split_name, records in (("train", train_records), ("internal_val", val_records)):
        for row in records:
            all_rows.append(
                {
                    "id": str(row.get("id", "")),
                    "split": split_name,
                    "center": str(row.get("center", "")),
                    "label": int(row.get("label", 0)),
                    "patient_cluster": str(row.get("patient_cluster", "unclustered")),
                }
            )
    with open(os.path.join(cluster_dir, "patient_clusters.csv"), "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["id", "split", "center", "label", "patient_cluster"],
        )
        writer.writeheader()
        writer.writerows(all_rows)

    summary_rows = []
    numeric_keys = [str(key) for key in cluster_info.get("numeric_keys", [])]
    for split_name, records in (("train", train_records), ("internal_val", val_records)):
        cluster_counts = _count_by(records, "patient_cluster")
        for cluster_id, count in cluster_counts.items():
            cluster_records = [row for row in records if str(row.get("patient_cluster")) == cluster_id]
            labels = _count_by(cluster_records, "label")
            centers = _count_by(cluster_records, "center")
            numeric_means = {}
            for key in numeric_keys:
                values = [_parse_float(row.get(key)) for row in cluster_records]
                values = [value for value in values if value is not None]
                if values:
                    numeric_means[key] = float(np.mean(values))
            summary_rows.append(
                {
                    "split": split_name,
                    "patient_cluster": cluster_id,
                    "count": int(count),
                    "label_0": int(labels.get("0", labels.get(0, 0))),
                    "label_1": int(labels.get("1", labels.get(1, 0))),
                    "centers": ";".join(f"{key}:{value}" for key, value in centers.items()),
                    "numeric_means": ";".join(
                        f"{key}:{value:.4f}" for key, value in numeric_means.items()
                    ),
                }
            )
    with open(os.path.join(cluster_dir, "patient_cluster_summary.csv"), "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "split",
                "patient_cluster",
                "count",
                "label_0",
                "label_1",
                "centers",
                "numeric_means",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    save_json(os.path.join(cluster_dir, "patient_cluster_config.json"), cluster_info)
    return cluster_dir


def _assign_patient_clusters(config, train_records, val_records, run_dir, seed):
    cluster_cfg = config.get("train", {}).get("patient_clustering", {})
    if not bool(cluster_cfg.get("enabled", False)):
        for row in train_records + val_records:
            row["patient_cluster"] = "0"
        return {"enabled": False, "cluster_dir": None}

    if not train_records:
        raise ValueError("Patient clustering requires at least one training record.")

    n_clusters = int(cluster_cfg.get("n_clusters", 4))
    n_clusters = max(1, min(n_clusters, len(train_records)))
    numeric_keys = _resolve_cluster_numeric_keys(config, train_records)
    center_levels = sorted({str(row.get("center", "")) for row in train_records})
    train_matrix, numeric_stats = _build_patient_cluster_matrix(
        train_records,
        numeric_keys,
        center_levels,
    )
    val_matrix, _ = _build_patient_cluster_matrix(
        val_records,
        numeric_keys,
        center_levels,
        numeric_stats=numeric_stats,
    )

    if train_matrix.shape[1] == 0 or n_clusters == 1:
        train_labels = np.zeros(len(train_records), dtype=np.int64)
        val_labels = np.zeros(len(val_records), dtype=np.int64)
        scaler_mean = []
        scaler_scale = []
    else:
        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train_matrix)
        val_scaled = scaler.transform(val_matrix) if len(val_records) else np.empty((0, train_matrix.shape[1]))
        kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
        train_labels = kmeans.fit_predict(train_scaled)
        val_labels = kmeans.predict(val_scaled) if len(val_records) else np.zeros(0, dtype=np.int64)
        scaler_mean = [float(value) for value in scaler.mean_]
        scaler_scale = [float(value) for value in scaler.scale_]

    for row, cluster_id in zip(train_records, train_labels):
        row["patient_cluster"] = str(int(cluster_id))
    for row, cluster_id in zip(val_records, val_labels):
        row["patient_cluster"] = str(int(cluster_id))

    cluster_info = {
        "enabled": True,
        "fit_split": "train",
        "n_clusters": int(n_clusters),
        "numeric_keys": numeric_keys,
        "center_levels": center_levels,
        "numeric_imputation": numeric_stats,
        "scaler_mean": scaler_mean,
        "scaler_scale": scaler_scale,
    }
    cluster_dir = _write_patient_cluster_artifacts(
        run_dir, train_records, val_records, cluster_info
    )
    cluster_info["cluster_dir"] = cluster_dir
    return cluster_info


def _save_split_artifacts(run_dir, train_records, val_records, seed, val_size, stratify_mode):
    split_dir = os.path.join(run_dir, "random_split")
    os.makedirs(split_dir, exist_ok=True)
    _write_records_csv(os.path.join(split_dir, "train_records.csv"), train_records)
    _write_records_csv(os.path.join(split_dir, "val_records.csv"), val_records)
    train_groups = {_patient_group(row) for row in train_records}
    val_groups = {_patient_group(row) for row in val_records}
    patient_group_overlap = sorted(train_groups & val_groups)

    summary = {
        "seed": int(seed),
        "val_size": float(val_size),
        "stratify_mode": stratify_mode,
        "train_size": int(len(train_records)),
        "val_size_count": int(len(val_records)),
        "train_patient_group_count": int(len(train_groups)),
        "val_patient_group_count": int(len(val_groups)),
        "patient_group_overlap_count": int(len(patient_group_overlap)),
        "patient_group_overlap_preview": patient_group_overlap[:20],
        "train_label_counts": _count_by(train_records, "label"),
        "val_label_counts": _count_by(val_records, "label"),
        "train_center_counts": _count_by(train_records, "center"),
        "val_center_counts": _count_by(val_records, "center"),
        "train_patient_cluster_counts": _count_by(train_records, "patient_cluster"),
        "val_patient_cluster_counts": _count_by(val_records, "patient_cluster"),
    }
    save_json(os.path.join(split_dir, "split_summary.json"), summary)
    return split_dir


def _write_oof_predictions(path, rows):
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("id,label,prob,fold\n")
        for row in rows:
            handle.write(
                f"{row['id']},{int(row['label'])},{float(row['prob']):.8f},{row['fold']}\n"
            )


def _write_prediction_rows(path, rows):
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("id,label,prob,split\n")
        for row in rows:
            handle.write(
                f"{row['id']},{int(row['label'])},{float(row['prob']):.8f},{row['split']}\n"
            )


def _write_validation_error_analysis(run_dir, validation_rows, val_records, threshold):
    record_by_id = {str(row["id"]): row for row in val_records}
    analysis_rows = []
    summary = {
        "threshold": float(threshold),
        "num_samples": int(len(validation_rows)),
        "num_errors": 0,
        "error_rate": 0.0,
        "false_positive_count": 0,
        "false_negative_count": 0,
        "high_confidence_error_count": 0,
        "error_by_center": {},
        "error_by_cluster": {},
        "error_by_label": {},
    }
    cluster_stats = {}

    for row in validation_rows:
        sample_id = str(row["id"])
        label = int(row["label"])
        prob = float(row["prob"])
        pred = int(prob >= threshold)
        is_error = int(pred != label)
        confidence = prob if pred == 1 else 1.0 - prob
        margin = abs(prob - threshold)
        metadata = record_by_id.get(sample_id, {})
        center = str(metadata.get("center", "unknown"))
        patient_cluster = str(metadata.get("patient_cluster", "unclustered"))
        if pred == 1 and label == 0:
            error_type = "false_positive"
        elif pred == 0 and label == 1:
            error_type = "false_negative"
        elif pred == 1:
            error_type = "true_positive"
        else:
            error_type = "true_negative"

        if is_error:
            summary["num_errors"] += 1
            summary["error_by_center"][center] = summary["error_by_center"].get(center, 0) + 1
            summary["error_by_cluster"][patient_cluster] = (
                summary["error_by_cluster"].get(patient_cluster, 0) + 1
            )
            label_key = str(label)
            summary["error_by_label"][label_key] = summary["error_by_label"].get(label_key, 0) + 1
            if error_type == "false_positive":
                summary["false_positive_count"] += 1
            elif error_type == "false_negative":
                summary["false_negative_count"] += 1
            if confidence >= 0.75:
                summary["high_confidence_error_count"] += 1
        cluster_row = cluster_stats.setdefault(
            patient_cluster,
            {
                "patient_cluster": patient_cluster,
                "count": 0,
                "errors": 0,
                "false_positive": 0,
                "false_negative": 0,
                "high_confidence_errors": 0,
                "mean_prob": 0.0,
                "centers": {},
            },
        )
        cluster_row["count"] += 1
        cluster_row["errors"] += is_error
        cluster_row["false_positive"] += int(error_type == "false_positive")
        cluster_row["false_negative"] += int(error_type == "false_negative")
        cluster_row["high_confidence_errors"] += int(is_error and confidence >= 0.75)
        cluster_row["mean_prob"] += prob
        cluster_row["centers"][center] = cluster_row["centers"].get(center, 0) + 1

        analysis_rows.append(
            {
                "id": sample_id,
                "center": center,
                "patient_cluster": patient_cluster,
                "label": label,
                "prob": prob,
                "pred": pred,
                "threshold": float(threshold),
                "is_error": is_error,
                "error_type": error_type,
                "confidence": confidence,
                "margin_to_threshold": margin,
            }
        )

    if validation_rows:
        summary["error_rate"] = float(summary["num_errors"] / len(validation_rows))
    analysis_rows.sort(
        key=lambda item: (
            -int(item["is_error"]),
            -float(item["confidence"]),
            float(item["margin_to_threshold"]),
            str(item["id"]),
        )
    )

    csv_path = os.path.join(run_dir, "validation_error_analysis.csv")
    json_path = os.path.join(run_dir, "validation_error_summary.json")
    cluster_csv_path = os.path.join(run_dir, "validation_cluster_error_summary.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "id",
            "center",
            "patient_cluster",
            "label",
            "prob",
            "pred",
            "threshold",
            "is_error",
            "error_type",
            "confidence",
            "margin_to_threshold",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(analysis_rows)
    cluster_rows = []
    for cluster_id, row in cluster_stats.items():
        count = max(int(row["count"]), 1)
        cluster_rows.append(
            {
                "patient_cluster": cluster_id,
                "count": int(row["count"]),
                "errors": int(row["errors"]),
                "error_rate": float(row["errors"] / count),
                "false_positive": int(row["false_positive"]),
                "false_negative": int(row["false_negative"]),
                "high_confidence_errors": int(row["high_confidence_errors"]),
                "mean_prob": float(row["mean_prob"] / count),
                "centers": ";".join(
                    f"{key}:{value}" for key, value in sorted(row["centers"].items())
                ),
            }
        )
    cluster_rows.sort(
        key=lambda item: (-float(item["error_rate"]), -int(item["errors"]), str(item["patient_cluster"]))
    )
    with open(cluster_csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "patient_cluster",
                "count",
                "errors",
                "error_rate",
                "false_positive",
                "false_negative",
                "high_confidence_errors",
                "mean_prob",
                "centers",
            ],
        )
        writer.writeheader()
        writer.writerows(cluster_rows)
    save_json(json_path, summary)
    return {
        "csv": csv_path,
        "summary_json": json_path,
        "cluster_csv": cluster_csv_path,
        "summary": summary,
    }


def _load_best_model_for_eval(model, ema_model, fold_dir, fold_idx, device):
    best_model_path = os.path.join(fold_dir, f"best_model_fold_{fold_idx}.pth")
    if not os.path.exists(best_model_path):
        raise FileNotFoundError(f"Best checkpoint not found: {best_model_path}")
    try:
        checkpoint = torch.load(best_model_path, map_location=device, weights_only=True)
    except Exception:
        checkpoint = torch.load(best_model_path, map_location=device)

    eval_model = ema_model.module if ema_model else model
    raw_eval_model = eval_model._orig_mod if hasattr(eval_model, "_orig_mod") else eval_model
    state_dict_key = (
        "ema_model_state_dict"
        if ema_model and "ema_model_state_dict" in checkpoint
        else "model_state_dict"
    )
    raw_eval_model.load_state_dict(checkpoint[state_dict_key])
    eval_model.eval()
    return eval_model


def _predict_loader(eval_model, loader, criterions, config, device, split_name):
    amp_dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float16
    )
    all_labels = []
    all_probs = []
    rows = []
    total_loss = 0.0
    valid_batches = 0
    with torch.no_grad():
        for batch in loader:
            sample_ids = batch["id"]
            images = batch["image"]
            if images.nelement() == 0:
                continue
            kinetics = batch.get("kinetics")
            masks = batch["mask"]
            labels = batch["label"]
            clinical = batch.get("clinical")

            images = images.to(device)
            kinetics = kinetics.to(device) if kinetics is not None and kinetics.nelement() > 0 else None
            masks = masks.to(device)
            labels_device = labels.to(device)
            clinical = clinical.to(device) if clinical is not None and clinical.nelement() > 0 else None

            with autocast_context(
                enabled=config["train"].get("amp", True),
                dtype=amp_dtype,
            ):
                outputs = eval_model(
                    images,
                    kinetics=kinetics,
                    lesion_mask=masks,
                    clinical=clinical,
                )
                logits = outputs[0] if isinstance(outputs, tuple) else outputs
                loss = criterions["pcr"](logits, labels_device)
                probs = F.softmax(logits, dim=1)[:, 1].float().cpu().numpy()

            total_loss += float(loss.item())
            valid_batches += 1
            labels_np = labels.cpu().numpy()
            all_labels.extend(labels_np)
            all_probs.extend(probs)
            for sample_id, label, prob in zip(sample_ids, labels_np, probs):
                rows.append(
                    {
                        "id": str(sample_id),
                        "label": int(label),
                        "prob": float(prob),
                        "split": split_name,
                    }
                )

    metrics = calculate_fast_metrics(np.array(all_labels), np.array(all_probs))
    metrics["PCR_Loss"] = float(total_loss / max(valid_batches, 1))
    return rows, metrics


def _plot_random_split_history_curves(run_dir):
    history_path = os.path.join(run_dir, "fold_1", "history_fold_1.csv")
    if not os.path.exists(history_path):
        return
    try:
        import pandas as pd
    except ImportError:
        return
    history_df = pd.read_csv(history_path)
    if history_df.empty or "Epoch" not in history_df.columns:
        return

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.reshape(-1)

    axes[0].plot(history_df["Epoch"], history_df["Train_Loss"], label="Train Loss", linewidth=2)
    axes[0].plot(
        history_df["Epoch"],
        history_df["InternalVal_PCR_Loss"],
        label="Val Loss",
        linewidth=2,
    )
    axes[0].set_title("Loss Curves")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()

    axes[1].plot(history_df["Epoch"], history_df["Train_AUC"], label="Train AUC", linewidth=2)
    axes[1].plot(history_df["Epoch"], history_df["InternalVal_AUC"], label="Val AUC", linewidth=2)
    axes[1].set_title("AUC Curves")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("AUC")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()

    axes[2].plot(history_df["Epoch"], history_df["Train_PRAUC"], label="Train PR-AUC", linewidth=2)
    axes[2].plot(history_df["Epoch"], history_df["InternalVal_PRAUC"], label="Val PR-AUC", linewidth=2)
    axes[2].set_title("PR-AUC Curves")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("PR-AUC")
    axes[2].set_ylim(0.0, 1.0)
    axes[2].grid(True, alpha=0.25)
    axes[2].legend()

    axes[3].plot(history_df["Epoch"], history_df["Train_F1"], label="Train F1", linewidth=2)
    axes[3].plot(history_df["Epoch"], history_df["InternalVal_F1_Fix"], label="Val F1", linewidth=2)
    axes[3].plot(history_df["Epoch"], history_df["Train_ACC"], label="Train ACC", linewidth=2)
    axes[3].plot(history_df["Epoch"], history_df["InternalVal_ACC_Fix"], label="Val ACC", linewidth=2)
    axes[3].set_title("F1 / ACC Curves")
    axes[3].set_xlabel("Epoch")
    axes[3].set_ylabel("Score")
    axes[3].set_ylim(0.0, 1.0)
    axes[3].grid(True, alpha=0.25)
    axes[3].legend()

    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "random_split_training_curves.png"), dpi=200)
    plt.close(fig)


def _plot_random_split_summary(run_dir, results):
    metric_keys = [
        ("auc", "AUC"),
        ("pr_auc", "PR-AUC"),
        ("f1", "F1"),
        ("acc", "ACC"),
        ("sensitivity", "Sensitivity"),
        ("specificity", "Specificity"),
        ("precision", "Precision"),
        ("recall", "Recall"),
    ]
    labels = [label for _, label in metric_keys]
    train_metrics = results.get("train_metrics", {})
    validation_metrics = results.get(
        "validation_metrics", results.get("holdout_metrics", {})
    )
    train_values = [float(train_metrics.get(key, 0.0)) for key, _ in metric_keys]
    validation_values = [
        float(validation_metrics.get(key, 0.0)) for key, _ in metric_keys
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    x_pos = np.arange(len(labels))
    width = 0.38
    axes[0].bar(x_pos - width / 2, train_values, width=width, label="Train", color="#2a6f97")
    axes[0].bar(x_pos + width / 2, validation_values, width=width, label="Internal Val", color="#d62828")
    axes[0].set_title("Random Split Internal Validation Metrics")
    axes[0].set_ylabel("Score")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(labels, rotation=30, ha="right")
    axes[0].grid(True, axis="y", alpha=0.25)
    axes[0].legend()

    axes[1].bar(
        ["Train", "Internal Val"],
        [int(results.get("train_size", 0)), int(results.get("val_size_count", 0))],
        color=["#2a6f97", "#d62828"],
    )
    axes[1].set_title("Random Split Sample Counts")
    axes[1].set_ylabel("Samples")
    axes[1].grid(True, axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "random_split_summary_plots.png"), dpi=200)
    plt.close(fig)


def _metric_payload(metrics):
    return {
        "pcr_loss": float(metrics.get("PCR_Loss", 0.0)),
        "auc": float(metrics.get("AUC", 0.0)),
        "pr_auc": float(metrics.get("PR-AUC", 0.0)),
        "precision": float(metrics.get("Precision", 0.0)),
        "recall": float(metrics.get("Recall", 0.0)),
        "f1": float(metrics.get("F1", 0.0)),
        "acc": float(metrics.get("ACC", 0.0)),
        "micro_acc": float(metrics.get("Micro-ACC", 0.0)),
        "npv": float(metrics.get("NPV", 0.0)),
        "kappa": float(metrics.get("Kappa", 0.0)),
        "sensitivity": float(metrics.get("Sens", 0.0)),
        "specificity": float(metrics.get("Spec", 0.0)),
        "threshold": float(metrics.get("Thresh", 0.0)),
    }


def _with_auc_ci(metrics, auc_ci):
    payload = _metric_payload(metrics)
    payload["auc_95ci"] = auc_ci
    return payload


def _fmt_metric(value):
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "n/a"


def _fmt_ci(values):
    if not values or len(values) != 2 or values[0] is None or values[1] is None:
        return "n/a"
    return f"[{_fmt_metric(values[0])}, {_fmt_metric(values[1])}]"


def _print_best_random_split_result(run_dir, results):
    train_metrics = results.get("train_metrics", {})
    val_metrics = results.get("validation_metrics", {})
    print("\n" + "=" * 80)
    print("BEST TRAINING RESULT")
    print("=" * 80)
    print(f"Run directory: {run_dir}")
    print(
        "Best internal validation AUC: "
        f"{_fmt_metric(results.get('best_auc'))} "
        "(best epoch selected by InternalVal_AUC)"
    )
    print(f"Best threshold: {_fmt_metric(results.get('best_threshold'))}")
    print(
        "Validation at best checkpoint: "
        f"AUC={_fmt_metric(val_metrics.get('auc'))}, "
        f"95% CI={_fmt_ci(val_metrics.get('auc_95ci'))}, "
        f"PR-AUC={_fmt_metric(val_metrics.get('pr_auc'))}, "
        f"ACC={_fmt_metric(val_metrics.get('acc'))}, "
        f"F1={_fmt_metric(val_metrics.get('f1'))}, "
        f"Sens={_fmt_metric(val_metrics.get('sensitivity'))}, "
        f"Spec={_fmt_metric(val_metrics.get('specificity'))}"
    )
    print(
        "Train at best checkpoint: "
        f"AUC={_fmt_metric(train_metrics.get('auc'))}, "
        f"95% CI={_fmt_ci(train_metrics.get('auc_95ci'))}, "
        f"PR-AUC={_fmt_metric(train_metrics.get('pr_auc'))}, "
        f"ACC={_fmt_metric(train_metrics.get('acc'))}, "
        f"F1={_fmt_metric(train_metrics.get('f1'))}"
    )
    print("=" * 80)


def run_random_split():
    parser = argparse.ArgumentParser(
        description="Train one strict patient-level random train/internal-validation split."
    )
    parser.add_argument("-c", "--config", required=True, help="Training config path.")
    parser.add_argument(
        "--internal_centers",
        nargs="+",
        default=None,
        help="Centers to include, e.g. ISPY1 ISPY2.",
    )
    parser.add_argument("--val_size", type=float, default=0.2, help="Validation ratio.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed override.")
    parser.add_argument(
        "--split_dir",
        default=None,
        help="Directory containing predefined train_records.csv and val_records.csv.",
    )
    args = parser.parse_args()

    if not 0.0 < float(args.val_size) < 1.0:
        raise ValueError("--val_size must be in (0, 1).")

    project_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = str(args.config if os.path.isabs(args.config) else os.path.join(project_dir, args.config))
    config_manager = ConfigManager(config_path)
    config = config_manager.config
    validation_mode = str(config.get("train", {}).get("validation_mode", "")).lower()
    if validation_mode != "random_split":
        raise ValueError(
            "run_random_split.py requires train.validation_mode: random_split. "
            f"Got {validation_mode!r}."
        )
    seed = int(args.seed if args.seed is not None else config["project"].get("seed", 2026))
    set_deterministic_environment(seed)

    centers = _resolve_centers(config, args.internal_centers)
    all_records = load_metadata_records(
        config["paths"]["metadata_path"],
        include_ihc=True,
        require_unique_ids=True,
        clinical_keys=config.get("data", {}).get("clinical_keys", []),
    )
    if centers:
        all_records = [row for row in all_records if str(row["center"]) in centers]
    if len(all_records) < 10:
        raise ValueError("Too few records for a random train/internal-validation split.")

    if args.split_dir:
        train_records, val_records = _load_predefined_split_records(args.split_dir, all_records)
        stratify_mode = "predefined"
        actual_val_size = len(val_records) / max(len(train_records) + len(val_records), 1)
    else:
        train_records, val_records, stratify_mode = _split_patient_group_records(
            all_records,
            val_size=args.val_size,
            seed=seed,
        )
        actual_val_size = float(args.val_size)

    center_label = "+".join(centers) if centers else "All"
    base_log_dir = config["paths"].get("log_dir", "logs")
    run_dir = os.path.join(
        base_log_dir,
        f"RandomSplit_({center_label})_seed_{seed}_val_{float(actual_val_size):.2f}",
    )
    os.makedirs(run_dir, exist_ok=True)
    config = deepcopy(config)
    config["paths"]["log_dir"] = run_dir

    logger = get_logger("random_split", run_dir, filename="random_split.log")
    logger.info(
        f"Random split | samples={len(all_records)}, train={len(train_records)}, "
        f"val={len(val_records)}, centers={center_label}, seed={seed}, stratify={stratify_mode}"
    )
    cluster_info = _assign_patient_clusters(
        config,
        train_records,
        val_records,
        run_dir,
        seed,
    )
    if cluster_info.get("enabled"):
        logger.info(
            f"Patient clustering enabled | n_clusters={cluster_info.get('n_clusters')} | "
            f"artifacts={cluster_info.get('cluster_dir')}"
        )
    split_dir = _save_split_artifacts(
        run_dir, train_records, val_records, seed, actual_val_size, stratify_mode
    )
    logger.info(f"Split artifacts saved to: {split_dir}")

    device = torch.device(config["project"].get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    target_records, target_summaries = load_domain_adaptation_records(config)
    center_names = {row["center"] for row in all_records}
    center_names.update(row["center"] for row in target_records)
    center_to_id = {name: idx for idx, name in enumerate(sorted(center_names))}
    config.setdefault("data", {})["num_centers"] = len(center_to_id)
    if target_summaries:
        logger.info(
            "Domain adaptation targets: "
            + ", ".join(
                f"{row['name']}({row['center']}): n={row['num_records']}, pos={row['positive']}"
                for row in target_summaries
            )
        )
    train_dataset, val_dataset = build_datasets(
        train_records, val_records, config, center_to_id
    )
    train_cfg = config.get("train", {})
    train_loader = make_loader(
        train_dataset,
        train_cfg["batch_size"],
        train_cfg.get("num_workers", config.get("data", {}).get("num_workers", 4)),
        shuffle=True,
        seed=seed,
        use_multi_center_sampler=bool(train_cfg.get("use_domain_debias", False)),
        use_weighted_sampler=bool(train_cfg.get("use_weighted_sampler", False)),
        class_balance_power=float(train_cfg.get("class_balance_power", 1.0)),
        sampler_num_samples_multiplier=float(train_cfg.get("sampler_num_samples_multiplier", 1.0)),
    )
    val_loader = make_loader(
        val_dataset,
        train_cfg["batch_size"],
        train_cfg.get("num_workers", config.get("data", {}).get("num_workers", 4)),
        shuffle=False,
        seed=seed,
    )
    _, train_eval_dataset = build_datasets(
        train_records,
        train_records,
        config,
        center_to_id,
        train_split_name="train-aug",
        val_split_name="train-eval",
    )
    train_eval_loader = make_loader(
        train_eval_dataset,
        train_cfg["batch_size"],
        train_cfg.get("num_workers", config.get("data", {}).get("num_workers", 4)),
        shuffle=False,
        seed=seed,
    )
    target_loader, _ = build_domain_adaptation_loader(
        config,
        center_to_id,
        seed,
        train_records,
        target_records=target_records,
    )

    class_counts = get_class_counts(train_records)
    model, ema_model, criterions, optimizer, scheduler, scaler = build_training_components(
        config, device, class_counts
    )
    writer = SummaryWriter(log_dir=os.path.join(run_dir, "tensorboard", "random_split"))
    trainer = KFoldTrainer(
        fold=1,
        model=model,
        ema_model=ema_model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        criterions=criterions,
        config=config,
        device=device,
        logger=logger,
        writer=writer,
        target_loader=target_loader,
        train_ids=[row["id"] for row in train_records],
        val_ids=[row["id"] for row in val_records],
        split_label="RandomSplit",
    )
    fold_out = trainer.fit()
    writer.close()

    best_eval_model = _load_best_model_for_eval(
        model, ema_model, trainer.fold_dir, 1, device
    )
    train_rows, train_metrics = _predict_loader(
        best_eval_model,
        train_eval_loader,
        criterions,
        config,
        device,
        "train",
    )
    validation_rows = fold_out.get("oof_rows", [])
    _write_oof_predictions(os.path.join(run_dir, "random_split_predictions.csv"), validation_rows)
    _write_oof_predictions(os.path.join(run_dir, "validation_predictions.csv"), validation_rows)
    _write_prediction_rows(os.path.join(run_dir, "train_predictions.csv"), train_rows)
    _write_prediction_rows(os.path.join(run_dir, "val_predictions.csv"), [
        {**row, "split": "internal_val"} for row in validation_rows
    ])
    y_true = np.array([row["label"] for row in validation_rows])
    y_prob = np.array([row["prob"] for row in validation_rows])
    validation_metrics = calculate_fast_metrics(y_true, y_prob)
    validation_metrics["PCR_Loss"] = 0.0
    n_boot = max(1, int(config.get("test", {}).get("bootstrap_iterations", 1000)))
    validation_auc_ci = bootstrap_auc_ci(
        y_true,
        y_prob,
        seed=seed,
        n_boot=n_boot,
    )
    train_y_true = np.array([row["label"] for row in train_rows])
    train_y_prob = np.array([row["prob"] for row in train_rows])
    train_auc_ci = bootstrap_auc_ci(
        train_y_true,
        train_y_prob,
        seed=seed,
        n_boot=n_boot,
    )
    train_metrics_payload = _with_auc_ci(train_metrics, train_auc_ci)
    validation_metrics_payload = _with_auc_ci(validation_metrics, validation_auc_ci)
    error_analysis = _write_validation_error_analysis(
        run_dir,
        validation_rows,
        val_records,
        float(validation_metrics.get("Thresh", 0.5)),
    )
    results = {
        "mode": "random_internal_validation",
        "seed": int(seed),
        "val_size": float(args.val_size),
        "stratify_mode": stratify_mode,
        "train_size": int(len(train_records)),
        "val_size_count": int(len(val_records)),
        "best_auc": float(fold_out.get("best_auc", 0.0)),
        "best_threshold": float(fold_out.get("best_threshold", 0.0)),
        "train_metrics": train_metrics_payload,
        "validation_metrics": validation_metrics_payload,
        "auc_ci_method": "bootstrap_percentile",
        "auc_ci_level": 0.95,
        "auc_ci_bootstrap_iterations": int(n_boot),
        "validation_note": (
            "The validation split is used for early stopping, best-epoch selection, "
            "and Youden threshold selection; these metrics are not an independent holdout estimate."
        ),
        "patient_clustering": cluster_info,
        "validation_error_analysis": {
            "csv": error_analysis["csv"],
            "summary_json": error_analysis["summary_json"],
            "cluster_csv": error_analysis["cluster_csv"],
            "summary": error_analysis["summary"],
        },
    }
    results.update(
        {
            "validation_auc": results["validation_metrics"]["auc"],
            "validation_pr_auc": results["validation_metrics"]["pr_auc"],
            "validation_precision": results["validation_metrics"]["precision"],
            "validation_recall": results["validation_metrics"]["recall"],
            "validation_f1": results["validation_metrics"]["f1"],
            "validation_acc": results["validation_metrics"]["acc"],
            "validation_micro_acc": results["validation_metrics"]["micro_acc"],
            "validation_npv": results["validation_metrics"]["npv"],
            "validation_kappa": results["validation_metrics"]["kappa"],
            "validation_sensitivity": results["validation_metrics"]["sensitivity"],
            "validation_specificity": results["validation_metrics"]["specificity"],
            "validation_threshold": results["validation_metrics"]["threshold"],
            "validation_auc_95ci": results["validation_metrics"]["auc_95ci"],
            "train_auc_95ci": results["train_metrics"]["auc_95ci"],
            "deprecated_holdout_metrics_alias": results["validation_metrics"],
        }
    )
    save_json(os.path.join(run_dir, "random_split_results.json"), results)
    _plot_fold_history_curves(run_dir, 1)
    _plot_random_split_history_curves(run_dir)
    _plot_random_split_summary(run_dir, results)
    logger.info(
        "Random split summary | "
        f"Train AUC={train_metrics.get('AUC', 0.0):.4f}, Train AUC 95% CI={_fmt_ci(train_auc_ci)}, "
        f"Train PR-AUC={train_metrics.get('PR-AUC', 0.0):.4f}, "
        f"Internal Val AUC={validation_metrics.get('AUC', 0.0):.4f}, "
        f"Internal Val AUC 95% CI={_fmt_ci(validation_auc_ci)}, "
        f"Internal Val PR-AUC={validation_metrics.get('PR-AUC', 0.0):.4f}, "
        f"Internal Val F1={validation_metrics.get('F1', 0.0):.4f}, Internal Val ACC={validation_metrics.get('ACC', 0.0):.4f}, "
        f"Internal Val NPV={validation_metrics.get('NPV', 0.0):.4f}, Internal Val Kappa={validation_metrics.get('Kappa', 0.0):.4f}, "
        f"Internal Val Precision={validation_metrics.get('Precision', 0.0):.4f}, Internal Val Recall={validation_metrics.get('Recall', 0.0):.4f}, "
        f"threshold={validation_metrics.get('Thresh', 0.0):.4f} | not an independent holdout estimate"
    )
    _print_best_random_split_result(run_dir, results)


if __name__ == "__main__":
    run_random_split()
