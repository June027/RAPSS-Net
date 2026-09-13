import os
import json
import sys

_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    _cwd = os.getcwd()
    sys.path[:] = [
        entry
        for entry in sys.path
        if os.path.abspath(entry or _cwd) != _PROJECT_DIR
    ]

import random
import argparse
import numpy as np


def _ensure_numpy_matrix_alias():
    if not hasattr(np, "matrix"):
        matrix_cls = None
        try:
            from numpy.matrixlib.defmatrix import matrix as matrix_cls
        except Exception:
            try:
                matrix_cls = np.matrixlib.matrix
            except Exception:
                class _NumpyMatrixCompat:
                    pass

                matrix_cls = _NumpyMatrixCompat
        np.matrix = matrix_cls
    if not hasattr(np, "mat"):
        np.mat = np.matrix


_ensure_numpy_matrix_alias()

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    auc,
    confusion_matrix,
    brier_score_loss,
)

if __name__ == "__main__":
    sys.path.insert(0, _PROJECT_DIR)

from utils.config_manager import ConfigManager
from utils.logger import get_logger
from utils.experiment import set_deterministic_environment
from utils.metadata import load_metadata_records
from utils.metrics import (
    binary_metrics_from_probs,
    bootstrap_auc_ci,
    bootstrap_metric_cis,
)
from datasets.mri_dataset import MRIDataset, safe_collate
from models.builder import build_model
from engine.evaluator import ClinicalEvaluator

_ensure_numpy_matrix_alias()


AXIS_NAME_TO_DIM = {"D": 2, "H": 3, "W": 4}


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _normalize_clinical_stats(clinical_stats, clinical_keys):
    normalized = {}
    for key in clinical_keys:
        stats = clinical_stats.get(key) if isinstance(clinical_stats, dict) else None
        if not isinstance(stats, dict):
            raise ValueError(f"Missing data.clinical_stats for clinical key: {key}")
        try:
            mean = float(stats["mean"])
            std = float(stats["std"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid data.clinical_stats entry for clinical key: {key}"
            ) from exc
        normalized[key] = {"mean": mean, "std": std if std > 1.0e-6 else 1.0}
    return normalized


def resolve_eval_clinical_stats(config, clinical_keys, checkpoints):
    if not clinical_keys or not bool(config.get("model", {}).get("use_clinical", False)):
        return {}

    configured_stats = config.get("data", {}).get("clinical_stats")
    if configured_stats:
        return _normalize_clinical_stats(configured_stats, clinical_keys)

    if not checkpoints:
        raise ValueError(
            "Clinical evaluation requires data.clinical_stats in the eval config "
            "or checkpoints that include training clinical_stats."
        )

    checkpoint_stats = []
    for checkpoint in checkpoints:
        stats = checkpoint.get("clinical_stats") if isinstance(checkpoint, dict) else None
        if stats:
            checkpoint_stats.append(_normalize_clinical_stats(stats, clinical_keys))
    if len(checkpoint_stats) != len(checkpoints):
        raise ValueError(
            "Clinical evaluation requires training clinical_stats for every checkpoint. "
            "Use checkpoints produced after this fix or provide data.clinical_stats in the eval config."
        )

    unique_stats = {
        json.dumps(stats, sort_keys=True): stats for stats in checkpoint_stats
    }
    if len(unique_stats) > 1:
        raise ValueError(
            "Clinical CV ensemble evaluation has fold-specific clinical_stats. "
            "Evaluate each fold with its own clinical preprocessing or provide a single fixed "
            "data.clinical_stats contract in the eval config."
        )
    return next(iter(unique_stats.values()))


def build_loader(config, data_list, clinical_stats=None):
    clinical_keys = [str(key) for key in config.get("data", {}).get("clinical_keys", [])]
    if clinical_stats is None:
        clinical_stats = resolve_eval_clinical_stats(config, clinical_keys, [])
    dataset = MRIDataset(
        data_list,
        config.get("paths", {}).get("data_dir", "data/processed"),
        model_config=config.get("model", {}),
        is_train=False,
        aug_cfg=config.get("data", {}).get("augmentation", {}),
        use_mask=True,
        mask_drop_prob=0.0,
        strict_file_check=True,
        cache_in_memory=False,
        split_name="test",
        clinical_keys=clinical_keys,
        clinical_stats=clinical_stats,
        clinical_missing_indicator=config.get("data", {}).get(
            "clinical_missing_indicator", True
        ),
    )
    gen = torch.Generator()
    gen.manual_seed(int(config.get("project", {}).get("seed", 42)))
    num_workers = config.get("train", {}).get(
        "num_workers", config.get("data", {}).get("num_workers", 4)
    )
    return DataLoader(
        dataset,
        batch_size=config["train"]["batch_size"],
        shuffle=False,
        collate_fn=safe_collate,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=gen,
    )


def load_checkpoint(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except Exception:
        return torch.load(path, map_location=device, weights_only=False)


def resolve_tta_views(config):
    test_cfg = config.get("test", {})
    if not bool(test_cfg.get("tta_enabled", False)):
        return [()]
    raw_views = test_cfg.get("tta_flip_axes", [])
    views = [()]
    for raw_view in raw_views:
        if isinstance(raw_view, str):
            raw_view = [raw_view]
        dims = []
        for axis_name in raw_view:
            norm_name = str(axis_name).upper()
            if norm_name not in AXIS_NAME_TO_DIM:
                raise ValueError(
                    f'Unsupported TTA axis "{axis_name}". Expected one of D/H/W.'
                )
            dims.append(AXIS_NAME_TO_DIM[norm_name])
        dims_tuple = tuple(sorted(set(dims)))
        if dims_tuple not in views:
            views.append(dims_tuple)
    return views


def apply_tta_view(images, flip_dims):
    if not flip_dims:
        return images
    return torch.flip(images, dims=list(flip_dims))


def apply_tta_volume(volume, flip_dims):
    if volume is None or not flip_dims:
        return volume
    return torch.flip(volume, dims=[dim - 1 for dim in flip_dims])


def build_center_metrics(all_ids, y_true, y_prob, threshold, center_lookup):
    center_rows = {}
    for sample_id, label, prob in zip(all_ids, y_true, y_prob):
        center_name = center_lookup.get(sample_id, "unknown")
        center_rows.setdefault(center_name, {"labels": [], "probs": []})
        center_rows[center_name]["labels"].append(int(label))
        center_rows[center_name]["probs"].append(float(prob))
    center_metrics = {}
    for center_name, rows in center_rows.items():
        metrics = binary_metrics_from_probs(rows["labels"], rows["probs"], threshold)
        metrics["num_samples"] = int(len(rows["labels"]))
        metrics["positive_cases"] = int(sum(rows["labels"]))
        center_metrics[center_name] = metrics
    return center_metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluation for RAPSS-Net/TriDual3D")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="configs/server/random/train_config_RAPSS_Net.yaml",
    )
    parser.add_argument(
        "--test_csv", type=str, required=True, help="Metadata CSV for external evaluation."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        action="append",
        required=True,
        help="Checkpoint path. Pass multiple times to ensemble CV folds.",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5, help="Fixed classification threshold."
    )
    parser.add_argument(
        "--output_json", type=str, default=None, help="Optional path for metrics JSON output."
    )
    args = parser.parse_args()
    config_mgr = ConfigManager(args.config)
    config = config_mgr.config

    seed = int(config.get("project", {}).get("seed", 42))
    set_deterministic_environment(seed)

    output_dir = (
        os.path.dirname(args.output_json)
        if args.output_json
        else config.get("paths", {}).get("log_dir", None)
    )
    logger = get_logger(
        "Test",
        output_dir or config.get("paths", {}).get("log_dir", None),
        filename="test.log",
    )
    logger.info(
        f"Project seed globally locked to {seed} with strictly deterministic execution."
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Launching evaluation mode...")
    if not os.path.exists(args.test_csv):
        raise FileNotFoundError(f"Test CSV not found: {args.test_csv}")
    data_list = load_metadata_records(
        args.test_csv,
        include_ihc=True,
        require_unique_ids=False,
        clinical_keys=config.get("data", {}).get("clinical_keys", []),
    )
    center_lookup = {str(item["id"]): str(item["center"]) for item in data_list}
    checkpoint_payloads = []
    for model_path in args.model_path:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
        checkpoint_payloads.append(load_checkpoint(model_path, device))
    clinical_keys = [str(key) for key in config.get("data", {}).get("clinical_keys", [])]
    clinical_stats = resolve_eval_clinical_stats(
        config, clinical_keys, checkpoint_payloads
    )
    loader = build_loader(config, data_list, clinical_stats=clinical_stats)
    tta_views = resolve_tta_views(config)
    models = []
    for model_path, checkpoint in zip(args.model_path, checkpoint_payloads):
        model = build_model(config, device)
        state_dict_key = (
            "ema_model_state_dict"
            if "ema_model_state_dict" in checkpoint
            else "model_state_dict"
        )
        model.load_state_dict(checkpoint[state_dict_key])
        model.eval()
        models.append(model)
    logger.info(f"Loaded {len(models)} model(s) for evaluation.")
    logger.info(f"Using {len(tta_views)} evaluation view(s).")
    all_ids = []
    all_labels = []
    all_probs = []
    all_feature_rows = []
    amp_dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float16
    )
    with torch.no_grad():
        for batch in loader:
            sample_ids = batch["id"]
            images = batch["image"]
            kinetics = batch.get("kinetics")
            labels = batch["label"]
            ihc_targets = batch.get("ihc")
            has_ihc = batch.get("has_ihc")
            if images.nelement() == 0:
                raise RuntimeError(
                    f"Empty test batch encountered for IDs: {sample_ids}"
                )
            images = images.to(device)
            masks = batch["mask"].to(device)
            kinetics = (
                kinetics.to(device)
                if kinetics is not None and kinetics.nelement() > 0
                else None
            )
            clinical = batch.get("clinical")
            clinical = (
                clinical.to(device)
                if clinical is not None and clinical.nelement() > 0
                else None
            )
            probs_ensemble = []
            global_ensemble = []
            core_ensemble = []
            peri_ensemble = []
            with torch.cuda.amp.autocast(
                enabled=config["train"].get("amp", True), dtype=amp_dtype
            ):
                for model in models:
                    probs_tta = []
                    global_tta = []
                    core_tta = []
                    peri_tta = []
                    for flip_dims in tta_views:
                        aug_images = apply_tta_view(images, flip_dims)
                        aug_masks = apply_tta_view(masks, flip_dims)
                        aug_kinetics = apply_tta_volume(kinetics, flip_dims)
                        outputs = model(
                            aug_images,
                            kinetics=aug_kinetics,
                            lesion_mask=aug_masks,
                            clinical=clinical,
                        )
                        logits = outputs[0] if isinstance(outputs, tuple) else outputs
                        probs_tta.append(F.softmax(logits, dim=1)[:, 1])
                        if isinstance(outputs, tuple) and len(outputs) >= 5:
                            global_tta.append(outputs[4])
                        if isinstance(outputs, tuple) and len(outputs) >= 4:
                            core_tta.append(outputs[2])
                            peri_tta.append(outputs[3])
                    probs_ensemble.append(torch.stack(probs_tta, dim=0).mean(dim=0))
                    if global_tta:
                        global_ensemble.append(torch.stack(global_tta, dim=0).mean(dim=0))
                    if core_tta:
                        core_ensemble.append(torch.stack(core_tta, dim=0).mean(dim=0))
                    if peri_tta:
                        peri_ensemble.append(torch.stack(peri_tta, dim=0).mean(dim=0))
            probs = torch.stack(probs_ensemble, dim=0).mean(dim=0)
            all_ids.extend(sample_ids)
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
            if global_ensemble or core_ensemble or peri_ensemble:
                global_feats = (
                    torch.stack(global_ensemble, dim=0).mean(dim=0).float().cpu().numpy()
                    if global_ensemble
                    else None
                )
                core_feats = (
                    torch.stack(core_ensemble, dim=0).mean(dim=0).float().cpu().numpy()
                    if core_ensemble
                    else None
                )
                peri_feats = (
                    torch.stack(peri_ensemble, dim=0).mean(dim=0).float().cpu().numpy()
                    if peri_ensemble
                    else None
                )
                ihc_np = (
                    ihc_targets.numpy()
                    if ihc_targets is not None and ihc_targets.nelement() > 0
                    else None
                )
                has_ihc_np = (
                    has_ihc.numpy()
                    if has_ihc is not None and has_ihc.nelement() > 0
                    else None
                )
                for idx, sample_id in enumerate(sample_ids):
                    row = {
                        "id": sample_id,
                        "label": int(labels[idx]),
                        "prob": float(probs[idx].cpu().item()),
                    }
                    if has_ihc_np is not None:
                        row["has_ihc"] = float(has_ihc_np[idx])
                    if ihc_np is not None:
                        row["ihc_cd8"] = float(ihc_np[idx, 0])
                        row["ihc_cd34"] = float(ihc_np[idx, 1])
                        row["ihc_ki67"] = float(ihc_np[idx, 2])
                        row["ihc_cd163"] = float(ihc_np[idx, 3])
                    if global_feats is not None:
                        for feat_idx, feat_val in enumerate(global_feats[idx]):
                            row[f"global_feat_{feat_idx:03d}"] = float(feat_val)
                    if core_feats is not None:
                        for feat_idx, feat_val in enumerate(core_feats[idx]):
                            row[f"core_feat_{feat_idx:03d}"] = float(feat_val)
                    if peri_feats is not None:
                        for feat_idx, feat_val in enumerate(peri_feats[idx]):
                            row[f"peri_feat_{feat_idx:03d}"] = float(feat_val)
                    all_feature_rows.append(row)
    if not all_labels:
        raise RuntimeError("No valid test samples were collected during evaluation.")
    y_true = np.array(all_labels)
    y_probs = np.array(all_probs)
    y_pred = (y_probs >= args.threshold).astype(int)
    try:
        auc_val = roc_auc_score(y_true, y_probs)
    except ValueError:
        auc_val = 0.5
    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    pr_auc = auc(recall, precision)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn + 1e-8)
    spec = tn / (tn + fp + 1e-8)
    acc = (tp + tn) / len(y_true)
    auc_ci = bootstrap_auc_ci(
        y_true,
        y_probs,
        seed=int(config.get("project", {}).get("seed", 42)),
        n_boot=int(config.get("test", {}).get("bootstrap_iterations", 1000)),
    )
    metric_cis = bootstrap_metric_cis(
        y_true,
        y_probs,
        args.threshold,
        seed=int(config.get("project", {}).get("seed", 42)),
        n_boot=int(config.get("test", {}).get("bootstrap_iterations", 1000)),
    )
    ppv = tp / (tp + fp + 1e-8)
    npv = tn / (tn + fn + 1e-8)
    brier = brier_score_loss(y_true, y_probs)
    center_metrics = build_center_metrics(
        all_ids, y_true, y_probs, args.threshold, center_lookup
    )
    logger.info("\n" + "=" * 40)
    logger.info(f"Test report ({len(y_true)} cases)")
    logger.info("=" * 40)
    logger.info(f"AUC         : {auc_val:.4f} | 95% CI {auc_ci}")
    logger.info(f"PR-AUC      : {pr_auc:.4f}")
    logger.info(f"Accuracy    : {acc:.4f}")
    logger.info(f"Sensitivity : {sens:.4f}")
    logger.info(f"Specificity : {spec:.4f}")
    logger.info(f"PPV         : {ppv:.4f}")
    logger.info(f"NPV         : {npv:.4f}")
    logger.info(f"Brier       : {brier:.4f}")
    logger.info(f"Threshold   : {args.threshold:.4f}")
    logger.info("=" * 40)
    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
        evaluator = ClinicalEvaluator(
            y_true=y_true,
            y_probs=y_probs,
            save_dir=os.path.dirname(args.output_json),
            seed=int(config.get("project", {}).get("seed", 42)),
            fixed_threshold=float(args.threshold),
            allow_posthoc=False,
        )
        clinical_report = evaluator.generate_report()
        payload = {
            "num_samples": int(len(y_true)),
            "auc": float(auc_val),
            "auc_95ci": auc_ci,
            "pr_auc": float(pr_auc),
            "accuracy": float(acc),
            "sensitivity": float(sens),
            "specificity": float(spec),
            "ppv": float(ppv),
            "npv": float(npv),
            "brier_score": float(brier),
            "threshold": float(args.threshold),
            "model_paths": args.model_path,
            "ensemble_size": int(len(args.model_path)),
            "tta_views": int(len(tta_views)),
            "test_csv": args.test_csv,
            "metric_95ci": metric_cis,
            "per_center_metrics": center_metrics,
            "clinical_report": clinical_report,
        }
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        pd.DataFrame(
            {"id": all_ids, "label": y_true, "prob": y_probs, "pred": y_pred}
        ).to_csv(
            os.path.join(os.path.dirname(args.output_json), "predictions.csv"),
            index=False,
        )
        if all_feature_rows:
            pd.DataFrame(all_feature_rows).to_csv(
                os.path.join(os.path.dirname(args.output_json), "feature_vectors.csv"),
                index=False,
            )
        logger.info(f"Metrics saved to: {args.output_json}")


if __name__ == "__main__":
    main()
