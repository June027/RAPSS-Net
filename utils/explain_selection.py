import json
import os


def load_fold_validation_ids(base_dir, fold):
    val_ids_path = os.path.join(base_dir, f"fold_{fold}", "val_ids.txt")
    if not os.path.exists(val_ids_path):
        return None
    with open(val_ids_path, "r", encoding="utf-8") as f:
        ids = {line.strip() for line in f if line.strip()}
    return ids or None


def load_cv_threshold(base_dir, default=0.5):
    threshold_path = os.path.join(base_dir, "best_threshold.json")
    if not os.path.exists(threshold_path):
        return float(default), "default_missing_threshold"
    try:
        with open(threshold_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return float(payload.get("best_threshold", default)), "pooled_oof_threshold"
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return float(default), "default_invalid_threshold"


def select_explain_records(all_data, center, base_dir, fold, strict_fold_match=False):
    center_records = [d for d in all_data if d["center"] == center]
    val_ids = load_fold_validation_ids(base_dir, fold)
    if strict_fold_match and not val_ids:
        raise ValueError(
            f"Missing validation ID file for fold {fold}; cannot safely restrict internal explainability samples."
        )
    if not val_ids:
        return center_records, "center_only"
    fold_records = [d for d in center_records if str(d["id"]) in val_ids]
    if fold_records:
        return fold_records, "fold_validation_subset"
    if strict_fold_match:
        raise ValueError(
            f"Center '{center}' has no overlap with fold {fold} validation IDs; refusing to mix in training samples."
        )
    return center_records, "center_only_no_fold_overlap"
