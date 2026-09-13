import argparse
import csv
import json
import math
import os
import shutil
import sys
from copy import deepcopy

import pandas as pd
import yaml
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit

from utils.config_manager import ConfigManager
from utils.metadata import (
    OPTIONAL_PATIENT_GROUP_ALIASES,
    resolve_required_metadata_columns,
)
from utils.process_runner import run_logged_command


def _resolve_internal_centers(config, cli_centers):
    if cli_centers:
        return [str(center) for center in cli_centers]
    train_centers = config.get("train", {}).get("internal_centers", None)
    if train_centers:
        return [str(center) for center in train_centers]
    return []


def _resolve_internal_test_size(config, cli_test_size):
    if cli_test_size is not None:
        return float(cli_test_size)
    return float(config.get("train", {}).get("internal_test_size", 0.2))


def _read_filtered_metadata_rows(metadata_path, internal_centers):
    with open(metadata_path, "r", encoding="utf-8") as fin:
        reader = csv.DictReader(fin)
        if not reader.fieldnames:
            raise ValueError("Metadata CSV is missing a header row.")
        fieldnames = list(reader.fieldnames)
        resolved_columns = resolve_required_metadata_columns(fieldnames)
        center_column = resolved_columns["center"]
        rows = [
            dict(row)
            for row in reader
            if not internal_centers or row.get(center_column) in internal_centers
        ]
    return rows, fieldnames, resolved_columns


def _resolve_split_targets(rows, label_column, center_column, test_size):
    label_targets = [str(row[label_column]) for row in rows]
    joint_targets = [f"{row[label_column]}__{row[center_column]}" for row in rows]
    total_count = len(rows)
    test_count = max(1, int(math.ceil(total_count * float(test_size))))
    train_count = total_count - test_count
    for target_name, targets in (("joint label-center", joint_targets), ("label", label_targets)):
        unique_targets = sorted(set(targets))
        min_count = min(targets.count(value) for value in unique_targets)
        if min_count < 2:
            continue
        if test_count < len(unique_targets) or train_count < len(unique_targets):
            continue
        return targets, target_name
    return None, "random"


def _resolve_optional_metadata_column(fieldnames, aliases):
    return next((column for column in aliases if column in fieldnames), None)


def _row_patient_group(row, id_column, patient_group_column=None):
    if patient_group_column:
        raw_group = row.get(patient_group_column)
        if raw_group is not None and str(raw_group).strip() != "":
            return str(raw_group).strip()
    return str(row[id_column]).strip()


def _build_patient_group_records(
    rows,
    id_column,
    label_column,
    center_column,
    patient_group_column=None,
):
    grouped = {}
    order = []
    for row in rows:
        group_id = _row_patient_group(row, id_column, patient_group_column)
        if group_id not in grouped:
            grouped[group_id] = {
                "group": group_id,
                "rows": [],
                "labels": set(),
                "centers": set(),
            }
            order.append(group_id)
        grouped[group_id]["rows"].append(row)
        grouped[group_id]["labels"].add(str(row[label_column]))
        grouped[group_id]["centers"].add(str(row[center_column]))

    group_records = []
    for group_id in order:
        record = grouped[group_id]
        if len(record["labels"]) != 1:
            raise ValueError(
                "Patient-level split is impossible because patient group "
                f"{group_id!r} has conflicting labels: {sorted(record['labels'])}"
            )
        label = next(iter(record["labels"]))
        center = (
            next(iter(record["centers"]))
            if len(record["centers"]) == 1
            else "mixed-center"
        )
        group_records.append(
            {
                "group": group_id,
                "rows": record["rows"],
                "label": label,
                "center": center,
            }
        )
    return group_records


def _resolve_group_split_targets(group_records, test_size):
    label_targets = [str(group["label"]) for group in group_records]
    joint_targets = [
        f"{group['label']}__{group['center']}" for group in group_records
    ]
    total_count = len(group_records)
    test_count = max(1, int(math.ceil(total_count * float(test_size))))
    train_count = total_count - test_count
    for target_name, targets in (
        ("joint label-center patient-group", joint_targets),
        ("label patient-group", label_targets),
    ):
        unique_targets = sorted(set(targets))
        min_count = min(targets.count(value) for value in unique_targets)
        if min_count < 2:
            continue
        if test_count < len(unique_targets) or train_count < len(unique_targets):
            continue
        return targets, target_name
    return None, "random patient-group"


def _split_rows_patient_grouped(
    rows,
    fieldnames,
    resolved_columns,
    test_size,
    seed,
):
    patient_group_column = _resolve_optional_metadata_column(
        fieldnames, OPTIONAL_PATIENT_GROUP_ALIASES
    )
    group_records = _build_patient_group_records(
        rows,
        resolved_columns["id"],
        resolved_columns["label"],
        resolved_columns["center"],
        patient_group_column=patient_group_column,
    )
    if len(group_records) < 2:
        raise ValueError("At least two patient groups are required for an internal split.")

    stratify_targets, stratify_mode = _resolve_group_split_targets(
        group_records, test_size
    )
    group_indices = list(range(len(group_records)))
    if stratify_targets is None:
        splitter = ShuffleSplit(
            n_splits=1, test_size=float(test_size), random_state=int(seed)
        )
        train_group_idx, test_group_idx = next(
            splitter.split(group_indices)
        )
    else:
        splitter = StratifiedShuffleSplit(
            n_splits=1, test_size=float(test_size), random_state=int(seed)
        )
        train_group_idx, test_group_idx = next(
            splitter.split(group_indices, stratify_targets)
        )

    def collect_rows(indices):
        selected = []
        groups = []
        for idx in indices:
            group = group_records[int(idx)]
            selected.extend(group["rows"])
            groups.append(group["group"])
        return selected, groups

    train_rows, train_groups = collect_rows(train_group_idx)
    test_rows, test_groups = collect_rows(test_group_idx)
    group_summary = {
        "patient_group_column": patient_group_column or resolved_columns["id"],
        "num_patient_groups": int(len(group_records)),
        "train_patient_groups": int(len(set(train_groups))),
        "internal_test_patient_groups": int(len(set(test_groups))),
        "patient_group_overlap": int(len(set(train_groups) & set(test_groups))),
    }
    return train_rows, test_rows, stratify_mode, group_summary


def _write_metadata_csv(path, fieldnames, rows):
    with open(path, "w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _count_by(rows, column_name):
    counts = {}
    for row in rows:
        key = str(row.get(column_name, ""))
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _parse_float(value):
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def _compute_clinical_stats(rows, clinical_keys):
    stats = {}
    for key in clinical_keys:
        values = [
            parsed
            for parsed in (_parse_float(row.get(key)) for row in rows)
            if parsed is not None
        ]
        if not values:
            stats[key] = {"mean": 0.0, "std": 1.0}
            continue
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        std = math.sqrt(variance)
        stats[key] = {"mean": float(mean), "std": float(std if std > 1.0e-6 else 1.0)}
    return stats


def _export_internal_split_artifacts(
    run_output_dir,
    train_rows,
    test_rows,
    fieldnames,
    resolved_columns,
    seed,
    test_size,
    stratify_mode,
    group_summary=None,
):
    split_dir = os.path.join(run_output_dir, "internal_split")
    os.makedirs(split_dir, exist_ok=True)

    train_df = pd.DataFrame(train_rows, columns=fieldnames)
    test_df = pd.DataFrame(test_rows, columns=fieldnames)
    summary_rows = [
        {"metric": "seed", "value": int(seed)},
        {"metric": "internal_test_size", "value": float(test_size)},
        {"metric": "stratify_mode", "value": stratify_mode},
        {"metric": "train_pool_size", "value": int(len(train_rows))},
        {"metric": "internal_test_count", "value": int(len(test_rows))},
    ]
    for key, value in (group_summary or {}).items():
        summary_rows.append({"metric": key, "value": value})
    center_rows = []
    for split_name, rows in (("train_pool", train_rows), ("internal_test", test_rows)):
        for center_name, count in _count_by(rows, resolved_columns["center"]).items():
            center_rows.append(
                {"split": split_name, "center": center_name, "count": int(count)}
            )
    label_rows = []
    for split_name, rows in (("train_pool", train_rows), ("internal_test", test_rows)):
        for label_name, count in _count_by(rows, resolved_columns["label"]).items():
            label_rows.append(
                {"split": split_name, "label": label_name, "count": int(count)}
            )

    workbook_path = os.path.join(split_dir, "internal_split.xlsx")
    with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
        pd.DataFrame(summary_rows).to_excel(writer, sheet_name="summary", index=False)
        pd.DataFrame(center_rows).to_excel(writer, sheet_name="center_stats", index=False)
        pd.DataFrame(label_rows).to_excel(writer, sheet_name="label_stats", index=False)
        train_df.to_excel(writer, sheet_name="train_pool", index=False)
        test_df.to_excel(writer, sheet_name="internal_test", index=False)

    train_df.to_excel(os.path.join(split_dir, "train_pool.xlsx"), index=False)
    test_df.to_excel(os.path.join(split_dir, "internal_test.xlsx"), index=False)
    train_df.to_csv(os.path.join(split_dir, "train_pool.csv"), index=False)
    test_df.to_csv(os.path.join(split_dir, "internal_test.csv"), index=False)
    return split_dir


def _resolve_cv_model_paths(run_output_dir, cv_results_path):
    with open(cv_results_path, "r", encoding="utf-8") as f:
        cv_info = json.load(f)
    model_paths = []
    for fold in cv_info.get("folds", []):
        fold_id = fold.get("fold")
        if fold_id is None:
            continue
        model_path = os.path.join(
            run_output_dir,
            f"fold_{fold_id}",
            f"best_model_fold_{fold_id}.pth",
        )
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Expected fold model not found: {model_path}")
        model_paths.append(model_path)
    if not model_paths:
        raise FileNotFoundError(
            f"No fold models were listed in CV results: {cv_results_path}"
        )
    return model_paths


def _load_json_if_exists(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _fmt_metric(value):
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "n/a"


def _print_best_training_result(run_output_dir, internal_test_result=None):
    cv_results = _load_json_if_exists(os.path.join(run_output_dir, "cv_results.json")) or {}
    threshold_info = _load_json_if_exists(os.path.join(run_output_dir, "best_threshold.json")) or {}
    internal_test = None
    if internal_test_result is not None:
        internal_test = _load_json_if_exists(internal_test_result["metrics_path"])

    folds = cv_results.get("folds", []) or []
    best_fold = None
    if folds:
        best_fold = max(
            folds,
            key=lambda row: (
                float(row.get("best_auc", 0.0)),
                -int(row.get("fold", 0)),
            ),
        )
    mean_best_auc = cv_results.get(
        "mean_best_auc",
        cv_results.get("best_auc", threshold_info.get("mean_best_auc")),
    )
    std_best_auc = cv_results.get("std_best_auc", threshold_info.get("std_best_auc"))
    pooled_oof_auc = cv_results.get(
        "pooled_oof_auc", threshold_info.get("pooled_oof_auc")
    )
    fixed_threshold = threshold_info.get(
        "best_threshold", cv_results.get("pooled_oof_threshold")
    )

    print("\n" + "=" * 80)
    print("BEST TRAINING RESULT")
    print("=" * 80)
    print(f"Run directory: {run_output_dir}")
    print(
        "CV best validation AUC: "
        f"{_fmt_metric(mean_best_auc)} +/- {_fmt_metric(std_best_auc)} "
        "(mean of per-fold best InternalVal_AUC)"
    )
    if best_fold is not None:
        print(
            "Best fold: "
            f"fold {best_fold.get('fold')} | "
            f"AUC={_fmt_metric(best_fold.get('best_auc'))} | "
            f"threshold={_fmt_metric(best_fold.get('best_threshold'))}"
        )
    if folds:
        fold_text = ", ".join(
            f"fold {row.get('fold')}={_fmt_metric(row.get('best_auc'))}"
            for row in sorted(folds, key=lambda item: int(item.get("fold", 0)))
        )
        print(f"Per-fold best AUC: {fold_text}")
    print(f"Fixed evaluation threshold: {_fmt_metric(fixed_threshold)}")
    print(f"Pooled OOF AUC (auxiliary): {_fmt_metric(pooled_oof_auc)}")
    if internal_test is not None:
        print(
            "Internal test: "
            f"AUC={_fmt_metric(internal_test.get('auc'))}, "
            f"PR-AUC={_fmt_metric(internal_test.get('pr_auc'))}, "
            f"ACC={_fmt_metric(internal_test.get('accuracy'))}, "
            f"Sens={_fmt_metric(internal_test.get('sensitivity'))}, "
            f"Spec={_fmt_metric(internal_test.get('specificity'))}, "
            f"N={internal_test.get('num_samples', 'n/a')}"
        )
    print("=" * 80)


def _evaluate_internal_test_set(
    project_dir,
    run_output_dir,
    base_config,
    test_metadata_path,
    test_rows,
):
    if not test_rows:
        return None

    threshold_path = os.path.join(run_output_dir, "best_threshold.json")
    cv_results_path = os.path.join(run_output_dir, "cv_results.json")
    if not os.path.exists(threshold_path):
        raise FileNotFoundError(f"Missing pooled CV threshold: {threshold_path}")
    if not os.path.exists(cv_results_path):
        raise FileNotFoundError(f"Missing CV results: {cv_results_path}")

    with open(threshold_path, "r", encoding="utf-8") as f:
        threshold_info = json.load(f)
    threshold = float(threshold_info.get("best_threshold", 0.5))
    model_paths = _resolve_cv_model_paths(run_output_dir, cv_results_path)

    output_dir = os.path.join(run_output_dir, "InternalTest")
    os.makedirs(output_dir, exist_ok=True)
    archived_metadata_path = os.path.join(output_dir, "metadata_internal_test.csv")
    shutil.copyfile(test_metadata_path, archived_metadata_path)

    eval_config = deepcopy(base_config)
    eval_config.setdefault("paths", {})
    eval_config["paths"]["metadata_path"] = archived_metadata_path
    eval_config["paths"]["log_dir"] = output_dir
    eval_config.setdefault("data", {})["cache_in_memory"] = False
    eval_config_path = os.path.join(output_dir, "internal_test_eval_config.yaml")
    with open(eval_config_path, "w", encoding="utf-8") as f:
        yaml.dump(eval_config, f, sort_keys=False)

    cmd_test = [
        sys.executable,
        "test.py",
        "-c",
        eval_config_path,
        "--test_csv",
        archived_metadata_path,
        "--threshold",
        str(threshold),
        "--output_json",
        os.path.join(output_dir, "metrics.json"),
    ]
    for model_path in model_paths:
        cmd_test.extend(["--model_path", model_path])

    manifest = {
        "mode": "cv_ensemble_internal_test",
        "metadata_path": archived_metadata_path,
        "config_path": eval_config_path,
        "model_paths": model_paths,
        "threshold": threshold,
        "threshold_path": threshold_path,
        "cv_results_path": cv_results_path,
        "command": cmd_test,
    }
    manifest_path = os.path.join(output_dir, "internal_test_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    stdout_log = os.path.join(output_dir, "test_console.log")
    stderr_log = os.path.join(output_dir, "test_error.log")
    print("\n" + "=" * 25 + " Starting Internal Test Evaluation " + "=" * 25)
    print(f"Running command: {' '.join(cmd_test)}")
    return_code = run_logged_command(cmd_test, project_dir, stdout_log, stderr_log)
    if return_code != 0:
        raise RuntimeError(
            f"Internal test evaluation failed with exit code {return_code}. "
            f"Check logs in {output_dir}"
        )
    print("Internal test evaluation completed successfully.")
    print(f"Internal test reports are saved in: {output_dir}")
    return {
        "output_dir": output_dir,
        "metrics_path": os.path.join(output_dir, "metrics.json"),
        "manifest_path": manifest_path,
    }


def run_internal_cv():
    parser = argparse.ArgumentParser(
        description="Step 1: split internal data into train/test, then run 5-fold CV on the train pool."
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="configs/server/cv/train_config_RAPSS_Net.yaml",
        help="Path to the base configuration file.",
    )
    parser.add_argument(
        "--internal_centers",
        nargs="+",
        default=None,
        help="Centers to pool for 5-fold internal cross-validation.",
    )
    parser.add_argument(
        "--internal_test_size",
        type=float,
        default=None,
        help="Random hold-out ratio for the internal test set. Set to 0 to skip hold-out and run 5-fold CV on the full internal pool.",
    )
    args = parser.parse_args()

    project_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = (
        args.config
        if os.path.isabs(args.config)
        else os.path.join(project_dir, args.config)
    )
    config = ConfigManager(config_path).config
    base_log_dir = config["paths"].get("log_dir", "logs")
    os.makedirs(base_log_dir, exist_ok=True)

    metadata_path = config["paths"].get("metadata_path", "")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Base metadata file not found: {metadata_path}")

    internal_centers = _resolve_internal_centers(config, args.internal_centers)
    internal_test_size = _resolve_internal_test_size(config, args.internal_test_size)
    if not 0.0 <= internal_test_size < 1.0:
        raise ValueError("internal_test_size must be in [0, 1).")
    target_seed = config["project"].get("seed", 2026)

    temp_dir = os.path.join(base_log_dir, "_temp_files")
    os.makedirs(temp_dir, exist_ok=True)
    filtered_rows, fieldnames, resolved_columns = _read_filtered_metadata_rows(
        metadata_path, internal_centers
    )
    record_count = len(filtered_rows)
    if internal_test_size > 0:
        expected_test_count = max(1, int(math.ceil(record_count * internal_test_size)))
        expected_train_count = record_count - expected_test_count
    else:
        expected_test_count = 0
        expected_train_count = record_count

    if not record_count:
        filter_hint = (
            f", centers={internal_centers}" if internal_centers else " (no center filter)"
        )
        raise ValueError(
            f"No data found in metadata{filter_hint}. Check your dataset metadata."
        )
    if expected_train_count < 1 or (internal_test_size > 0 and expected_test_count < 1):
        raise ValueError(
            "internal_test_size leads to an empty train or internal test split. Adjust the ratio or add more samples."
        )

    center_label = "+".join(internal_centers) if internal_centers else "All"
    if internal_test_size > 0:
        experiment_name = f"InternalCV_({center_label})_seed_{target_seed}"
    else:
        experiment_name = f"InternalCVFull_({center_label})_seed_{target_seed}"
    run_output_dir = os.path.join(base_log_dir, experiment_name)
    os.makedirs(run_output_dir, exist_ok=True)

    if internal_test_size > 0:
        train_rows, test_rows, stratify_mode, group_summary = _split_rows_patient_grouped(
            filtered_rows,
            fieldnames,
            resolved_columns,
            internal_test_size,
            target_seed,
        )
    else:
        stratify_mode = "none (full internal CV)"
        train_rows = list(filtered_rows)
        test_rows = []
        patient_group_column = _resolve_optional_metadata_column(
            fieldnames, OPTIONAL_PATIENT_GROUP_ALIASES
        )
        group_values = {
            _row_patient_group(row, resolved_columns["id"], patient_group_column)
            for row in train_rows
        }
        group_summary = {
            "patient_group_column": patient_group_column or resolved_columns["id"],
            "num_patient_groups": int(len(group_values)),
            "train_patient_groups": int(len(group_values)),
            "internal_test_patient_groups": 0,
            "patient_group_overlap": 0,
        }
    train_rows = sorted(train_rows, key=lambda row: str(row[resolved_columns["id"]]))
    test_rows = sorted(test_rows, key=lambda row: str(row[resolved_columns["id"]]))

    train_metadata_path = os.path.join(temp_dir, "metadata_internal_train_pool.csv")
    test_metadata_path = os.path.join(temp_dir, "metadata_internal_test.csv")
    _write_metadata_csv(train_metadata_path, fieldnames, train_rows)
    _write_metadata_csv(test_metadata_path, fieldnames, test_rows)
    split_dir = _export_internal_split_artifacts(
        run_output_dir,
        train_rows,
        test_rows,
        fieldnames,
        resolved_columns,
        target_seed,
        internal_test_size,
        stratify_mode,
        group_summary,
    )

    if internal_test_size > 0:
        print(
            f"Created internal split for {record_count} samples ({center_label}) "
            f"with seed={target_seed}, test_size={internal_test_size:.2f}, stratify={stratify_mode}."
        )
    else:
        print(
            f"Prepared full internal CV for {record_count} samples ({center_label}) "
            f"with seed={target_seed}, hold-out skipped, stratify={stratify_mode}."
        )
    print(f"Train pool metadata: {train_metadata_path}")
    print(f"Internal test metadata: {test_metadata_path}")
    print(f"Split artifacts: {split_dir}")

    temp_config = deepcopy(config)
    temp_config["paths"]["metadata_path"] = train_metadata_path
    temp_config["paths"]["log_dir"] = run_output_dir
    temp_config.setdefault("paths", {})["internal_test_metadata_path"] = test_metadata_path
    temp_config.setdefault("train", {})["validation_mode"] = "global_cv"
    clinical_keys = [
        str(key) for key in temp_config.get("data", {}).get("clinical_keys", [])
    ]
    if clinical_keys:
        temp_config.setdefault("data", {})["clinical_stats"] = _compute_clinical_stats(
            train_rows, clinical_keys
        )
        temp_config["data"].setdefault("clinical_missing_indicator", True)
    temp_config_path = os.path.join(temp_dir, f"config_{experiment_name}.yaml")
    with open(temp_config_path, "w", encoding="utf-8") as f:
        yaml.dump(temp_config, f, sort_keys=False)

    print(f"Created temporary config for internal CV at: {temp_config_path}")
    print("\n" + "=" * 25 + " Starting Internal 5-Fold CV " + "=" * 25)

    python_exec = sys.executable
    cmd_train = [
        python_exec,
        "train.py",
        "-c",
        temp_config_path,
        "--train_csv",
        train_metadata_path,
        "--seed",
        str(target_seed),
    ]
    train_stdout_log = os.path.join(run_output_dir, "train_console.log")
    train_stderr_log = os.path.join(run_output_dir, "train_error.log")
    try:
        print(f"Running command: {' '.join(cmd_train)}")
        return_code = run_logged_command(
            cmd_train, project_dir, train_stdout_log, train_stderr_log
        )
        if return_code != 0:
            print(
                f"Training failed with exit code {return_code}. Check logs in {run_output_dir}",
                file=sys.stderr,
            )
            sys.exit(1)
        internal_test_result = _evaluate_internal_test_set(
            project_dir,
            run_output_dir,
            temp_config,
            test_metadata_path,
            test_rows,
        )
        print("\n" + "=" * 80)
        print("Internal split + 5-fold CV training completed successfully.")
        print("Artifacts are saved in:")
        print(run_output_dir)
        if internal_test_result is not None:
            print("Internal test metrics:")
            print(internal_test_result["metrics_path"])
        print("=" * 80)
        _print_best_training_result(run_output_dir, internal_test_result)
    except Exception as e:
        print(f"Unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    run_internal_cv()
