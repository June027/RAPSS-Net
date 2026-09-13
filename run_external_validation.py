import argparse
import csv
import json
import os
import re
import shutil
import sys
from copy import deepcopy

_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    _cwd = os.getcwd()
    sys.path[:] = [
        entry
        for entry in sys.path
        if os.path.abspath(entry or _cwd) != _PROJECT_DIR
    ]

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
import yaml

_ensure_numpy_matrix_alias()

if __name__ == "__main__":
    sys.path.insert(0, _PROJECT_DIR)

from utils.config_manager import ConfigManager
from utils.metadata import resolve_required_metadata_columns
from utils.process_runner import run_logged_command


LABEL_ALIASES = ("label", "pCR", "PCR", "y")
PRIVATE_LABEL_HINTS = ("G5=1", "pCR", "PCR", "NAT")
ENV_VALUE_RE = re.compile(r"^\$\{(.*?)(?::-(.*?))?\}$")


def _load_raw_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _expand_env_value(value):
    if not isinstance(value, str):
        return value
    match = ENV_VALUE_RE.match(value)
    if not match:
        return value
    env_var, default_val = match.groups()
    return os.getenv(env_var, default_val)


def _as_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _resolve_optional_path(value):
    value = _expand_env_value(value)
    if not value:
        return None
    return os.path.abspath(str(value))


def _coerce_binary_label(value):
    if pd.isna(value):
        raise ValueError("Cannot parse binary label from missing value.")
    text = str(value).strip()
    if not text:
        raise ValueError("Cannot parse binary label from empty value.")
    normalized = text.casefold()
    if normalized in {"0", "0.0", "false", "no", "non-pcr", "non_pcr", "non pcr", "nonpcr"}:
        return 0
    if normalized in {"1", "1.0", "true", "yes", "pcr"}:
        return 1
    try:
        numeric_value = float(text)
    except (TypeError, ValueError):
        numeric_value = None
    if numeric_value in {0.0, 1.0}:
        return int(numeric_value)
    key_value_match = re.search(r"(?:^|[=:\s,;])([01])(?:$|[\s,;)\]])", text)
    if key_value_match:
        return int(key_value_match.group(1))
    raise ValueError(f"Cannot parse binary label from value: {value!r}")


def _normalize_column_name(value):
    return str(value).replace("\ufeff", "").strip()


def _normalize_column_key(value):
    return _normalize_column_name(value).lower()


def _resolve_label_column(columns, requested=None):
    normalized_columns = {
        _normalize_column_key(column): column for column in columns
    }
    requested_label_missing = False
    if requested:
        requested_key = _normalize_column_key(requested)
        if requested_key in normalized_columns:
            return normalized_columns[requested_key]
        if requested_key == "pcr":
            for alias in ("pCR", "PCR", "pcr", "NAT", "label", "y"):
                alias_key = _normalize_column_key(alias)
                if alias_key in normalized_columns:
                    return normalized_columns[alias_key]
        requested_label_missing = True
    for alias in LABEL_ALIASES:
        alias_key = _normalize_column_key(alias)
        if alias_key in normalized_columns:
            return normalized_columns[alias_key]
    for column in columns:
        if any(hint in str(column) for hint in PRIVATE_LABEL_HINTS):
            return column
    available = ", ".join(str(column) for column in columns)
    if requested_label_missing:
        raise ValueError(
            f"Requested label column not found and automatic inference also failed: {requested}. "
            f"Available columns: {available}"
        )
    raise ValueError(
        f"Could not infer label column. Available columns: {available}"
    )


def _standardize_external_metadata(
    source_csv,
    output_csv,
    test_center=None,
    label_column=None,
    filter_center=True,
    default_center="External",
):
    _ensure_numpy_matrix_alias()
    source_ext = os.path.splitext(str(source_csv))[1].lower()
    if source_ext in {".xlsx", ".xls"}:
        df = pd.read_excel(source_csv)
    else:
        df = pd.read_csv(source_csv, encoding="utf-8-sig")
    df.columns = [_normalize_column_name(column) for column in df.columns]
    _ensure_numpy_matrix_alias()
    if "id" not in df.columns:
        raise ValueError(f"External metadata missing required id column: {source_csv}")
    has_center_column = "center" in df.columns
    center_fill_value = str(test_center or default_center or "External")
    if not has_center_column:
        if not center_fill_value:
            raise ValueError(
                "External metadata has no center column. Pass --test_center to fill it."
            )

    resolved_label_column = _resolve_label_column(df.columns, label_column)
    label_source_column = resolved_label_column
    records = df.to_dict(orient="records")
    for record in records:
        if not has_center_column:
            record["center"] = center_fill_value
        record["label"] = _coerce_binary_label(record[label_source_column])
    if filter_center and test_center:
        records = [
            record
            for record in records
            if str(record.get("center", "")).strip() == str(test_center)
        ]
    if not records:
        raise ValueError(f"No external records remain for center={test_center!r}.")

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    fieldnames = list(df.columns)
    if "center" not in fieldnames:
        fieldnames.append("center")
    if "label" not in fieldnames:
        fieldnames.append("label")
    with open(output_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)

    labels = [int(record["label"]) for record in records]
    return {
        "path": output_csv,
        "num_records": int(len(records)),
        "label_column": str(resolved_label_column),
        "positive": int(sum(labels)),
        "negative": int(sum(label == 0 for label in labels)),
    }


def _infer_run_dir_from_model_path(model_path):
    model_dir = os.path.dirname(os.path.abspath(model_path))
    if os.path.basename(model_dir).startswith("fold_"):
        return os.path.dirname(model_dir)
    return model_dir


def _resolve_threshold_from_run(run_dir, fallback=0.5):
    candidates = [
        os.path.join(run_dir, "random_split_results.json"),
        os.path.join(run_dir, "best_threshold.json"),
    ]
    for path in candidates:
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        for key in ("best_threshold", "threshold"):
            if key in payload:
                return float(payload[key])
    return float(fallback)


def _write_external_config(config_path, output_dir, data_dir, metadata_path):
    raw_config = _load_raw_config(config_path)
    external_config = deepcopy(raw_config)
    external_config.setdefault("paths", {})
    external_config["paths"]["data_dir"] = data_dir
    external_config["paths"]["metadata_path"] = metadata_path
    external_config["paths"]["log_dir"] = output_dir
    external_config.setdefault("data", {})
    external_config["data"]["cache_in_memory"] = False
    config_out = os.path.join(output_dir, "external_eval_config.yaml")
    with open(config_out, "w", encoding="utf-8") as f:
        yaml.safe_dump(external_config, f, allow_unicode=True, sort_keys=False)
    return config_out


def _resolve_internal_cv_dir(base_log_dir, target_seed, experiment_name=None):
    if experiment_name:
        candidate = os.path.join(base_log_dir, experiment_name)
        if os.path.isdir(candidate):
            return candidate
        raise FileNotFoundError(
            f"Requested experiment directory not found: '{candidate}'"
        )

    suffix = f"_seed_{target_seed}"
    prefixes = ("InternalCV_", "InternalCVFull_")
    candidates = []
    if os.path.isdir(base_log_dir):
        for name in os.listdir(base_log_dir):
            full_path = os.path.join(base_log_dir, name)
            if (
                os.path.isdir(full_path)
                and name.startswith(prefixes)
                and name.endswith(suffix)
                and os.path.exists(os.path.join(full_path, "best_threshold.json"))
                and os.path.exists(os.path.join(full_path, "cv_results.json"))
            ):
                candidates.append(full_path)
    if not candidates:
        raise FileNotFoundError(
            f"No InternalCV experiment with seed {target_seed} was found in '{base_log_dir}'."
        )
    candidates.sort(key=os.path.getmtime, reverse=True)
    return candidates[0]


def run_external_validation():
    parser = argparse.ArgumentParser(
        description="Evaluate checkpoints on an external center or a direct external CSV."
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="configs/server/random/train_config_RAPSS_Net.yaml",
        help="Path to the base configuration file.",
    )
    parser.add_argument(
        "--test_center",
        type=str,
        default=None,
        help="External center name to evaluate.",
    )
    parser.add_argument(
        "--test_csv",
        type=str,
        default=None,
        help="External metadata CSV/XLSX. When provided with --model_path, runs direct external evaluation.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        action="append",
        default=None,
        help="Checkpoint path for direct external evaluation. Pass multiple times to ensemble.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for direct external evaluation.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="External dataset root containing images/ and masks/. Defaults to parent of --test_csv.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Fixed threshold. Direct mode defaults to random_split_results.json best_threshold.",
    )
    parser.add_argument(
        "--label_column",
        type=str,
        default=None,
        help="Label column for external CSV if it is not named label/pCR/PCR/y.",
    )
    parser.add_argument(
        "--prepare_only",
        action="store_true",
        help="Write standardized metadata/config and print the test command without running it.",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        help="Specific internal CV experiment directory name. Defaults to the latest matching run.",
    )
    args = parser.parse_args()

    project_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = (
        args.config
        if os.path.isabs(args.config)
        else os.path.join(project_dir, args.config)
    )
    config = ConfigManager(config_path).config
    external_cfg = config.get("external", {}) or {}
    external_model_paths = [
        _expand_env_value(path) for path in _as_list(external_cfg.get("model_paths"))
    ]
    direct_external_mode = bool(
        args.model_path
        or args.test_csv
        or external_cfg.get("enabled", False)
        or external_model_paths
    )

    if direct_external_mode:
        model_path_values = args.model_path or external_model_paths
        if not model_path_values:
            print(
                "Error: direct external mode needs --model_path or external.model_paths in the config.",
                file=sys.stderr,
            )
            sys.exit(1)
        test_csv = _resolve_optional_path(
            args.test_csv
            or external_cfg.get("test_csv")
            or config.get("paths", {}).get("metadata_path")
        )
        if not test_csv:
            print(
                "Error: direct external mode needs --test_csv, external.test_csv, or paths.metadata_path.",
                file=sys.stderr,
            )
            sys.exit(1)
        if not os.path.exists(test_csv):
            print(f"Error: external metadata not found: {test_csv}", file=sys.stderr)
            sys.exit(1)
        model_paths = [
            os.path.abspath(str(_expand_env_value(path))) for path in model_path_values
        ]
        for model_path in model_paths:
            if not args.prepare_only and not os.path.exists(model_path):
                print(f"Error: model checkpoint not found: {model_path}", file=sys.stderr)
                sys.exit(1)

        run_dir = _infer_run_dir_from_model_path(model_paths[0])
        requested_test_center = args.test_center or external_cfg.get("test_center")
        test_center = requested_test_center or "External"
        configured_output_dir = args.output_dir or external_cfg.get("output_dir")
        test_output_dir = (
            _resolve_optional_path(configured_output_dir)
            if configured_output_dir
            else os.path.join(run_dir, f"ExternalTest_{test_center}")
        )
        os.makedirs(test_output_dir, exist_ok=True)
        standardized_csv = os.path.join(test_output_dir, f"{test_center}_metadata_standardized.csv")
        metadata_info = _standardize_external_metadata(
            test_csv,
            standardized_csv,
            test_center=requested_test_center,
            label_column=args.label_column or external_cfg.get("label_column"),
            filter_center=requested_test_center is not None,
            default_center=test_center,
        )
        report_metadata_csv = os.path.join(test_output_dir, "dataset_metadata.csv")
        shutil.copyfile(standardized_csv, report_metadata_csv)
        data_dir = _resolve_optional_path(
            args.data_dir
            or external_cfg.get("data_dir")
            or config.get("paths", {}).get("data_dir")
            or os.path.dirname(test_csv)
        )
        external_config_path = _write_external_config(
            config_path,
            test_output_dir,
            data_dir,
            standardized_csv,
        )
        threshold_cfg = external_cfg.get("threshold")
        threshold = (
            float(args.threshold)
            if args.threshold is not None
            else (
                float(_expand_env_value(threshold_cfg))
                if threshold_cfg is not None and not args.model_path
                else _resolve_threshold_from_run(run_dir, fallback=0.5)
            )
        )
        cmd_test = [
            sys.executable,
            "test.py",
            "-c",
            external_config_path,
            "--test_csv",
            standardized_csv,
            "--threshold",
            str(threshold),
            "--output_json",
            os.path.join(test_output_dir, "metrics.json"),
        ]
        for model_path in model_paths:
            cmd_test.extend(["--model_path", model_path])

        manifest = {
            "mode": "direct_external",
            "test_center": test_center,
            "center_filter": requested_test_center,
            "metadata": metadata_info,
            "report_metadata_path": report_metadata_csv,
            "data_dir": data_dir,
            "config_path": external_config_path,
            "model_paths": model_paths,
            "threshold": threshold,
            "command": cmd_test,
        }
        manifest_path = os.path.join(test_output_dir, "external_validation_manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        print(f"Prepared external metadata: {standardized_csv}")
        print(
            f"Records={metadata_info['num_records']} "
            f"positive={metadata_info['positive']} negative={metadata_info['negative']} "
            f"label_column={metadata_info['label_column']}"
        )
        print(f"Running command: {' '.join(cmd_test)}")
        if args.prepare_only:
            print(f"Prepare-only mode. Manifest saved to: {manifest_path}")
            return

        test_stdout_log = os.path.join(test_output_dir, "test_console.log")
        test_stderr_log = os.path.join(test_output_dir, "test_error.log")
        return_code = run_logged_command(
            cmd_test, project_dir, test_stdout_log, test_stderr_log
        )
        if return_code != 0:
            print(
                f"External test on '{test_center}' failed with exit code {return_code}. "
                f"Check logs in {test_output_dir}",
                file=sys.stderr,
            )
            sys.exit(1)
        print("\n" + "=" * 80)
        print(f"External validation on '{test_center}' completed successfully.")
        print(f"Reports are saved in: {test_output_dir}")
        print("=" * 80)
        return

    if not args.test_center:
        print("Error: --test_center is required in InternalCV mode.", file=sys.stderr)
        sys.exit(1)

    base_log_dir = config["paths"].get("log_dir", "logs")
    target_seed = config["project"].get("seed", 2026)
    try:
        internal_cv_dir = _resolve_internal_cv_dir(
            base_log_dir, target_seed, args.experiment_name
        )
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        print("Please run 'run_internal_cv.py' first.", file=sys.stderr)
        sys.exit(1)

    threshold_path = os.path.join(internal_cv_dir, "best_threshold.json")
    cv_results_path = os.path.join(internal_cv_dir, "cv_results.json")
    if not os.path.exists(threshold_path):
        print(
            f"Error: best_threshold.json not found in '{internal_cv_dir}'.",
            file=sys.stderr,
        )
        sys.exit(1)
    if not os.path.exists(cv_results_path):
        print(
            f"Error: cv_results.json not found in '{internal_cv_dir}'.",
            file=sys.stderr,
        )
        sys.exit(1)

    with open(threshold_path, "r", encoding="utf-8") as f:
        threshold_info = json.load(f)
    with open(cv_results_path, "r", encoding="utf-8") as f:
        cv_info = json.load(f)

    model_paths = []
    for fold in cv_info.get("folds", []):
        fold_id = fold.get("fold")
        model_path = os.path.join(
            internal_cv_dir, f"fold_{fold_id}", f"best_model_fold_{fold_id}.pth"
        )
        if not os.path.exists(model_path):
            print(
                f"Error: expected fold model not found at '{model_path}'",
                file=sys.stderr,
            )
            sys.exit(1)
        model_paths.append(model_path)
    if not model_paths:
        print("Error: no fold models were found for ensembling.", file=sys.stderr)
        sys.exit(1)

    threshold = float(threshold_info.get("best_threshold", 0.5))
    print(f"Found {len(model_paths)} fold models for ensemble external validation.")

    metadata_path = config["paths"].get("metadata_path", "")
    if not os.path.exists(metadata_path):
        print(f"Error: metadata file not found at '{metadata_path}'", file=sys.stderr)
        sys.exit(1)

    test_output_dir = os.path.join(internal_cv_dir, f"ExternalTest_{args.test_center}")
    os.makedirs(test_output_dir, exist_ok=True)
    filtered_test_csv = os.path.join(
        test_output_dir, f"{args.test_center}_metadata.csv"
    )

    record_count = 0
    with (
        open(metadata_path, "r", encoding="utf-8") as infile,
        open(filtered_test_csv, "w", newline="", encoding="utf-8") as outfile,
    ):
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        header = next(reader)
        writer.writerow(header)
        try:
            resolved_columns = resolve_required_metadata_columns(header)
            center_idx = header.index(resolved_columns["center"])
        except ValueError:
            print(
                "Error: metadata CSV is missing a required center column.",
                file=sys.stderr,
            )
            sys.exit(1)
        for row in reader:
            if row[center_idx].strip() == args.test_center:
                writer.writerow(row)
                record_count += 1
    if record_count == 0:
        print(
            f"Error: no records found for center '{args.test_center}'",
            file=sys.stderr,
        )
        sys.exit(1)
    report_metadata_csv = os.path.join(test_output_dir, "dataset_metadata.csv")
    shutil.copyfile(filtered_test_csv, report_metadata_csv)
    external_config_path = _write_external_config(
        config_path,
        test_output_dir,
        config["paths"].get("data_dir", ""),
        filtered_test_csv,
    )

    print(
        "\n"
        + "=" * 25
        + f" Starting External Test on [ {args.test_center} ] "
        + "=" * 25
    )
    python_exec = sys.executable
    cmd_test = [
        python_exec,
        "test.py",
        "-c",
        external_config_path,
        "--test_csv",
        filtered_test_csv,
        "--threshold",
        str(threshold),
        "--output_json",
        os.path.join(test_output_dir, "metrics.json"),
    ]
    for model_path in model_paths:
        cmd_test.extend(["--model_path", model_path])

    manifest = {
        "mode": "cv_ensemble_external",
        "test_center": args.test_center,
        "metadata_path": filtered_test_csv,
        "report_metadata_path": report_metadata_csv,
        "data_dir": config["paths"].get("data_dir", ""),
        "config_path": external_config_path,
        "internal_cv_dir": internal_cv_dir,
        "model_paths": model_paths,
        "threshold": threshold,
        "threshold_path": threshold_path,
        "cv_results_path": cv_results_path,
        "command": cmd_test,
    }
    manifest_path = os.path.join(test_output_dir, "external_validation_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    test_stdout_log = os.path.join(test_output_dir, "test_console.log")
    test_stderr_log = os.path.join(test_output_dir, "test_error.log")
    try:
        print(f"Running command: {' '.join(cmd_test)}")
        return_code = run_logged_command(
            cmd_test, project_dir, test_stdout_log, test_stderr_log
        )
        if return_code != 0:
            print(
                f"External test on '{args.test_center}' failed with exit code {return_code}. Check logs in {test_output_dir}",
                file=sys.stderr,
            )
            sys.exit(1)
        print("\n" + "=" * 80)
        print(f"External validation on '{args.test_center}' completed successfully.")
        print(f"Reports are saved in: {test_output_dir}")
        print("=" * 80)
    except FileNotFoundError:
        print(
            "Error: 'test.py' not found. Make sure you are in the project root.",
            file=sys.stderr,
        )
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    run_external_validation()
