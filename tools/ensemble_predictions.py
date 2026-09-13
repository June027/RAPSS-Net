import argparse
import csv
import itertools
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from utils.metrics import binary_metrics_from_probs, calculate_fast_metrics


@dataclass(frozen=True)
class PredictionRun:
    name: str
    path: str
    ids: Tuple[str, ...]
    labels: np.ndarray
    probs: np.ndarray


def infer_run_name(path: str) -> str:
    normalized = os.path.normpath(path)
    filename = os.path.basename(normalized).lower()
    if filename in {"random_split_predictions.csv", "predictions.csv"}:
        return os.path.basename(os.path.dirname(normalized))
    return os.path.splitext(os.path.basename(normalized))[0]


def load_prediction_run(path: str, name: Optional[str] = None) -> PredictionRun:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Prediction CSV not found: {path}")

    ids: List[str] = []
    labels: List[int] = []
    probs: List[float] = []
    seen_ids = set()

    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"id", "label", "prob"}
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise ValueError(
                f"Prediction CSV must contain columns {sorted(required)}: {path}"
            )
        for row in reader:
            sample_id = str(row["id"]).strip()
            if not sample_id:
                raise ValueError(f"Encountered empty sample id in: {path}")
            if sample_id in seen_ids:
                raise ValueError(f"Duplicate sample id '{sample_id}' in: {path}")
            seen_ids.add(sample_id)
            ids.append(sample_id)
            labels.append(int(float(row["label"])))
            probs.append(float(row["prob"]))

    if not ids:
        raise ValueError(f"No prediction rows found in: {path}")

    return PredictionRun(
        name=str(name or infer_run_name(path)),
        path=os.path.abspath(path),
        ids=tuple(ids),
        labels=np.asarray(labels, dtype=np.int64),
        probs=np.asarray(probs, dtype=np.float64),
    )


def align_prediction_runs(
    runs: Sequence[PredictionRun],
) -> Tuple[List[str], np.ndarray, Dict[str, np.ndarray]]:
    if len(runs) < 2:
        raise ValueError("At least two prediction runs are required for ensembling.")

    base_run = runs[0]
    base_ids = list(base_run.ids)
    base_labels = np.asarray(base_run.labels, dtype=np.int64)
    base_index = {sample_id: idx for idx, sample_id in enumerate(base_ids)}

    aligned_probs: Dict[str, np.ndarray] = {base_run.name: np.asarray(base_run.probs)}

    for run in runs[1:]:
        if len(run.ids) != len(base_ids):
            raise ValueError(
                f"Prediction size mismatch: '{run.name}' has {len(run.ids)} rows, "
                f"expected {len(base_ids)}."
            )
        if len(set(run.ids)) != len(base_ids):
            raise ValueError(f"Duplicate ids detected after loading run '{run.name}'.")

        current_map = {
            sample_id: (int(label), float(prob))
            for sample_id, label, prob in zip(run.ids, run.labels, run.probs)
        }
        if set(current_map.keys()) != set(base_ids):
            missing = sorted(set(base_ids) - set(current_map.keys()))
            extra = sorted(set(current_map.keys()) - set(base_ids))
            raise ValueError(
                f"Sample id mismatch for run '{run.name}'. Missing={missing[:5]}, extra={extra[:5]}"
            )

        ordered_probs = np.zeros(len(base_ids), dtype=np.float64)
        for sample_id, base_idx in base_index.items():
            run_label, run_prob = current_map[sample_id]
            if run_label != int(base_labels[base_idx]):
                raise ValueError(
                    f"Label mismatch on sample '{sample_id}' between base run "
                    f"'{base_run.name}' and run '{run.name}'."
                )
            ordered_probs[base_idx] = run_prob
        aligned_probs[run.name] = ordered_probs

    return base_ids, base_labels, aligned_probs


def evaluate_probabilities(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    fixed_threshold: Optional[float] = None,
) -> Dict[str, float]:
    if fixed_threshold is None:
        return calculate_fast_metrics(y_true, y_prob)

    metrics = binary_metrics_from_probs(y_true, y_prob, threshold=float(fixed_threshold))
    return {
        "AUC": float(metrics["auc"]),
        "PR-AUC": float(metrics["pr_auc"]),
        "Precision": float(metrics["precision"]),
        "Recall": float(metrics["recall"]),
        "F1": float(metrics["f1"]),
        "ACC": float(metrics["accuracy"]),
        "Micro-ACC": float(metrics["micro_accuracy"]),
        "NPV": float(metrics["npv"]),
        "Kappa": float(metrics["kappa"]),
        "Sens": float(metrics["sensitivity"]),
        "Spec": float(metrics["specificity"]),
        "Thresh": float(fixed_threshold),
    }


def evaluate_combinations(
    ids: Sequence[str],
    y_true: np.ndarray,
    aligned_probs: Dict[str, np.ndarray],
    min_size: int = 1,
    max_size: Optional[int] = None,
    fixed_threshold: Optional[float] = None,
) -> List[Dict[str, object]]:
    model_names = list(aligned_probs.keys())
    if max_size is None:
        max_size = len(model_names)
    max_size = min(max_size, len(model_names))
    if min_size < 1 or min_size > max_size:
        raise ValueError(
            f"Invalid combination range: min_size={min_size}, max_size={max_size}."
        )

    rows: List[Dict[str, object]] = []
    for combo_size in range(min_size, max_size + 1):
        for combo_names in itertools.combinations(model_names, combo_size):
            combo_matrix = np.stack([aligned_probs[name] for name in combo_names], axis=0)
            mean_prob = combo_matrix.mean(axis=0)
            metrics = evaluate_probabilities(
                y_true, mean_prob, fixed_threshold=fixed_threshold
            )
            rows.append(
                {
                    "combo_size": int(combo_size),
                    "combo_names": list(combo_names),
                    "combo_key": " + ".join(combo_names),
                    "auc": float(metrics["AUC"]),
                    "pr_auc": float(metrics["PR-AUC"]),
                    "precision": float(metrics["Precision"]),
                    "recall": float(metrics["Recall"]),
                    "f1": float(metrics["F1"]),
                    "acc": float(metrics["ACC"]),
                    "micro_acc": float(metrics["Micro-ACC"]),
                    "npv": float(metrics["NPV"]),
                    "kappa": float(metrics["Kappa"]),
                    "sens": float(metrics["Sens"]),
                    "spec": float(metrics["Spec"]),
                    "threshold": float(metrics["Thresh"]),
                    "threshold_mode": "fixed" if fixed_threshold is not None else "dynamic",
                    "probs": mean_prob,
                    "ids": list(ids),
                    "labels": np.asarray(y_true, dtype=np.int64),
                }
            )

    rows.sort(
        key=lambda row: (
            -float(np.nan_to_num(row["auc"], nan=-1.0)),
            row["combo_size"],
            row["combo_key"],
        )
    )
    return rows


def write_predictions_csv(path: str, ids: Sequence[str], labels: np.ndarray, probs: np.ndarray):
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "label", "prob"])
        for sample_id, label, prob in zip(ids, labels, probs):
            writer.writerow([sample_id, int(label), float(prob)])


def write_ranking_csv(path: str, rows: Sequence[Dict[str, object]]):
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "rank",
                "combo_size",
                "combo_key",
                "auc",
                "pr_auc",
                "precision",
                "recall",
                "f1",
                "acc",
                "npv",
                "kappa",
                "sens",
                "spec",
                "threshold",
                "threshold_mode",
            ]
        )
        for rank, row in enumerate(rows, start=1):
            writer.writerow(
                [
                    rank,
                    row["combo_size"],
                    row["combo_key"],
                    row["auc"],
                    row["pr_auc"],
                    row["precision"],
                    row["recall"],
                    row["f1"],
                    row["acc"],
                    row["npv"],
                    row["kappa"],
                    row["sens"],
                    row["spec"],
                    row["threshold"],
                    row["threshold_mode"],
                ]
            )


def build_summary_payload(
    runs: Sequence[PredictionRun],
    rows: Sequence[Dict[str, object]],
    best_row: Dict[str, object],
    fixed_threshold: Optional[float],
) -> Dict[str, object]:
    serializable_rows = []
    for rank, row in enumerate(rows, start=1):
        serializable_rows.append(
            {
                "rank": int(rank),
                "combo_size": int(row["combo_size"]),
                "combo_names": list(row["combo_names"]),
                "combo_key": str(row["combo_key"]),
                "auc": float(row["auc"]),
                "pr_auc": float(row["pr_auc"]),
                "precision": float(row["precision"]),
                "recall": float(row["recall"]),
                "f1": float(row["f1"]),
                "acc": float(row["acc"]),
                "micro_acc": float(row["micro_acc"]),
                "npv": float(row["npv"]),
                "kappa": float(row["kappa"]),
                "sens": float(row["sens"]),
                "spec": float(row["spec"]),
                "threshold": float(row["threshold"]),
                "threshold_mode": str(row["threshold_mode"]),
            }
        )

    return {
        "num_models": int(len(runs)),
        "num_samples": int(len(best_row["ids"])),
        "fixed_threshold": None if fixed_threshold is None else float(fixed_threshold),
        "inputs": [{"name": run.name, "path": run.path} for run in runs],
        "best_combo": serializable_rows[0] if serializable_rows else None,
        "ranking": serializable_rows,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Average prediction CSVs and rank soft-voting ensembles."
    )
    parser.add_argument(
        "--prediction",
        action="append",
        required=True,
        help="Prediction CSV path. Pass multiple times.",
    )
    parser.add_argument(
        "--name",
        action="append",
        default=None,
        help="Optional display name aligned with each --prediction.",
    )
    parser.add_argument(
        "--min_size",
        type=int,
        default=1,
        help="Minimum ensemble size to evaluate.",
    )
    parser.add_argument(
        "--max_size",
        type=int,
        default=3,
        help="Maximum ensemble size to evaluate.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Optional fixed threshold for threshold-based metrics.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory for reports and best-ensemble predictions.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Number of top-ranked combinations to print.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.name is not None and len(args.name) != len(args.prediction):
        raise ValueError("--name count must match --prediction count.")
    if args.threshold is not None and not 0.0 <= float(args.threshold) <= 1.0:
        raise ValueError("--threshold must be in [0, 1].")

    runs = [
        load_prediction_run(path, None if args.name is None else args.name[idx])
        for idx, path in enumerate(args.prediction)
    ]
    ids, labels, aligned_probs = align_prediction_runs(runs)
    rows = evaluate_combinations(
        ids=ids,
        y_true=labels,
        aligned_probs=aligned_probs,
        min_size=int(args.min_size),
        max_size=int(args.max_size),
        fixed_threshold=args.threshold,
    )
    if not rows:
        raise RuntimeError("No ensemble combinations were evaluated.")

    best_row = rows[0]
    os.makedirs(args.output_dir, exist_ok=True)

    ranking_csv = os.path.join(args.output_dir, "ensemble_ranking.csv")
    summary_json = os.path.join(args.output_dir, "ensemble_summary.json")
    best_pred_csv = os.path.join(args.output_dir, "best_ensemble_predictions.csv")

    write_ranking_csv(ranking_csv, rows)
    write_predictions_csv(
        best_pred_csv,
        best_row["ids"],
        best_row["labels"],
        best_row["probs"],
    )
    summary = build_summary_payload(runs, rows, best_row, fixed_threshold=args.threshold)
    with open(summary_json, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(f"Evaluated {len(rows)} combination(s) across {len(runs)} model(s).")
    print(f"Best combo: {best_row['combo_key']}")
    print(
        "Best metrics: "
        f"AUC={best_row['auc']:.4f}, PR-AUC={best_row['pr_auc']:.4f}, "
        f"F1={best_row['f1']:.4f}, ACC={best_row['acc']:.4f}, "
        f"Thresh={best_row['threshold']:.4f} ({best_row['threshold_mode']})"
    )
    print(f"Saved ranking CSV: {ranking_csv}")
    print(f"Saved summary JSON: {summary_json}")
    print(f"Saved best predictions: {best_pred_csv}")

    for rank, row in enumerate(rows[: max(1, int(args.top_k))], start=1):
        print(
            f"[{rank}] {row['combo_key']} | size={row['combo_size']} | "
            f"AUC={row['auc']:.4f} | PR-AUC={row['pr_auc']:.4f} | "
            f"F1={row['f1']:.4f} | ACC={row['acc']:.4f}"
        )


if __name__ == "__main__":
    main()
