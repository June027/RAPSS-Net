from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


MetricFunc = Callable[[np.ndarray, np.ndarray, np.ndarray], float]


def _read_predictions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"id", "label", "prob"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} missing required columns: {sorted(missing)}")

    out = df.copy()
    out["id"] = out["id"].astype(str)
    out["label"] = out["label"].astype(int)
    out["prob"] = out["prob"].astype(float)
    if not set(out["label"].dropna().unique()).issubset({0, 1}):
        raise ValueError(f"{path} label column must be binary 0/1.")
    return out


def _resolve_pred(
    df: pd.DataFrame,
    threshold: float,
    pred_source: str,
) -> tuple[np.ndarray, str]:
    if pred_source == "pred_column":
        if "pred" not in df.columns:
            raise ValueError("--pred_source pred_column was requested, but CSV has no pred column.")
        return df["pred"].astype(int).to_numpy(), "pred_column"

    if pred_source == "threshold":
        return (df["prob"].to_numpy(float) >= float(threshold)).astype(int), "threshold"

    if pred_source == "auto" and "pred" in df.columns:
        return df["pred"].astype(int).to_numpy(), "pred_column"

    return (df["prob"].to_numpy(float) >= float(threshold)).astype(int), "threshold"


def _safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))


def _safe_ap(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_prob))


def _classification_counts(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[int, int, int, int]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return int(tn), int(fp), int(fn), int(tp)


def _accuracy(y_true: np.ndarray, y_prob: np.ndarray, y_pred: np.ndarray) -> float:
    return float((y_pred == y_true).mean())


def _sensitivity(y_true: np.ndarray, y_prob: np.ndarray, y_pred: np.ndarray) -> float:
    _, _, fn, tp = _classification_counts(y_true, y_pred)
    return float(tp / (tp + fn)) if tp + fn else 0.0


def _specificity(y_true: np.ndarray, y_prob: np.ndarray, y_pred: np.ndarray) -> float:
    tn, fp, _, _ = _classification_counts(y_true, y_pred)
    return float(tn / (tn + fp)) if tn + fp else 0.0


def _ppv(y_true: np.ndarray, y_prob: np.ndarray, y_pred: np.ndarray) -> float:
    _, fp, _, tp = _classification_counts(y_true, y_pred)
    return float(tp / (tp + fp)) if tp + fp else 0.0


def _npv(y_true: np.ndarray, y_prob: np.ndarray, y_pred: np.ndarray) -> float:
    tn, _, fn, _ = _classification_counts(y_true, y_pred)
    return float(tn / (tn + fn)) if tn + fn else 0.0


def _f1(y_true: np.ndarray, y_prob: np.ndarray, y_pred: np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, zero_division=0))


def _brier(y_true: np.ndarray, y_prob: np.ndarray, y_pred: np.ndarray) -> float:
    return float(brier_score_loss(y_true, y_prob))


def _metric_value(metric: str, y_true: np.ndarray, y_prob: np.ndarray, y_pred: np.ndarray) -> float:
    metric_funcs: dict[str, MetricFunc] = {
        "auc": lambda y, p, pred: _safe_auc(y, p),
        "pr_auc": lambda y, p, pred: _safe_ap(y, p),
        "accuracy": _accuracy,
        "sensitivity": _sensitivity,
        "specificity": _specificity,
        "ppv": _ppv,
        "npv": _npv,
        "f1": _f1,
        "brier": _brier,
    }
    return metric_funcs[metric](y_true, y_prob, y_pred)


def _bootstrap_ci(
    metric: str,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    y_pred: np.ndarray,
    n_bootstrap: int,
    seed: int,
) -> tuple[float | None, float | None]:
    if n_bootstrap <= 0:
        return None, None

    rng = np.random.default_rng(seed)
    n = len(y_true)
    values: list[float] = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        sample_y = y_true[idx]
        sample_prob = y_prob[idx]
        sample_pred = y_pred[idx]
        if metric in {"auc", "pr_auc"} and len(np.unique(sample_y)) < 2:
            continue
        values.append(_metric_value(metric, sample_y, sample_prob, sample_pred))

    if not values:
        return None, None
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def _format_ci(value: float, low: float | None, high: float | None, digits: int) -> str:
    if low is None or high is None:
        return f"{value:.{digits}f}"
    return f"{value:.{digits}f} ({low:.{digits}f}-{high:.{digits}f})"


def compute_metrics(
    df: pd.DataFrame,
    dataset_name: str,
    threshold: float,
    pred_source: str,
    n_bootstrap: int,
    seed: int,
    digits: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    y_true = df["label"].to_numpy(int)
    y_prob = df["prob"].to_numpy(float)
    y_pred, resolved_pred_source = _resolve_pred(df, threshold, pred_source)

    pred_out = df.copy()
    pred_out["pred"] = y_pred
    pred_out["dataset"] = dataset_name
    pred_out["threshold"] = float(threshold)
    pred_out["pred_source"] = resolved_pred_source

    tn, fp, fn, tp = _classification_counts(y_true, y_pred)
    raw: dict[str, object] = {
        "dataset": dataset_name,
        "n": int(len(y_true)),
        "positive": int(y_true.sum()),
        "negative": int((y_true == 0).sum()),
        "threshold": float(threshold),
        "pred_source": resolved_pred_source,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "predicted_positive": int(y_pred.sum()),
    }

    formatted: dict[str, object] = {
        "Dataset": dataset_name,
        "N": int(len(y_true)),
        "Positive": int(y_true.sum()),
        "Negative": int((y_true == 0).sum()),
        "Threshold": float(threshold),
        "Prediction source": resolved_pred_source,
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
    }

    display_names = {
        "auc": "AUC (95% CI)",
        "pr_auc": "PR-AUC (95% CI)",
        "accuracy": "Accuracy (95% CI)",
        "sensitivity": "Sensitivity (95% CI)",
        "specificity": "Specificity (95% CI)",
        "ppv": "PPV (95% CI)",
        "npv": "NPV (95% CI)",
        "f1": "F1 (95% CI)",
        "brier": "Brier score (95% CI)",
    }

    for offset, metric in enumerate(display_names):
        value = _metric_value(metric, y_true, y_prob, y_pred)
        low, high = _bootstrap_ci(
            metric,
            y_true,
            y_prob,
            y_pred,
            n_bootstrap=n_bootstrap,
            seed=seed + offset,
        )
        raw[metric] = value
        raw[f"{metric}_ci_low"] = low
        raw[f"{metric}_ci_high"] = high
        formatted[display_names[metric]] = _format_ci(value, low, high, digits)

    raw_df = pd.DataFrame([raw])
    formatted_df = pd.DataFrame([formatted])
    return pred_out, pd.concat([formatted_df, raw_df], axis=1)


def _write_outputs(
    prediction_tables: list[pd.DataFrame],
    metrics_tables: list[pd.DataFrame],
    output_dir: Path,
    output_prefix: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions = pd.concat(prediction_tables, ignore_index=True)
    metrics = pd.concat(metrics_tables, ignore_index=True)

    predictions.to_csv(output_dir / f"{output_prefix}_predictions.csv", index=False)
    metrics.to_csv(output_dir / f"{output_prefix}_metrics_summary.csv", index=False)

    with pd.ExcelWriter(output_dir / f"{output_prefix}_prediction_metrics.xlsx", engine="openpyxl") as writer:
        predictions.to_excel(writer, sheet_name="predictions", index=False)
        metrics.to_excel(writer, sheet_name="metrics", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export prediction CSV/XLSX and metrics table with bootstrap 95% CI."
    )
    parser.add_argument(
        "--prediction_csv",
        type=Path,
        action="append",
        required=True,
        help="Prediction CSV with id,label,prob and optional pred. Pass multiple times for multiple datasets.",
    )
    parser.add_argument(
        "--dataset_name",
        action="append",
        default=None,
        help="Dataset display name. Pass once per --prediction_csv. Defaults to file stem.",
    )
    parser.add_argument("--output_dir", type=Path, default=Path("prediction_metrics_export"))
    parser.add_argument("--output_prefix", default="prediction_metrics")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--pred_source",
        choices=["auto", "pred_column", "threshold"],
        default="auto",
        help="auto uses pred column when present; otherwise prob >= threshold.",
    )
    parser.add_argument("--n_bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2040)
    parser.add_argument("--digits", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths: list[Path] = args.prediction_csv
    names: list[str]
    if args.dataset_name is None:
        names = [path.stem for path in paths]
    else:
        if len(args.dataset_name) != len(paths):
            raise SystemExit("--dataset_name count must match --prediction_csv count.")
        names = args.dataset_name

    prediction_tables: list[pd.DataFrame] = []
    metrics_tables: list[pd.DataFrame] = []
    for path, name in zip(paths, names):
        df = _read_predictions(path)
        pred_out, metrics = compute_metrics(
            df,
            dataset_name=name,
            threshold=args.threshold,
            pred_source=args.pred_source,
            n_bootstrap=args.n_bootstrap,
            seed=args.seed,
            digits=args.digits,
        )
        prediction_tables.append(pred_out)
        metrics_tables.append(metrics)

    _write_outputs(prediction_tables, metrics_tables, args.output_dir, args.output_prefix)
    print(args.output_dir / f"{args.output_prefix}_prediction_metrics.xlsx")
    print(args.output_dir / f"{args.output_prefix}_metrics_summary.csv")
    print(args.output_dir / f"{args.output_prefix}_predictions.csv")


if __name__ == "__main__":
    main()
