from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


DEFAULT_RUN_DIR = (
    Path("output")
    / "MRIUpper_NoTTA_Seed2040_Random"
    / "RandomSplit_(ISPY1+ISPY2)_seed_2040_val_0.20"
)
DEFAULT_EVAL_CONFIG = Path("configs/server/random/MRIUpper_NoTTA_Seed2040_Random_eval.yaml")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


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
    if not set(out["label"].unique()).issubset({0, 1}):
        raise ValueError(f"{path} label column must contain only 0/1.")
    return out


def _read_metadata(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path, encoding="utf-8-sig")
    if "id" not in df.columns:
        raise ValueError(f"{path} missing required column: id")
    df = df.copy()
    df["id"] = df["id"].astype(str)
    return df


def _apply_metadata_label(pred_df: pd.DataFrame, metadata_path: Path) -> pd.DataFrame:
    metadata = _read_metadata(metadata_path)
    if "pCR" not in metadata.columns:
        raise ValueError(f"{metadata_path} missing pCR column for metadata label mode.")
    labels = metadata[["id", "pCR"]].copy()
    labels["label_metadata_pcr"] = pd.to_numeric(labels["pCR"], errors="coerce").astype("Int64")
    merged = pred_df.drop(columns=["label"]).merge(
        labels[["id", "label_metadata_pcr"]],
        on="id",
        how="left",
    )
    if merged["label_metadata_pcr"].isna().any():
        missing = merged.loc[merged["label_metadata_pcr"].isna(), "id"].head(8).tolist()
        raise ValueError(f"{metadata_path} has no pCR label for prediction IDs: {missing}")
    merged["label"] = merged["label_metadata_pcr"].astype(int)
    merged = merged.drop(columns=["label_metadata_pcr"])
    front = ["id", "label", "prob"]
    if "pred" in merged.columns:
        front.append("pred")
    return merged[front + [col for col in merged.columns if col not in front]]


def _best_youden_threshold(y_true: np.ndarray, prob: np.ndarray) -> float:
    fpr, tpr, thresholds = roc_curve(y_true, prob)
    valid = np.isfinite(thresholds)
    if not valid.any():
        return 0.5
    idx = int(np.argmax(tpr[valid] - fpr[valid]))
    return float(thresholds[valid][idx])


def _metrics(y_true: np.ndarray, prob: np.ndarray, pred: np.ndarray, threshold: float) -> dict[str, Any]:
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if tp + fn else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    ppv = tp / (tp + fp) if tp + fp else 0.0
    npv = tn / (tn + fn) if tn + fn else 0.0
    return {
        "n": int(len(y_true)),
        "positive": int(y_true.sum()),
        "negative": int((y_true == 0).sum()),
        "threshold": float(threshold),
        "auc": float(roc_auc_score(y_true, prob)) if len(np.unique(y_true)) == 2 else np.nan,
        "average_precision": float(average_precision_score(y_true, prob))
        if len(np.unique(y_true)) == 2
        else np.nan,
        "accuracy": float((tp + tn) / len(y_true)),
        "sensitivity": float(sens),
        "specificity": float(spec),
        "ppv": float(ppv),
        "npv": float(npv),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "predicted_positive": int(pred.sum()),
    }


def _prepare_predictions(
    path: Path,
    threshold: float,
    prefer_existing_pred: bool,
) -> tuple[pd.DataFrame, np.ndarray, dict[str, Any]]:
    df = _read_predictions(path)
    y = df["label"].to_numpy(int)
    prob = df["prob"].to_numpy(float)
    if prefer_existing_pred and "pred" in df.columns:
        pred = df["pred"].astype(int).to_numpy()
        pred_source = "csv_pred"
    else:
        pred = (prob >= threshold).astype(int)
        pred_source = "threshold"
    df = df.copy()
    df["pred"] = pred
    return df, pred, {"pred_source": pred_source, **_metrics(y, prob, pred, threshold)}


def _plot_confusion_panel(
    ax: plt.Axes,
    title: str,
    y_true: np.ndarray,
    pred: np.ndarray,
    metric: dict[str, Any],
) -> None:
    mat = confusion_matrix(y_true, pred, labels=[0, 1])
    ax.imshow(mat, cmap="Blues")
    cutoff = (mat.max() + mat.min()) / 2.0
    for (row, col), value in np.ndenumerate(mat):
        color = "white" if value > cutoff else "#111827"
        ax.text(
            col,
            row,
            str(int(value)),
            ha="center",
            va="center",
            fontsize=13,
            fontweight="bold",
            color=color,
        )
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Pred 0", "Pred 1"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["True 0", "True 1"])
    ax.set_title(
        f"{title}\nAUC={metric['auc']:.3f}, ACC={metric['accuracy']:.3f}, T={metric['threshold']:.3f}",
        fontsize=10,
    )
    ax.tick_params(axis="both", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#374151")
        spine.set_linewidth(0.8)


def _save_confusion_figures(
    outputs: dict[str, tuple[pd.DataFrame, np.ndarray, dict[str, Any]]],
    output_dir: Path,
) -> None:
    names = list(outputs)
    fig, axes = plt.subplots(1, len(names), figsize=(4.2 * len(names), 4.2), dpi=220)
    if len(names) == 1:
        axes = [axes]
    for ax, name in zip(axes, names):
        df, pred, metric = outputs[name]
        _plot_confusion_panel(ax, name, df["label"].to_numpy(int), pred, metric)
    fig.suptitle("Confusion Matrices", y=1.04, fontsize=14)
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrices_train_test_external.png", bbox_inches="tight")
    fig.savefig(output_dir / "confusion_matrices_train_test_external.pdf", bbox_inches="tight")
    plt.close(fig)

    single_dir = output_dir / "single_confusion_matrices"
    single_dir.mkdir(parents=True, exist_ok=True)
    for name, (df, pred, metric) in outputs.items():
        fig, ax = plt.subplots(figsize=(4.2, 4.2), dpi=220)
        _plot_confusion_panel(ax, name, df["label"].to_numpy(int), pred, metric)
        fig.tight_layout()
        slug = name.lower().replace(" ", "_").replace("/", "_")
        fig.savefig(single_dir / f"{slug}_confusion_matrix.png", bbox_inches="tight")
        fig.savefig(single_dir / f"{slug}_confusion_matrix.pdf", bbox_inches="tight")
        plt.close(fig)


def _save_roc_figure(
    outputs: dict[str, tuple[pd.DataFrame, np.ndarray, dict[str, Any]]],
    output_dir: Path,
) -> None:
    colors = ["#2563eb", "#dc2626", "#7c3aed", "#059669"]
    fig, ax = plt.subplots(figsize=(6.2, 5.4), dpi=220)
    for (name, (df, _, metric)), color in zip(outputs.items(), colors):
        y = df["label"].to_numpy(int)
        prob = df["prob"].to_numpy(float)
        fpr, tpr, _ = roc_curve(y, prob)
        ax.plot(fpr, tpr, linewidth=2.0, color=color, label=f"{name} AUC={metric['auc']:.3f}")
    ax.plot([0, 1], [0, 1], color="#9ca3af", linestyle="--", linewidth=1.2)
    ax.set_xlabel("1-Specificity")
    ax.set_ylabel("Sensitivity")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(True, color="#e5e7eb", linewidth=0.8)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "roc_curves_train_test_external.png", bbox_inches="tight")
    fig.savefig(output_dir / "roc_curves_train_test_external.pdf", bbox_inches="tight")
    plt.close(fig)


def _save_pr_figure(
    outputs: dict[str, tuple[pd.DataFrame, np.ndarray, dict[str, Any]]],
    output_dir: Path,
) -> None:
    colors = ["#2563eb", "#dc2626", "#7c3aed", "#059669"]
    fig, ax = plt.subplots(figsize=(6.2, 5.4), dpi=220)
    for (name, (df, _, metric)), color in zip(outputs.items(), colors):
        y = df["label"].to_numpy(int)
        prob = df["prob"].to_numpy(float)
        precision, recall, _ = precision_recall_curve(y, prob)
        ax.plot(
            recall,
            precision,
            linewidth=2.0,
            color=color,
            label=f"{name} AP={metric['average_precision']:.3f}",
        )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(True, color="#e5e7eb", linewidth=0.8)
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "pr_curves_train_test_external.png", bbox_inches="tight")
    fig.savefig(output_dir / "pr_curves_train_test_external.pdf", bbox_inches="tight")
    plt.close(fig)


def _merge_dataset_for_xlsx(
    records_path: Path | None,
    metadata_path: Path | None,
    pred_df: pd.DataFrame,
    split_name: str,
) -> pd.DataFrame:
    out = pred_df[["id", "label", "prob", "pred"]].copy()
    out.insert(0, "split", split_name)
    if records_path is not None and records_path.exists():
        records = pd.read_csv(records_path)
        records["id"] = records["id"].astype(str)
        out = out.merge(records, on=["id", "label"], how="left", suffixes=("", "_record"))
    if metadata_path is not None and metadata_path.exists():
        metadata = _read_metadata(metadata_path)
        keep = [col for col in ["id", "pCR", "center", "age", "ER", "PR", "HER2", "her-2", "Ki-67"] if col in metadata]
        metadata = metadata[keep].drop_duplicates("id")
        out = out.merge(metadata, on="id", how="left", suffixes=("", "_metadata"))
    return out


def _write_dataset_workbooks(
    output_dir: Path,
    run_dir: Path,
    datasets: dict[str, pd.DataFrame],
    metadata_paths: dict[str, Path],
) -> list[Path]:
    dataset_dir = output_dir / "datasets_xlsx"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    generated = []

    train_xlsx = _merge_dataset_for_xlsx(
        run_dir / "random_split" / "train_records.csv",
        Path("dataset/dataset_metadata.csv"),
        datasets["Train"],
        "Train",
    )
    test_xlsx = _merge_dataset_for_xlsx(
        run_dir / "random_split" / "val_records.csv",
        Path("dataset/dataset_metadata.csv"),
        datasets["Internal Test"],
        "Internal Test",
    )
    internal_path = dataset_dir / "internal_train_test_datasets.xlsx"
    with pd.ExcelWriter(internal_path, engine="openpyxl") as writer:
        train_xlsx.to_excel(writer, sheet_name="Train", index=False)
        test_xlsx.to_excel(writer, sheet_name="Internal_Test", index=False)
    generated.append(internal_path)

    external_path = dataset_dir / "external_duke_private_datasets.xlsx"
    with pd.ExcelWriter(external_path, engine="openpyxl") as writer:
        for name in ["Duke External", "Private External"]:
            metadata_path = metadata_paths[name]
            merged = _merge_dataset_for_xlsx(None, metadata_path, datasets[name], name)
            sheet = name.replace(" External", "")[:31]
            merged.to_excel(writer, sheet_name=sheet, index=False)
    generated.append(external_path)
    return generated


def _run_external_prediction(
    cohort: str,
    metadata_path: Path,
    output_dir: Path,
    config_path: Path,
    model_path: Path,
    data_dir: Path,
    threshold: float,
) -> Path:
    cohort_out = output_dir / f"ExternalTest_{cohort}_generated"
    cmd = [
        sys.executable,
        "run_external_validation.py",
        "-c",
        str(config_path),
        "--test_csv",
        str(metadata_path),
        "--test_center",
        cohort,
        "--model_path",
        str(model_path),
        "--output_dir",
        str(cohort_out),
        "--data_dir",
        str(data_dir),
        "--threshold",
        str(threshold),
    ]
    subprocess.run(cmd, check=True)
    pred_path = cohort_out / "predictions.csv"
    if not pred_path.exists():
        raise FileNotFoundError(f"External inference finished but predictions not found: {pred_path}")
    return pred_path


def generate_report(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = args.run_dir
    output_dir = args.output_dir or run_dir / "confusion_report_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    train_pred_path = args.train_predictions or run_dir / "fold_1" / "best_epoch_train_predictions.csv"
    test_pred_path = args.test_predictions or run_dir / "fold_1" / "best_epoch_val_predictions.csv"
    train_raw = _read_predictions(train_pred_path)
    test_raw = _read_predictions(test_pred_path)
    threshold = args.threshold
    threshold_source = "fixed"
    if threshold is None:
        threshold = _best_youden_threshold(test_raw["label"].to_numpy(int), test_raw["prob"].to_numpy(float))
        threshold_source = "internal_test_youden"

    duke_pred_path = args.duke_predictions
    if duke_pred_path is None:
        default_duke = run_dir / "yuanyuan" / "fold_1-yuanyuan" / "predictions_duke.csv"
        duke_pred_path = default_duke if default_duke.exists() else None
    private_pred_path = args.private_predictions

    if duke_pred_path is None and args.run_missing_external:
        duke_pred_path = _run_external_prediction(
            "Duke",
            args.duke_metadata,
            output_dir,
            args.eval_config,
            args.model_path,
            args.data_dir,
            threshold,
        )
    if private_pred_path is None and args.run_missing_external:
        private_pred_path = _run_external_prediction(
            "Private",
            args.private_metadata,
            output_dir,
            args.eval_config,
            args.model_path,
            args.data_dir,
            threshold,
        )
    if duke_pred_path is None or not duke_pred_path.exists():
        raise FileNotFoundError("Duke predictions not found. Pass --duke_predictions or use --run_missing_external.")
    if private_pred_path is None or not private_pred_path.exists():
        raise FileNotFoundError(
            "Private predictions not found. Pass --private_predictions or use --run_missing_external."
        )

    train_df, train_pred, train_metric = _prepare_predictions(train_pred_path, threshold, False)
    test_df, test_pred, test_metric = _prepare_predictions(test_pred_path, threshold, False)
    duke_df = _read_predictions(duke_pred_path)
    private_df = _read_predictions(private_pred_path)
    if args.external_label_source == "metadata_pCR":
        duke_df = _apply_metadata_label(duke_df, args.duke_metadata)
        private_df = _apply_metadata_label(private_df, args.private_metadata)
        duke_tmp = output_dir / "duke_predictions_metadata_pcr_label.csv"
        private_tmp = output_dir / "private_predictions_metadata_pcr_label.csv"
        duke_df.to_csv(duke_tmp, index=False)
        private_df.to_csv(private_tmp, index=False)
        duke_pred_path = duke_tmp
        private_pred_path = private_tmp

    duke_df, duke_pred, duke_metric = _prepare_predictions(
        duke_pred_path,
        threshold,
        args.prefer_existing_external_pred,
    )
    private_df, private_pred, private_metric = _prepare_predictions(
        private_pred_path,
        threshold,
        args.prefer_existing_external_pred,
    )

    outputs = {
        "Train": (train_df, train_pred, train_metric),
        "Internal Test": (test_df, test_pred, test_metric),
        "Duke External": (duke_df, duke_pred, duke_metric),
        "Private External": (private_df, private_pred, private_metric),
    }
    _save_confusion_figures(outputs, output_dir)
    _save_roc_figure(outputs, output_dir)
    _save_pr_figure(outputs, output_dir)

    rows = []
    for name, (_, _, metric) in outputs.items():
        rows.append({"dataset": name, **metric})
    metrics_df = pd.DataFrame(rows)
    metrics_path = output_dir / "metrics_summary.csv"
    metrics_df.to_csv(metrics_path, index=False)

    for name, (df, _, _) in outputs.items():
        slug = name.lower().replace(" ", "_")
        df.to_csv(output_dir / f"{slug}_predictions_with_pred.csv", index=False)

    dataset_paths = _write_dataset_workbooks(
        output_dir,
        run_dir,
        {
            "Train": train_df,
            "Internal Test": test_df,
            "Duke External": duke_df,
            "Private External": private_df,
        },
        {
            "Duke External": args.duke_metadata,
            "Private External": args.private_metadata,
        },
    )

    summary = {
        "run_dir": run_dir,
        "output_dir": output_dir,
        "threshold": threshold,
        "threshold_source": threshold_source,
        "external_label_source": args.external_label_source,
        "paths": {
            "train_predictions": train_pred_path,
            "test_predictions": test_pred_path,
            "duke_predictions": duke_pred_path,
            "private_predictions": private_pred_path,
            "metrics": metrics_path,
            "confusion_matrix": output_dir / "confusion_matrices_train_test_external.png",
            "roc": output_dir / "roc_curves_train_test_external.png",
            "pr": output_dir / "pr_curves_train_test_external.png",
            "dataset_workbooks": dataset_paths,
        },
        "metrics": rows,
    }
    with (output_dir / "report_summary.json").open("w", encoding="utf-8") as f:
        json.dump(_json_safe(summary), f, ensure_ascii=False, indent=2)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate train/test/external result tables and confusion matrices."
    )
    parser.add_argument("--run_dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--train_predictions", type=Path, default=None)
    parser.add_argument("--test_predictions", type=Path, default=None)
    parser.add_argument("--duke_predictions", type=Path, default=None)
    parser.add_argument("--private_predictions", type=Path, default=None)
    parser.add_argument("--duke_metadata", type=Path, default=Path("dataset/Duke/dataset_metadata.csv"))
    parser.add_argument("--private_metadata", type=Path, default=Path("dataset/Private/dataset_metadata.csv"))
    parser.add_argument("--data_dir", type=Path, default=Path("dataset"))
    parser.add_argument("--eval_config", type=Path, default=DEFAULT_EVAL_CONFIG)
    parser.add_argument(
        "--model_path",
        type=Path,
        default=DEFAULT_RUN_DIR / "fold_1" / "best_model_fold_1.pth",
    )
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument(
        "--external_label_source",
        choices=["prediction_csv", "metadata_pCR"],
        default="prediction_csv",
    )
    parser.add_argument(
        "--prefer_existing_external_pred",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use pred column from external prediction CSVs when present.",
    )
    parser.add_argument(
        "--run_missing_external",
        action="store_true",
        help="Run run_external_validation.py when Duke/Private predictions are missing.",
    )
    return parser.parse_args()


def main() -> None:
    summary = generate_report(parse_args())
    print(json.dumps(_json_safe(summary["paths"]), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
