from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parent
if __name__ == "__main__":
    cwd = Path.cwd().resolve()
    sys.path[:] = [
        entry
        for entry in sys.path
        if Path(entry or cwd).resolve() != PROJECT_DIR
    ]

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

if __name__ == "__main__":
    sys.path.insert(0, str(PROJECT_DIR))

from utils.metadata import read_metadata_csv


DEFAULT_NUMERIC = ["age", "ER", "PR", "her-2", "Ki-67"]
DEFAULT_CATEGORICAL = ["T period", "N_period", "clinicalperiod"]
LEAKAGE_COLUMNS = {"label", "pCR", "p", "MP"}


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _make_one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _clean_stage_text(value):
    if pd.isna(value):
        return np.nan
    text = str(value).replace("\t", "").strip()
    text = (
        text.replace("Ⅰ", "I")
        .replace("Ⅱ", "II")
        .replace("Ⅲ", "III")
        .replace("Ⅳ", "IV")
    )
    return text if text else np.nan


def _read_predictions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"id", "label", "prob"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    out = df[["id", "label", "prob"]].copy()
    out["id"] = out["id"].astype(str)
    out["label"] = out["label"].astype(int)
    out["mri_prob"] = out["prob"].astype(float)
    return out.drop(columns=["prob"])


def _prepare_clinical_frame(metadata_path: Path, ids: pd.Series) -> tuple[pd.DataFrame, list[str], list[str]]:
    metadata = read_metadata_csv(str(metadata_path))
    if "id" not in metadata.columns:
        raise ValueError(f"{metadata_path} does not contain an id column.")
    metadata["id"] = metadata["id"].astype(str)

    numeric_cols = [col for col in DEFAULT_NUMERIC if col in metadata.columns]
    categorical_cols = [col for col in DEFAULT_CATEGORICAL if col in metadata.columns]
    if not numeric_cols and not categorical_cols:
        raise ValueError("No default clinical columns were found in metadata.")

    feature_df = pd.DataFrame({"id": ids.astype(str)})
    feature_df = feature_df.merge(
        metadata[["id", *numeric_cols, *categorical_cols]], on="id", how="left"
    )
    missing_rows = feature_df[numeric_cols + categorical_cols].isna().all(axis=1)
    if bool(missing_rows.any()):
        examples = feature_df.loc[missing_rows, "id"].head(5).tolist()
        raise ValueError(
            f"{int(missing_rows.sum())} prediction IDs have no clinical metadata, examples={examples}"
        )
    for col in numeric_cols:
        feature_df[col] = pd.to_numeric(feature_df[col], errors="coerce")
    for col in categorical_cols:
        feature_df[col] = feature_df[col].map(_clean_stage_text)
    return feature_df[numeric_cols + categorical_cols], numeric_cols, categorical_cols


def _build_lr_pipeline(numeric_cols: list[str], categorical_cols: list[str], c_value: float) -> Pipeline:
    transformers = []
    if numeric_cols:
        transformers.append(
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_cols,
            )
        )
    if categorical_cols:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", _make_one_hot_encoder()),
                    ]
                ),
                categorical_cols,
            )
        )
    return Pipeline(
        [
            ("preprocess", ColumnTransformer(transformers)),
            (
                "lr",
                LogisticRegression(
                    C=float(c_value),
                    class_weight="balanced",
                    solver="liblinear",
                    max_iter=5000,
                    random_state=42,
                ),
            ),
        ]
    )


def _best_c_by_inner_auc(X: pd.DataFrame, y: np.ndarray, numeric_cols: list[str], categorical_cols: list[str], seed: int) -> float:
    c_grid = [0.01, 0.03, 0.1, 0.3, 1.0]
    inner = StratifiedKFold(n_splits=4, shuffle=True, random_state=seed)
    best_c = c_grid[0]
    best_auc = -np.inf
    for c_value in c_grid:
        prob = np.zeros(len(y), dtype=float)
        for train_idx, val_idx in inner.split(X, y):
            model = _build_lr_pipeline(numeric_cols, categorical_cols, c_value)
            model.fit(X.iloc[train_idx], y[train_idx])
            prob[val_idx] = model.predict_proba(X.iloc[val_idx])[:, 1]
        auc = roc_auc_score(y, prob)
        if auc > best_auc:
            best_auc = float(auc)
            best_c = float(c_value)
    return best_c


def _oof_clinical_and_fusion(
    clinical_x: pd.DataFrame,
    mri_prob: np.ndarray,
    y: np.ndarray,
    numeric_cols: list[str],
    categorical_cols: list[str],
    seed: int,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    clinical_prob = np.zeros(len(y), dtype=float)
    fusion_prob = np.zeros(len(y), dtype=float)
    fold_rows = []

    fusion_x = clinical_x.copy()
    fusion_x["mri_prob"] = mri_prob
    fusion_numeric_cols = ["mri_prob", *numeric_cols]

    for fold, (train_idx, val_idx) in enumerate(outer.split(clinical_x, y), start=1):
        c_clin = _best_c_by_inner_auc(
            clinical_x.iloc[train_idx],
            y[train_idx],
            numeric_cols,
            categorical_cols,
            seed + fold,
        )
        c_fusion = _best_c_by_inner_auc(
            fusion_x.iloc[train_idx],
            y[train_idx],
            fusion_numeric_cols,
            categorical_cols,
            seed + 100 + fold,
        )
        clinical_model = _build_lr_pipeline(numeric_cols, categorical_cols, c_clin)
        fusion_model = _build_lr_pipeline(fusion_numeric_cols, categorical_cols, c_fusion)
        clinical_model.fit(clinical_x.iloc[train_idx], y[train_idx])
        fusion_model.fit(fusion_x.iloc[train_idx], y[train_idx])
        clinical_prob[val_idx] = clinical_model.predict_proba(clinical_x.iloc[val_idx])[:, 1]
        fusion_prob[val_idx] = fusion_model.predict_proba(fusion_x.iloc[val_idx])[:, 1]
        fold_rows.append(
            {
                "fold": int(fold),
                "n_train": int(len(train_idx)),
                "n_val": int(len(val_idx)),
                "clinical_C": float(c_clin),
                "fusion_C": float(c_fusion),
                "clinical_auc": float(roc_auc_score(y[val_idx], clinical_prob[val_idx])),
                "fusion_auc": float(roc_auc_score(y[val_idx], fusion_prob[val_idx])),
            }
        )
    return clinical_prob, fusion_prob, fold_rows


def _binary_metrics(y_true: np.ndarray, prob: np.ndarray, threshold: float) -> dict[str, Any]:
    pred = (prob >= float(threshold)).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if tp + fn else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    ppv = tp / (tp + fp) if tp + fp else 0.0
    npv = tn / (tn + fn) if tn + fn else 0.0
    return {
        "threshold": float(threshold),
        "auc": float(roc_auc_score(y_true, prob)),
        "pr_auc": float(average_precision_score(y_true, prob)),
        "brier": float(brier_score_loss(y_true, prob)),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "sensitivity": float(sens),
        "specificity": float(spec),
        "ppv": float(ppv),
        "npv": float(npv),
        "accuracy": float((tp + tn) / len(y_true)),
        "f1": float(2.0 * ppv * sens / (ppv + sens) if ppv + sens else 0.0),
        "balanced_accuracy": float((sens + spec) / 2.0),
        "predicted_positive": int(pred.sum()),
    }


def _best_threshold(y_true: np.ndarray, prob: np.ndarray, mode: str = "youden") -> float:
    thresholds = np.unique(np.concatenate([prob, [0.0, 0.5, 1.0]]))
    best_score = -np.inf
    best_threshold = 0.5
    for threshold in thresholds:
        metrics = _binary_metrics(y_true, prob, float(threshold))
        if mode == "f1":
            score = metrics["f1"]
        else:
            score = metrics["sensitivity"] + metrics["specificity"] - 1.0
        if score > best_score or (score == best_score and threshold > best_threshold):
            best_score = score
            best_threshold = float(threshold)
    return best_threshold


def _net_benefit(y_true: np.ndarray, prob: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    n = float(len(y_true))
    out = []
    for threshold in thresholds:
        pred_pos = prob >= threshold
        tp = float(np.logical_and(pred_pos, y_true == 1).sum()) / n
        fp = float(np.logical_and(pred_pos, y_true == 0).sum()) / n
        out.append(tp - fp * threshold / (1.0 - threshold))
    return np.asarray(out)


def _plot_roc_pr(pred_df: pd.DataFrame, out_path: Path) -> None:
    y = pred_df["label"].to_numpy(int)
    models = [
        ("MRI", "mri_prob", "#2563eb"),
        ("Clinical", "clinical_prob", "#059669"),
        ("Fusion", "fusion_prob", "#dc2626"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.4), dpi=220)
    ax = axes[0]
    for name, col, color in models:
        prob = pred_df[col].to_numpy(float)
        fpr, tpr, _ = roc_curve(y, prob)
        ax.plot(fpr, tpr, color=color, linewidth=2.3, label=f"{name} AUC={roc_auc_score(y, prob):.3f}")
    ax.plot([0, 1], [0, 1], color="#8c8c8c", linestyle="--", linewidth=1.2)
    ax.set_title("External ROC")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(True, color="#e5e7eb", linewidth=0.8)
    ax.legend(frameon=True, loc="lower right")

    ax = axes[1]
    prevalence = float(y.mean())
    for name, col, color in models:
        prob = pred_df[col].to_numpy(float)
        precision, recall, _ = precision_recall_curve(y, prob)
        ap = average_precision_score(y, prob)
        ax.plot(recall, precision, color=color, linewidth=2.3, label=f"{name} AP={ap:.3f}")
    ax.axhline(prevalence, color="#8c8c8c", linestyle="--", linewidth=1.2, label=f"Prevalence={prevalence:.3f}")
    ax.set_title("External Precision-Recall")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(True, color="#e5e7eb", linewidth=0.8)
    ax.legend(frameon=True, loc="lower left")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_dca(pred_df: pd.DataFrame, out_path: Path, out_csv: Path) -> None:
    y = pred_df["label"].to_numpy(int)
    thresholds = np.arange(0.01, 0.80 + 0.005, 0.01)
    prevalence = float(y.mean())
    dca = pd.DataFrame(
        {
            "threshold": thresholds,
            "none": np.zeros_like(thresholds),
            "all": prevalence - (1.0 - prevalence) * thresholds / (1.0 - thresholds),
            "mri": _net_benefit(y, pred_df["mri_prob"].to_numpy(float), thresholds),
            "clinical": _net_benefit(y, pred_df["clinical_prob"].to_numpy(float), thresholds),
            "fusion": _net_benefit(y, pred_df["fusion_prob"].to_numpy(float), thresholds),
        }
    )
    dca.to_csv(out_csv, index=False)
    fig, ax = plt.subplots(figsize=(7.5, 5.4), dpi=220)
    styles = {
        "none": ("None", "#6b7280", "-", 1.7),
        "all": ("All", "#111827", "--", 1.5),
        "mri": (f"MRI AUC={roc_auc_score(y, pred_df['mri_prob']):.3f}", "#2563eb", "-", 2.2),
        "clinical": (f"Clinical AUC={roc_auc_score(y, pred_df['clinical_prob']):.3f}", "#059669", "-", 2.2),
        "fusion": (f"Fusion AUC={roc_auc_score(y, pred_df['fusion_prob']):.3f}", "#dc2626", "-", 2.8),
    }
    for col, (label, color, linestyle, linewidth) in styles.items():
        ax.plot(dca["threshold"], dca[col], label=label, color=color, linestyle=linestyle, linewidth=linewidth)
    ax.axhline(0, color="#9ca3af", linewidth=0.8)
    ax.set_xlim(0.01, 0.80)
    ax.set_title("External Decision Curve Analysis")
    ax.set_xlabel("Threshold probability")
    ax.set_ylabel("Net benefit")
    ax.grid(True, color="#e5e7eb", linewidth=0.8)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_confusion(metrics_by_model: dict[str, dict[str, Any]], out_path: Path) -> None:
    names = ["MRI", "Clinical", "Fusion"]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), dpi=220)
    for ax, name in zip(axes, names):
        m = metrics_by_model[name]
        mat = np.asarray([[m["tn"], m["fp"]], [m["fn"], m["tp"]]], dtype=int)
        ax.imshow(mat, cmap="Blues")
        for (i, j), value in np.ndenumerate(mat):
            ax.text(j, i, str(value), ha="center", va="center", fontsize=13, fontweight="bold")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Pred 0", "Pred 1"])
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["True 0", "True 1"])
        ax.set_title(f"{name}\nT={m['threshold']:.3f}, ACC={m['accuracy']:.3f}")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def make_report(input_dir: Path, output_dir: Path, seed: int) -> dict[str, Any]:
    pred = _read_predictions(input_dir / "predictions.csv")
    clinical_x, numeric_cols, categorical_cols = _prepare_clinical_frame(
        input_dir / "dataset_metadata.csv",
        pred["id"],
    )
    y = pred["label"].to_numpy(int)
    mri_prob = pred["mri_prob"].to_numpy(float)
    clinical_prob, fusion_prob, fold_rows = _oof_clinical_and_fusion(
        clinical_x,
        mri_prob,
        y,
        numeric_cols,
        categorical_cols,
        seed,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    out_pred = pred.copy()
    out_pred["clinical_prob"] = clinical_prob
    out_pred["fusion_prob"] = fusion_prob

    models = {
        "MRI": mri_prob,
        "Clinical": clinical_prob,
        "Fusion": fusion_prob,
    }
    metrics_rows = []
    adaptive_metrics = {}
    fixed_metrics = {}
    for name, prob in models.items():
        adaptive_threshold = _best_threshold(y, prob, "youden")
        f1_threshold = _best_threshold(y, prob, "f1")
        adaptive = _binary_metrics(y, prob, adaptive_threshold)
        fixed = _binary_metrics(y, prob, 0.5)
        f1_best = _binary_metrics(y, prob, f1_threshold)
        adaptive_metrics[name] = adaptive
        fixed_metrics[name] = fixed
        for mode, row in [("adaptive_youden", adaptive), ("fixed_0.5", fixed), ("adaptive_f1", f1_best)]:
            metrics_rows.append({"model": name, "threshold_mode": mode, **row})
        out_pred[f"{name.lower()}_pred_adaptive"] = (prob >= adaptive_threshold).astype(int)
        out_pred[f"{name.lower()}_pred_fixed_0_5"] = (prob >= 0.5).astype(int)

    out_pred.to_csv(output_dir / "external_mri_clinical_fusion_predictions.csv", index=False)
    pd.DataFrame(metrics_rows).to_csv(output_dir / "external_metrics_summary.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(output_dir / "external_lr_oof_folds.csv", index=False)
    _plot_roc_pr(out_pred, output_dir / "external_roc_pr_mri_clinical_fusion.png")
    _plot_dca(
        out_pred,
        output_dir / "external_dca_mri_clinical_fusion.png",
        output_dir / "external_dca_mri_clinical_fusion.csv",
    )
    _plot_confusion(
        adaptive_metrics,
        output_dir / "external_confusion_matrices_adaptive_youden.png",
    )
    _plot_confusion(
        fixed_metrics,
        output_dir / "external_confusion_matrices_fixed_0_5.png",
    )
    summary = {
        "input_dir": input_dir,
        "output_dir": output_dir,
        "num_samples": int(len(y)),
        "positive": int(y.sum()),
        "negative": int((y == 0).sum()),
        "clinical_numeric_features": numeric_cols,
        "clinical_categorical_features": categorical_cols,
        "leakage_columns_excluded": sorted(LEAKAGE_COLUMNS),
        "metrics_adaptive_youden": adaptive_metrics,
        "metrics_fixed_0_5": fixed_metrics,
        "outputs": {
            "predictions": output_dir / "external_mri_clinical_fusion_predictions.csv",
            "metrics": output_dir / "external_metrics_summary.csv",
            "roc_pr": output_dir / "external_roc_pr_mri_clinical_fusion.png",
            "dca": output_dir / "external_dca_mri_clinical_fusion.png",
            "confusion_adaptive": output_dir / "external_confusion_matrices_adaptive_youden.png",
            "confusion_fixed": output_dir / "external_confusion_matrices_fixed_0_5.png",
        },
    }
    with (output_dir / "external_report_summary.json").open("w", encoding="utf-8") as f:
        json.dump(_json_safe(summary), f, ensure_ascii=False, indent=2)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Create external validation figures and metrics.")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else input_dir / "figures_external_report"
    summary = make_report(input_dir, output_dir, args.seed)
    print(json.dumps(_json_safe(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
