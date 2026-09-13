from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

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

from utils.metadata import read_metadata_csv


DEFAULT_FOLD_DIR = (
    PROJECT_DIR
    / "output"
    / "MRIUpper_NoTTA_Seed2040_Random"
    / "fold_1"
)


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    kind: str
    aliases: tuple[str, ...]


FEATURE_SPECS = (
    FeatureSpec("age", "numeric", ("age", "Age", "AGE", "age_at_diagnosis", "年龄")),
    FeatureSpec("ER", "numeric", ("ER", "er", "ER_status", "ER表达：阴为0，阳为1")),
    FeatureSpec("PR", "numeric", ("PR", "pr", "PR_status", "PR表达：阴为0，阳为1")),
    FeatureSpec("HER2", "numeric", ("HER2", "her-2", "HER-2", "HER2_status", "her-2阴：0；阳：1")),
    FeatureSpec("Ki67", "numeric", ("Ki-67", "Ki67", "KI67", "Ki-67表达情况： <30%为0，>=30%为1")),
    FeatureSpec("T_stage", "categorical", ("T period", "T_period", "stage_tum_s", "T分期")),
    FeatureSpec("N_stage", "categorical", ("N_period", "N period", "N_stage", "N分期")),
    FeatureSpec("Clinical_stage", "categorical", ("clinicalperiod", "clinical_period", "Clinical_stage", "临床分期")),
    FeatureSpec("Tumor_grade", "categorical", ("Tumor_Grade", "Nottingham_grade", "grade")),
)

MODEL_SPECS = (
    ("MRI", "mri_prob", "#2563eb"),
    ("Clinical", "clinical_prob", "#059669"),
    ("Fusion", "fusion_prob", "#dc2626"),
)

DATASET_SPECS = (
    ("Train", "best_epoch_train_predictions.xlsx", "dataset/dataset_metadata.csv"),
    ("Internal Val", "best_epoch_val_predictions.xlsx", "dataset/dataset_metadata.csv"),
    ("Duke", "predictions_duke.xlsx", "dataset/Duke/dataset_metadata.csv"),
    ("Private", "predictions_private.xlsx", "dataset/Private/dataset_metadata.csv"),
)


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


def _read_prediction_xlsx(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_excel(path)
    required = {"id", "label", "prob"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} missing required columns: {sorted(missing)}")
    out = df.copy()
    out["id"] = out["id"].astype(str)
    out["label"] = pd.to_numeric(out["label"], errors="raise").astype(int)
    out["mri_prob"] = pd.to_numeric(out["prob"], errors="raise").astype(float)
    if not set(out["label"].unique()).issubset({0, 1}):
        raise ValueError(f"{path} label column must be binary 0/1.")
    front = ["id", "label", "mri_prob"]
    if "pred" in out.columns:
        out["source_pred"] = pd.to_numeric(out["pred"], errors="coerce").astype("Int64")
        front.append("source_pred")
    keep = front + [c for c in out.columns if c not in {*front, "prob", "pred"}]
    return out[keep]


def _read_metadata(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    else:
        df = read_metadata_csv(str(path))
    if "id" not in df.columns:
        raise ValueError(f"{path} missing required column: id")
    df = df.copy()
    df["id"] = df["id"].astype(str)
    return df.drop_duplicates(subset=["id"], keep="first")


def _resolve_column(columns: pd.Index, aliases: tuple[str, ...]) -> str | None:
    for alias in aliases:
        if alias in columns:
            return str(alias)
    lower_to_column = {str(column).lower(): str(column) for column in columns}
    for alias in aliases:
        column = lower_to_column.get(alias.lower())
        if column is not None:
            return column
    return None


def _clean_category(value: Any) -> Any:
    if pd.isna(value):
        return np.nan
    text = str(value).replace("\t", "").strip()
    if not text:
        return np.nan
    replacements = {
        "一期": "I",
        "二期": "II",
        "三期": "III",
        "四期": "IV",
        "Ⅰ": "I",
        "Ⅱ": "II",
        "Ⅲ": "III",
        "Ⅳ": "IV",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def _feature_frame(
    predictions: pd.DataFrame,
    metadata: pd.DataFrame,
    dataset_name: str,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    merged = predictions[["id"]].merge(metadata, on="id", how="left", indicator=True)
    missing_metadata = merged["_merge"].eq("left_only")
    if bool(missing_metadata.any()):
        examples = merged.loc[missing_metadata, "id"].head(10).tolist()
        raise ValueError(
            f"{dataset_name}: {int(missing_metadata.sum())} prediction IDs have no clinical metadata. "
            f"Examples: {examples}"
        )
    merged = merged.drop(columns=["_merge"])

    features = pd.DataFrame(index=predictions.index)
    manifest_rows: list[dict[str, Any]] = []
    for spec in FEATURE_SPECS:
        source_col = _resolve_column(metadata.columns, spec.aliases)
        if source_col is None:
            features[spec.name] = np.nan
            manifest_rows.append(
                {
                    "dataset": dataset_name,
                    "feature": spec.name,
                    "kind": spec.kind,
                    "source_column": None,
                    "non_missing": 0,
                }
            )
            continue
        series = merged[source_col]
        if spec.kind == "numeric":
            series = pd.to_numeric(series, errors="coerce")
        else:
            series = series.map(_clean_category)
        features[spec.name] = series
        manifest_rows.append(
            {
                "dataset": dataset_name,
                "feature": spec.name,
                "kind": spec.kind,
                "source_column": source_col,
                "non_missing": int(series.notna().sum()),
            }
        )
    return features, manifest_rows


def _make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _select_usable_features(train_x: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    numeric_cols: list[str] = []
    categorical_cols: list[str] = []
    dropped_cols: list[str] = []
    for spec in FEATURE_SPECS:
        if not train_x[spec.name].notna().any():
            dropped_cols.append(spec.name)
            continue
        if spec.kind == "numeric":
            numeric_cols.append(spec.name)
        else:
            categorical_cols.append(spec.name)
    if not numeric_cols and not categorical_cols:
        raise ValueError("No usable clinical features were found in the training metadata.")
    return numeric_cols, categorical_cols, dropped_cols


def _build_lr_pipeline(
    numeric_cols: list[str],
    categorical_cols: list[str],
    c_value: float,
    seed: int,
) -> Pipeline:
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
                    random_state=seed,
                ),
            ),
        ]
    )


def _safe_auc(y_true: np.ndarray, prob: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, prob))


def _safe_ap(y_true: np.ndarray, prob: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, prob))


def _best_c_by_inner_auc(
    x: pd.DataFrame,
    y: np.ndarray,
    numeric_cols: list[str],
    categorical_cols: list[str],
    seed: int,
) -> float:
    class_counts = np.bincount(y.astype(int), minlength=2)
    n_splits = int(min(5, class_counts.min()))
    if n_splits < 2:
        return 1.0
    c_grid = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]
    inner = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    best_c = 1.0
    best_auc = -np.inf
    for c_value in c_grid:
        prob = np.zeros(len(y), dtype=float)
        for train_idx, val_idx in inner.split(x, y):
            model = _build_lr_pipeline(numeric_cols, categorical_cols, c_value, seed)
            model.fit(x.iloc[train_idx], y[train_idx])
            prob[val_idx] = model.predict_proba(x.iloc[val_idx])[:, 1]
        auc_value = _safe_auc(y, prob)
        if np.isfinite(auc_value) and auc_value > best_auc:
            best_auc = auc_value
            best_c = float(c_value)
    return best_c


def _fit_models(
    train_x: pd.DataFrame,
    train_y: np.ndarray,
    train_mri_prob: np.ndarray,
    numeric_cols: list[str],
    categorical_cols: list[str],
    seed: int,
) -> tuple[Pipeline, Pipeline, dict[str, Any]]:
    clinical_c = _best_c_by_inner_auc(train_x, train_y, numeric_cols, categorical_cols, seed)
    clinical_model = _build_lr_pipeline(numeric_cols, categorical_cols, clinical_c, seed)
    clinical_model.fit(train_x, train_y)

    fusion_x = train_x.copy()
    fusion_x["mri_prob"] = train_mri_prob
    fusion_numeric_cols = ["mri_prob", *numeric_cols]
    fusion_c = _best_c_by_inner_auc(
        fusion_x,
        train_y,
        fusion_numeric_cols,
        categorical_cols,
        seed + 101,
    )
    fusion_model = _build_lr_pipeline(fusion_numeric_cols, categorical_cols, fusion_c, seed + 101)
    fusion_model.fit(fusion_x, train_y)
    return clinical_model, fusion_model, {"clinical_C": clinical_c, "fusion_C": fusion_c}


def _append_model_probs(
    pred: pd.DataFrame,
    x: pd.DataFrame,
    clinical_model: Pipeline,
    fusion_model: Pipeline,
) -> pd.DataFrame:
    out = pred.copy()
    out["clinical_prob"] = clinical_model.predict_proba(x)[:, 1]
    fusion_x = x.copy()
    fusion_x["mri_prob"] = out["mri_prob"].to_numpy(float)
    out["fusion_prob"] = fusion_model.predict_proba(fusion_x)[:, 1]
    return out


def _binary_metrics(y_true: np.ndarray, prob: np.ndarray, threshold: float) -> dict[str, Any]:
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if tp + fn else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    ppv = tp / (tp + fp) if tp + fp else 0.0
    npv = tn / (tn + fn) if tn + fn else 0.0
    f1 = 2.0 * ppv * sens / (ppv + sens) if ppv + sens else 0.0
    return {
        "threshold": float(threshold),
        "auc": _safe_auc(y_true, prob),
        "average_precision": _safe_ap(y_true, prob),
        "brier": float(brier_score_loss(y_true, prob)),
        "accuracy": float((tp + tn) / len(y_true)),
        "sensitivity": float(sens),
        "specificity": float(spec),
        "ppv": float(ppv),
        "npv": float(npv),
        "f1": float(f1),
        "balanced_accuracy": float((sens + spec) / 2.0),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "predicted_positive": int(pred.sum()),
    }


def _best_threshold_youden(y_true: np.ndarray, prob: np.ndarray) -> float:
    thresholds = np.unique(np.concatenate([prob, [0.0, 0.5, 1.0]]))
    best_score = -np.inf
    best_threshold = 0.5
    for threshold in thresholds:
        m = _binary_metrics(y_true, prob, float(threshold))
        score = m["sensitivity"] + m["specificity"] - 1.0
        if score > best_score or (score == best_score and threshold > best_threshold):
            best_score = score
            best_threshold = float(threshold)
    return best_threshold


def _bootstrap_ci(
    y_true: np.ndarray,
    prob: np.ndarray,
    threshold: float,
    seed: int,
    n_bootstrap: int,
) -> dict[str, tuple[float | None, float | None]]:
    keys = [
        "auc",
        "average_precision",
        "accuracy",
        "sensitivity",
        "specificity",
        "ppv",
        "npv",
        "f1",
        "brier",
    ]
    if n_bootstrap <= 0:
        return {key: (None, None) for key in keys}
    rng = np.random.default_rng(seed)
    values = {key: [] for key in keys}
    n = len(y_true)
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        sample_y = y_true[idx]
        if len(np.unique(sample_y)) < 2:
            continue
        sample_prob = prob[idx]
        m = _binary_metrics(sample_y, sample_prob, threshold)
        for key in keys:
            values[key].append(m[key])
    return {
        key: (
            float(np.percentile(vals, 2.5)) if vals else None,
            float(np.percentile(vals, 97.5)) if vals else None,
        )
        for key, vals in values.items()
    }


def _metrics_row(
    dataset: str,
    model: str,
    threshold_mode: str,
    threshold_source: str,
    threshold: float,
    y_true: np.ndarray,
    prob: np.ndarray,
    seed: int,
    n_bootstrap: int,
) -> dict[str, Any]:
    row = {
        "dataset": dataset,
        "model": model,
        "threshold_mode": threshold_mode,
        "threshold_source": threshold_source,
        "n": int(len(y_true)),
        "positive": int(y_true.sum()),
        "negative": int((y_true == 0).sum()),
        **_binary_metrics(y_true, prob, threshold),
    }
    ci = _bootstrap_ci(y_true, prob, threshold, seed, n_bootstrap)
    for key, (low, high) in ci.items():
        row[f"{key}_ci_low"] = low
        row[f"{key}_ci_high"] = high
    return row


def _save_figure(fig: plt.Figure, png_path: Path) -> None:
    fig.tight_layout()
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(png_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _slug(text: str) -> str:
    return (
        text.lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("+", "plus")
        .replace(".", "_")
    )


def _plot_roc_pr(pred_df: pd.DataFrame, dataset_name: str, out_path: Path) -> None:
    y = pred_df["label"].to_numpy(int)
    prevalence = float(y.mean())
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))

    ax = axes[0]
    for name, col, color in MODEL_SPECS:
        prob = pred_df[col].to_numpy(float)
        fpr, tpr, _ = roc_curve(y, prob)
        ax.plot(fpr, tpr, color=color, linewidth=2.2, label=f"{name} AUC={_safe_auc(y, prob):.3f}")
    ax.plot([0, 1], [0, 1], color="#8c8c8c", linestyle="--", linewidth=1.0)
    ax.set_title(f"{dataset_name} ROC")
    ax.set_xlabel("False Positive Rate (1-Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(True, color="#e5e7eb", linewidth=0.8)
    ax.legend(frameon=True, loc="lower right", fontsize=8)

    ax = axes[1]
    for name, col, color in MODEL_SPECS:
        prob = pred_df[col].to_numpy(float)
        precision, recall, _ = precision_recall_curve(y, prob)
        ax.plot(recall, precision, color=color, linewidth=2.2, label=f"{name} AP={_safe_ap(y, prob):.3f}")
    ax.axhline(prevalence, color="#8c8c8c", linestyle="--", linewidth=1.0, label=f"Prevalence={prevalence:.3f}")
    ax.set_title(f"{dataset_name} Precision-Recall")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(True, color="#e5e7eb", linewidth=0.8)
    ax.legend(frameon=True, loc="lower left", fontsize=8)
    _save_figure(fig, out_path)


def _plot_confusion(pred_df: pd.DataFrame, dataset_name: str, thresholds: dict[str, float], out_path: Path) -> None:
    y = pred_df["label"].to_numpy(int)
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2))
    fig.suptitle(f"{dataset_name} Confusion Matrices", y=1.04, fontsize=14)
    for ax, (name, col, _) in zip(axes, MODEL_SPECS):
        prob = pred_df[col].to_numpy(float)
        threshold = thresholds[name]
        pred = (prob >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        mat = np.asarray([[tn, fn], [fp, tp]], dtype=int)
        ax.imshow(mat, cmap="Blues")
        cutoff = (mat.max() + mat.min()) / 2.0
        for (row, col_idx), value in np.ndenumerate(mat):
            color = "white" if value > cutoff else "#111827"
            ax.text(
                col_idx,
                row,
                str(int(value)),
                ha="center",
                va="center",
                fontsize=13,
                fontweight="bold",
                color=color,
            )
        m = _binary_metrics(y, prob, threshold)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Non-pCR", "pCR"])
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Non-pCR", "pCR"])
        ax.set_xlabel("Actual Treatment Response")
        ax.set_ylabel("Predicted Treatment Response")
        ax.set_title(f"{name}\nAUC={m['auc']:.3f}, ACC={m['accuracy']:.3f}, T={threshold:.3f}", fontsize=10)
        ax.tick_params(axis="both", labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#374151")
            spine.set_linewidth(0.8)
    _save_figure(fig, out_path)


def _net_benefit(y_true: np.ndarray, prob: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    n = float(len(y_true))
    out = []
    for threshold in thresholds:
        pred_pos = prob >= threshold
        tp = float(np.logical_and(pred_pos, y_true == 1).sum()) / n
        fp = float(np.logical_and(pred_pos, y_true == 0).sum()) / n
        out.append(tp - fp * threshold / (1.0 - threshold))
    return np.asarray(out)


def _plot_dca(pred_df: pd.DataFrame, dataset_name: str, out_path: Path, csv_path: Path) -> None:
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
    dca.to_csv(csv_path, index=False)

    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    styles = {
        "none": ("None", "#6b7280", "-", 1.7),
        "all": ("All", "#111827", "--", 1.5),
        "mri": (f"MRI AUC={_safe_auc(y, pred_df['mri_prob'].to_numpy(float)):.3f}", "#2563eb", "-", 2.1),
        "clinical": (
            f"Clinical AUC={_safe_auc(y, pred_df['clinical_prob'].to_numpy(float)):.3f}",
            "#059669",
            "-",
            2.1,
        ),
        "fusion": (
            f"Fusion AUC={_safe_auc(y, pred_df['fusion_prob'].to_numpy(float)):.3f}",
            "#dc2626",
            "-",
            2.6,
        ),
    }
    for col, (label, color, linestyle, linewidth) in styles.items():
        ax.plot(dca["threshold"], dca[col], label=label, color=color, linestyle=linestyle, linewidth=linewidth)
    ax.axhline(0, color="#9ca3af", linewidth=0.8)
    ax.set_xlim(0.01, 0.80)
    display_cols = ["none", "mri", "clinical", "fusion"]
    low = float(np.nanmin(dca[display_cols].to_numpy()))
    high = float(np.nanmax(dca[["all", *display_cols]].to_numpy()))
    ax.set_ylim(min(-0.05, low - 0.03), max(0.05, high + 0.03))
    ax.set_title(f"{dataset_name} Decision Curve Analysis")
    ax.set_xlabel("Threshold Probability")
    ax.set_ylabel("Net Benefit")
    ax.grid(True, color="#e5e7eb", linewidth=0.8)
    ax.legend(frameon=False, loc="best", fontsize=8)
    _save_figure(fig, out_path)


def _plot_radar(
    metrics_df: pd.DataFrame,
    dataset_name: str,
    threshold_mode: str,
    out_path: Path,
) -> None:
    axes_labels = ["AUC", "Sensitivity", "Specificity", "Accuracy", "PPV", "NPV"]
    metric_cols = ["auc", "sensitivity", "specificity", "accuracy", "ppv", "npv"]
    angles = np.linspace(0, 2 * np.pi, len(axes_labels), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6.2, 6.2), subplot_kw={"polar": True})
    rows = metrics_df[
        (metrics_df["dataset"] == dataset_name)
        & (metrics_df["threshold_mode"] == threshold_mode)
    ]
    for name, _, color in MODEL_SPECS:
        row = rows[rows["model"] == name]
        if row.empty:
            continue
        values = [float(row.iloc[0][col]) for col in metric_cols]
        values += values[:1]
        ax.plot(angles, values, color=color, linewidth=2.0, label=name)
        ax.scatter(angles, values, color=color, s=22)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(axes_labels)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["25%", "50%", "75%", "100%"])
    ax.set_ylim(0, 1)
    ax.set_title(f"{dataset_name} Metrics Radar")
    ax.grid(True, color="#d1d5db", linewidth=0.8)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.18), ncol=3, frameon=False)
    _save_figure(fig, out_path)


def _write_prediction_workbook(output_dir: Path, predictions: dict[str, pd.DataFrame]) -> Path:
    path = output_dir / "mri_clinical_fusion_predictions.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for dataset_name, df in predictions.items():
            sheet = dataset_name.replace(" ", "_")[:31]
            df.to_excel(writer, sheet_name=sheet, index=False)
    return path


def make_report(
    fold_dir: Path,
    output_dir: Path,
    threshold_mode: str,
    seed: int,
    n_bootstrap: int,
) -> dict[str, Any]:
    fold_dir = fold_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_predictions: dict[str, pd.DataFrame] = {}
    features: dict[str, pd.DataFrame] = {}
    manifest_rows: list[dict[str, Any]] = []

    for dataset_name, pred_name, metadata_name in DATASET_SPECS:
        pred = _read_prediction_xlsx(fold_dir / pred_name)
        metadata = _read_metadata(PROJECT_DIR / metadata_name)
        x, manifest = _feature_frame(pred, metadata, dataset_name)
        raw_predictions[dataset_name] = pred
        features[dataset_name] = x
        manifest_rows.extend(manifest)

    train_y = raw_predictions["Train"]["label"].to_numpy(int)
    numeric_cols, categorical_cols, dropped_cols = _select_usable_features(features["Train"])
    used_cols = [*numeric_cols, *categorical_cols]

    clinical_model, fusion_model, model_info = _fit_models(
        features["Train"][used_cols],
        train_y,
        raw_predictions["Train"]["mri_prob"].to_numpy(float),
        numeric_cols,
        categorical_cols,
        seed,
    )

    predictions: dict[str, pd.DataFrame] = {}
    for dataset_name, pred in raw_predictions.items():
        predictions[dataset_name] = _append_model_probs(
            pred,
            features[dataset_name][used_cols],
            clinical_model,
            fusion_model,
        )

    train_pred = predictions["Train"]
    thresholds = {
        name: _best_threshold_youden(train_y, train_pred[col].to_numpy(float))
        for name, col, _ in MODEL_SPECS
    }
    if threshold_mode == "fixed_0.5":
        confusion_thresholds = {name: 0.5 for name, _, _ in MODEL_SPECS}
        threshold_source = "fixed"
    else:
        confusion_thresholds = thresholds
        threshold_source = "train_youden"

    evaluation_names = ["Internal Val", "Duke", "Private"]
    overall = pd.concat(
        [predictions[name].assign(source_dataset=name) for name in evaluation_names],
        ignore_index=True,
    )
    all_eval_predictions = {**predictions, "Overall": overall}

    for dataset_name, df in list(all_eval_predictions.items()):
        for name, col, _ in MODEL_SPECS:
            df[f"{name.lower()}_pred_train_youden"] = (
                df[col].to_numpy(float) >= thresholds[name]
            ).astype(int)
            df[f"{name.lower()}_pred_fixed_0_5"] = (df[col].to_numpy(float) >= 0.5).astype(int)
        all_eval_predictions[dataset_name] = df
        if dataset_name in predictions:
            predictions[dataset_name] = df

    prediction_xlsx = _write_prediction_workbook(output_dir, predictions)
    overall.to_csv(output_dir / "overall_eval_mri_clinical_fusion_predictions.csv", index=False)

    metrics_rows = []
    for dataset_idx, (dataset_name, df) in enumerate(all_eval_predictions.items()):
        y = df["label"].to_numpy(int)
        for model_idx, (name, col, _) in enumerate(MODEL_SPECS):
            prob = df[col].to_numpy(float)
            metrics_rows.append(
                _metrics_row(
                    dataset_name,
                    name,
                    "train_youden",
                    "train",
                    thresholds[name],
                    y,
                    prob,
                    seed + dataset_idx * 100 + model_idx,
                    n_bootstrap,
                )
            )
            metrics_rows.append(
                _metrics_row(
                    dataset_name,
                    name,
                    "fixed_0.5",
                    "fixed",
                    0.5,
                    y,
                    prob,
                    seed + dataset_idx * 100 + model_idx + 50,
                    n_bootstrap,
                )
            )
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_csv = output_dir / "metrics_summary.csv"
    metrics_xlsx = output_dir / "metrics_summary.xlsx"
    metrics_df.to_csv(metrics_csv, index=False)
    metrics_df.to_excel(metrics_xlsx, index=False)
    pd.DataFrame(manifest_rows).to_csv(output_dir / "clinical_feature_manifest.csv", index=False)

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    plotted_names = ["Train", "Internal Val", "Duke", "Private", "Overall"]
    for dataset_name in plotted_names:
        df = all_eval_predictions[dataset_name]
        slug = _slug(dataset_name)
        _plot_roc_pr(df, dataset_name, figures_dir / f"{slug}_roc_pr_mri_clinical_fusion.png")
        _plot_confusion(
            df,
            dataset_name,
            confusion_thresholds,
            figures_dir / f"{slug}_confusion_matrices_{threshold_mode}.png",
        )
        _plot_dca(
            df,
            dataset_name,
            figures_dir / f"{slug}_dca_mri_clinical_fusion.png",
            figures_dir / f"{slug}_dca_mri_clinical_fusion.csv",
        )
        _plot_radar(
            metrics_df,
            dataset_name,
            "train_youden" if threshold_mode == "train_youden" else "fixed_0.5",
            figures_dir / f"{slug}_radar_mri_clinical_fusion.png",
        )

    summary = {
        "fold_dir": fold_dir,
        "output_dir": output_dir,
        "label_source": "The label column in each fold_1 xlsx is used as ground truth.",
        "overall_definition": "Overall = Internal Val + Duke + Private; Train is not included.",
        "threshold_mode_for_confusion": threshold_mode,
        "threshold_source": threshold_source,
        "train_youden_thresholds": thresholds,
        "model_info": model_info,
        "used_numeric_features": numeric_cols,
        "used_categorical_features": categorical_cols,
        "dropped_train_empty_features": dropped_cols,
        "outputs": {
            "prediction_workbook": prediction_xlsx,
            "overall_predictions_csv": output_dir / "overall_eval_mri_clinical_fusion_predictions.csv",
            "metrics_csv": metrics_csv,
            "metrics_xlsx": metrics_xlsx,
            "figures_dir": figures_dir,
            "overall_roc_pr": figures_dir / "overall_roc_pr_mri_clinical_fusion.png",
            "overall_confusion": figures_dir / f"overall_confusion_matrices_{threshold_mode}.png",
            "overall_dca": figures_dir / "overall_dca_mri_clinical_fusion.png",
            "overall_radar": figures_dir / "overall_radar_mri_clinical_fusion.png",
        },
    }
    with (output_dir / "report_summary.json").open("w", encoding="utf-8") as f:
        json.dump(_json_safe(summary), f, ensure_ascii=False, indent=2)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read fold_1 xlsx predictions and export MRI, clinical, and MRI+clinical "
            "fusion metrics, predictions, ROC/PR, confusion matrices, radar plots, and DCA."
        )
    )
    parser.add_argument("--fold_dir", type=Path, default=DEFAULT_FOLD_DIR)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument(
        "--threshold_mode",
        choices=["train_youden", "fixed_0.5"],
        default="train_youden",
        help="Threshold used by confusion matrices and default prediction labels.",
    )
    parser.add_argument("--seed", type=int, default=2040)
    parser.add_argument(
        "--n_bootstrap",
        type=int,
        default=1000,
        help="Bootstrap repeats for metric 95%% CIs. Use 0 to disable.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else args.fold_dir.resolve() / "mri_clinical_fusion_report"
    )
    summary = make_report(
        fold_dir=args.fold_dir,
        output_dir=output_dir,
        threshold_mode=args.threshold_mode,
        seed=args.seed,
        n_bootstrap=args.n_bootstrap,
    )
    print(json.dumps(_json_safe(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
