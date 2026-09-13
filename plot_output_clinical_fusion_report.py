from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
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


LEAKAGE_COLUMNS = {
    "label",
    "pCR",
    "PCR",
    "NAT response",
    "NAT_after_response",
    "NAT后反应（G2-G4=0、G5=1）",
}


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    kind: str
    aliases: tuple[str, ...]


FEATURE_SPECS = (
    FeatureSpec("age", "numeric", ("age", "Age", "年龄", "骞撮緞")),
    FeatureSpec("ER", "numeric", ("ER", "ER表达：阴为0，阳为1")),
    FeatureSpec("PR", "numeric", ("PR", "PR表达：阴为0，阳为1")),
    FeatureSpec("HER2", "numeric", ("HER2", "her-2", "HER-2", "her-2阴：0；阳：1")),
    FeatureSpec("Ki-67", "numeric", ("Ki-67", "Ki67", "KI67", "Ki-67表达情况： <30%为0，>=30%为1")),
    FeatureSpec("T_stage", "categorical", ("T period", "T分期", "stage_tum_s")),
    FeatureSpec("N_stage", "categorical", ("N_period", "N period", "N分期")),
    FeatureSpec("Clinical_stage", "categorical", ("clinicalperiod", "clinical_period", "临床分期")),
    FeatureSpec("Tumor_grade", "categorical", ("Tumor_Grade", "Nottingham_grade")),
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


def _make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


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
    if not set(out["label"].unique()).issubset({0, 1}):
        raise ValueError(f"{path} label column must be binary 0/1.")
    return out.drop(columns=["prob"])


def _resolve_column(columns: pd.Index, aliases: tuple[str, ...]) -> str | None:
    for alias in aliases:
        if alias in columns:
            return alias
    lower_to_column = {str(col).lower(): col for col in columns}
    for alias in aliases:
        column = lower_to_column.get(alias.lower())
        if column is not None:
            return str(column)
    return None


def _clean_category(value: Any) -> Any:
    if pd.isna(value):
        return np.nan
    text = str(value).replace("\t", "").strip()
    replacements = {
        "I期": "I",
        "II期": "II",
        "III期": "III",
        "IV期": "IV",
        "一期": "I",
        "二期": "II",
        "三期": "III",
        "四期": "IV",
        "Άρ": "I",
        "Άς": "II",
        "Άσ": "III",
        "Άτ": "IV",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text if text else np.nan


def _load_metadata(path: Path) -> pd.DataFrame:
    df = read_metadata_csv(str(path))
    if "id" not in df.columns:
        raise ValueError(f"{path} does not contain an id column.")
    df = df.copy()
    df["id"] = df["id"].astype(str)
    return df.drop_duplicates(subset=["id"], keep="first")


def _feature_frame(
    predictions: pd.DataFrame,
    metadata: pd.DataFrame,
    dataset_role: str,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    merged = predictions[["id"]].merge(metadata, on="id", how="left", indicator=True)
    missing_metadata = merged["_merge"].eq("left_only")
    if bool(missing_metadata.any()):
        examples = merged.loc[missing_metadata, "id"].head(8).tolist()
        raise ValueError(
            f"{int(missing_metadata.sum())} {dataset_role} prediction IDs have no clinical metadata; "
            f"examples={examples}"
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
                    "feature": spec.name,
                    "kind": spec.kind,
                    f"{dataset_role}_source_column": None,
                    f"{dataset_role}_non_missing": 0,
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
                "feature": spec.name,
                "kind": spec.kind,
                f"{dataset_role}_source_column": source_col,
                f"{dataset_role}_non_missing": int(series.notna().sum()),
            }
        )
    return features, manifest_rows


def _select_usable_features(
    train_x: pd.DataFrame,
    train_manifest: list[dict[str, Any]],
    test_manifest: list[dict[str, Any]],
) -> tuple[list[str], list[str], list[dict[str, Any]]]:
    train_sources = {row["feature"]: row for row in train_manifest}
    test_sources = {row["feature"]: row for row in test_manifest}
    numeric_cols: list[str] = []
    categorical_cols: list[str] = []
    resolved: list[dict[str, Any]] = []
    for spec in FEATURE_SPECS:
        used = bool(train_x[spec.name].notna().any())
        if used and spec.kind == "numeric":
            numeric_cols.append(spec.name)
        elif used:
            categorical_cols.append(spec.name)
        row = {
            "feature": spec.name,
            "kind": spec.kind,
            "used": used,
            **train_sources.get(spec.name, {}),
            **test_sources.get(spec.name, {}),
        }
        resolved.append(row)
    if not numeric_cols and not categorical_cols:
        raise ValueError("No usable clinical features were found in the training metadata.")
    return numeric_cols, categorical_cols, resolved


def _build_lr_pipeline(
    numeric_cols: list[str],
    categorical_cols: list[str],
    c_value: float,
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
                    random_state=42,
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
    best_c = c_grid[0]
    best_auc = -np.inf
    for c_value in c_grid:
        prob = np.zeros(len(y), dtype=float)
        for train_idx, val_idx in inner.split(x, y):
            model = _build_lr_pipeline(numeric_cols, categorical_cols, c_value)
            model.fit(x.iloc[train_idx], y[train_idx])
            prob[val_idx] = model.predict_proba(x.iloc[val_idx])[:, 1]
        auc = _safe_auc(y, prob)
        if np.isfinite(auc) and auc > best_auc:
            best_auc = float(auc)
            best_c = float(c_value)
    return float(best_c)


def _fit_predict_logistic(
    train_x: pd.DataFrame,
    test_x: pd.DataFrame,
    train_y: np.ndarray,
    train_mri_prob: np.ndarray,
    test_mri_prob: np.ndarray,
    numeric_cols: list[str],
    categorical_cols: list[str],
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    clinical_c = _best_c_by_inner_auc(train_x, train_y, numeric_cols, categorical_cols, seed)
    clinical_model = _build_lr_pipeline(numeric_cols, categorical_cols, clinical_c)
    clinical_model.fit(train_x, train_y)

    train_fusion_x = train_x.copy()
    test_fusion_x = test_x.copy()
    train_fusion_x["mri_prob"] = train_mri_prob
    test_fusion_x["mri_prob"] = test_mri_prob
    fusion_numeric_cols = ["mri_prob", *numeric_cols]
    fusion_c = _best_c_by_inner_auc(
        train_fusion_x,
        train_y,
        fusion_numeric_cols,
        categorical_cols,
        seed + 101,
    )
    fusion_model = _build_lr_pipeline(fusion_numeric_cols, categorical_cols, fusion_c)
    fusion_model.fit(train_fusion_x, train_y)

    train_prob = pd.DataFrame(
        {
            "clinical_prob": clinical_model.predict_proba(train_x)[:, 1],
            "fusion_prob": fusion_model.predict_proba(train_fusion_x)[:, 1],
        }
    )
    test_prob = pd.DataFrame(
        {
            "clinical_prob": clinical_model.predict_proba(test_x)[:, 1],
            "fusion_prob": fusion_model.predict_proba(test_fusion_x)[:, 1],
        }
    )
    return train_prob, test_prob, {"clinical_C": clinical_c, "fusion_C": fusion_c}


def _binary_metrics(y_true: np.ndarray, prob: np.ndarray, threshold: float) -> dict[str, Any]:
    pred = (prob >= float(threshold)).astype(int)
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
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "sensitivity": float(sens),
        "specificity": float(spec),
        "ppv": float(ppv),
        "npv": float(npv),
        "accuracy": float((tp + tn) / len(y_true)),
        "f1": float(f1),
        "balanced_accuracy": float((sens + spec) / 2.0),
        "predicted_positive": int(pred.sum()),
    }


def _best_threshold_youden(y_true: np.ndarray, prob: np.ndarray) -> float:
    thresholds = np.unique(np.concatenate([prob, [0.0, 0.5, 1.0]]))
    best_score = -np.inf
    best_threshold = 0.5
    for threshold in thresholds:
        metrics = _binary_metrics(y_true, prob, float(threshold))
        score = metrics["sensitivity"] + metrics["specificity"] - 1.0
        if score > best_score or (score == best_score and threshold > best_threshold):
            best_score = float(score)
            best_threshold = float(threshold)
    return best_threshold


def _bootstrap_cis(
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
    values = {key: [] for key in keys}
    if n_bootstrap <= 0:
        return {key: (None, None) for key in keys}
    rng = np.random.default_rng(seed)
    n = len(y_true)
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        sample_y = y_true[idx]
        sample_prob = prob[idx]
        if len(np.unique(sample_y)) < 2:
            continue
        metrics = _binary_metrics(sample_y, sample_prob, threshold)
        for key in keys:
            values[key].append(metrics[key])
    return {
        key: (
            float(np.percentile(vals, 2.5)) if vals else None,
            float(np.percentile(vals, 97.5)) if vals else None,
        )
        for key, vals in values.items()
    }


def _metrics_row(
    split: str,
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
        "split": split,
        "model": model,
        "threshold_mode": threshold_mode,
        "threshold_source": threshold_source,
        **_binary_metrics(y_true, prob, threshold),
    }
    cis = _bootstrap_cis(y_true, prob, threshold, seed, n_bootstrap)
    for key, (lower, upper) in cis.items():
        row[f"{key}_ci_low"] = lower
        row[f"{key}_ci_high"] = upper
    return row


def _net_benefit(y_true: np.ndarray, prob: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    n = float(len(y_true))
    values = []
    for threshold in thresholds:
        pred_pos = prob >= threshold
        tp = float(np.logical_and(pred_pos, y_true == 1).sum()) / n
        fp = float(np.logical_and(pred_pos, y_true == 0).sum()) / n
        values.append(tp - fp * threshold / (1.0 - threshold))
    return np.asarray(values)


def _save_figure(fig: plt.Figure, png_path: Path) -> None:
    fig.tight_layout()
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(png_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _plot_confusion(
    pred_df: pd.DataFrame,
    thresholds: dict[str, float],
    split_name: str,
    threshold_mode: str,
    out_path: Path,
) -> None:
    model_specs = [
        ("MRI", "mri_prob"),
        ("Clinical", "clinical_prob"),
        ("Fusion", "fusion_prob"),
    ]
    y = pred_df["label"].to_numpy(int)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    fig.suptitle(f"{split_name} Confusion Matrices ({threshold_mode})", y=1.02, fontsize=14)
    for ax, (name, col) in zip(axes, model_specs):
        prob = pred_df[col].to_numpy(float)
        m = _binary_metrics(y, prob, thresholds[name])
        mat = np.asarray([[m["tn"], m["fp"]], [m["fn"], m["tp"]]], dtype=int)
        im = ax.imshow(mat, cmap="Blues")
        cutoff = (mat.max() + mat.min()) / 2.0
        for (i, j), value in np.ndenumerate(mat):
            color = "white" if value > cutoff else "#111827"
            ax.text(j, i, str(value), ha="center", va="center", fontsize=13, fontweight="bold", color=color)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Pred 0", "Pred 1"])
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["True 0", "True 1"])
        ax.set_title(f"{name}\nAUC={m['auc']:.3f}, ACC={m['accuracy']:.3f}, T={m['threshold']:.3f}", fontsize=10)
        ax.tick_params(axis="both", labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#374151")
            spine.set_linewidth(0.8)
        im.set_clim(vmin=0, vmax=max(1, mat.max()))
    _save_figure(fig, out_path)


def _plot_roc_pr(pred_df: pd.DataFrame, split_name: str, out_path: Path) -> None:
    y = pred_df["label"].to_numpy(int)
    prevalence = float(y.mean())
    model_specs = [
        ("MRI", "mri_prob", "#2563eb"),
        ("Clinical", "clinical_prob", "#059669"),
        ("Fusion", "fusion_prob", "#dc2626"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.2))
    ax = axes[0]
    for name, col, color in model_specs:
        prob = pred_df[col].to_numpy(float)
        fpr, tpr, _ = roc_curve(y, prob)
        ax.plot(fpr, tpr, color=color, linewidth=2.2, label=f"{name} AUC={_safe_auc(y, prob):.3f}")
    ax.plot([0, 1], [0, 1], color="#8c8c8c", linestyle="--", linewidth=1.0)
    ax.set_title(f"{split_name} ROC (logistic)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(True, color="#e5e7eb", linewidth=0.8)
    ax.legend(frameon=True, loc="lower right", fontsize=8)

    ax = axes[1]
    for name, col, color in model_specs:
        prob = pred_df[col].to_numpy(float)
        precision, recall, _ = precision_recall_curve(y, prob)
        ax.plot(recall, precision, color=color, linewidth=2.2, label=f"{name} AP={_safe_ap(y, prob):.3f}")
    ax.axhline(prevalence, color="#8c8c8c", linestyle="--", linewidth=1.0, label=f"Prevalence={prevalence:.3f}")
    ax.set_title(f"{split_name} PR (logistic)")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(True, color="#e5e7eb", linewidth=0.8)
    ax.legend(frameon=True, loc="lower left", fontsize=8)
    _save_figure(fig, out_path)


def _plot_dca(pred_df: pd.DataFrame, split_name: str, out_path: Path, csv_path: Path) -> None:
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

    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    styles = {
        "none": ("None", "#6b7280", "-", 1.7),
        "all": ("All", "#111827", "--", 1.5),
        "mri": (f"MRI AUC={_safe_auc(y, pred_df['mri_prob']):.3f}", "#2563eb", "-", 2.1),
        "clinical": (f"Clinical AUC={_safe_auc(y, pred_df['clinical_prob']):.3f}", "#059669", "-", 2.1),
        "fusion": (f"Fusion AUC={_safe_auc(y, pred_df['fusion_prob']):.3f}", "#dc2626", "-", 2.7),
    }
    for col, (label, color, linestyle, linewidth) in styles.items():
        ax.plot(dca["threshold"], dca[col], label=label, color=color, linestyle=linestyle, linewidth=linewidth)
    ax.axhline(0, color="#9ca3af", linewidth=0.8)
    ax.set_xlim(0.01, 0.80)
    ax.set_title(f"{split_name} Decision Curve Analysis")
    ax.set_xlabel("Threshold probability")
    ax.set_ylabel("Net benefit")
    ax.grid(True, color="#e5e7eb", linewidth=0.8)
    ax.legend(frameon=False, loc="best", fontsize=8)
    _save_figure(fig, out_path)


def make_report(
    train_csv: Path,
    test_csv: Path,
    train_metadata: Path,
    test_metadata: Path,
    output_dir: Path,
    train_name: str,
    test_name: str,
    seed: int,
    n_bootstrap: int,
) -> dict[str, Any]:
    train_pred = _read_predictions(train_csv)
    test_pred = _read_predictions(test_csv)
    train_meta = _load_metadata(train_metadata)
    test_meta = _load_metadata(test_metadata)

    train_x_all, train_manifest = _feature_frame(train_pred, train_meta, "train")
    test_x_all, test_manifest = _feature_frame(test_pred, test_meta, "test")
    numeric_cols, categorical_cols, resolved_features = _select_usable_features(
        train_x_all,
        train_manifest,
        test_manifest,
    )
    used_cols = [*numeric_cols, *categorical_cols]
    train_x = train_x_all[used_cols]
    test_x = test_x_all[used_cols]
    train_y = train_pred["label"].to_numpy(int)
    test_y = test_pred["label"].to_numpy(int)
    train_mri_prob = train_pred["mri_prob"].to_numpy(float)
    test_mri_prob = test_pred["mri_prob"].to_numpy(float)

    train_extra_prob, test_extra_prob, model_info = _fit_predict_logistic(
        train_x,
        test_x,
        train_y,
        train_mri_prob,
        test_mri_prob,
        numeric_cols,
        categorical_cols,
        seed,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    train_out = pd.concat([train_pred.reset_index(drop=True), train_extra_prob], axis=1)
    test_out = pd.concat([test_pred.reset_index(drop=True), test_extra_prob], axis=1)

    models = {
        "MRI": "mri_prob",
        "Clinical": "clinical_prob",
        "Fusion": "fusion_prob",
    }
    thresholds = {
        name: _best_threshold_youden(train_y, train_out[col].to_numpy(float))
        for name, col in models.items()
    }
    for name, col in models.items():
        train_out[f"{name.lower()}_pred_adaptive_youden_train"] = (
            train_out[col].to_numpy(float) >= thresholds[name]
        ).astype(int)
        test_out[f"{name.lower()}_pred_adaptive_youden_train"] = (
            test_out[col].to_numpy(float) >= thresholds[name]
        ).astype(int)
        train_out[f"{name.lower()}_pred_fixed_0_5"] = (train_out[col].to_numpy(float) >= 0.5).astype(int)
        test_out[f"{name.lower()}_pred_fixed_0_5"] = (test_out[col].to_numpy(float) >= 0.5).astype(int)

    train_out.to_csv(output_dir / "train_predictions_mri_clinical_fusion.csv", index=False)
    test_out.to_csv(output_dir / "test_predictions_mri_clinical_fusion.csv", index=False)

    metrics_rows = []
    for split, split_name, df, y in [
        ("train", train_name, train_out, train_y),
        ("test", test_name, test_out, test_y),
    ]:
        for model_idx, (name, col) in enumerate(models.items()):
            prob = df[col].to_numpy(float)
            metrics_rows.append(
                _metrics_row(
                    split,
                    name,
                    "adaptive_youden",
                    "train",
                    thresholds[name],
                    y,
                    prob,
                    seed + model_idx + (0 if split == "train" else 1000),
                    n_bootstrap,
                )
            )
            metrics_rows.append(
                _metrics_row(
                    split,
                    name,
                    "fixed_0.5",
                    "fixed",
                    0.5,
                    y,
                    prob,
                    seed + model_idx + (10 if split == "train" else 1010),
                    n_bootstrap,
                )
            )
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(output_dir / "metrics_summary.csv", index=False)

    _plot_confusion(
        train_out,
        thresholds,
        train_name,
        "adaptive_youden",
        output_dir / "train_confusion_matrices_adaptive_youden.png",
    )
    _plot_confusion(
        test_out,
        thresholds,
        test_name,
        "adaptive_youden",
        output_dir / "test_confusion_matrices_adaptive_youden.png",
    )
    _plot_roc_pr(test_out, test_name, output_dir / "test_roc_pr_logistic.png")
    _plot_dca(
        test_out,
        test_name,
        output_dir / "test_dca_mri_clinical_fusion.png",
        output_dir / "test_dca_mri_clinical_fusion.csv",
    )

    feature_manifest = {
        "label_source": "The label column in each prediction CSV is used as ground truth. Metadata label/pCR fields are ignored.",
        "leakage_columns_excluded": sorted(LEAKAGE_COLUMNS),
        "numeric_features": numeric_cols,
        "categorical_features": categorical_cols,
        "resolved_features": resolved_features,
    }
    with (output_dir / "clinical_feature_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(_json_safe(feature_manifest), f, ensure_ascii=False, indent=2)

    summary = {
        "train_csv": train_csv,
        "test_csv": test_csv,
        "train_metadata": train_metadata,
        "test_metadata": test_metadata,
        "output_dir": output_dir,
        "train_name": train_name,
        "test_name": test_name,
        "seed": seed,
        "n_bootstrap": n_bootstrap,
        "train_samples": int(len(train_y)),
        "train_positive": int(train_y.sum()),
        "train_negative": int((train_y == 0).sum()),
        "test_samples": int(len(test_y)),
        "test_positive": int(test_y.sum()),
        "test_negative": int((test_y == 0).sum()),
        "selected_thresholds": thresholds,
        "logistic_model": model_info,
        "outputs": {
            "train_predictions": output_dir / "train_predictions_mri_clinical_fusion.csv",
            "test_predictions": output_dir / "test_predictions_mri_clinical_fusion.csv",
            "metrics": output_dir / "metrics_summary.csv",
            "feature_manifest": output_dir / "clinical_feature_manifest.json",
            "train_confusion_png": output_dir / "train_confusion_matrices_adaptive_youden.png",
            "test_confusion_png": output_dir / "test_confusion_matrices_adaptive_youden.png",
            "test_roc_pr_png": output_dir / "test_roc_pr_logistic.png",
            "test_dca_png": output_dir / "test_dca_mri_clinical_fusion.png",
        },
    }
    with (output_dir / "report_summary.json").open("w", encoding="utf-8") as f:
        json.dump(_json_safe(summary), f, ensure_ascii=False, indent=2)
    return summary


def _default_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir:
        return Path(args.output_dir).resolve()
    if args.run_dir:
        slug = args.test_name.lower().replace(" ", "_")
        return Path(args.run_dir).resolve() / "figures_output_clinical_fusion" / slug
    return Path(args.test_csv).resolve().parent / "figures_output_clinical_fusion"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export MRI, clinical, and MRI+clinical fusion figures from prediction CSVs."
    )
    parser.add_argument("--run_dir", default=None, help="Run directory containing fold_1 train/val predictions.")
    parser.add_argument("--train_csv", default=None, help="Training prediction CSV with id,label,prob.")
    parser.add_argument("--test_csv", default=None, help="Test/external prediction CSV with id,label,prob.")
    parser.add_argument("--train_metadata", default="dataset/dataset_metadata.csv")
    parser.add_argument("--test_metadata", default=None, help="Defaults to --train_metadata.")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--train_name", default="Train")
    parser.add_argument("--test_name", default="Test")
    parser.add_argument("--seed", type=int, default=2040)
    parser.add_argument("--n_bootstrap", type=int, default=1000)
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve() if args.run_dir else None
    train_csv = Path(args.train_csv).resolve() if args.train_csv else None
    test_csv = Path(args.test_csv).resolve() if args.test_csv else None
    if run_dir is not None:
        train_csv = train_csv or run_dir / "fold_1" / "best_epoch_train_predictions.csv"
        test_csv = test_csv or run_dir / "fold_1" / "best_epoch_val_predictions.csv"
    if train_csv is None or test_csv is None:
        raise SystemExit("Pass --run_dir or both --train_csv and --test_csv.")

    train_metadata = Path(args.train_metadata).resolve()
    test_metadata = Path(args.test_metadata).resolve() if args.test_metadata else train_metadata
    output_dir = _default_output_dir(args)

    summary = make_report(
        train_csv=train_csv,
        test_csv=test_csv,
        train_metadata=train_metadata,
        test_metadata=test_metadata,
        output_dir=output_dir,
        train_name=args.train_name,
        test_name=args.test_name,
        seed=args.seed,
        n_bootstrap=args.n_bootstrap,
    )
    print(json.dumps(_json_safe(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
