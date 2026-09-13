from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import mannwhitneyu, pearsonr, spearmanr
from sklearn.metrics import brier_score_loss, roc_auc_score


DEFAULT_REPORT_DIR = (
    Path(__file__).resolve().parents[1]
    / "output"
    / "MRIUpper_NoTTA_Seed2040_Random"
    / "fold_1"
    / "mri_clinical_fusion_report"
)
DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "dataset"

MODEL_COLUMNS = {
    "MRI": "mri_prob",
    "Clinical": "clinical_prob",
    "Fusion": "fusion_prob",
}
MODEL_COLORS = {
    "MRI": "#2563eb",
    "Clinical": "#059669",
    "Fusion": "#dc2626",
}
LABEL_NAMES = {0: "Non-pCR", 1: "pCR"}
LABEL_COLORS = {"Non-pCR": "#4f7df3", "pCR": "#e85d68"}
DATASET_ORDER = ["Train", "Internal Val", "Duke", "Private", "Overall"]
PLOT_DATASET_ORDER = ["Internal Val", "Duke", "Private", "Overall", "Train"]
METRIC_ORDER = [
    "auc",
    "accuracy",
    "sensitivity",
    "specificity",
    "ppv",
    "npv",
    "f1",
    "brier",
]


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def _slug(text: str) -> str:
    return text.lower().replace(" ", "_").replace("/", "_").replace(".", "_")


def _save_figure(fig: plt.Figure, png_path: Path) -> None:
    fig.tight_layout()
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(png_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _read_predictions(report_dir: Path) -> dict[str, pd.DataFrame]:
    pred_path = report_dir / "mri_clinical_fusion_predictions.xlsx"
    if not pred_path.exists():
        raise FileNotFoundError(
            f"Prediction workbook not found: {pred_path}. "
            "Run tools/export_fold1_mri_clinical_fusion_report.py first."
        )
    sheets = pd.read_excel(pred_path, sheet_name=None)
    predictions: dict[str, pd.DataFrame] = {}
    sheet_map = {
        "Train": "Train",
        "Internal_Val": "Internal Val",
        "Duke": "Duke",
        "Private": "Private",
    }
    for sheet_name, dataset_name in sheet_map.items():
        if sheet_name not in sheets:
            raise ValueError(f"{pred_path} missing sheet: {sheet_name}")
        df = sheets[sheet_name].copy()
        df["dataset"] = dataset_name
        df["id"] = df["id"].astype(str)
        df["label"] = df["label"].astype(int)
        df["label_name"] = df["label"].map(LABEL_NAMES)
        predictions[dataset_name] = df

    overall_path = report_dir / "overall_eval_mri_clinical_fusion_predictions.csv"
    if overall_path.exists():
        overall = pd.read_csv(overall_path)
        overall["id"] = overall["id"].astype(str)
        overall["label"] = overall["label"].astype(int)
        overall["label_name"] = overall["label"].map(LABEL_NAMES)
        overall["dataset"] = "Overall"
    else:
        overall = pd.concat(
            [
                predictions["Internal Val"].assign(source_dataset="Internal Val"),
                predictions["Duke"].assign(source_dataset="Duke"),
                predictions["Private"].assign(source_dataset="Private"),
            ],
            ignore_index=True,
        )
        overall["dataset"] = "Overall"
    predictions["Overall"] = overall
    return predictions


def _read_metrics(report_dir: Path, threshold_mode: str) -> pd.DataFrame:
    metrics_path = report_dir / "metrics_summary.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"Metrics file not found: {metrics_path}. "
            "Run tools/export_fold1_mri_clinical_fusion_report.py first."
        )
    metrics = pd.read_csv(metrics_path)
    metrics = metrics[metrics["threshold_mode"] == threshold_mode].copy()
    if metrics.empty:
        raise ValueError(f"No metrics rows found for threshold_mode={threshold_mode}")
    return metrics


def _long_scores(predictions: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for dataset_name in DATASET_ORDER:
        df = predictions[dataset_name]
        for model_name, col in MODEL_COLUMNS.items():
            rows.append(
                pd.DataFrame(
                    {
                        "dataset": dataset_name,
                        "id": df["id"].astype(str),
                        "label": df["label"].astype(int),
                        "label_name": df["label_name"],
                        "model": model_name,
                        "score": df[col].astype(float),
                    }
                )
            )
    return pd.concat(rows, ignore_index=True)


def _score_distribution_summary(long_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = (
        long_df.groupby(["dataset", "model", "label_name"], observed=False)["score"]
        .agg(["count", "mean", "std", "median", "min", "max"])
        .reset_index()
    )
    q = (
        long_df.groupby(["dataset", "model", "label_name"], observed=False)["score"]
        .quantile([0.25, 0.75])
        .unstack()
        .reset_index()
        .rename(columns={0.25: "q1", 0.75: "q3"})
    )
    summary = summary.merge(q, on=["dataset", "model", "label_name"], how="left")

    tests = []
    for dataset_name in DATASET_ORDER:
        for model_name in MODEL_COLUMNS:
            sub = long_df[(long_df["dataset"] == dataset_name) & (long_df["model"] == model_name)]
            neg = sub.loc[sub["label"] == 0, "score"].to_numpy(float)
            pos = sub.loc[sub["label"] == 1, "score"].to_numpy(float)
            if len(neg) > 0 and len(pos) > 0:
                stat, p_value = mannwhitneyu(pos, neg, alternative="two-sided")
                delta_mean = float(np.mean(pos) - np.mean(neg))
                delta_median = float(np.median(pos) - np.median(neg))
            else:
                stat, p_value, delta_mean, delta_median = np.nan, np.nan, np.nan, np.nan
            tests.append(
                {
                    "dataset": dataset_name,
                    "model": model_name,
                    "n_non_pcr": int(len(neg)),
                    "n_pcr": int(len(pos)),
                    "non_pcr_mean": float(np.mean(neg)) if len(neg) else np.nan,
                    "pcr_mean": float(np.mean(pos)) if len(pos) else np.nan,
                    "delta_mean_pcr_minus_non_pcr": delta_mean,
                    "non_pcr_median": float(np.median(neg)) if len(neg) else np.nan,
                    "pcr_median": float(np.median(pos)) if len(pos) else np.nan,
                    "delta_median_pcr_minus_non_pcr": delta_median,
                    "mannwhitney_u": float(stat) if np.isfinite(stat) else np.nan,
                    "mannwhitney_p": float(p_value) if np.isfinite(p_value) else np.nan,
                }
            )
    return summary, pd.DataFrame(tests)


def _expected_calibration_error(y_true: np.ndarray, prob: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        low, high = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (prob >= low) & (prob <= high)
        else:
            mask = (prob >= low) & (prob < high)
        if not np.any(mask):
            continue
        ece += float(np.sum(mask) / n) * abs(float(np.mean(y_true[mask])) - float(np.mean(prob[mask])))
    return float(ece)


def _calibration_points(
    y_true: np.ndarray,
    prob: np.ndarray,
    n_bins: int,
) -> pd.DataFrame:
    order = np.argsort(prob)
    splits = np.array_split(order, min(n_bins, len(order)))
    rows = []
    for idx in splits:
        if len(idx) == 0:
            continue
        rows.append(
            {
                "n": int(len(idx)),
                "predicted_probability": float(np.mean(prob[idx])),
                "true_probability": float(np.mean(y_true[idx])),
            }
        )
    return pd.DataFrame(rows)


def _calibration_summary(predictions: dict[str, pd.DataFrame], n_bins: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    point_rows = []
    for dataset_name in DATASET_ORDER:
        df = predictions[dataset_name]
        y = df["label"].to_numpy(int)
        for model_name, col in MODEL_COLUMNS.items():
            prob = df[col].to_numpy(float)
            summary_rows.append(
                {
                    "dataset": dataset_name,
                    "model": model_name,
                    "n": int(len(y)),
                    "brier": float(brier_score_loss(y, prob)),
                    "ece_uniform_10_bins": _expected_calibration_error(y, prob, 10),
                }
            )
            points = _calibration_points(y, prob, n_bins=n_bins)
            points.insert(0, "model", model_name)
            points.insert(0, "dataset", dataset_name)
            point_rows.append(points)
    return pd.DataFrame(summary_rows), pd.concat(point_rows, ignore_index=True)


def _mri_clinical_correlation(predictions: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for dataset_name in DATASET_ORDER:
        df = predictions[dataset_name]
        x = df["mri_prob"].to_numpy(float)
        y = df["clinical_prob"].to_numpy(float)
        pearson = pearsonr(x, y)
        spearman = spearmanr(x, y)
        rows.append(
            {
                "dataset": dataset_name,
                "n": int(len(df)),
                "pearson_r": float(pearson.statistic),
                "pearson_p": float(pearson.pvalue),
                "spearman_r": float(spearman.statistic),
                "spearman_p": float(spearman.pvalue),
            }
        )
    return pd.DataFrame(rows)


def _plot_metric_heatmaps(metrics: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(18.0, 8.6))
    axes = axes.ravel()
    for ax, metric_name in zip(axes, METRIC_ORDER):
        pivot = (
            metrics.pivot_table(
                index="dataset",
                columns="model",
                values=metric_name,
                aggfunc="first",
            )
            .reindex(DATASET_ORDER)
            .reindex(columns=list(MODEL_COLUMNS))
        )
        cmap = "viridis_r" if metric_name == "brier" else "YlGnBu"
        im = ax.imshow(pivot.to_numpy(float), cmap=cmap, aspect="auto")
        ax.set_title(metric_name.upper())
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=25, ha="right")
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        for row_idx in range(pivot.shape[0]):
            for col_idx in range(pivot.shape[1]):
                value = pivot.iloc[row_idx, col_idx]
                if pd.isna(value):
                    text = "NA"
                else:
                    text = f"{value:.3f}"
                ax.text(col_idx, row_idx, text, ha="center", va="center", fontsize=8, color="#111827")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    fig.suptitle("MRI, Clinical, and Fusion Metrics by Dataset", fontsize=15, y=1.02)
    _save_figure(fig, out_dir / "metric_heatmaps_mri_clinical_fusion.png")


def _plot_patient_score_heatmap(dataset_name: str, df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    cols = list(MODEL_COLUMNS.values())
    model_names = list(MODEL_COLUMNS)
    ordered = df.sort_values(["label", "fusion_prob", "mri_prob"], ascending=[True, True, True]).copy()
    matrix = ordered[cols].to_numpy(float)
    labels = ordered["label_name"].tolist()
    fig = plt.figure(figsize=(7.5, max(5.0, min(18.0, 0.035 * len(ordered) + 3.0))))
    grid = fig.add_gridspec(1, 2, width_ratios=[0.18, 1.0], wspace=0.05)
    ax_bar = fig.add_subplot(grid[0, 0])
    ax = fig.add_subplot(grid[0, 1])

    label_values = ordered["label"].to_numpy(int).reshape(-1, 1)
    label_cmap = matplotlib.colors.ListedColormap([LABEL_COLORS["Non-pCR"], LABEL_COLORS["pCR"]])
    ax_bar.imshow(label_values, cmap=label_cmap, aspect="auto", vmin=0, vmax=1)
    ax_bar.set_xticks([0])
    ax_bar.set_xticklabels(["Label"], rotation=90)
    ax_bar.set_yticks([])
    ax_bar.set_title("True", fontsize=10)

    im = ax.imshow(matrix, cmap="magma", aspect="auto", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(model_names)))
    ax.set_xticklabels(model_names)
    ax.set_yticks([])
    ax.set_title(f"{dataset_name} Patient Score Heatmap")
    ax.set_xlabel("Model score")
    ax.set_ylabel("")
    n_non_pcr = int((ordered["label"] == 0).sum())
    if 0 < n_non_pcr < len(ordered):
        ax.axhline(n_non_pcr - 0.5, color="white", linewidth=1.8)
        ax_bar.axhline(n_non_pcr - 0.5, color="white", linewidth=1.8)
    if len(ordered) <= 60:
        ax.set_yticks(np.arange(len(ordered)))
        ax.set_yticklabels(ordered["id"].tolist(), fontsize=6)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("Predicted pCR probability")
    handles = [
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=LABEL_COLORS[name], markersize=9, label=name)
        for name in ["Non-pCR", "pCR"]
    ]
    ax.legend(handles=handles, loc="upper right", frameon=True, fontsize=8)
    _save_figure(fig, out_dir / f"{_slug(dataset_name)}_patient_score_heatmap.png")

    values = ordered[["dataset", "id", "label", "label_name", *cols]].copy()
    values = values.rename(columns={v: k for k, v in MODEL_COLUMNS.items()})
    values.insert(0, "sort_order", np.arange(1, len(values) + 1))
    values["label_display"] = labels
    return values


def _center_from_id(sample_id: str) -> str | None:
    if sample_id.startswith("ISPY1_"):
        return "ISPY1"
    if sample_id.startswith("ISPY2_"):
        return "ISPY2"
    if sample_id.startswith("Duke_"):
        return "Duke"
    if sample_id.startswith("Private_"):
        return "Private"
    return None


def _case_image_paths(data_dir: Path, sample_id: str) -> tuple[Path, Path] | None:
    center = _center_from_id(sample_id)
    if center is None:
        return None
    image_path = data_dir / center / "images" / f"{sample_id}.npy"
    mask_path = data_dir / center / "masks" / f"{sample_id}.npy"
    if not image_path.exists() or not mask_path.exists():
        return None
    return image_path, mask_path


def _load_case_slice(image_path: Path, mask_path: Path) -> tuple[np.ndarray, np.ndarray, int]:
    image = np.load(image_path)
    mask = np.load(mask_path)
    if image.ndim == 4:
        image_3d = image[-1]
    elif image.ndim == 3:
        image_3d = image
    else:
        raise ValueError(f"Unsupported image shape for {image_path}: {image.shape}")
    if mask.ndim == 4:
        mask_3d = mask[0]
    elif mask.ndim == 3:
        mask_3d = mask
    else:
        raise ValueError(f"Unsupported mask shape for {mask_path}: {mask.shape}")
    mask_3d = mask_3d > 0.5
    areas = mask_3d.reshape(mask_3d.shape[0], -1).sum(axis=1)
    z = int(np.argmax(areas)) if np.any(areas) else int(mask_3d.shape[0] // 2)
    img_slice = np.asarray(image_3d[z], dtype=float)
    mask_slice = np.asarray(mask_3d[z], dtype=bool)
    lo, hi = np.nanpercentile(img_slice, [1, 99])
    if hi <= lo:
        hi = lo + 1.0
    img_slice = np.clip((img_slice - lo) / (hi - lo), 0.0, 1.0)
    return img_slice, mask_slice, z


def _select_case_heatmap_rows(
    predictions: dict[str, pd.DataFrame],
    data_dir: Path,
    cases_per_label: int,
) -> pd.DataFrame:
    rows = []
    for dataset_name in ["Train", "Internal Val", "Duke", "Private"]:
        df = predictions[dataset_name].copy()
        for label, label_name, ascending in [
            (0, "Non-pCR", True),
            (1, "pCR", False),
        ]:
            sub = df[df["label"] == label].sort_values("fusion_prob", ascending=ascending)
            selected = 0
            for _, row in sub.iterrows():
                paths = _case_image_paths(data_dir, str(row["id"]))
                if paths is None:
                    continue
                rows.append(
                    {
                        "dataset": dataset_name,
                        "id": str(row["id"]),
                        "label": int(row["label"]),
                        "label_name": label_name,
                        "mri_prob": float(row["mri_prob"]),
                        "clinical_prob": float(row["clinical_prob"]),
                        "fusion_prob": float(row["fusion_prob"]),
                        "image_path": paths[0],
                        "mask_path": paths[1],
                        "selection_rule": "lowest_fusion_for_non_pcr_highest_fusion_for_pcr",
                    }
                )
                selected += 1
                if selected >= cases_per_label:
                    break
    return pd.DataFrame(rows)


def _plot_mri_case_heatmaps(
    case_rows: pd.DataFrame,
    out_dir: Path,
) -> dict[str, Path]:
    outputs: dict[str, Path] = {}
    case_dir = out_dir / "mri_case_heatmaps"
    case_dir.mkdir(parents=True, exist_ok=True)
    for dataset_name, dataset_rows in case_rows.groupby("dataset", sort=False):
        labels_present = [name for name in ["Non-pCR", "pCR"] if name in set(dataset_rows["label_name"])]
        max_cols = int(dataset_rows.groupby("label_name").size().max())
        fig, axes = plt.subplots(
            len(labels_present),
            max_cols,
            figsize=(4.1 * max_cols, 3.8 * len(labels_present)),
            squeeze=False,
        )
        for ax in axes.ravel():
            ax.axis("off")
        for row_idx, label_name in enumerate(labels_present):
            label_rows = dataset_rows[dataset_rows["label_name"] == label_name].reset_index(drop=True)
            for col_idx, (_, row) in enumerate(label_rows.iterrows()):
                ax = axes[row_idx, col_idx]
                img_slice, mask_slice, z = _load_case_slice(Path(row["image_path"]), Path(row["mask_path"]))
                overlay = np.where(mask_slice, img_slice, np.nan)
                ax.imshow(img_slice, cmap="gray", vmin=0, vmax=1)
                ax.imshow(overlay, cmap="turbo", alpha=0.68, vmin=0, vmax=1)
                if np.any(mask_slice):
                    ax.contour(mask_slice.astype(float), levels=[0.5], colors="white", linewidths=0.7)
                short_id = str(row["id"]).replace("ISPY2_", "").replace("ISPY1_", "")
                ax.set_title(
                    f"{short_id}\nL={int(row['label'])} MRI={row['mri_prob']:.2f} "
                    f"Clin={row['clinical_prob']:.2f} Fus={row['fusion_prob']:.2f} z={z}",
                    fontsize=8,
                )
                ax.axis("off")
            axes[row_idx, 0].set_ylabel(label_name, fontsize=11)
        fig.suptitle(f"{dataset_name} Representative MRI Tumor Heatmaps", fontsize=14, y=1.02)
        out_path = case_dir / f"{_slug(dataset_name)}_mri_case_heatmaps_by_label.png"
        _save_figure(fig, out_path)
        outputs[dataset_name] = out_path
    return outputs


def _plot_score_violin(long_df: pd.DataFrame, out_dir: Path) -> None:
    plot_df = long_df[long_df["dataset"].isin(PLOT_DATASET_ORDER)].copy()
    plot_df["dataset"] = pd.Categorical(plot_df["dataset"], PLOT_DATASET_ORDER, ordered=True)
    plot_df["model"] = pd.Categorical(plot_df["model"], list(MODEL_COLUMNS), ordered=True)
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 8.6), sharey=True)
    axes = axes.ravel()
    for ax, dataset_name in zip(axes, PLOT_DATASET_ORDER):
        sub = plot_df[plot_df["dataset"] == dataset_name]
        sns.violinplot(
            data=sub,
            x="model",
            y="score",
            hue="label_name",
            hue_order=["Non-pCR", "pCR"],
            palette=LABEL_COLORS,
            split=True,
            inner="quartile",
            linewidth=0.8,
            cut=0,
            ax=ax,
        )
        sns.stripplot(
            data=sub,
            x="model",
            y="score",
            hue="label_name",
            hue_order=["Non-pCR", "pCR"],
            palette=LABEL_COLORS,
            dodge=True,
            alpha=0.35,
            size=1.7,
            legend=False,
            ax=ax,
        )
        ax.set_title(dataset_name)
        ax.set_xlabel("")
        ax.set_ylabel("Predicted pCR probability")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, axis="y", color="#e5e7eb", linewidth=0.8)
        if ax.get_legend() is not None:
            ax.get_legend().remove()
    axes[-1].axis("off")
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=LABEL_COLORS[name], markersize=8, label=name)
        for name in ["Non-pCR", "pCR"]
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False)
    fig.suptitle("Score Distributions by True Response", fontsize=15, y=1.02)
    _save_figure(fig, out_dir / "score_violin_by_dataset_model_label.png")


def _plot_mri_clinical_scatter(
    predictions: dict[str, pd.DataFrame],
    metrics: pd.DataFrame,
    out_dir: Path,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 8.6), sharex=True, sharey=True)
    axes = axes.ravel()
    for ax, dataset_name in zip(axes, PLOT_DATASET_ORDER):
        df = predictions[dataset_name]
        for label_name, sub in df.groupby("label_name"):
            ax.scatter(
                sub["mri_prob"],
                sub["clinical_prob"],
                s=22,
                alpha=0.62,
                color=LABEL_COLORS[label_name],
                label=label_name,
                edgecolor="none",
            )
        x = df["mri_prob"].to_numpy(float)
        y = df["clinical_prob"].to_numpy(float)
        r = pearsonr(x, y).statistic
        auc_mri = metrics.loc[(metrics["dataset"] == dataset_name) & (metrics["model"] == "MRI"), "auc"].iloc[0]
        auc_clinical = metrics.loc[
            (metrics["dataset"] == dataset_name) & (metrics["model"] == "Clinical"), "auc"
        ].iloc[0]
        t_mri = metrics.loc[(metrics["dataset"] == dataset_name) & (metrics["model"] == "MRI"), "threshold"].iloc[0]
        t_clin = metrics.loc[
            (metrics["dataset"] == dataset_name) & (metrics["model"] == "Clinical"), "threshold"
        ].iloc[0]
        ax.axvline(t_mri, color="#9ca3af", linestyle="--", linewidth=0.9)
        ax.axhline(t_clin, color="#9ca3af", linestyle="--", linewidth=0.9)
        ax.text(
            0.04,
            0.96,
            f"AUC MRI={auc_mri:.3f}\nAUC Clinical={auc_clinical:.3f}\nr={r:.2f}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.75, "edgecolor": "#d1d5db"},
        )
        ax.set_title(dataset_name)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, color="#e5e7eb", linewidth=0.8)
        ax.set_xlabel("MRI score")
        ax.set_ylabel("Clinical score")
    axes[-1].axis("off")
    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", color=LABEL_COLORS[name], label=name)
        for name in ["Non-pCR", "pCR"]
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False)
    fig.suptitle("MRI Score vs Clinical Score", fontsize=15, y=1.02)
    _save_figure(fig, out_dir / "mri_vs_clinical_score_scatter.png")


def _plot_calibration_curves(
    calibration_points: pd.DataFrame,
    calibration_summary: pd.DataFrame,
    out_dir: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16.2, 5.2), sharex=True, sharey=True)
    dataset_colors = {
        "Train": "#111827",
        "Internal Val": "#f97316",
        "Duke": "#2563eb",
        "Private": "#7c3aed",
        "Overall": "#dc2626",
    }
    for ax, model_name in zip(axes, MODEL_COLUMNS):
        ax.plot([0, 1], [0, 1], linestyle="--", color="#6b7280", linewidth=1.0, label="Perfect")
        for dataset_name in DATASET_ORDER:
            points = calibration_points[
                (calibration_points["dataset"] == dataset_name)
                & (calibration_points["model"] == model_name)
            ]
            summary = calibration_summary[
                (calibration_summary["dataset"] == dataset_name)
                & (calibration_summary["model"] == model_name)
            ].iloc[0]
            ax.plot(
                points["predicted_probability"],
                points["true_probability"],
                marker="o",
                linewidth=1.7,
                color=dataset_colors[dataset_name],
                label=f"{dataset_name} ECE={summary['ece_uniform_10_bins']:.3f}",
            )
        ax.set_title(f"{model_name} Calibration")
        ax.set_xlabel("Predicted probability")
        ax.set_ylabel("True probability")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, color="#e5e7eb", linewidth=0.8)
        ax.legend(fontsize=7, loc="best", frameon=True)
    fig.suptitle("Calibration Curves", fontsize=15, y=1.02)
    _save_figure(fig, out_dir / "calibration_curves_by_model_dataset.png")


def _write_value_tables(
    out_dir: Path,
    metrics: pd.DataFrame,
    long_df: pd.DataFrame,
    distribution_summary: pd.DataFrame,
    distribution_tests: pd.DataFrame,
    calibration_summary: pd.DataFrame,
    calibration_points: pd.DataFrame,
    correlation: pd.DataFrame,
    patient_heatmap_values: dict[str, pd.DataFrame],
    case_heatmap_rows: pd.DataFrame,
) -> dict[str, Path]:
    tables_dir = out_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    metric_values_path = tables_dir / "metric_heatmap_values.xlsx"
    with pd.ExcelWriter(metric_values_path, engine="openpyxl") as writer:
        metrics.to_excel(writer, sheet_name="metrics_long", index=False)
        for metric_name in METRIC_ORDER:
            pivot = (
                metrics.pivot_table(
                    index="dataset",
                    columns="model",
                    values=metric_name,
                    aggfunc="first",
                )
                .reindex(DATASET_ORDER)
                .reindex(columns=list(MODEL_COLUMNS))
            )
            pivot.to_excel(writer, sheet_name=metric_name[:31])

    score_tables_path = tables_dir / "score_distribution_and_patient_values.xlsx"
    with pd.ExcelWriter(score_tables_path, engine="openpyxl") as writer:
        long_df.to_excel(writer, sheet_name="all_scores_long", index=False)
        distribution_summary.to_excel(writer, sheet_name="group_summary", index=False)
        distribution_tests.to_excel(writer, sheet_name="pcr_vs_nonpcr_tests", index=False)
        for dataset_name, values in patient_heatmap_values.items():
            values.to_excel(writer, sheet_name=_slug(dataset_name)[:31], index=False)

    calibration_path = tables_dir / "calibration_and_correlation_values.xlsx"
    with pd.ExcelWriter(calibration_path, engine="openpyxl") as writer:
        calibration_summary.to_excel(writer, sheet_name="calibration_summary", index=False)
        calibration_points.to_excel(writer, sheet_name="calibration_points", index=False)
        correlation.to_excel(writer, sheet_name="mri_clinical_correlation", index=False)

    case_heatmap_path = tables_dir / "selected_mri_case_heatmap_cases.xlsx"
    case_heatmap_rows.to_excel(case_heatmap_path, index=False)

    distribution_tests.to_csv(tables_dir / "pcr_vs_nonpcr_score_tests.csv", index=False)
    calibration_summary.to_csv(tables_dir / "calibration_summary.csv", index=False)
    correlation.to_csv(tables_dir / "mri_clinical_score_correlation.csv", index=False)
    case_heatmap_rows.to_csv(tables_dir / "selected_mri_case_heatmap_cases.csv", index=False)
    return {
        "metric_values": metric_values_path,
        "score_values": score_tables_path,
        "calibration_correlation_values": calibration_path,
        "selected_mri_case_heatmaps": case_heatmap_path,
    }


def make_extended_figures(
    report_dir: Path,
    output_dir: Path,
    threshold_mode: str,
    calibration_bins: int,
    data_dir: Path,
    cases_per_label: int,
) -> dict[str, Any]:
    report_dir = report_dir.resolve()
    output_dir = output_dir.resolve()
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    predictions = _read_predictions(report_dir)
    metrics = _read_metrics(report_dir, threshold_mode=threshold_mode)
    long_df = _long_scores(predictions)
    distribution_summary, distribution_tests = _score_distribution_summary(long_df)
    calibration_summary, calibration_points = _calibration_summary(predictions, n_bins=calibration_bins)
    correlation = _mri_clinical_correlation(predictions)

    _plot_metric_heatmaps(metrics, figures_dir)
    patient_heatmap_values = {
        dataset_name: _plot_patient_score_heatmap(dataset_name, predictions[dataset_name], figures_dir)
        for dataset_name in DATASET_ORDER
    }
    _plot_score_violin(long_df, figures_dir)
    _plot_mri_clinical_scatter(predictions, metrics, figures_dir)
    _plot_calibration_curves(calibration_points, calibration_summary, figures_dir)
    case_heatmap_rows = _select_case_heatmap_rows(
        predictions,
        data_dir.resolve(),
        cases_per_label=cases_per_label,
    )
    case_heatmap_outputs = _plot_mri_case_heatmaps(case_heatmap_rows, figures_dir)

    tables = _write_value_tables(
        output_dir,
        metrics,
        long_df,
        distribution_summary,
        distribution_tests,
        calibration_summary,
        calibration_points,
        correlation,
        patient_heatmap_values,
        case_heatmap_rows,
    )

    overall_metrics = metrics[
        (metrics["dataset"] == "Overall")
        & (metrics["model"].isin(MODEL_COLUMNS))
    ][
        [
            "dataset",
            "model",
            "n",
            "positive",
            "negative",
            "threshold",
            "auc",
            "accuracy",
            "sensitivity",
            "specificity",
            "ppv",
            "npv",
            "f1",
            "brier",
        ]
    ].copy()
    summary = {
        "report_dir": report_dir,
        "output_dir": output_dir,
        "threshold_mode": threshold_mode,
        "overall_metrics": overall_metrics.to_dict(orient="records"),
        "tables": tables,
        "figures": {
            "metric_heatmaps": figures_dir / "metric_heatmaps_mri_clinical_fusion.png",
            "score_violin": figures_dir / "score_violin_by_dataset_model_label.png",
            "mri_clinical_scatter": figures_dir / "mri_vs_clinical_score_scatter.png",
            "calibration": figures_dir / "calibration_curves_by_model_dataset.png",
            "patient_score_heatmaps": {
                dataset_name: figures_dir / f"{_slug(dataset_name)}_patient_score_heatmap.png"
                for dataset_name in DATASET_ORDER
            },
            "mri_case_heatmaps": case_heatmap_outputs,
        },
    }
    with (output_dir / "extended_report_summary.json").open("w", encoding="utf-8") as f:
        json.dump(_json_safe(summary), f, ensure_ascii=False, indent=2)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export extended numeric tables and statistical figures from fold_1 MRI/clinical/fusion predictions."
    )
    parser.add_argument("--report_dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument(
        "--threshold_mode",
        choices=["train_youden", "fixed_0.5"],
        default="train_youden",
        help="Which rows to read from metrics_summary.csv.",
    )
    parser.add_argument("--calibration_bins", type=int, default=8)
    parser.add_argument("--data_dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--cases_per_label", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else args.report_dir.resolve() / "extended_stat_figures"
    )
    summary = make_extended_figures(
        report_dir=args.report_dir,
        output_dir=output_dir,
        threshold_mode=args.threshold_mode,
        calibration_bins=args.calibration_bins,
        data_dir=args.data_dir,
        cases_per_label=args.cases_per_label,
    )
    print(json.dumps(_json_safe(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
