from __future__ import annotations

import argparse
import json
import sys
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
import seaborn as sns
import torch
import torch.nn.functional as F
import yaml
from scipy.stats import mannwhitneyu, pearsonr
from sklearn.metrics import brier_score_loss

from datasets.mri_dataset import MRIDataset
from models.builder import build_model


DEFAULT_REPORT_DIR = (
    PROJECT_DIR
    / "output"
    / "MRIUpper_NoTTA_Seed2040_Random"
    / "fold_1"
    / "mri_clinical_fusion_report"
)
DEFAULT_FOLD_DIR = DEFAULT_REPORT_DIR.parent
DEFAULT_DATA_DIR = PROJECT_DIR / "dataset"
DEFAULT_CONFIG = PROJECT_DIR / "configs" / "server" / "random" / "MRIUpper_NoTTA_Seed2040_Random_eval.yaml"

DATASET_ORDER = ["Train", "Internal Val", "Duke", "Private", "Overall"]
MODEL_COLUMNS = {"MRI": "mri_prob", "Clinical": "clinical_prob", "Fusion": "fusion_prob"}
METRIC_COLUMNS = ["auc", "accuracy", "sensitivity", "specificity", "ppv", "npv", "f1", "brier"]
LABEL_NAMES = {0: "Non-pCR", 1: "pCR"}
LABEL_COLORS = {"Non-pCR": "#4f7df3", "pCR": "#e85d68"}
MODEL_COLORS = {"MRI": "#2563eb", "Clinical": "#059669", "Fusion": "#dc2626"}


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


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _read_predictions(report_dir: Path) -> dict[str, pd.DataFrame]:
    workbook = report_dir / "mri_clinical_fusion_predictions.xlsx"
    if not workbook.exists():
        raise FileNotFoundError(f"Missing {workbook}. Run export_fold1_mri_clinical_fusion_report.py first.")
    sheets = pd.read_excel(workbook, sheet_name=None)
    sheet_to_dataset = {
        "Train": "Train",
        "Internal_Val": "Internal Val",
        "Duke": "Duke",
        "Private": "Private",
    }
    out = {}
    for sheet, dataset in sheet_to_dataset.items():
        df = sheets[sheet].copy()
        df["id"] = df["id"].astype(str)
        df["label"] = df["label"].astype(int)
        df["label_name"] = df["label"].map(LABEL_NAMES)
        df["dataset"] = dataset
        out[dataset] = df
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
                out["Internal Val"].assign(source_dataset="Internal Val"),
                out["Duke"].assign(source_dataset="Duke"),
                out["Private"].assign(source_dataset="Private"),
            ],
            ignore_index=True,
        )
        overall["dataset"] = "Overall"
    out["Overall"] = overall
    return out


def _read_metrics(report_dir: Path, threshold_mode: str) -> pd.DataFrame:
    metrics = pd.read_csv(report_dir / "metrics_summary.csv")
    metrics = metrics[metrics["threshold_mode"] == threshold_mode].copy()
    if metrics.empty:
        raise ValueError(f"No metrics rows for threshold_mode={threshold_mode}")
    return metrics


def _long_scores(predictions: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for dataset, df in predictions.items():
        for model, col in MODEL_COLUMNS.items():
            rows.append(
                pd.DataFrame(
                    {
                        "dataset": dataset,
                        "id": df["id"],
                        "label": df["label"],
                        "label_name": df["label_name"],
                        "model": model,
                        "score": df[col].astype(float),
                    }
                )
            )
    return pd.concat(rows, ignore_index=True)


def _score_group_stats(long_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    group_stats = (
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
    group_stats = group_stats.merge(q, on=["dataset", "model", "label_name"], how="left")
    tests = []
    for dataset in DATASET_ORDER:
        for model in MODEL_COLUMNS:
            sub = long_df[(long_df["dataset"] == dataset) & (long_df["model"] == model)]
            neg = sub.loc[sub["label"] == 0, "score"].to_numpy(float)
            pos = sub.loc[sub["label"] == 1, "score"].to_numpy(float)
            stat, p_value = mannwhitneyu(pos, neg, alternative="two-sided")
            tests.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "non_pcr_n": int(len(neg)),
                    "pcr_n": int(len(pos)),
                    "non_pcr_mean": float(np.mean(neg)),
                    "pcr_mean": float(np.mean(pos)),
                    "delta_mean_pcr_minus_non_pcr": float(np.mean(pos) - np.mean(neg)),
                    "non_pcr_median": float(np.median(neg)),
                    "pcr_median": float(np.median(pos)),
                    "delta_median_pcr_minus_non_pcr": float(np.median(pos) - np.median(neg)),
                    "mannwhitney_u": float(stat),
                    "mannwhitney_p": float(p_value),
                }
            )
    return group_stats, pd.DataFrame(tests)


def _calibration_points(df: pd.DataFrame, model: str, bins: int) -> pd.DataFrame:
    y = df["label"].to_numpy(int)
    prob = df[MODEL_COLUMNS[model]].to_numpy(float)
    order = np.argsort(prob)
    parts = np.array_split(order, min(bins, len(order)))
    rows = []
    for part in parts:
        if len(part) == 0:
            continue
        rows.append(
            {
                "model": model,
                "n": int(len(part)),
                "mean_predicted_probability": float(np.mean(prob[part])),
                "observed_pcr_rate": float(np.mean(y[part])),
            }
        )
    return pd.DataFrame(rows)


def _calibration_summary(predictions: dict[str, pd.DataFrame], bins: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    point_rows = []
    for dataset in DATASET_ORDER:
        df = predictions[dataset]
        y = df["label"].to_numpy(int)
        for model, col in MODEL_COLUMNS.items():
            prob = df[col].to_numpy(float)
            bin_edges = np.linspace(0.0, 1.0, 11)
            ece = 0.0
            for idx in range(10):
                low, high = bin_edges[idx], bin_edges[idx + 1]
                mask = (prob >= low) & (prob <= high if idx == 9 else prob < high)
                if np.any(mask):
                    ece += float(mask.mean()) * abs(float(y[mask].mean()) - float(prob[mask].mean()))
            summary_rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "brier": float(brier_score_loss(y, prob)),
                    "ece_10_uniform_bins": float(ece),
                }
            )
            points = _calibration_points(df, model, bins)
            points.insert(0, "dataset", dataset)
            point_rows.append(points)
    return pd.DataFrame(summary_rows), pd.concat(point_rows, ignore_index=True)


def _plot_dataset_metrics(dataset: str, metrics: pd.DataFrame, out_dir: Path) -> None:
    sub = metrics[metrics["dataset"] == dataset].set_index("model").reindex(list(MODEL_COLUMNS))
    matrix = sub[METRIC_COLUMNS].T
    fig, ax = plt.subplots(figsize=(6.4, 5.8))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".3f",
        cmap="YlGnBu",
        cbar=True,
        linewidths=0.5,
        linecolor="white",
        ax=ax,
    )
    ax.set_title(f"{dataset} Metrics")
    ax.set_xlabel("Model")
    ax.set_ylabel("Metric")
    _save(fig, out_dir / "metrics_heatmap.png")


def _plot_dataset_score_heatmap(dataset: str, df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    ordered = df.sort_values(["label", "fusion_prob"], ascending=[True, True]).reset_index(drop=True)
    values = ordered[["mri_prob", "clinical_prob", "fusion_prob"]].to_numpy(float)
    fig = plt.figure(figsize=(6.4, max(4.6, min(16.0, 0.035 * len(ordered) + 2.2))))
    gs = fig.add_gridspec(1, 2, width_ratios=[0.16, 1.0], wspace=0.04)
    ax_label = fig.add_subplot(gs[0, 0])
    ax = fig.add_subplot(gs[0, 1])
    label_colors = matplotlib.colors.ListedColormap([LABEL_COLORS["Non-pCR"], LABEL_COLORS["pCR"]])
    ax_label.imshow(ordered["label"].to_numpy(int).reshape(-1, 1), cmap=label_colors, vmin=0, vmax=1, aspect="auto")
    ax_label.set_xticks([0])
    ax_label.set_xticklabels(["True"], rotation=90)
    ax_label.set_yticks([])
    ax_label.set_title("Label", fontsize=9)
    im = ax.imshow(values, cmap="magma", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["MRI", "Clinical", "Fusion"])
    ax.set_yticks([])
    ax.set_title(f"{dataset} Patient-Level Scores")
    split = int((ordered["label"] == 0).sum())
    if 0 < split < len(ordered):
        ax.axhline(split - 0.5, color="white", linewidth=1.5)
        ax_label.axhline(split - 0.5, color="white", linewidth=1.5)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("Predicted pCR probability")
    _save(fig, out_dir / "patient_score_heatmap.png")
    table = ordered[["id", "label", "label_name", "mri_prob", "clinical_prob", "fusion_prob"]].copy()
    table.insert(0, "sort_order", np.arange(1, len(table) + 1))
    table.to_excel(out_dir / "patient_score_heatmap_values.xlsx", index=False)
    return table


def _plot_dataset_violin(dataset: str, long_df: pd.DataFrame, out_dir: Path) -> None:
    sub = long_df[long_df["dataset"] == dataset].copy()
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    sns.violinplot(
        data=sub,
        x="model",
        y="score",
        hue="label_name",
        hue_order=["Non-pCR", "pCR"],
        order=list(MODEL_COLUMNS),
        palette=LABEL_COLORS,
        split=True,
        inner="quartile",
        cut=0,
        linewidth=0.8,
        ax=ax,
    )
    sns.stripplot(
        data=sub,
        x="model",
        y="score",
        hue="label_name",
        hue_order=["Non-pCR", "pCR"],
        order=list(MODEL_COLUMNS),
        palette=LABEL_COLORS,
        dodge=True,
        alpha=0.32,
        size=2.0,
        legend=False,
        ax=ax,
    )
    ax.set_title(f"{dataset} Non-pCR vs pCR Score Distribution")
    ax.set_xlabel("Model")
    ax.set_ylabel("Predicted pCR probability")
    ax.set_ylim(0, 1)
    ax.grid(True, axis="y", color="#e5e7eb", linewidth=0.8)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:2], labels[:2], loc="upper left", frameon=True)
    _save(fig, out_dir / "violin_nonpcr_vs_pcr.png")


def _plot_dataset_scatter(dataset: str, df: pd.DataFrame, metrics: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.1, 5.5))
    for label_name, group in df.groupby("label_name"):
        ax.scatter(
            group["mri_prob"],
            group["clinical_prob"],
            color=LABEL_COLORS[label_name],
            s=24,
            alpha=0.62,
            label=label_name,
            edgecolor="none",
        )
    r = pearsonr(df["mri_prob"].to_numpy(float), df["clinical_prob"].to_numpy(float)).statistic
    auc_mri = metrics[(metrics["dataset"] == dataset) & (metrics["model"] == "MRI")]["auc"].iloc[0]
    auc_clin = metrics[(metrics["dataset"] == dataset) & (metrics["model"] == "Clinical")]["auc"].iloc[0]
    t_mri = metrics[(metrics["dataset"] == dataset) & (metrics["model"] == "MRI")]["threshold"].iloc[0]
    t_clin = metrics[(metrics["dataset"] == dataset) & (metrics["model"] == "Clinical")]["threshold"].iloc[0]
    ax.axvline(t_mri, color="#9ca3af", linestyle="--", linewidth=0.9)
    ax.axhline(t_clin, color="#9ca3af", linestyle="--", linewidth=0.9)
    ax.text(
        0.04,
        0.96,
        f"AUC MRI={auc_mri:.3f}\nAUC Clinical={auc_clin:.3f}\nr={r:.2f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.78, "edgecolor": "#d1d5db"},
    )
    ax.set_title(f"{dataset} MRI Score vs Clinical Score")
    ax.set_xlabel("MRI score")
    ax.set_ylabel("Clinical score")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, color="#e5e7eb", linewidth=0.8)
    ax.legend(frameon=True, loc="lower right")
    _save(fig, out_dir / "mri_vs_clinical_scatter.png")


def _plot_dataset_calibration(dataset: str, points: pd.DataFrame, summary: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.1, 5.5))
    ax.plot([0, 1], [0, 1], color="#6b7280", linestyle="--", linewidth=1.0, label="Perfect")
    for model in MODEL_COLUMNS:
        p = points[(points["dataset"] == dataset) & (points["model"] == model)]
        s = summary[(summary["dataset"] == dataset) & (summary["model"] == model)].iloc[0]
        ax.plot(
            p["mean_predicted_probability"],
            p["observed_pcr_rate"],
            marker="o",
            linewidth=1.8,
            color=MODEL_COLORS[model],
            label=f"{model} ECE={s['ece_10_uniform_bins']:.3f}",
        )
    ax.set_title(f"{dataset} Calibration")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed pCR rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, color="#e5e7eb", linewidth=0.8)
    ax.legend(frameon=True, fontsize=8)
    _save(fig, out_dir / "calibration_curves.png")


def _center_from_id(sample_id: str) -> str:
    if sample_id.startswith("ISPY1_"):
        return "ISPY1"
    if sample_id.startswith("ISPY2_"):
        return "ISPY2"
    if sample_id.startswith("Duke_"):
        return "Duke"
    if sample_id.startswith("Private_"):
        return "Private"
    raise ValueError(f"Cannot infer center from id={sample_id}")


def _select_heatmap_cases(
    predictions: dict[str, pd.DataFrame],
    data_dir: Path,
    cases_per_label: int,
) -> pd.DataFrame:
    rows = []
    for dataset in DATASET_ORDER:
        df = predictions[dataset].copy()
        for label, ascending in [(0, True), (1, False)]:
            sub = df[df["label"] == label].sort_values("fusion_prob", ascending=ascending)
            kept = 0
            for _, row in sub.iterrows():
                try:
                    center = _center_from_id(str(row["id"]))
                except ValueError:
                    continue
                image_path = data_dir / center / "images" / f"{row['id']}.npy"
                mask_path = data_dir / center / "masks" / f"{row['id']}.npy"
                if not image_path.exists() or not mask_path.exists():
                    continue
                rows.append(
                    {
                        "dataset": dataset,
                        "id": str(row["id"]),
                        "center": center,
                        "label": int(row["label"]),
                        "label_name": LABEL_NAMES[int(row["label"])],
                        "mri_prob": float(row["mri_prob"]),
                        "clinical_prob": float(row["clinical_prob"]),
                        "fusion_prob": float(row["fusion_prob"]),
                        "image_path": image_path,
                        "mask_path": mask_path,
                    }
                )
                kept += 1
                if kept >= cases_per_label:
                    break
    return pd.DataFrame(rows)


def _load_model(config_path: Path, model_path: Path, data_dir: Path) -> tuple[dict[str, Any], torch.nn.Module, torch.device]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["paths"]["data_dir"] = str(data_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config, device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    state = checkpoint.get("ema_model_state_dict", checkpoint.get("model_state_dict", checkpoint))
    model.load_state_dict(state, strict=True)
    model.eval()
    return config, model, device


def _attention_maps_for_case(
    row: pd.Series,
    config: dict[str, Any],
    model: torch.nn.Module,
    device: torch.device,
    data_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, float]:
    record = {
        "id": str(row["id"]),
        "label": int(row["label"]),
        "center": str(row["center"]),
    }
    dataset = MRIDataset(
        [record],
        str(data_dir),
        model_config=config.get("model", {}),
        is_train=False,
        use_mask=True,
        strict_file_check=True,
        clinical_keys=[],
        clinical_stats={},
    )
    sample = dataset[0]
    image = sample["image"].unsqueeze(0).to(device)
    kinetics = sample["kinetics"].unsqueeze(0).to(device)
    mask = sample["mask"].unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(image, kinetics=kinetics, lesion_mask=mask, return_attention_maps=True)
        logits = out[0]
        prob = float(torch.softmax(logits, dim=1)[0, 1].item())
        attention = out[-1]
        core = attention["core"]
        peri = attention["peri"]
        target_size = tuple(image.shape[-3:])
        core_up = F.interpolate(core, size=target_size, mode="trilinear", align_corners=False)[0, 0]
        peri_up = F.interpolate(peri, size=target_size, mode="trilinear", align_corners=False)[0, 0]
        combined = torch.maximum(core_up, peri_up)
    img3d = image[0, -1].detach().cpu().numpy()
    mask3d = mask[0, 0].detach().cpu().numpy() > 0.5
    core_np = core_up.detach().cpu().numpy()
    peri_np = peri_up.detach().cpu().numpy()
    combined_np = combined.detach().cpu().numpy()
    areas = mask3d.reshape(mask3d.shape[0], -1).sum(axis=1)
    z = int(np.argmax(areas)) if np.any(areas) else int(mask3d.shape[0] // 2)
    img_slice = img3d[z]
    lo, hi = np.nanpercentile(img_slice, [1, 99])
    if hi <= lo:
        hi = lo + 1.0
    img_slice = np.clip((img_slice - lo) / (hi - lo), 0, 1)
    return img_slice, mask3d[z], core_np[z], peri_np[z], combined_np[z], z, prob


def _norm_map(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    lo, hi = np.nanmin(values), np.nanmax(values)
    if hi <= lo:
        return np.zeros_like(values)
    return (values - lo) / (hi - lo)


def _plot_model_heatmaps(
    selected: pd.DataFrame,
    config: dict[str, Any],
    model: torch.nn.Module,
    device: torch.device,
    data_dir: Path,
    output_dir: Path,
) -> pd.DataFrame:
    rows = []
    for dataset in DATASET_ORDER:
        drows = selected[selected["dataset"] == dataset].reset_index(drop=True)
        if drows.empty:
            continue
        fig, axes = plt.subplots(len(drows), 4, figsize=(12.5, 3.2 * len(drows)), squeeze=False)
        for row_idx, (_, row) in enumerate(drows.iterrows()):
            img, mask, core, peri, combined, z, model_prob = _attention_maps_for_case(row, config, model, device, data_dir)
            maps = [
                ("DCE-MRI", None),
                ("Core attention", _norm_map(core)),
                ("Peri attention", _norm_map(peri)),
                ("Combined attention", _norm_map(combined)),
            ]
            for col_idx, (title, heat) in enumerate(maps):
                ax = axes[row_idx, col_idx]
                ax.imshow(img, cmap="gray", vmin=0, vmax=1)
                if heat is not None:
                    heat_masked = np.where(mask, heat, np.nan)
                    ax.imshow(heat_masked, cmap="turbo", alpha=0.68, vmin=0, vmax=1)
                if np.any(mask):
                    ax.contour(mask.astype(float), levels=[0.5], colors="white", linewidths=0.7)
                if row_idx == 0:
                    ax.set_title(title, fontsize=10)
                if col_idx == 0:
                    ax.set_ylabel(f"{row['label_name']}\n{row['id']}", fontsize=8)
                ax.set_xticks([])
                ax.set_yticks([])
            rows.append(
                {
                    **{k: row[k] for k in ["dataset", "id", "center", "label", "label_name"]},
                    "model_attention_prob": model_prob,
                    "xlsx_mri_prob": float(row["mri_prob"]),
                    "clinical_prob": float(row["clinical_prob"]),
                    "fusion_prob": float(row["fusion_prob"]),
                    "slice_z": z,
                    "heatmap_type": "model_core_peri_attention",
                }
            )
        fig.suptitle(f"{dataset} Model Attention Heatmaps", fontsize=14, y=1.01)
        _save(fig, output_dir / dataset / "model_attention_heatmaps.png")
    return pd.DataFrame(rows)


def _write_dataset_tables(
    dataset: str,
    metrics: pd.DataFrame,
    long_df: pd.DataFrame,
    group_stats: pd.DataFrame,
    tests: pd.DataFrame,
    calibration_summary: pd.DataFrame,
    calibration_points: pd.DataFrame,
    output_dir: Path,
) -> None:
    out_dir = output_dir / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_dir / "all_numeric_results.xlsx", engine="openpyxl") as writer:
        metrics[metrics["dataset"] == dataset].to_excel(writer, sheet_name="metrics", index=False)
        group_stats[group_stats["dataset"] == dataset].to_excel(writer, sheet_name="score_group_stats", index=False)
        tests[tests["dataset"] == dataset].to_excel(writer, sheet_name="nonpcr_vs_pcr_tests", index=False)
        calibration_summary[calibration_summary["dataset"] == dataset].to_excel(
            writer, sheet_name="calibration_summary", index=False
        )
        calibration_points[calibration_points["dataset"] == dataset].to_excel(
            writer, sheet_name="calibration_points", index=False
        )
        long_df[long_df["dataset"] == dataset].to_excel(writer, sheet_name="scores_long", index=False)


def export_requested_results(
    report_dir: Path,
    fold_dir: Path,
    config_path: Path,
    data_dir: Path,
    output_dir: Path,
    threshold_mode: str,
    calibration_bins: int,
    cases_per_label: int,
    skip_model_heatmaps: bool,
) -> dict[str, Any]:
    report_dir = report_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions = _read_predictions(report_dir)
    metrics = _read_metrics(report_dir, threshold_mode)
    long_df = _long_scores(predictions)
    group_stats, tests = _score_group_stats(long_df)
    cal_summary, cal_points = _calibration_summary(predictions, calibration_bins)

    selected_cases = _select_heatmap_cases(predictions, data_dir.resolve(), cases_per_label)
    heatmap_info = pd.DataFrame()
    if not skip_model_heatmaps and not selected_cases.empty:
        model_path = fold_dir.resolve() / "best_model_fold_1.pth"
        config, model, device = _load_model(config_path.resolve(), model_path, data_dir.resolve())
        heatmap_info = _plot_model_heatmaps(selected_cases, config, model, device, data_dir.resolve(), output_dir)

    for dataset in DATASET_ORDER:
        out_dir = output_dir / dataset
        out_dir.mkdir(parents=True, exist_ok=True)
        _write_dataset_tables(dataset, metrics, long_df, group_stats, tests, cal_summary, cal_points, output_dir)
        predictions[dataset][["id", "label", "label_name", "mri_prob", "clinical_prob", "fusion_prob"]].to_excel(
            out_dir / "patient_probabilities.xlsx", index=False
        )
        _plot_dataset_metrics(dataset, metrics, out_dir)
        _plot_dataset_score_heatmap(dataset, predictions[dataset], out_dir)
        _plot_dataset_violin(dataset, long_df, out_dir)
        _plot_dataset_scatter(dataset, predictions[dataset], metrics, out_dir)
        _plot_dataset_calibration(dataset, cal_points, cal_summary, out_dir)

    with pd.ExcelWriter(output_dir / "ALL_DATASETS_NUMERIC_RESULTS.xlsx", engine="openpyxl") as writer:
        metrics.to_excel(writer, sheet_name="metrics_all", index=False)
        group_stats.to_excel(writer, sheet_name="score_group_stats", index=False)
        tests.to_excel(writer, sheet_name="nonpcr_vs_pcr_tests", index=False)
        cal_summary.to_excel(writer, sheet_name="calibration_summary", index=False)
        cal_points.to_excel(writer, sheet_name="calibration_points", index=False)
        selected_cases.to_excel(writer, sheet_name="selected_heatmap_cases", index=False)
        if not heatmap_info.empty:
            heatmap_info.to_excel(writer, sheet_name="generated_model_heatmaps", index=False)

    summary = {
        "output_dir": output_dir,
        "threshold_mode": threshold_mode,
        "datasets": DATASET_ORDER,
        "per_dataset_outputs": {
            dataset: {
                "numeric_results": output_dir / dataset / "all_numeric_results.xlsx",
                "patient_probabilities": output_dir / dataset / "patient_probabilities.xlsx",
                "metrics_heatmap": output_dir / dataset / "metrics_heatmap.png",
                "patient_score_heatmap": output_dir / dataset / "patient_score_heatmap.png",
                "violin": output_dir / dataset / "violin_nonpcr_vs_pcr.png",
                "scatter": output_dir / dataset / "mri_vs_clinical_scatter.png",
                "calibration": output_dir / dataset / "calibration_curves.png",
                "model_attention_heatmaps": output_dir / dataset / "model_attention_heatmaps.png",
            }
            for dataset in DATASET_ORDER
        },
        "all_numeric_results": output_dir / "ALL_DATASETS_NUMERIC_RESULTS.xlsx",
        "heatmap_note": "model_attention_heatmaps are true model core/peri attention maps upsampled to image size; they are not mask-only overlays.",
    }
    with (output_dir / "requested_results_summary.json").open("w", encoding="utf-8") as f:
        json.dump(_json_safe(summary), f, ensure_ascii=False, indent=2)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export per-dataset MRI/Clinical/Fusion numeric results and figures.")
    parser.add_argument("--report_dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--fold_dir", type=Path, default=DEFAULT_FOLD_DIR)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data_dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--threshold_mode", choices=["train_youden", "fixed_0.5"], default="train_youden")
    parser.add_argument("--calibration_bins", type=int, default=8)
    parser.add_argument("--cases_per_label", type=int, default=2)
    parser.add_argument("--skip_model_heatmaps", action="store_true")
    argv = [arg for arg in sys.argv[1:] if arg != "`"]
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else args.report_dir.resolve() / "requested_per_dataset_results"
    )
    summary = export_requested_results(
        report_dir=args.report_dir,
        fold_dir=args.fold_dir,
        config_path=args.config,
        data_dir=args.data_dir,
        output_dir=output_dir,
        threshold_mode=args.threshold_mode,
        calibration_bins=args.calibration_bins,
        cases_per_label=args.cases_per_label,
        skip_model_heatmaps=args.skip_model_heatmaps,
    )
    print(json.dumps(_json_safe(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
