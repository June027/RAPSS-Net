import argparse
import json
import math
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class NumericShift:
    column: str
    train_mean: float
    val_mean: float
    train_std: float
    val_std: float
    smd: float
    ks: float
    risk: str


@dataclass
class CategoricalShift:
    column: str
    train_counts: dict
    val_counts: dict
    train_props: dict
    val_props: dict
    total_variation: float
    overlap: float
    risk: str


def _empirical_ks(train_vals: np.ndarray, val_vals: np.ndarray) -> float:
    train_vals = np.sort(train_vals.astype(np.float64))
    val_vals = np.sort(val_vals.astype(np.float64))
    if train_vals.size == 0 or val_vals.size == 0:
        return 0.0
    support = np.unique(np.concatenate([train_vals, val_vals]))
    train_cdf = np.searchsorted(train_vals, support, side="right") / float(train_vals.size)
    val_cdf = np.searchsorted(val_vals, support, side="right") / float(val_vals.size)
    return float(np.max(np.abs(train_cdf - val_cdf)))


def _categorical_distribution(series: pd.Series) -> tuple[dict, dict]:
    counts = series.fillna("<NA>").astype(str).value_counts().sort_index()
    props = (counts / max(int(counts.sum()), 1)).to_dict()
    return counts.to_dict(), props


def _align_probabilities(train_props: dict, val_props: dict) -> tuple[np.ndarray, np.ndarray, list[str]]:
    keys = sorted(set(train_props) | set(val_props))
    train = np.array([float(train_props.get(k, 0.0)) for k in keys], dtype=np.float64)
    val = np.array([float(val_props.get(k, 0.0)) for k in keys], dtype=np.float64)
    return train, val, keys


def _categorical_risk(total_variation: float) -> str:
    if total_variation >= 0.30:
        return "high"
    if total_variation >= 0.15:
        return "medium"
    return "low"


def _numeric_risk(smd: float, ks: float) -> str:
    if smd >= 0.50 or ks >= 0.35:
        return "high"
    if smd >= 0.25 or ks >= 0.20:
        return "medium"
    return "low"


def compute_categorical_shift(train_df: pd.DataFrame, val_df: pd.DataFrame, column: str) -> CategoricalShift:
    train_counts, train_props = _categorical_distribution(train_df[column])
    val_counts, val_props = _categorical_distribution(val_df[column])
    train_p, val_p, keys = _align_probabilities(train_props, val_props)
    tvd = 0.5 * float(np.abs(train_p - val_p).sum())
    overlap = float(np.minimum(train_p, val_p).sum())
    return CategoricalShift(
        column=column,
        train_counts=train_counts,
        val_counts=val_counts,
        train_props={k: float(train_props.get(k, 0.0)) for k in keys},
        val_props={k: float(val_props.get(k, 0.0)) for k in keys},
        total_variation=tvd,
        overlap=overlap,
        risk=_categorical_risk(tvd),
    )


def compute_numeric_shift(train_df: pd.DataFrame, val_df: pd.DataFrame, column: str) -> NumericShift | None:
    train_vals = pd.to_numeric(train_df[column], errors="coerce").dropna().to_numpy()
    val_vals = pd.to_numeric(val_df[column], errors="coerce").dropna().to_numpy()
    if train_vals.size == 0 or val_vals.size == 0:
        return None
    train_mean = float(train_vals.mean())
    val_mean = float(val_vals.mean())
    train_std = float(train_vals.std(ddof=0))
    val_std = float(val_vals.std(ddof=0))
    pooled_std = math.sqrt(max((train_std**2 + val_std**2) / 2.0, 1.0e-12))
    smd = abs(train_mean - val_mean) / pooled_std
    ks = _empirical_ks(train_vals, val_vals)
    return NumericShift(
        column=column,
        train_mean=train_mean,
        val_mean=val_mean,
        train_std=train_std,
        val_std=val_std,
        smd=float(smd),
        ks=float(ks),
        risk=_numeric_risk(float(smd), float(ks)),
    )


def _resolve_sample_path(data_dir: str, sample_id: str, center: str | None) -> str | None:
    direct_path = os.path.join(data_dir, "images", f"{sample_id}.npy")
    if os.path.exists(direct_path):
        return direct_path
    if center:
        center_path = os.path.join(data_dir, str(center), "images", f"{sample_id}.npy")
        if os.path.exists(center_path):
            return center_path
    return None


def add_image_stats(df: pd.DataFrame, data_dir: str, max_samples: int | None = None) -> pd.DataFrame:
    if "id" not in df.columns:
        raise ValueError("Dataframe must include an 'id' column for image-stat analysis.")
    sample_df = df.copy()
    if max_samples is not None and len(sample_df) > max_samples:
        sample_df = sample_df.sample(n=int(max_samples), random_state=42).sort_index()

    image_rows = []
    for _, row in sample_df.iterrows():
        sample_id = str(row["id"])
        center = str(row["center"]) if "center" in row and not pd.isna(row["center"]) else None
        img_path = _resolve_sample_path(data_dir, sample_id, center)
        if img_path is None:
            continue
        raw_img = np.load(img_path, mmap_mode="r")
        finite_mask = np.isfinite(raw_img)
        valid = raw_img[finite_mask]
        nonzero = valid[valid != 0]
        if nonzero.size == 0:
            nonzero = valid
        if nonzero.size == 0:
            continue
        image_rows.append(
            {
                "id": sample_id,
                "img_phase_count": float(raw_img.shape[0]),
                "img_nonzero_mean": float(nonzero.mean()),
                "img_nonzero_std": float(nonzero.std()),
                "img_nonzero_p95": float(np.percentile(nonzero, 95)),
                "img_foreground_fraction": float(nonzero.size / max(raw_img.size, 1)),
            }
        )
    if not image_rows:
        return df
    image_df = pd.DataFrame(image_rows)
    return df.merge(image_df, on="id", how="left")


def summarize_consistency(train_df: pd.DataFrame, val_df: pd.DataFrame) -> dict:
    categorical_columns = [col for col in ("label", "center") if col in train_df.columns and col in val_df.columns]
    numeric_columns = []
    for col in sorted(set(train_df.columns) & set(val_df.columns)):
        if col in {"id", "label", "center"}:
            continue
        if pd.api.types.is_numeric_dtype(train_df[col]) or pd.api.types.is_numeric_dtype(val_df[col]):
            numeric_columns.append(col)

    categorical = [
        compute_categorical_shift(train_df, val_df, col).__dict__
        for col in categorical_columns
    ]
    numeric = []
    for col in numeric_columns:
        shift = compute_numeric_shift(train_df, val_df, col)
        if shift is not None:
            numeric.append(shift.__dict__)

    high_risk_reasons = []
    medium_risk_reasons = []
    for item in categorical + numeric:
        reason = f"{item['column']} ({item['risk']})"
        if item["risk"] == "high":
            high_risk_reasons.append(reason)
        elif item["risk"] == "medium":
            medium_risk_reasons.append(reason)

    if high_risk_reasons:
        overall_risk = "high"
    elif medium_risk_reasons:
        overall_risk = "medium"
    else:
        overall_risk = "low"

    return {
        "train_size": int(len(train_df)),
        "val_size": int(len(val_df)),
        "categorical": categorical,
        "numeric": numeric,
        "overall_risk": overall_risk,
        "high_risk_reasons": high_risk_reasons,
        "medium_risk_reasons": medium_risk_reasons,
    }


def _print_summary(report: dict):
    print("=== Split Consistency Report ===")
    print(f"train_size={report['train_size']} val_size={report['val_size']}")
    print(f"overall_risk={report['overall_risk']}")
    if report["high_risk_reasons"]:
        print("high_risk_reasons=" + ", ".join(report["high_risk_reasons"]))
    if report["medium_risk_reasons"]:
        print("medium_risk_reasons=" + ", ".join(report["medium_risk_reasons"]))

    if report["categorical"]:
        print("\n[categorical]")
        for item in report["categorical"]:
            print(
                f"{item['column']}: tvd={item['total_variation']:.3f}, "
                f"overlap={item['overlap']:.3f}, risk={item['risk']}"
            )
    if report["numeric"]:
        print("\n[numeric]")
        for item in report["numeric"]:
            print(
                f"{item['column']}: smd={item['smd']:.3f}, ks={item['ks']:.3f}, "
                f"train_mean={item['train_mean']:.4f}, val_mean={item['val_mean']:.4f}, risk={item['risk']}"
            )


def main():
    parser = argparse.ArgumentParser(
        description="Compare train/validation split consistency and flag shift risk."
    )
    parser.add_argument("--train_csv", help="Path to train_records.csv")
    parser.add_argument("--val_csv", help="Path to val_records.csv")
    parser.add_argument(
        "--split_dir",
        help="Directory containing train_records.csv and val_records.csv",
    )
    parser.add_argument(
        "--data_dir",
        default=None,
        help="Optional processed data root for image-stat consistency checks.",
    )
    parser.add_argument(
        "--max_image_samples",
        type=int,
        default=None,
        help="Optional cap for image-stat sampling per split.",
    )
    parser.add_argument(
        "--output_json",
        default=None,
        help="Optional path to save the report as JSON.",
    )
    args = parser.parse_args()

    train_csv = args.train_csv
    val_csv = args.val_csv
    if args.split_dir:
        train_csv = train_csv or os.path.join(args.split_dir, "train_records.csv")
        val_csv = val_csv or os.path.join(args.split_dir, "val_records.csv")
    if not train_csv or not val_csv:
        raise ValueError("Provide --split_dir or both --train_csv and --val_csv.")

    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    if args.data_dir:
        train_df = add_image_stats(train_df, args.data_dir, max_samples=args.max_image_samples)
        val_df = add_image_stats(val_df, args.data_dir, max_samples=args.max_image_samples)

    report = summarize_consistency(train_df, val_df)
    _print_summary(report)
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
