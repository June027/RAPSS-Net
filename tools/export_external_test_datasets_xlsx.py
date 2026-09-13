from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_RUN_DIR = (
    Path("output")
    / "MRIUpper_NoTTA_Seed2040_Random"
    / "RandomSplit_(ISPY1+ISPY2)_seed_2040_val_0.20"
)


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    return pd.read_csv(path, encoding="utf-8-sig")


def _resolve_previous_prediction(path: Path | None) -> Path | None:
    if path is None:
        return None
    return path if path.exists() else None


def _build_dataset(
    cohort: str,
    metadata_path: Path,
    previous_prediction_path: Path | None,
    data_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = _read_table(metadata_path)
    if "id" not in df.columns:
        raise ValueError(f"{metadata_path} missing required column: id")
    if "pCR" not in df.columns:
        raise ValueError(f"{metadata_path} missing required column: pCR")

    df = df.copy()
    df["id"] = df["id"].astype(str)
    df.insert(0, "dataset_name", cohort)
    df.insert(1, "label_current_metadata_pcr", pd.to_numeric(df["pCR"], errors="coerce").astype("Int64"))

    previous_prediction_path = _resolve_previous_prediction(previous_prediction_path)
    if previous_prediction_path is not None:
        previous = pd.read_csv(previous_prediction_path)[["id", "label", "prob", "pred"]].copy()
        previous["id"] = previous["id"].astype(str)
        previous = previous.rename(
            columns={
                "label": "label_previous_output",
                "prob": "previous_output_prob",
                "pred": "previous_output_pred",
            }
        )
        df = df.merge(previous, on="id", how="left")
    else:
        df["label_previous_output"] = pd.NA
        df["previous_output_prob"] = pd.NA
        df["previous_output_pred"] = pd.NA

    has_previous = df["label_previous_output"].notna()
    df["label"] = df["label_previous_output"].where(
        has_previous,
        df["label_current_metadata_pcr"],
    ).astype("Int64")
    df["label_source_used"] = has_previous.map(
        {
            True: "previous_output_label",
            False: "current_metadata_pCR",
        }
    )
    df["label_mismatch_previous_vs_current"] = (
        has_previous
        & (
            df["label_previous_output"].astype("Int64")
            != df["label_current_metadata_pcr"].astype("Int64")
        )
    )

    image_paths = []
    mask_paths = []
    image_exists = []
    mask_exists = []
    for sample_id in df["id"].astype(str):
        image_path = data_root / cohort / "images" / f"{sample_id}.npy"
        mask_path = data_root / cohort / "masks" / f"{sample_id}.npy"
        image_paths.append(str(image_path))
        mask_paths.append(str(mask_path))
        image_exists.append(image_path.exists())
        mask_exists.append(mask_path.exists())
    df["image_path"] = image_paths
    df["mask_path"] = mask_paths
    df["image_exists"] = image_exists
    df["mask_exists"] = mask_exists

    front_columns = [
        "dataset_name",
        "id",
        "label",
        "label_source_used",
        "label_current_metadata_pcr",
        "label_previous_output",
        "label_mismatch_previous_vs_current",
        "previous_output_prob",
        "previous_output_pred",
        "center",
        "image_path",
        "mask_path",
        "image_exists",
        "mask_exists",
    ]
    front_columns = [column for column in front_columns if column in df.columns]
    df = df[front_columns + [column for column in df.columns if column not in front_columns]]

    summary = pd.DataFrame(
        [
            {"field": "cohort", "value": cohort},
            {"field": "metadata_path", "value": str(metadata_path)},
            {
                "field": "previous_output_path",
                "value": str(previous_prediction_path) if previous_prediction_path else "",
            },
            {"field": "num_samples", "value": int(len(df))},
            {
                "field": "current_metadata_pcr_positive",
                "value": int(df["label_current_metadata_pcr"].sum()),
            },
            {
                "field": "previous_output_positive",
                "value": int(df["label_previous_output"].dropna().astype(int).sum())
                if df["label_previous_output"].notna().any()
                else "",
            },
            {"field": "label_used_positive", "value": int(df["label"].sum())},
            {
                "field": "label_mismatch_previous_vs_current",
                "value": int(df["label_mismatch_previous_vs_current"].sum()),
            },
            {"field": "missing_images", "value": int((~df["image_exists"]).sum())},
            {"field": "missing_masks", "value": int((~df["mask_exists"]).sum())},
        ]
    )
    return df, summary


def _write_single_workbook(path: Path, data: pd.DataFrame, summary: pd.DataFrame) -> None:
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        data.to_excel(writer, sheet_name="test_dataset", index=False)
        summary.to_excel(writer, sheet_name="summary", index=False)


def _versioned_labels(df: pd.DataFrame, version: str) -> pd.DataFrame:
    out = df.copy()
    if version == "current_metadata_pCR":
        out["label"] = out["label_current_metadata_pcr"].astype("Int64")
        out["label_source_used"] = "current_metadata_pCR"
    elif version == "previous_output_label":
        has_previous = out["label_previous_output"].notna()
        out["label"] = out["label_previous_output"].where(
            has_previous,
            out["label_current_metadata_pcr"],
        ).astype("Int64")
        out["label_source_used"] = has_previous.map(
            {
                True: "previous_output_label",
                False: "current_metadata_pCR_fallback_no_previous_output",
            }
        )
    else:
        raise ValueError(f"Unsupported label version: {version}")
    return out


def _summary_for_version(cohort: str, data: pd.DataFrame, version: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"field": "label_version", "value": version},
            {"field": "cohort", "value": cohort},
            {"field": "num_samples", "value": int(len(data))},
            {"field": "label_positive", "value": int(data["label"].sum())},
            {
                "field": "current_metadata_pcr_positive",
                "value": int(data["label_current_metadata_pcr"].sum()),
            },
            {
                "field": "previous_output_nonmissing",
                "value": int(data["label_previous_output"].notna().sum()),
            },
            {
                "field": "previous_output_positive",
                "value": int(data["label_previous_output"].dropna().astype(int).sum())
                if data["label_previous_output"].notna().any()
                else "",
            },
            {
                "field": "label_mismatch_previous_vs_current",
                "value": int(data["label_mismatch_previous_vs_current"].sum()),
            },
            {"field": "missing_images", "value": int((~data["image_exists"]).sum())},
            {"field": "missing_masks", "value": int((~data["mask_exists"]).sum())},
        ]
    )


def export_workbooks(
    run_dir: Path,
    output_dir: Path,
    data_root: Path,
    duke_metadata: Path,
    private_metadata: Path,
    duke_previous_predictions: Path | None,
    private_previous_predictions: Path | None,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cohorts = {
        "Duke": {
            "metadata": duke_metadata,
            "previous": duke_previous_predictions,
        },
        "Private": {
            "metadata": private_metadata,
            "previous": private_previous_predictions,
        },
    }

    generated: list[Path] = []
    datasets: dict[str, pd.DataFrame] = {}
    summaries: dict[str, pd.DataFrame] = {}
    for cohort, cfg in cohorts.items():
        data, summary = _build_dataset(
            cohort=cohort,
            metadata_path=Path(cfg["metadata"]),
            previous_prediction_path=cfg["previous"],
            data_root=data_root,
        )
        datasets[cohort] = data
        summaries[cohort] = summary
        path = output_dir / f"{cohort}_external_test_dataset.xlsx"
        _write_single_workbook(path, data, summary)
        generated.append(path)

    combined_path = output_dir / "external_test_datasets_duke_private.xlsx"
    with pd.ExcelWriter(combined_path, engine="openpyxl") as writer:
        for cohort in ["Duke", "Private"]:
            datasets[cohort].to_excel(writer, sheet_name=cohort, index=False)
            summaries[cohort].to_excel(writer, sheet_name=f"{cohort}_summary", index=False)
    generated.append(combined_path)

    for version in ["current_metadata_pCR", "previous_output_label"]:
        version_path = output_dir / f"external_test_dataset_{version}.xlsx"
        with pd.ExcelWriter(version_path, engine="openpyxl") as writer:
            for cohort in ["Duke", "Private"]:
                versioned = _versioned_labels(datasets[cohort], version)
                versioned.to_excel(writer, sheet_name=cohort, index=False)
                _summary_for_version(cohort, versioned, version).to_excel(
                    writer,
                    sheet_name=f"{cohort}_summary",
                    index=False,
                )
        generated.append(version_path)

    return generated


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Duke/Private external test datasets as XLSX.")
    parser.add_argument("--run_dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--data_root", type=Path, default=Path("dataset"))
    parser.add_argument("--duke_metadata", type=Path, default=Path("dataset/Duke/dataset_metadata.csv"))
    parser.add_argument("--private_metadata", type=Path, default=Path("dataset/Private/dataset_metadata.csv"))
    parser.add_argument(
        "--duke_previous_predictions",
        type=Path,
        default=None,
        help="Previous Duke output predictions.csv. Defaults to run_dir/yuanyuan/fold_1-yuanyuan/predictions_duke.csv.",
    )
    parser.add_argument(
        "--private_previous_predictions",
        type=Path,
        default=None,
        help="Previous Private output predictions.csv, if available.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir or args.run_dir / "test_datasets_xlsx"
    duke_previous = args.duke_previous_predictions
    if duke_previous is None:
        duke_previous = args.run_dir / "yuanyuan" / "fold_1-yuanyuan" / "predictions_duke.csv"

    generated = export_workbooks(
        run_dir=args.run_dir,
        output_dir=output_dir,
        data_root=args.data_root,
        duke_metadata=args.duke_metadata,
        private_metadata=args.private_metadata,
        duke_previous_predictions=duke_previous,
        private_previous_predictions=args.private_previous_predictions,
    )
    for path in generated:
        print(path)


if __name__ == "__main__":
    main()
