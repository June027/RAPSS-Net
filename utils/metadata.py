from __future__ import annotations

import logging
from typing import Iterable

import pandas as pd

logger = logging.getLogger(__name__)


REQUIRED_COLUMN_ALIASES = {
    "id": ("id", "ID", "patient_id", "case_id", "pid"),
    "label": ("label", "pCR", "PCR", "y"),
    "center": ("center", "Center", "site"),
}

OPTIONAL_PATIENT_GROUP_ALIASES = (
    "patient_group",
    "patient_id",
    "PatientID",
    "patient",
    "subject_id",
    "case_id",
    "pid",
)

OPTIONAL_IHC_ALIASES = {
    "ihc_cd8": ("ihc_cd8", "CD8", "cd8"),
    "ihc_cd34": ("ihc_cd34", "CD34", "cd34"),
    "ihc_ki67": ("ihc_ki67", "Ki67", "ki67", "KI67"),
    "ihc_cd163": ("ihc_cd163", "CD163", "cd163"),
}

METADATA_ENCODINGS = ("utf-8-sig", "utf-8", "gb18030", "gbk", "latin1")


def _resolve_column_name(columns: Iterable[str], aliases: Iterable[str]) -> str | None:
    return next((column for column in aliases if column in columns), None)


def read_metadata_csv(csv_path: str) -> pd.DataFrame:
    last_error = None
    for encoding in METADATA_ENCODINGS:
        try:
            df = pd.read_csv(csv_path, encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
            continue
        if encoding != METADATA_ENCODINGS[0]:
            logger.warning("Read metadata CSV with encoding=%s: %s", encoding, csv_path)
        return df
    raise UnicodeDecodeError(
        last_error.encoding,
        last_error.object,
        last_error.start,
        last_error.end,
        last_error.reason,
    )


def resolve_required_metadata_columns(columns: Iterable[str]) -> dict[str, str]:
    resolved_columns = {
        field_name: _resolve_column_name(columns, aliases)
        for field_name, aliases in REQUIRED_COLUMN_ALIASES.items()
    }
    missing_fields = [
        field_name
        for field_name, column_name in resolved_columns.items()
        if column_name is None
    ]
    if missing_fields:
        raise ValueError(f"Missing required metadata columns: {missing_fields}")
    return {field_name: column_name for field_name, column_name in resolved_columns.items()}


def _coerce_optional_float(value) -> float | None:
    if pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_required_string(value, field_name: str) -> str:
    if pd.isna(value) or str(value).strip() == "":
        raise ValueError(f"missing required value for '{field_name}'")
    return str(value)


def _coerce_required_label(value) -> int:
    if pd.isna(value) or str(value).strip() == "":
        raise ValueError("missing required value for 'label'")
    label_value = float(value)
    if not label_value.is_integer():
        raise ValueError(f"label must be an integer class, got {value!r}")
    label_int = int(label_value)
    if label_int not in {0, 1}:
        raise ValueError(f"label must be binary 0/1, got {value!r}")
    return label_int


def infer_patient_group(sample_id: str) -> str:
    """Return the patient-level grouping key used by split code.

    Public metadata normally has one row per patient, so the full sample id is the
    safest default. If a metadata CSV contains an explicit patient column, that
    value overrides this inference in load_metadata_records().
    """
    return str(sample_id).strip()


def load_metadata_records(
    csv_path: str,
    include_ihc: bool = True,
    require_unique_ids: bool = False,
    clinical_keys: list[str] | tuple[str, ...] | None = None,
) -> list[dict]:
    df = read_metadata_csv(csv_path)
    resolved_columns = resolve_required_metadata_columns(df.columns)
    patient_group_column = _resolve_column_name(df.columns, OPTIONAL_PATIENT_GROUP_ALIASES)

    optional_ihc_columns = {}
    if include_ihc:
        optional_ihc_columns = {
            field_name: _resolve_column_name(df.columns, aliases)
            for field_name, aliases in OPTIONAL_IHC_ALIASES.items()
        }

    data_list = []
    row_errors = []
    skipped_missing_required = []
    dropped_ihc_values = []
    for _, row in df.iterrows():
        row_idx = int(getattr(row, "name", -1)) + 2
        try:
            item = {
                "id": _coerce_required_string(row[resolved_columns["id"]], "id"),
                "label": _coerce_required_label(row[resolved_columns["label"]]),
                "center": _coerce_required_string(
                    row[resolved_columns["center"]], "center"
                ),
            }
            if patient_group_column is not None:
                item["patient_group"] = _coerce_required_string(
                    row[patient_group_column], "patient_group"
                )
            else:
                item["patient_group"] = infer_patient_group(item["id"])
        except Exception as exc:
            if "missing required value" in str(exc):
                skipped_missing_required.append(f"row {row_idx}: {exc}")
                continue
            row_errors.append(f"row {row_idx}: {exc}")
            continue

        for field_name, column_name in optional_ihc_columns.items():
            if column_name is None:
                continue
            parsed_value = _coerce_optional_float(row[column_name])
            if parsed_value is None:
                if not pd.isna(row[column_name]):
                    dropped_ihc_values.append(
                        f"row {row_idx} column '{column_name}'"
                    )
                continue
            item[field_name] = parsed_value
        for key in clinical_keys or []:
            if key not in df.columns:
                continue
            value = row[key]
            if pd.isna(value):
                item[key] = None
            else:
                item[key] = value
        data_list.append(item)

    if row_errors:
        preview = "; ".join(row_errors[:5])
        if len(row_errors) > 5:
            preview += "; ..."
        raise ValueError(
            f"Found {len(row_errors)} invalid metadata row(s): {preview}"
        )

    if not data_list:
        raise ValueError("No valid records were loaded from metadata CSV.")

    if skipped_missing_required:
        preview = "; ".join(skipped_missing_required[:5])
        if len(skipped_missing_required) > 5:
            preview += "; ..."
        logger.warning(
            "Skipped %d metadata row(s) with missing required values; affected entries: %s",
            len(skipped_missing_required),
            preview,
        )

    if dropped_ihc_values:
        preview = ", ".join(dropped_ihc_values[:5])
        if len(dropped_ihc_values) > 5:
            preview += ", ..."
        logger.warning(
            "Dropped %d non-numeric IHC value(s); affected entries: %s",
            len(dropped_ihc_values),
            preview,
        )

    if require_unique_ids:
        ids = [record["id"] for record in data_list]
        if len(ids) != len(set(ids)):
            raise ValueError(
                "Duplicate sample IDs detected in metadata. Group-safe splitting cannot be guaranteed."
            )

    return data_list
