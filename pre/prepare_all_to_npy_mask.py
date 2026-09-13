import os
import glob
import numpy as np
import pandas as pd
import nibabel as nib
import yaml
import argparse
import SimpleITK as sitk
import shutil
import re
import json
from scipy.ndimage import zoom
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)


def resolve_config_values(node, project_root):
    env_var_pattern = re.compile(r"\$\{(.*?)(?::-(.*?))?\}")
    if isinstance(node, dict):
        return {k: resolve_config_values(v, project_root) for k, v in node.items()}
    if isinstance(node, list):
        return [resolve_config_values(v, project_root) for v in node]
    if isinstance(node, str):
        match = env_var_pattern.fullmatch(node)
        if match:
            env_var, default_val = match.groups()
            node = os.getenv(env_var, default_val if default_val is not None else "")
        if isinstance(node, str) and node.startswith(("./", "../")):
            return os.path.normpath(os.path.join(project_root, node))
    return node


def get_config():
    parser = argparse.ArgumentParser(description="RAPSS-Net ROI Data Factory")
    parser.add_argument("--config", "-c", type=str, default="configs/server/prep_config_public_server.yaml")
    args = parser.parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        raw_cfg = yaml.safe_load(f)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return resolve_config_values(raw_cfg, project_root)


def fast_n4_bias_field_correction(image_np: np.ndarray, n4_iters) -> np.ndarray | None:
    try:
        thresh = (
            np.percentile(image_np[image_np > 0], 50) if np.any(image_np > 0) else 0
        )
        mask_np = (image_np > thresh).astype(np.uint8)
    except Exception:
        mask_np = (image_np > 0).astype(np.uint8)

    if not np.any(mask_np):
        return image_np

    sitk_img = sitk.GetImageFromArray(image_np.astype(np.float32))
    sitk_mask = sitk.GetImageFromArray(mask_np)

    shrinkFactor = 4
    if all(sz > shrinkFactor for sz in sitk_img.GetSize()):
        inputImage, maskImage = (
            sitk.Shrink(sitk_img, [shrinkFactor] * 3),
            sitk.Shrink(sitk_mask, [shrinkFactor] * 3),
        )
    else:
        inputImage, maskImage = sitk_img, sitk_mask

    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations(n4_iters)
    try:
        corrector.Execute(inputImage, maskImage)
        logBiasField = corrector.GetLogBiasFieldAsImage(sitk_img)
        corrected = sitk_img / sitk.Exp(logBiasField)
        return sitk.GetArrayFromImage(corrected)
    except Exception:
        return None


def robust_standardize(image):
    foreground_mask = image > 1e-6
    if not np.any(foreground_mask):
        return image
    fg_pixels = image[foreground_mask]
    p1, p99 = np.percentile(fg_pixels, [1, 99])
    image_clipped = np.clip(image, p1, p99)
    fg_clipped = image_clipped[foreground_mask]
    mean, std = np.mean(fg_clipped), np.std(fg_clipped)
    out = np.zeros_like(image_clipped)
    out[foreground_mask] = (image_clipped[foreground_mask] - mean) / (std + 1e-8)
    return out


def robust_standardize_multiphase(vols_list):
    stack = np.stack(vols_list, axis=0).astype(np.float32)
    foreground_mask = np.any(stack > 1e-6, axis=0)
    if not np.any(foreground_mask):
        return list(stack)
    fg_pixels = stack[:, foreground_mask].reshape(-1)
    p1, p99 = np.percentile(fg_pixels, [1, 99])
    stack_clipped = np.clip(stack, p1, p99)
    fg_clipped = stack_clipped[:, foreground_mask].reshape(-1)
    mean, std = np.mean(fg_clipped), np.std(fg_clipped)
    out = np.zeros_like(stack_clipped)
    out[:, foreground_mask] = (
        stack_clipped[:, foreground_mask] - mean
    ) / (std + 1e-8)
    return [out[idx] for idx in range(out.shape[0])]


def standardize_volumes(vols_list, mode="per_phase"):
    mode = str(mode or "per_phase").lower()
    if mode in {"none", "raw"}:
        return [np.asarray(v, dtype=np.float32) for v in vols_list]
    if mode in {"joint", "joint_zscore", "multiphase"}:
        return robust_standardize_multiphase(vols_list)
    if mode in {"per_phase", "phase", "robust_zscore"}:
        return [robust_standardize(v) for v in vols_list]
    raise ValueError(f"Unsupported standardization_mode: {mode}")


def calculate_clinical_kinetics(early, late):
    sub = np.clip(late - early, 0, None)
    tissue_threshold = np.percentile(early[early > 0], 5) if np.any(early > 0) else 0.01
    eps = np.maximum(early * 0.05, tissue_threshold) + 1e-8
    return sub / (early + eps)


def align_to_ras(nii_img):
    orig_ornt = nib.io_orientation(nii_img.affine)
    ras_ornt = nib.orientations.axcodes2ornt("RAS")
    transform = nib.orientations.ornt_transform(orig_ornt, ras_ornt)
    return nii_img.as_reoriented(transform)


def _coerce_phase_index(value):
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            return None
        return int(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def _build_casefold_row_lookup(row_data):
    return {str(key).strip().casefold(): value for key, value in row_data.items()}


def _resolve_case_insensitive_column(df, column_name):
    lookup = {str(col).strip().casefold(): col for col in df.columns}
    return lookup.get(str(column_name).strip().casefold())


def _candidate_patient_keys(patient_name):
    raw_name = str(patient_name).strip()
    candidates = []

    def add_candidate(value):
        value = str(value).strip()
        if value and value.casefold() not in {
            existing.casefold() for existing in candidates
        }:
            candidates.append(value)

    add_candidate(raw_name)
    if "_" in raw_name:
        add_candidate(raw_name.split("_", 1)[1])
    if "-" in raw_name:
        add_candidate(raw_name.split("-", 1)[1])

    stripped_prefix = re.sub(r"^[A-Za-z]+[_-]*", "", raw_name)
    add_candidate(stripped_prefix)

    digits_only = re.sub(r"\D", "", raw_name)
    add_candidate(digits_only)
    return candidates


def _lookup_patient_row(labels_dict, patient_name):
    for candidate in _candidate_patient_keys(patient_name):
        row_data = labels_dict.get(candidate)
        if row_data is not None:
            return row_data, candidate
    return None, None


def _resolve_table_path(preferred_path):
    if os.path.exists(preferred_path):
        return preferred_path

    parent_dir = os.path.dirname(preferred_path)
    if not os.path.isdir(parent_dir):
        return preferred_path

    candidate_files = []
    for name in sorted(os.listdir(parent_dir)):
        if name.startswith("~$"):
            continue
        full_path = os.path.join(parent_dir, name)
        if not os.path.isfile(full_path):
            continue
        if not name.lower().endswith((".xlsx", ".xls", ".csv")):
            continue
        candidate_files.append(full_path)

    preferred_stem = os.path.splitext(os.path.basename(preferred_path))[0].casefold()
    for candidate in candidate_files:
        candidate_stem = os.path.splitext(os.path.basename(candidate))[0].casefold()
        if preferred_stem and (
            preferred_stem in candidate_stem or candidate_stem in preferred_stem
        ):
            return candidate

    if len(candidate_files) == 1:
        return candidate_files[0]
    return preferred_path


def _strip_nii_extension(name):
    lower_name = name.lower()
    if lower_name.endswith(".nii.gz"):
        return name[:-7]
    return os.path.splitext(name)[0]


def _extract_phase_index_from_filename(path, filename_phase_regex):
    basename = os.path.basename(path)
    match = re.search(filename_phase_regex, basename, flags=re.IGNORECASE)
    if not match:
        return None
    phase_value = match.groupdict().get("phase")
    if phase_value is None and match.groups():
        phase_value = match.group(1)
    return _coerce_phase_index(phase_value)


def _normalize_phase_value_candidates(value):
    if value is None or pd.isna(value):
        return []
    text = str(value).strip()
    if not text:
        return []

    candidates = []
    for candidate in (
        text,
        os.path.basename(text.replace("\\", "/")),
        _strip_nii_extension(os.path.basename(text.replace("\\", "/"))),
    ):
        candidate = str(candidate).strip()
        if candidate and candidate.casefold() not in {
            existing.casefold() for existing in candidates
        }:
            candidates.append(candidate)
    return candidates


def _match_metadata_value_to_path(value, image_entries):
    candidates = _normalize_phase_value_candidates(value)
    if not candidates:
        return None, None

    for matcher_name, matcher in (
        ("exact_filename", lambda token, entry: entry["basename"] == token),
        ("exact_stem", lambda token, entry: entry["stem"] == token),
        ("suffix_filename", lambda token, entry: entry["basename"].endswith(token)),
        ("suffix_stem", lambda token, entry: entry["stem"].endswith(token)),
    ):
        for candidate in candidates:
            token = candidate.casefold()
            if token.isdigit():
                continue
            matched = [entry for entry in image_entries if matcher(token, entry)]
            if len(matched) == 1:
                return matched[0]["path"], candidate
            if len(matched) > 1:
                return "__AMBIGUOUS__", candidate

    return None, None


def _match_phase_keyword_to_path(keyword_values, image_entries):
    if isinstance(keyword_values, str):
        keyword_values = [keyword_values]
    candidates = [str(value).strip().casefold() for value in (keyword_values or []) if str(value).strip()]
    if not candidates:
        return None, None

    ambiguous_matches = []
    for candidate in candidates:
        matched = [
            entry
            for entry in image_entries
            if candidate in entry["basename"] or candidate in entry["stem"]
        ]
        if len(matched) == 1:
            return matched[0]["path"], candidate
        if len(matched) > 1:
            ambiguous_matches.extend(matched)

    if len({entry["path"] for entry in ambiguous_matches}) > 1:
        return "__AMBIGUOUS__", "|".join(candidates)
    return None, None


def _match_phase_group_to_path(allowed_phase_values, phase_to_path):
    if isinstance(allowed_phase_values, (str, int, float)):
        allowed_phase_values = [allowed_phase_values]

    allowed_indices = []
    for value in allowed_phase_values or []:
        phase_index = _coerce_phase_index(value)
        if phase_index is not None:
            allowed_indices.append(phase_index)

    for phase_index in allowed_indices:
        matched_path = phase_to_path.get(phase_index)
        if matched_path is not None:
            return matched_path, phase_index
    return None, None

def resolve_phase_selected_images(all_nii, row_data, phase_selection_cfg):
    filename_phase_regex = phase_selection_cfg.get(
        "filename_phase_regex", r"(?:aqc|acq)[_-]?(?P<phase>\d+)"
    )
    row_lookup = _build_casefold_row_lookup(row_data)
    use_metadata_columns = phase_selection_cfg.get("use_metadata_columns", True)
    phase_filename_keywords = {
        str(key).strip().casefold(): value
        for key, value in phase_selection_cfg.get("phase_filename_keywords", {}).items()
    }
    fallback_phase_groups = {
        str(key).strip().casefold(): value
        for key, value in phase_selection_cfg.get("fallback_phase_groups", {}).items()
    }
    phase_columns = [
        str(col).strip()
        for col in phase_selection_cfg.get("phase_columns", ["post_early", "post_late"])
    ]

    image_entries = []
    phase_to_path = {}
    duplicate_phases = set()
    for nii_path in all_nii:
        if "mask" in os.path.basename(nii_path).lower():
            continue
        basename = os.path.basename(nii_path).casefold()
        stem = _strip_nii_extension(os.path.basename(nii_path)).casefold()
        image_entries.append({"path": nii_path, "basename": basename, "stem": stem})
        phase_index = _extract_phase_index_from_filename(nii_path, filename_phase_regex)
        if phase_index is None:
            continue
        if phase_index in phase_to_path:
            duplicate_phases.add(phase_index)
            continue
        phase_to_path[phase_index] = nii_path

    if duplicate_phases:
        return None, f"Duplicate image files found for phase indices: {sorted(duplicate_phases)}"

    selected_paths = []
    selected_phase_refs = []
    selected_phase_sources = []
    missing_columns = []
    missing_files = []
    duplicated_selected_paths = set()
    for col in phase_columns:
        col_key = col.casefold()
        raw_phase_value = row_lookup.get(col_key) if use_metadata_columns else None
        has_metadata_value = not (
            raw_phase_value is None
            or pd.isna(raw_phase_value)
            or not str(raw_phase_value).strip()
        )
        matched_path = None
        matched_ref = None
        matched_source = None

        if has_metadata_value:
            matched_path, matched_ref = _match_metadata_value_to_path(
                raw_phase_value, image_entries
            )
            if matched_path == "__AMBIGUOUS__":
                missing_files.append(f"{col}={raw_phase_value} (ambiguous match)")
                continue
            if matched_path is not None:
                matched_source = "metadata_filename"

            if matched_path is None:
                phase_index = _coerce_phase_index(raw_phase_value)
                if phase_index is None:
                    phase_index = _extract_phase_index_from_filename(
                        str(raw_phase_value), filename_phase_regex
                    )
                if phase_index is not None:
                    matched_path = phase_to_path.get(phase_index)
                    matched_ref = phase_index
                    if matched_path is not None:
                        matched_source = "metadata_phase_index"

        if matched_path is None and col_key in phase_filename_keywords:
            matched_path, matched_ref = _match_phase_keyword_to_path(
                phase_filename_keywords[col_key], image_entries
            )
            if matched_path == "__AMBIGUOUS__":
                missing_files.append(f"{col}=filename_keywords (ambiguous match)")
                continue
            if matched_path is not None:
                matched_source = "filename_keyword"

        if matched_path is None and col_key in fallback_phase_groups:
            matched_path, matched_ref = _match_phase_group_to_path(
                fallback_phase_groups[col_key], phase_to_path
            )
            if matched_path is not None:
                matched_source = "phase_group_fallback"

        if matched_path is None:
            if (
                use_metadata_columns
                and not has_metadata_value
                and col_key not in phase_filename_keywords
                and col_key not in fallback_phase_groups
            ):
                missing_columns.append(col)
            else:
                missing_files.append(
                    f"{col}={raw_phase_value if has_metadata_value else 'filename_keywords'}"
                )
            continue
        if matched_path in selected_paths:
            duplicated_selected_paths.add(os.path.basename(matched_path))
            continue
        selected_paths.append(matched_path)
        selected_phase_refs.append(matched_ref if matched_ref is not None else raw_phase_value)
        selected_phase_sources.append(matched_source if matched_source is not None else "unknown")

    if missing_columns:
        return None, f"Missing phase suffix columns or values in metadata: {missing_columns}"
    if missing_files:
        return None, f"Missing NIfTI files for metadata-selected phases: {missing_files}"
    if duplicated_selected_paths:
        return None, f"Duplicate metadata-selected phase files: {sorted(duplicated_selected_paths)}"
    if len(selected_paths) != len(phase_columns):
        return None, "Failed to resolve all metadata-selected phase files"

    return {
        "paths": selected_paths,
        "phase_indices": selected_phase_refs,
        "phase_sources": selected_phase_sources,
        "phase_columns": phase_columns,
    }, None


def crop_and_resize_4d_strict(vols_list, mask_vol, target_shape, interp_order):
    z_idx, y_idx, x_idx = np.where(mask_vol > 0.5)
    if len(z_idx) == 0:
        return None, None, None
    D, H, W = mask_vol.shape
    z_c, y_c, x_c = int(np.mean(z_idx)), int(np.mean(y_idx)), int(np.mean(x_idx))
    z_span = max(16, (np.max(z_idx) - np.min(z_idx)) // 2 + 5)
    y_span, x_span = 64, 64
    z_s, z_e = max(0, z_c - z_span), min(D, z_c + z_span)
    y_s, y_e = max(0, y_c - y_span), min(H, y_c + y_span)
    x_s, x_e = max(0, x_c - x_span), min(W, x_c + x_span)
    if z_e <= z_s:
        z_e = min(D, z_s + 1)
    if y_e <= y_s:
        y_e = min(H, y_s + 1)
    if x_e <= x_s:
        x_e = min(W, x_s + 1)

    c_mask = mask_vol[z_s:z_e, y_s:y_e, x_s:x_e]
    d_depth, d_height, d_width = c_mask.shape
    zoom_factors = (
        target_shape[0] / d_depth,
        target_shape[1] / d_height,
        target_shape[2] / d_width,
    )

    res_vols = []
    for vol in vols_list:
        c_vol = vol[z_s:z_e, y_s:y_e, x_s:x_e]
        res_vol = np.clip(
            zoom(
                c_vol, zoom_factors, order=interp_order, grid_mode=True, mode="reflect"
            ),
            0,
            None,
        )
        res_vols.append(res_vol)

    res_mask = (zoom(c_mask, zoom_factors, order=0, grid_mode=True) > 0.5).astype(
        np.float32
    )
    crop_metadata = {
        "original_depth": int(D),
        "original_height": int(H),
        "original_width": int(W),
        "crop_start_z": int(z_s),
        "crop_end_z": int(z_e),
        "crop_start_y": int(y_s),
        "crop_end_y": int(y_e),
        "crop_start_x": int(x_s),
        "crop_end_x": int(x_e),
        "crop_depth": int(d_depth),
        "crop_height": int(d_height),
        "crop_width": int(d_width),
        "target_depth": int(target_shape[0]),
        "target_height": int(target_shape[1]),
        "target_width": int(target_shape[2]),
        "zoom_factor_z": float(zoom_factors[0]),
        "zoom_factor_y": float(zoom_factors[1]),
        "zoom_factor_x": float(zoom_factors[2]),
        "roi_center_z": int(z_c),
        "roi_center_y": int(y_c),
        "roi_center_x": int(x_c),
        "roi_bbox_depth": int(np.max(z_idx) - np.min(z_idx) + 1),
        "roi_bbox_height": int(np.max(y_idx) - np.min(y_idx) + 1),
        "roi_bbox_width": int(np.max(x_idx) - np.min(x_idx) + 1),
        "roi_voxels_original": int(np.count_nonzero(mask_vol > 0.5)),
        "roi_voxels_cropped": int(np.count_nonzero(c_mask > 0.5)),
        "roi_voxels_resized": int(np.count_nonzero(res_mask > 0.5)),
        "num_phases": len(vols_list),
    }
    return res_vols, res_mask, crop_metadata


def process_single_patient(
    patient_dir,
    ds_name,
    row_data,
    pcr_col,
    out_root,
    problematic_raw_dir,
    target_shape,
    interp_order,
    n4_iters,
    n4_enabled,
    standardization_mode,
    phase_selection_cfg,
):
    pid = os.path.basename(patient_dir)
    problematic_case_id = f"{ds_name}_{pid}"

    def archive_problematic_case_and_report(reason):
        try:
            if problematic_raw_dir:
                os.makedirs(problematic_raw_dir, exist_ok=True)
                dest_path = os.path.join(problematic_raw_dir, problematic_case_id)
                if not os.path.exists(dest_path):
                    shutil.copytree(patient_dir, dest_path)
                with open(
                    os.path.join(dest_path, "_failure_reason.txt"),
                    "w",
                    encoding="utf-8",
                ) as f:
                    f.write(f"source_patient_dir: {patient_dir}\n")
                    f.write(f"reason: {reason}\n")
        except Exception as e:
            tqdm.write(
                f"  - Critical Error: Failed to archive problematic folder {problematic_case_id}. Reason: {e}"
            )

        report = row_data.copy()
        report["status"] = "failed"
        report["reason_for_failure"] = reason
        report["source_patient_dir"] = patient_dir
        if problematic_raw_dir:
            report["problematic_case_dir"] = os.path.join(
                problematic_raw_dir, problematic_case_id
            )
        return report

    try:
        all_nii = glob.glob(os.path.join(patient_dir, "**", "*.nii.gz"), recursive=True)
        masks = [f for f in all_nii if "mask" in f.lower()]

        if not masks:
            return archive_problematic_case_and_report("Mask file not found")

        if phase_selection_cfg.get("enabled", False):
            phase_selection, phase_error = resolve_phase_selected_images(
                all_nii, row_data, phase_selection_cfg
            )
            if phase_selection is None:
                return archive_problematic_case_and_report(phase_error)
            imgs = phase_selection["paths"]
        else:
            imgs = sorted([f for f in all_nii if "mask" not in f.lower()])
            if len(imgs) < 2:
                return archive_problematic_case_and_report("Not enough image phases (< 2)")

        if len(imgs) < 2:
            return archive_problematic_case_and_report(
                "Not enough metadata-selected image phases (< 2)"
            )

        v_m = align_to_ras(nib.load(masks[0])).get_fdata().squeeze()

        vols_n4 = []
        for img_path in imgs:
            vol = align_to_ras(nib.load(img_path)).get_fdata().squeeze()
            if n4_enabled:
                vol_n4 = fast_n4_bias_field_correction(vol, n4_iters)
                if vol_n4 is None:
                    return archive_problematic_case_and_report(
                        f"N4 correction failed on phase: {os.path.basename(img_path)}"
                    )
            else:
                vol_n4 = vol
            vols_n4.append(vol_n4)

        res_vols, re_m, crop_metadata = crop_and_resize_4d_strict(
            vols_n4, v_m, target_shape, interp_order
        )
        if res_vols is None:
            return archive_problematic_case_and_report("Mask is empty (all zeros)")

        # Stack to T x D x H x W. Joint z-score preserves cross-phase enhancement
        # scale better than independent per-phase standardization.
        std_vols = standardize_volumes(res_vols, standardization_mode)
        img_tensor = np.stack(std_vols, axis=0).astype(np.float32)
        mask_tensor = np.expand_dims(re_m, axis=0).astype(np.float32)

        final_id = f"{ds_name}_{pid}"
        np.save(os.path.join(out_root, "images", f"{final_id}.npy"), img_tensor)
        np.save(os.path.join(out_root, "masks", f"{final_id}.npy"), mask_tensor)

        report = row_data.copy()
        report["status"] = "success"
        report["id"] = final_id
        report["center"] = ds_name
        report[pcr_col] = int(float(row_data[pcr_col]))
        if phase_selection_cfg.get("enabled", False):
            report["selected_phase_columns"] = ",".join(phase_selection["phase_columns"])
            report["selected_phase_indices"] = ",".join(
                str(idx) for idx in phase_selection["phase_indices"]
            )
            report["selected_phase_sources"] = ",".join(
                str(src) for src in phase_selection.get("phase_sources", [])
            )
            report["selected_phase_files"] = "|".join(
                os.path.basename(path) for path in phase_selection["paths"]
            )
        report.update(crop_metadata)
        return report

    except Exception as e:
        return archive_problematic_case_and_report(f"Unexpected error: {str(e)}")


def build_ultimate_dataset():
    CONFIG = get_config()
    DATA_CFG = CONFIG["data_prep"]
    TARGET_SHAPE = tuple(DATA_CFG["target_shape"])
    PROC_PARAMS = DATA_CFG.get("processing_params", {})
    INTERP_ORDER = PROC_PARAMS.get("interpolation_order", 3)
    N4_ITERS = PROC_PARAMS.get("n4_iterations", [50, 50, 50, 50])
    N4_ENABLED = bool(PROC_PARAMS.get("n4_enabled", True))
    STANDARDIZATION_MODE = PROC_PARAMS.get("standardization_mode", "per_phase")
    MAX_WORKERS = PROC_PARAMS.get("max_workers", 8)
    PHASE_SELECTION_CFG = DATA_CFG.get("phase_selection", {})

    out_pool = DATA_CFG["unified_pool_dir"]
    problematic_dir = DATA_CFG.get("problematic_raw_dir", "problematic_data")

    os.makedirs(out_pool, exist_ok=True)
    os.makedirs(problematic_dir, exist_ok=True)

    print(f"⚡ Starting data factory! Workers: {MAX_WORKERS}")
    print(f"✔️ Clean data will be saved to: {out_pool}")
    print(f"❌ Problematic raw data will be moved to: {problematic_dir}")

    def load_safe_table(p, pid_c, pcr_c):
        resolved_path = _resolve_table_path(p)
        if not os.path.exists(resolved_path):
            return {}
        try:
            df = (
                pd.read_excel(resolved_path)
                if resolved_path.endswith((".xls", ".xlsx"))
                else pd.read_csv(resolved_path, sep=None, engine="python")
            )
            pid_col = _resolve_case_insensitive_column(df, pid_c)
            pcr_col = _resolve_case_insensitive_column(df, pcr_c)
            if pid_col is None or pcr_col is None:
                return {}
            df = df.dropna(subset=[pid_col, pcr_col])
            return {
                str(row[pid_col]).strip(): row.to_dict()
                for _, row in df.iterrows()
                if str(row[pid_col]).strip()
            }
        except Exception as e:
            print(f"⚠️ Failed to parse table {resolved_path}: {e}")
            return {}

    overall_successful = []
    overall_failed = []
    for ds_name, cfg in DATA_CFG["datasets"].items():
        print(f"\n🚀 Processing: {ds_name}")
        labels_dict = load_safe_table(cfg["table_path"], cfg["pid_col"], cfg["pcr_col"])

        if not labels_dict:
            print(f"  [!] No labels found for {ds_name}. Skipping.")
            continue

        ds_out_root = os.path.join(out_pool, ds_name)
        ds_problematic_dir = os.path.join(problematic_dir, ds_name)
        os.makedirs(os.path.join(ds_out_root, "images"), exist_ok=True)
        os.makedirs(os.path.join(ds_out_root, "masks"), exist_ok=True)
        os.makedirs(ds_problematic_dir, exist_ok=True)

        patient_dirs = [
            os.path.join(cfg["img_dir"], d)
            for d in os.listdir(cfg["img_dir"])
            if os.path.isdir(os.path.join(cfg["img_dir"], d))
        ]

        ds_successful_meta = []
        ds_failed_meta = []
        tasks = []
        for p_dir in patient_dirs:
            p_name = os.path.basename(p_dir)
            row_data, matched_pid = _lookup_patient_row(labels_dict, p_name)
            if row_data:
                ds_phase_selection_cfg = dict(PHASE_SELECTION_CFG)
                ds_phase_selection_cfg.update(cfg.get("phase_selection", {}))
                row_data = row_data.copy()
                row_data["matched_pid"] = matched_pid
                row_data["source_patient_dirname"] = p_name
                tasks.append(
                    (
                        p_dir,
                        ds_name,
                        row_data,
                        cfg["pcr_col"],
                        ds_out_root,
                        ds_problematic_dir,
                        TARGET_SHAPE,
                        INTERP_ORDER,
                        N4_ITERS,
                        N4_ENABLED,
                        STANDARDIZATION_MODE,
                        ds_phase_selection_cfg,
                    )
                )

        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [
                executor.submit(process_single_patient, *task_args)
                for task_args in tasks
            ]

            for fut in tqdm(as_completed(futures), total=len(futures), desc=ds_name):
                res = fut.result()
                if res:
                    status = res.pop("status", "failed")
                    if status == "success":
                        ds_successful_meta.append(res)
                        overall_successful.append(res)
                    else:
                        ds_failed_meta.append(res)
                        overall_failed.append(res)

        if ds_successful_meta:
            success_df = pd.DataFrame(ds_successful_meta)
            metadata_csv_path = os.path.join(ds_out_root, "dataset_metadata.csv")
            metadata_xlsx_path = os.path.join(ds_out_root, "dataset_metadata.xlsx")
            success_df.to_csv(metadata_csv_path, index=False)
            try:
                success_df.to_excel(metadata_xlsx_path, index=False)
            except Exception:
                metadata_xlsx_path = None

            qc_summary_path = os.path.join(ds_out_root, "roi_qc_summary.csv")
            qc_json_path = os.path.join(ds_out_root, "roi_qc_summary.json")
            qc_summary = (
                success_df.groupby("center")
                .agg(
                    num_cases=("id", "count"),
                    mean_roi_voxels_original=("roi_voxels_original", "mean"),
                    mean_roi_voxels_resized=("roi_voxels_resized", "mean"),
                    mean_crop_depth=("crop_depth", "mean"),
                    mean_crop_height=("crop_height", "mean"),
                    mean_crop_width=("crop_width", "mean"),
                    mean_zoom_factor_z=("zoom_factor_z", "mean"),
                    mean_zoom_factor_y=("zoom_factor_y", "mean"),
                    mean_zoom_factor_x=("zoom_factor_x", "mean"),
                )
                .reset_index()
            )
            qc_summary.to_csv(qc_summary_path, index=False)
            with open(qc_json_path, "w", encoding="utf-8") as f:
                json.dump(
                    qc_summary.to_dict(orient="records"), f, ensure_ascii=False, indent=2
                )
            print(f"  [OK] {ds_name}: processed {len(ds_successful_meta)} cases")
            print(f"       data dir: {ds_out_root}")
            print(f"       metadata csv: {metadata_csv_path}")
            if metadata_xlsx_path is not None:
                print(f"       metadata xlsx: {metadata_xlsx_path}")
        else:
            print(f"  [!] {ds_name}: no patients processed successfully")

        if ds_failed_meta:
            failed_df = pd.DataFrame(ds_failed_meta)
            failed_report_path = os.path.join(
                ds_problematic_dir, "problematic_data_report.csv"
            )
            failed_df.to_csv(failed_report_path, index=False)
            print(f"  [!] {ds_name}: archived {len(ds_failed_meta)} problematic cases")
            print(f"       problematic dir: {ds_problematic_dir}")
            print(f"       failure report: {failed_report_path}")

    successful_meta = overall_successful
    failed_meta = overall_failed

    print("\n" + "=" * 70)
    print("📊 FINAL REPORT")
    print("=" * 70)

    if successful_meta:
        success_df = pd.DataFrame(successful_meta)
        metadata_path = os.path.join(out_pool, "dataset_metadata.csv")
        success_df.to_csv(metadata_path, index=False)
        qc_summary_path = os.path.join(out_pool, "roi_qc_summary.csv")
        qc_json_path = os.path.join(out_pool, "roi_qc_summary.json")
        qc_summary = (
            success_df.groupby("center")
            .agg(
                num_cases=("id", "count"),
                mean_roi_voxels_original=("roi_voxels_original", "mean"),
                mean_roi_voxels_resized=("roi_voxels_resized", "mean"),
                mean_crop_depth=("crop_depth", "mean"),
                mean_crop_height=("crop_height", "mean"),
                mean_crop_width=("crop_width", "mean"),
                mean_zoom_factor_z=("zoom_factor_z", "mean"),
                mean_zoom_factor_y=("zoom_factor_y", "mean"),
                mean_zoom_factor_x=("zoom_factor_x", "mean"),
            )
            .reset_index()
        )
        qc_summary.to_csv(qc_summary_path, index=False)
        with open(qc_json_path, "w", encoding="utf-8") as f:
            json.dump(
                qc_summary.to_dict(orient="records"), f, ensure_ascii=False, indent=2
            )
        print(f"✅ Successfully processed {len(successful_meta)} patients.")
        print(f"   - Clean .npy dataset saved in: {out_pool}")
        print(f"   - Final metadata (with ROI crop details) saved to: {metadata_path}")
        print(f"   - ROI QC summary saved to: {qc_summary_path}")
    else:
        print("❌ No patients were processed successfully.")

    if failed_meta:
        failed_df = pd.DataFrame(failed_meta)
        failed_df.to_csv(
            os.path.join(problematic_dir, "problematic_data_report.csv"), index=False
        )
        print(f"\n❌ Moved {len(failed_meta)} problematic raw data folders.")
        print(f"   - Moved folders are located in: {problematic_dir}")
        print(
            f"   - Detailed failure report (with all info) saved to: {os.path.join(problematic_dir, 'problematic_data_report.csv')}"
        )
    else:
        print("\n✅ No problematic data was found.")

    print("\n🚀 Data factory finished!")


if __name__ == "__main__":
    build_ultimate_dataset()
