from __future__ import annotations

import argparse
import contextlib
import csv
import datetime as dt
import json
import math
import os
import platform
import socket
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_CONFIGS = [
    "configs/server/cv/train_config_Proposed_3D_Base.yaml",
    "configs/server/cv/train_config_RAPSS_Net.yaml",
]
DEFAULT_PREP_CONFIG = "configs/server/prep_config_public_server.yaml"


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a dictionary: {path}")
    return data


def _load_training_config(path: str | Path) -> dict[str, Any]:
    from utils.config_manager import ConfigManager

    return ConfigManager(str(path)).config


def _load_prep_target_shape(path: str | Path) -> tuple[int, int, int]:
    data = _load_yaml(path)
    try:
        raw_shape = data["data_prep"]["target_shape"]
    except KeyError as exc:
        raise KeyError(f"Missing data_prep.target_shape in {path}") from exc
    if not isinstance(raw_shape, (list, tuple)) or len(raw_shape) != 3:
        raise ValueError("data_prep.target_shape must contain [D, H, W].")
    shape = tuple(int(v) for v in raw_shape)
    if any(v <= 0 for v in shape):
        raise ValueError("data_prep.target_shape values must be positive.")
    return shape


def _sanitize_filename(value: str) -> str:
    clean = []
    for char in str(value):
        if char.isalnum() or char in {"-", "_", "."}:
            clean.append(char)
        else:
            clean.append("_")
    return "".join(clean).strip("_") or "profile"


def _resolve_configs(config_args: list[str] | None) -> list[Path]:
    raw_configs = config_args or DEFAULT_CONFIGS
    config_paths = []
    for raw_path in raw_configs:
        path = Path(raw_path)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if not path.exists():
            raise FileNotFoundError(f"Config not found: {path}")
        config_paths.append(path)
    return config_paths


def _resolve_prep_config(path_arg: str) -> Path:
    path = Path(path_arg)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"Preprocessing config not found: {path}")
    return path


def _resolve_output_dir(path_arg: str) -> Path:
    path = Path(path_arg)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path.mkdir(parents=True, exist_ok=True)
    return path


def summarize_rapss_modules(config: dict[str, Any]) -> dict[str, Any]:
    model_cfg = config.get("model", {}) or {}
    mamba_layers = int(model_cfg.get("mamba_layers", 0))
    enabled_labels = []
    if mamba_layers > 0:
        enabled_labels.append(f"{mamba_layers} DSTA-Mamba block(s)")
    if bool(model_cfg.get("use_peritumor", False)):
        enabled_labels.append("peritumoral dual stream")
    if bool(model_cfg.get("use_dynamic_scan_router", False)):
        enabled_labels.append("dynamic axial routing")
    if bool(model_cfg.get("use_explicit_peritumor_ring", False)):
        enabled_labels.append("explicit peritumoral-ring input")
    if bool(model_cfg.get("use_kinetics_channel", False)):
        enabled_labels.append("kinetics-aware input branch")
    if bool(model_cfg.get("use_multiscale_feature_aggregation", False)):
        enabled_labels.append("multi-scale feature aggregation")
    if bool(model_cfg.get("use_feature_recalibration", False)):
        enabled_labels.append("global feature recalibration")

    return {
        "dsta_mamba_blocks": mamba_layers,
        "peritumor_dual_stream": bool(model_cfg.get("use_peritumor", False)),
        "dynamic_axial_routing": bool(
            model_cfg.get("use_dynamic_scan_router", False)
        ),
        "explicit_peritumor_ring": bool(
            model_cfg.get("use_explicit_peritumor_ring", False)
        ),
        "kinetics_channel": bool(model_cfg.get("use_kinetics_channel", False)),
        "multiscale_feature_aggregation": bool(
            model_cfg.get("use_multiscale_feature_aggregation", False)
        ),
        "feature_recalibration": bool(
            model_cfg.get("use_feature_recalibration", False)
        ),
        "enabled_module_labels": enabled_labels,
    }


def _require_torch():
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "PyTorch is required for efficiency profiling. Install the project "
            "runtime environment before running tools/profile_efficiency.py."
        ) from exc
    return torch


def _resolve_device(torch_module, requested: str, config: dict[str, Any]):
    requested = str(requested or "auto").lower()
    config_device = str(config.get("project", {}).get("device", "cuda")).lower()
    if requested == "auto":
        requested = "cuda" if torch_module.cuda.is_available() else "cpu"
    elif requested == "config":
        requested = config_device

    if requested.startswith("cuda") and not torch_module.cuda.is_available():
        raise RuntimeError(
            f"Requested device '{requested}', but CUDA is not available in this environment."
        )
    return torch_module.device(requested)


def _resolve_amp_enabled(
    torch_module, amp_arg: str, config: dict[str, Any], device
) -> bool:
    if device.type != "cuda":
        return False
    if amp_arg == "on":
        return True
    if amp_arg == "off":
        return False
    return bool(config.get("train", {}).get("amp", False))


@contextlib.contextmanager
def _autocast_context(torch_module, enabled: bool, device):
    if not enabled or device.type != "cuda":
        yield
        return
    if hasattr(torch_module, "amp") and hasattr(torch_module.amp, "autocast"):
        with torch_module.amp.autocast(device_type="cuda", enabled=True):
            yield
    else:
        with torch_module.cuda.amp.autocast(enabled=True):
            yield


def _seed_torch(torch_module, seed: int) -> None:
    torch_module.manual_seed(seed)
    if torch_module.cuda.is_available():
        torch_module.cuda.manual_seed_all(seed)


def _make_center_mask(torch_module, batch_size: int, shape: tuple[int, int, int], device):
    d, h, w = shape
    mask = torch_module.zeros((batch_size, 1, d, h, w), device=device)
    d0, d1 = max(0, d // 2 - max(1, d // 8)), min(d, d // 2 + max(1, d // 8))
    h0, h1 = max(0, h // 2 - max(1, h // 8)), min(h, h // 2 + max(1, h // 8))
    w0, w1 = max(0, w // 2 - max(1, w // 8)), min(w, w // 2 + max(1, w // 8))
    mask[:, :, d0:d1, h0:h1, w0:w1] = 1.0
    return mask


def _build_synthetic_inputs(
    torch_module,
    config: dict[str, Any],
    shape: tuple[int, int, int],
    batch_size: int,
    device,
) -> dict[str, Any]:
    model_cfg = config.get("model", {}) or {}
    data_cfg = config.get("data", {}) or {}
    in_channels = int(model_cfg.get("in_channels", 2))
    image = torch_module.randn((batch_size, in_channels, *shape), device=device)
    kinetics = (
        torch_module.randn((batch_size, *shape), device=device)
        if bool(model_cfg.get("use_kinetics_channel", False))
        else None
    )
    lesion_mask = _make_center_mask(torch_module, batch_size, shape, device)

    clinical = None
    if bool(model_cfg.get("use_clinical", False)):
        clinical_dim = int(model_cfg.get("clinical_input_dim", 0))
        if clinical_dim <= 0:
            clinical_keys = data_cfg.get("clinical_keys", []) or []
            missing_indicator = bool(data_cfg.get("clinical_missing_indicator", True))
            clinical_dim = len(clinical_keys) * (2 if missing_indicator else 1)
        if clinical_dim > 0:
            clinical = torch_module.zeros((batch_size, clinical_dim), device=device)

    return {
        "image": image,
        "kinetics": kinetics,
        "lesion_mask": lesion_mask,
        "clinical": clinical,
    }


def _forward_model(model, inputs: dict[str, Any]):
    return model(
        inputs["image"],
        kinetics=inputs.get("kinetics"),
        lesion_mask=inputs.get("lesion_mask"),
        clinical=inputs.get("clinical"),
    )


def _sync_if_needed(torch_module, device) -> None:
    if device.type == "cuda":
        torch_module.cuda.synchronize(device)


def count_parameters(model) -> dict[str, int]:
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return {
        "total_params": int(total),
        "trainable_params": int(trainable),
        "non_trainable_params": int(total - trainable),
    }


def _first_tensor(torch_module, value):
    if isinstance(value, torch_module.Tensor):
        return value
    if isinstance(value, (list, tuple)):
        for item in value:
            found = _first_tensor(torch_module, item)
            if found is not None:
                return found
    if isinstance(value, dict):
        for item in value.values():
            found = _first_tensor(torch_module, item)
            if found is not None:
                return found
    return None


def _prod(values) -> int:
    result = 1
    for value in values:
        result *= int(value)
    return int(result)


def _mamba_macs(module, input_tensor) -> tuple[int, dict[str, Any]]:
    if input_tensor is None or input_tensor.dim() != 3:
        return 0, {"reason": "unexpected Mamba input shape"}
    batch, length, dim = [int(v) for v in input_tensor.shape]
    d_model = int(getattr(module, "d_model", dim))
    d_inner = int(getattr(module, "d_inner", d_model * 2))
    d_state = int(getattr(module, "d_state", 16))
    d_conv = int(getattr(module, "d_conv", 4))
    dt_rank_raw = getattr(module, "dt_rank", max(1, math.ceil(d_model / 16)))
    dt_rank = int(dt_rank_raw if isinstance(dt_rank_raw, int) else max(1, math.ceil(d_model / 16)))
    tokens = batch * length

    in_proj = tokens * d_model * (2 * d_inner)
    depthwise_conv = batch * d_inner * length * d_conv
    x_proj = tokens * d_inner * (dt_rank + 2 * d_state)
    dt_proj = tokens * dt_rank * d_inner
    selective_scan_equivalent = tokens * d_inner * d_state * 4
    out_proj = tokens * d_inner * d_model
    macs = (
        in_proj
        + depthwise_conv
        + x_proj
        + dt_proj
        + selective_scan_equivalent
        + out_proj
    )
    detail = {
        "d_model": d_model,
        "d_inner": d_inner,
        "d_state": d_state,
        "d_conv": d_conv,
        "dt_rank": dt_rank,
        "sequence_length": length,
        "batch_like": batch,
        "method": "analytic_mamba_estimate",
    }
    return int(macs), detail


def estimate_macs(
    torch_module,
    model,
    forward_fn: Callable[[], Any],
    amp_enabled: bool,
    device,
    include_breakdown: bool = False,
) -> dict[str, Any]:
    nn = torch_module.nn
    records: list[dict[str, Any]] = []
    handles = []

    mamba_prefixes = []
    for name, module in model.named_modules():
        if name and module.__class__.__name__ == "Mamba":
            mamba_prefixes.append(name)

    def is_inside_counted_mamba(module_name: str) -> bool:
        return any(module_name.startswith(prefix + ".") for prefix in mamba_prefixes)

    def add_record(module_name: str, module_type: str, macs: int, detail: dict[str, Any]):
        records.append(
            {
                "module": module_name,
                "type": module_type,
                "macs": int(macs),
                "flops": int(macs) * 2,
                "detail": detail,
            }
        )

    def hook_factory(module_name: str, module):
        def hook(_module, inputs, output):
            input_tensor = _first_tensor(torch_module, inputs)
            output_tensor = _first_tensor(torch_module, output)
            if output_tensor is None:
                return
            module_type = module.__class__.__name__
            macs = 0
            detail: dict[str, Any] = {}

            if module_type == "Mamba":
                macs, detail = _mamba_macs(module, input_tensor)
            elif isinstance(
                module,
                (
                    nn.Conv1d,
                    nn.Conv2d,
                    nn.Conv3d,
                    nn.ConvTranspose1d,
                    nn.ConvTranspose2d,
                    nn.ConvTranspose3d,
                ),
            ):
                kernel_ops = (int(module.in_channels) // int(module.groups)) * _prod(
                    module.kernel_size
                )
                macs = int(output_tensor.numel()) * kernel_ops
                detail = {
                    "kernel_size": list(module.kernel_size),
                    "groups": int(module.groups),
                    "in_channels": int(module.in_channels),
                    "out_shape": list(output_tensor.shape),
                    "method": "conv_macs",
                }
            elif isinstance(module, nn.Linear):
                macs = int(output_tensor.numel()) * int(module.in_features)
                detail = {
                    "in_features": int(module.in_features),
                    "out_features": int(module.out_features),
                    "out_shape": list(output_tensor.shape),
                    "method": "linear_macs",
                }

            if macs > 0:
                add_record(module_name, module_type, macs, detail)

        return hook

    supported = (
        nn.Conv1d,
        nn.Conv2d,
        nn.Conv3d,
        nn.ConvTranspose1d,
        nn.ConvTranspose2d,
        nn.ConvTranspose3d,
        nn.Linear,
    )

    for name, module in model.named_modules():
        if not name:
            continue
        if is_inside_counted_mamba(name):
            continue
        if module.__class__.__name__ == "Mamba" or isinstance(module, supported):
            handles.append(module.register_forward_hook(hook_factory(name, module)))

    try:
        with torch_module.inference_mode():
            with _autocast_context(torch_module, amp_enabled, device):
                forward_fn()
        _sync_if_needed(torch_module, device)
    finally:
        for handle in handles:
            handle.remove()

    total_macs = int(sum(row["macs"] for row in records))
    by_type: dict[str, int] = {}
    for row in records:
        by_type[row["type"]] = by_type.get(row["type"], 0) + int(row["macs"])

    result = {
        "estimated_macs": total_macs,
        "estimated_flops": total_macs * 2,
        "flops_definition": "FLOPs are reported as 2 x MACs.",
        "flops_scope": (
            "Hook-based Conv/Linear counting plus analytic estimates for detected "
            "mamba_ssm.Mamba modules. Functional elementwise ops, reductions, "
            "indexing, reshapes, and interpolation are not included."
        ),
        "macs_by_module_type": by_type,
        "mamba_modules_counted": int(sum(1 for row in records if row["type"] == "Mamba")),
    }
    if include_breakdown:
        result["module_breakdown"] = records
    return result


def measure_latency_ms(
    torch_module,
    forward_fn: Callable[[], Any],
    warmup: int,
    repeats: int,
    amp_enabled: bool,
    device,
) -> dict[str, float]:
    if repeats <= 0:
        raise ValueError("--repeats must be positive.")
    warmup = max(0, int(warmup))
    repeats = int(repeats)
    timings = []

    with torch_module.inference_mode():
        for _ in range(warmup):
            with _autocast_context(torch_module, amp_enabled, device):
                forward_fn()
        _sync_if_needed(torch_module, device)

        if device.type == "cuda":
            starter = torch_module.cuda.Event(enable_timing=True)
            ender = torch_module.cuda.Event(enable_timing=True)
            for _ in range(repeats):
                starter.record()
                with _autocast_context(torch_module, amp_enabled, device):
                    forward_fn()
                ender.record()
                torch_module.cuda.synchronize(device)
                timings.append(float(starter.elapsed_time(ender)))
        else:
            for _ in range(repeats):
                start = time.perf_counter()
                with _autocast_context(torch_module, amp_enabled, device):
                    forward_fn()
                timings.append((time.perf_counter() - start) * 1000.0)

    return {
        "mean_ms": float(statistics.mean(timings)),
        "median_ms": float(statistics.median(timings)),
        "std_ms": float(statistics.pstdev(timings)) if len(timings) > 1 else 0.0,
        "min_ms": float(min(timings)),
        "max_ms": float(max(timings)),
        "p90_ms": float(_percentile(timings, 90.0)),
        "repeats": repeats,
        "warmup": warmup,
    }


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return float("nan")
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (len(sorted_values) - 1) * (percentile / 100.0)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return sorted_values[int(rank)]
    weight = rank - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def measure_peak_cuda_memory_mb(
    torch_module,
    forward_fn: Callable[[], Any],
    amp_enabled: bool,
    device,
) -> dict[str, float | None]:
    if device.type != "cuda":
        return {
            "peak_allocated_mb": None,
            "peak_reserved_mb": None,
        }
    torch_module.cuda.empty_cache()
    torch_module.cuda.reset_peak_memory_stats(device)
    with torch_module.inference_mode():
        with _autocast_context(torch_module, amp_enabled, device):
            forward_fn()
    torch_module.cuda.synchronize(device)
    return {
        "peak_allocated_mb": float(
            torch_module.cuda.max_memory_allocated(device) / (1024.0**2)
        ),
        "peak_reserved_mb": float(
            torch_module.cuda.max_memory_reserved(device) / (1024.0**2)
        ),
    }


def collect_environment(torch_module, device) -> dict[str, Any]:
    cuda_info: dict[str, Any] = {
        "cuda_available": bool(torch_module.cuda.is_available()),
        "torch_cuda_version": getattr(torch_module.version, "cuda", None),
        "cudnn_version": torch_module.backends.cudnn.version()
        if hasattr(torch_module.backends, "cudnn")
        else None,
    }
    if device.type == "cuda":
        props = torch_module.cuda.get_device_properties(device)
        cuda_info.update(
            {
                "device_name": torch_module.cuda.get_device_name(device),
                "device_index": int(device.index or torch_module.cuda.current_device()),
                "total_memory_mb": float(props.total_memory / (1024.0**2)),
            }
        )
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "torch_version": torch_module.__version__,
        "device": str(device),
        "cuda": cuda_info,
    }


def _load_checkpoint_if_requested(torch_module, model, checkpoint_path: str | None, device):
    if checkpoint_path is None:
        return {"checkpoint_path": None, "checkpoint_state_key": None}
    checkpoint = torch_module.load(checkpoint_path, map_location=device)
    state_key = None
    if isinstance(checkpoint, dict):
        for candidate in ("ema_model_state_dict", "model_state_dict", "state_dict"):
            if candidate in checkpoint:
                state_key = candidate
                checkpoint = checkpoint[candidate]
                break
    model.load_state_dict(checkpoint)
    return {"checkpoint_path": str(checkpoint_path), "checkpoint_state_key": state_key}


def profile_one_config(
    config_path: Path,
    input_shape: tuple[int, int, int],
    args: argparse.Namespace,
) -> dict[str, Any]:
    torch_module = _require_torch()
    from models.builder import build_model

    config = _load_training_config(config_path)
    seed = int(config.get("project", {}).get("seed", 42))
    _seed_torch(torch_module, seed)
    if hasattr(torch_module.backends, "cudnn"):
        torch_module.backends.cudnn.benchmark = bool(args.cudnn_benchmark)

    device = _resolve_device(torch_module, args.device, config)
    amp_enabled = _resolve_amp_enabled(torch_module, args.amp, config, device)
    try:
        model = build_model(config, device)
    except ModuleNotFoundError as exc:
        if exc.name == "mamba_ssm":
            raise ModuleNotFoundError(
                f"{config_path.name} enables Mamba layers, but mamba_ssm is not "
                "installed in this environment. Install mamba-ssm or profile a "
                "configuration with model.mamba_layers=0."
            ) from exc
        raise
    model.eval()
    checkpoint_info = _load_checkpoint_if_requested(
        torch_module, model, args.checkpoint, device
    )
    if bool(args.compile):
        if not hasattr(torch_module, "compile"):
            raise RuntimeError("This PyTorch build does not provide torch.compile.")
        model = torch_module.compile(model)

    inputs = _build_synthetic_inputs(
        torch_module,
        config,
        input_shape,
        int(args.batch_size),
        device,
    )

    def forward_fn():
        return _forward_model(model, inputs)

    with torch_module.inference_mode():
        with _autocast_context(torch_module, amp_enabled, device):
            forward_fn()
    _sync_if_needed(torch_module, device)

    params = count_parameters(model)
    if args.flops == "skip":
        flops = {
            "estimated_macs": None,
            "estimated_flops": None,
            "flops_definition": None,
            "flops_scope": "Skipped by --flops skip.",
            "macs_by_module_type": {},
            "mamba_modules_counted": 0,
        }
    else:
        flops = estimate_macs(
            torch_module,
            model,
            forward_fn,
            amp_enabled,
            device,
            include_breakdown=bool(args.include_module_breakdown),
        )

    latency = measure_latency_ms(
        torch_module,
        forward_fn,
        warmup=int(args.warmup),
        repeats=int(args.repeats),
        amp_enabled=amp_enabled,
        device=device,
    )
    peak_memory = measure_peak_cuda_memory_mb(
        torch_module,
        forward_fn,
        amp_enabled=amp_enabled,
        device=device,
    )

    model_cfg = config.get("model", {}) or {}
    result = {
        "project_name": config.get("project", {}).get("name", config_path.stem),
        "config_path": str(config_path),
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "input": {
            "batch_size": int(args.batch_size),
            "in_channels": int(model_cfg.get("in_channels", 2)),
            "spatial_shape_dhw": list(input_shape),
            "synthetic_input": True,
            "kinetics_input_provided": bool(model_cfg.get("use_kinetics_channel", False)),
            "lesion_mask_input_provided": True,
            "clinical_input_provided": inputs.get("clinical") is not None,
        },
        "inference_settings": {
            "device": str(device),
            "amp_enabled": bool(amp_enabled),
            "cudnn_benchmark": bool(args.cudnn_benchmark),
            "torch_compile": bool(args.compile),
            "warmup": int(args.warmup),
            "repeats": int(args.repeats),
        },
        "parameters": params,
        "flops": flops,
        "latency": latency,
        "cuda_memory": peak_memory,
        "rapss_modules": summarize_rapss_modules(config),
        "checkpoint": checkpoint_info,
        "environment": collect_environment(torch_module, device),
    }
    return result


def _short_number(value: int | float | None) -> str:
    if value is None:
        return ""
    value = float(value)
    abs_value = abs(value)
    if abs_value >= 1e12:
        return f"{value / 1e12:.3f}T"
    if abs_value >= 1e9:
        return f"{value / 1e9:.3f}G"
    if abs_value >= 1e6:
        return f"{value / 1e6:.3f}M"
    if abs_value >= 1e3:
        return f"{value / 1e3:.3f}K"
    return f"{value:.3f}"


def _result_to_row(result: dict[str, Any]) -> dict[str, Any]:
    memory = result.get("cuda_memory", {})
    flops = result.get("flops", {})
    latency = result.get("latency", {})
    params = result.get("parameters", {})
    input_info = result.get("input", {})
    return {
        "project_name": result.get("project_name"),
        "config_path": result.get("config_path"),
        "input_shape_dhw": "x".join(str(v) for v in input_info.get("spatial_shape_dhw", [])),
        "batch_size": input_info.get("batch_size"),
        "device": result.get("inference_settings", {}).get("device"),
        "amp_enabled": result.get("inference_settings", {}).get("amp_enabled"),
        "total_params": params.get("total_params"),
        "trainable_params": params.get("trainable_params"),
        "estimated_macs": flops.get("estimated_macs"),
        "estimated_flops": flops.get("estimated_flops"),
        "latency_mean_ms": latency.get("mean_ms"),
        "latency_median_ms": latency.get("median_ms"),
        "latency_std_ms": latency.get("std_ms"),
        "latency_p90_ms": latency.get("p90_ms"),
        "peak_allocated_mb": memory.get("peak_allocated_mb"),
        "peak_reserved_mb": memory.get("peak_reserved_mb"),
        "warmup": latency.get("warmup"),
        "repeats": latency.get("repeats"),
        "mamba_modules_counted": flops.get("mamba_modules_counted"),
        "flops_scope": flops.get("flops_scope"),
    }


def write_outputs(results: list[dict[str, Any]], output_dir: Path, tag: str) -> dict[str, str]:
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    tag_part = f"_{_sanitize_filename(tag)}" if tag else ""
    stem = f"efficiency_profile{tag_part}_{timestamp}"

    json_path = output_dir / f"{stem}.json"
    csv_path = output_dir / f"{stem}.csv"
    md_path = output_dir / f"{stem}.md"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump({"results": results}, f, indent=2)

    rows = [_result_to_row(result) for result in results]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with md_path.open("w", encoding="utf-8") as f:
        f.write("# Efficiency Profiling Results\n\n")
        f.write(
            "FLOPs are estimated as `2 x MACs`. The estimate includes Conv/Linear "
            "modules and analytic estimates for detected `mamba_ssm.Mamba` modules; "
            "functional elementwise operations, reductions, indexing, reshapes, and "
            "interpolation are not included.\n\n"
        )
        f.write(
            "| Model | Params | FLOPs | Latency mean (ms) | Peak CUDA allocated (MB) | "
            "Input | Device | AMP |\n"
        )
        f.write("|---|---:|---:|---:|---:|---|---|---|\n")
        for result in results:
            row = _result_to_row(result)
            peak_allocated = (
                ""
                if row["peak_allocated_mb"] is None
                else f"{float(row['peak_allocated_mb']):.1f}"
            )
            f.write(
                f"| {row['project_name']} | "
                f"{_short_number(row['total_params'])} | "
                f"{_short_number(row['estimated_flops'])} | "
                f"{float(row['latency_mean_ms']):.3f} | "
                f"{peak_allocated} | "
                f"{row['batch_size']}x{row['input_shape_dhw']} | "
                f"{row['device']} | {row['amp_enabled']} |\n"
            )
        f.write("\n## RAPSS-Net Module Summary\n\n")
        for result in results:
            labels = result.get("rapss_modules", {}).get("enabled_module_labels", [])
            f.write(f"- **{result.get('project_name')}**: ")
            f.write(", ".join(labels) if labels else "Base configuration without RAPSS extra modules")
            f.write("\n")

    return {
        "json": str(json_path),
        "csv": str(csv_path),
        "markdown": str(md_path),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Profile RAPSS-Net/TriDual3D efficiency with reproducible synthetic "
            "single-sample inputs."
        )
    )
    parser.add_argument(
        "-c",
        "--config",
        action="append",
        default=None,
        help=(
            "Training config to profile. Pass multiple times to compare Base/Final. "
            "Defaults to the active server CV Base and Final configs."
        ),
    )
    parser.add_argument(
        "--prep_config",
        default=DEFAULT_PREP_CONFIG,
        help="Preprocessing config used to read data_prep.target_shape.",
    )
    parser.add_argument(
        "--input_shape",
        nargs=3,
        type=int,
        metavar=("D", "H", "W"),
        default=None,
        help="Override the spatial input shape. Defaults to prep_config target_shape.",
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument(
        "--device",
        default="auto",
        help="Profiling device: auto, config, cpu, cuda, cuda:0, etc.",
    )
    parser.add_argument(
        "--amp",
        choices=("config", "on", "off"),
        default="config",
        help="Use AMP during inference. 'config' follows train.amp on CUDA.",
    )
    parser.add_argument(
        "--flops",
        choices=("estimate", "skip"),
        default="estimate",
        help="Estimate MACs/FLOPs or skip FLOPs counting.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help=(
            "Optional checkpoint to load. Use only when profiling one compatible "
            "configuration."
        ),
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Profile torch.compile(model). Disabled by default for comparability.",
    )
    parser.add_argument(
        "--cudnn_benchmark",
        action="store_true",
        help="Enable cuDNN benchmark before timing.",
    )
    parser.add_argument(
        "--include_module_breakdown",
        action="store_true",
        help="Include per-module MAC/FLOP records in the JSON output.",
    )
    parser.add_argument("--output_dir", default="profiling_results")
    parser.add_argument("--tag", default="")
    args = parser.parse_args(argv)
    if int(args.batch_size) <= 0:
        parser.error("--batch_size must be positive.")
    if int(args.repeats) <= 0:
        parser.error("--repeats must be positive.")
    if args.checkpoint and len(args.config or DEFAULT_CONFIGS) != 1:
        parser.error("--checkpoint can only be used with exactly one --config.")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config_paths = _resolve_configs(args.config)
    prep_config = _resolve_prep_config(args.prep_config)
    input_shape = (
        tuple(int(v) for v in args.input_shape)
        if args.input_shape is not None
        else _load_prep_target_shape(prep_config)
    )
    if any(int(v) <= 0 for v in input_shape):
        raise ValueError("--input_shape values must be positive.")

    output_dir = _resolve_output_dir(args.output_dir)
    print(f"Input shape (D,H,W): {input_shape}")
    print(f"Output directory: {output_dir}")

    results = []
    for config_path in config_paths:
        print(f"\n[PROFILE] {config_path}")
        result = profile_one_config(config_path, input_shape, args)
        row = _result_to_row(result)
        print(
            "  params={params}, flops={flops}, latency_mean={latency:.3f} ms, "
            "peak_cuda_allocated={memory}".format(
                params=_short_number(row["total_params"]),
                flops=_short_number(row["estimated_flops"]),
                latency=float(row["latency_mean_ms"]),
                memory=(
                    "n/a"
                    if row["peak_allocated_mb"] is None
                    else f"{float(row['peak_allocated_mb']):.1f} MB"
                ),
            )
        )
        results.append(result)

    paths = write_outputs(results, output_dir, args.tag)
    print("\nSaved profiling outputs:")
    for label, path in paths.items():
        print(f"  {label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
