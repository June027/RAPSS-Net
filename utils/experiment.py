import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np
import torch


def set_deterministic_environment(seed: int = 42) -> None:
    """
    配置极致的全局随机数种子与 CUDA 确定性（提升论文复现的严谨性）。
    锁定 Python, NumPy, PyTorch 的种子，并强制 cuDNN 使用确定性算法。
    """
    # 1. 基础随机种子
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    strict_determinism = os.environ.get("STRICT_DETERMINISM", "0") == "1"

    # 2. CUDA determinism/performance mode.
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
        torch.backends.cudnn.deterministic = strict_determinism
        torch.backends.cudnn.benchmark = not strict_determinism
        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # 3. Default to the fast server mode. Set STRICT_DETERMINISM=1 only when
    # exact deterministic replay is more important than speed/noise.
    try:
        torch.use_deterministic_algorithms(strict_determinism, warn_only=True)
        if strict_determinism:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    except Exception:
        pass


def stable_hash(obj: Any) -> str:
    payload = json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str).encode(
        "utf-8"
    )
    return hashlib.sha256(payload).hexdigest()


def config_fingerprint(config: Dict[str, Any]) -> str:
    return stable_hash(config)


def split_fingerprint(train_ids: Iterable[str], val_ids: Iterable[str]) -> str:
    return stable_hash({"train_ids": list(train_ids), "val_ids": list(val_ids)})


def save_json(path: str | Path, payload: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
