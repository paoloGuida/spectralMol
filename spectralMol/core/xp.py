from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import numpy as np


def backend_name(xp_mod: Any) -> str:
    if xp_mod is np:
        return "numpy"
    name = getattr(xp_mod, "__name__", "")
    if "cupy" in name:
        return "cuda"
    return name or "unknown"


def get_array_module(use_cuda: bool, cuda_device: int = 0, *, use_mps: bool = False):
    """Return an array module/context pair, falling back to NumPy on CPU-only hosts."""
    _ = use_mps
    if use_cuda:
        try:
            import cupy as cp  # type: ignore

            device = cp.cuda.Device(int(cuda_device))
            return cp, device
        except Exception:
            pass
    return np, nullcontext()


def to_numpy(xp_mod: Any, arr: Any) -> np.ndarray:
    if xp_mod is np:
        return np.asarray(arr)
    try:
        return xp_mod.asnumpy(arr)
    except Exception:
        return np.asarray(arr)
