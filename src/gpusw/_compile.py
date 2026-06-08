"""Runtime kernel compilation + caching, and the (soft) CuPy boundary.

CuPy is imported lazily so that importing ``gpusw`` and using the CPU oracle never
require a GPU. Compiled :class:`cupy.RawModule` objects are cached process-wide keyed
on every source-affecting input *plus the GPU architecture*, and substitution tables
are uploaded once per ``(table, device)``.
"""
from __future__ import annotations

import numpy as np

from .errors import GpuSWError, NoGpuError
from .kernel import render_source

__all__ = ["gpu_available", "require_cupy", "bucket_maxq", "get_module", "get_sub"]

# Per-thread local-memory budget for the two MAXQ-sized DP buffers (Hp, F).
_LOCALMEM_BUDGET = 48 * 1024  # bytes

_MODULE_CACHE: dict = {}
_SUB_CACHE: dict = {}

_INSTALL_HINT = (
    "CuPy with a working NVIDIA GPU is required for GPU alignment. Install the wheel "
    "matching your CUDA major version, e.g.  pip install 'gpusw[cuda13]'  (or cuda12 / "
    "cuda11).  The CPU oracle gpusw.cpu_reference_score() needs no GPU."
)


def require_cupy():
    """Import CuPy or raise :class:`NoGpuError` with an actionable message."""
    try:
        import cupy as cp  # noqa: PLC0415
    except Exception as exc:  # ImportError or a CUDA init error
        raise NoGpuError(_INSTALL_HINT) from exc
    return cp


def gpu_available() -> bool:
    """True iff CuPy imports and at least one CUDA device is visible."""
    try:
        import cupy as cp  # noqa: PLC0415

        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def bucket_maxq(maxq: int) -> int:
    """Round ``maxq`` up to the next multiple of 128 (≥128) so near-equal query
    lengths reuse one compiled module."""
    return max(128, ((int(maxq) + 127) // 128) * 128)


def get_module(scheme, maxq: int, dtype: str):
    """Return ``(module, sw_cross, sw_pairs, source)`` for ``scheme`` (cached)."""
    cp = require_cupy()
    mf = scheme.module_fields()
    mq = bucket_maxq(maxq)
    store_bytes = 2 if dtype == "int16" else 4
    need = 2 * (mq + 1) * store_bytes
    if need > _LOCALMEM_BUDGET:
        cap = _LOCALMEM_BUDGET // (2 * store_bytes) - 1
        raise GpuSWError(
            f"query too long for the one-thread-per-pair kernel: MAXQ bucket {mq} needs "
            f"{need} bytes/thread of local memory (> {_LOCALMEM_BUDGET // 1024} KiB). "
            f"Max query length is ~{cap} residues at dtype={dtype}. Split long queries "
            f"into shorter windows, make them the shorter (query) side, or score on CPU "
            f"with gpusw.cpu_reference_score()."
        )
    arch = cp.cuda.Device().compute_capability
    key = (tuple(sorted(mf.items())), mq, dtype, arch)
    cached = _MODULE_CACHE.get(key)
    if cached is None:
        src = render_source(mf, mq, dtype)
        mod = cp.RawModule(code=src, options=("--std=c++14",))
        cached = (mod, mod.get_function("sw_cross"), mod.get_function("sw_pairs"), src)
        _MODULE_CACHE[key] = cached
    return cached


def get_sub(scheme):
    """Upload (once per device) and return the ``NS*NS`` int32 substitution table."""
    cp = require_cupy()
    dev = cp.cuda.Device().id
    key = (scheme.table_hash(), dev)
    arr = _SUB_CACHE.get(key)
    if arr is None:
        host = np.ascontiguousarray(scheme.substitution_table(), dtype=np.int32).ravel()
        arr = cp.asarray(host)
        _SUB_CACHE[key] = arr
    return arr
