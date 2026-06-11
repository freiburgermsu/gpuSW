"""Runtime Metal pipeline compilation + caching, and the (soft) PyObjC boundary.

The Apple-Silicon sibling of :mod:`gpusw._compile` (the CuPy/NVRTC boundary). PyObjC's
``Metal`` bindings are imported lazily so that importing ``gpusw`` and using the CPU
oracle never require a GPU or macOS. Compiled :class:`MTLComputePipelineState` objects
are cached process-wide keyed on every source-affecting input *plus the Metal device*,
and substitution tables are uploaded once per ``(table, device)``.

Like NVRTC for CUDA, the Metal *runtime* shader compiler
(``newLibraryWithSource:options:error:``) ships inside the OS Metal framework — it needs
**no Xcode and no offline ``metal``/``metallib`` toolchain**, and it targets the live
GPU. So the only thing a user installs is the PyObjC wheel.
"""
from __future__ import annotations

import numpy as np

from ._compile import bucket_maxq  # backend-agnostic MAXQ bucketing (shared with CUDA)
from .errors import GpuSWError, NoGpuError
from .metal_kernel import render_source

__all__ = [
    "metal_available",
    "require_metal",
    "device",
    "command_queue",
    "get_pipeline",
    "get_sub",
    "to_buffer",
    "empty_buffer",
    "buffer_to_numpy",
    "bucket_maxq",
]

# Per-thread private-memory budget for the two MAXQ-sized DP buffers (Hp, F). Mirrors
# the CUDA local-memory budget; on Apple GPUs these thread-address-space arrays spill to
# device-backed thread-private memory, so the cap is conservative rather than a hard wall.
_LOCALMEM_BUDGET = 48 * 1024  # bytes

_PIPELINE_CACHE: dict = {}
_SUB_CACHE: dict = {}
_DEVICE = None          # cached MTLDevice
_QUEUE = None           # cached MTLCommandQueue (one per process is plenty)

_INSTALL_HINT = (
    "PyObjC's Metal bindings and an Apple-Silicon (Metal) GPU are required for the Metal "
    "backend. Install the wheels with  pip install 'gpusw[metal]'  (pyobjc-framework-Metal "
    "+ pyobjc-framework-libdispatch) on macOS. The CPU oracle gpusw.cpu_reference_score() "
    "needs no GPU."
)


def require_metal():
    """Import the PyObjC ``Metal`` module or raise :class:`NoGpuError`."""
    try:
        import Metal  # noqa: PLC0415
    except Exception as exc:  # ImportError on non-mac / missing wheel
        raise NoGpuError(_INSTALL_HINT) from exc
    return Metal


def metal_available() -> bool:
    """True iff PyObjC's Metal imports and a default Metal device is present."""
    try:
        import Metal  # noqa: PLC0415

        return Metal.MTLCreateSystemDefaultDevice() is not None
    except Exception:
        return False


def device():
    """Return the cached default :class:`MTLDevice` (raising if none is present)."""
    global _DEVICE, _QUEUE
    if _DEVICE is None:
        Metal = require_metal()
        dev = Metal.MTLCreateSystemDefaultDevice()
        if dev is None:
            raise NoGpuError(_INSTALL_HINT)
        _DEVICE = dev
        _QUEUE = dev.newCommandQueue()
    return _DEVICE


def command_queue():
    """Return the cached :class:`MTLCommandQueue` for the default device."""
    if _QUEUE is None:
        device()
    return _QUEUE


def _device_key() -> int:
    """A stable per-device identity for cache keys (the CUDA ``arch`` analog)."""
    return int(device().registryID())


def get_pipeline(scheme, maxq: int, dtype: str):
    """Return ``(pso_cross, pso_pairs, source)`` for ``scheme`` (cached).

    ``pso_*`` are :class:`MTLComputePipelineState` objects for the ``sw_cross`` /
    ``sw_pairs`` entry points. The cache key mirrors the CUDA module cache: every
    source-affecting field plus ``maxq`` bucket, ``dtype`` and the device identity.
    """
    Metal = require_metal()
    dev = device()
    mf = scheme.module_fields()
    mq = bucket_maxq(maxq)
    store_bytes = 2 if dtype == "int16" else 4
    need = 2 * (mq + 1) * store_bytes
    if need > _LOCALMEM_BUDGET:
        # largest query length whose 128-aligned MAXQ bucket still fits the budget
        cap = ((_LOCALMEM_BUDGET // (2 * store_bytes)) - 1) // 128 * 128
        raise GpuSWError(
            f"query too long for the one-thread-per-pair kernel: MAXQ bucket {mq} needs "
            f"{need} bytes/thread of thread-private memory (> {_LOCALMEM_BUDGET // 1024} "
            f"KiB). Max query length is ~{cap} residues at dtype={dtype}. Split long "
            f"queries into shorter windows, make them the shorter (query) side, or score "
            f"on CPU with gpusw.cpu_reference_score()."
        )
    key = (tuple(sorted(mf.items())), mq, dtype, _device_key())
    cached = _PIPELINE_CACHE.get(key)
    if cached is None:
        src = render_source(mf, mq, dtype)
        opts = Metal.MTLCompileOptions.alloc().init()
        lib, err = dev.newLibraryWithSource_options_error_(src, opts, None)
        if lib is None:
            raise GpuSWError(f"Metal (MSL) compilation failed: {err}")
        pso_cross, e1 = dev.newComputePipelineStateWithFunction_error_(
            lib.newFunctionWithName_("sw_cross"), None
        )
        pso_pairs, e2 = dev.newComputePipelineStateWithFunction_error_(
            lib.newFunctionWithName_("sw_pairs"), None
        )
        if pso_cross is None or pso_pairs is None:
            raise GpuSWError(f"Metal pipeline-state creation failed: {e1 or e2}")
        cached = (pso_cross, pso_pairs, src)
        _PIPELINE_CACHE[key] = cached
    return cached


def get_sub(scheme):
    """Upload (once per device) and return the ``NS*NS`` int32 substitution-table buffer."""
    key = (scheme.table_hash(), _device_key())
    buf = _SUB_CACHE.get(key)
    if buf is None:
        host = np.ascontiguousarray(scheme.substitution_table(), dtype=np.int32).ravel()
        buf = to_buffer(host)
        _SUB_CACHE[key] = buf
    return buf


# --------------------------------------------------------------------- buffers
def to_buffer(arr: np.ndarray):
    """Copy a contiguous NumPy array into a shared (unified-memory) ``MTLBuffer``."""
    Metal = require_metal()
    dev = device()
    a = np.ascontiguousarray(arr)
    nbytes = max(int(a.nbytes), 1)  # Metal rejects zero-length buffers
    raw = a.tobytes() if a.nbytes else b"\x00"
    return dev.newBufferWithBytes_length_options_(
        raw, nbytes, Metal.MTLResourceStorageModeShared
    )


def empty_buffer(nbytes: int):
    """Allocate an uninitialised shared ``MTLBuffer`` of ``nbytes`` (≥1)."""
    Metal = require_metal()
    dev = device()
    return dev.newBufferWithLength_options_(
        max(int(nbytes), 1), Metal.MTLResourceStorageModeShared
    )


def buffer_to_numpy(buf, count: int, dtype) -> np.ndarray:
    """Read ``count`` elements of ``dtype`` back from a shared ``MTLBuffer`` → NumPy.

    On unified memory the buffer's bytes are host-visible after the command buffer
    completes; we copy them out so the result owns its memory independently of ``buf``.
    """
    dt = np.dtype(dtype)
    nbytes = int(count) * dt.itemsize
    if nbytes == 0:
        return np.empty(0, dtype=dt)
    mv = buf.contents().as_buffer(nbytes)
    return np.frombuffer(mv, dtype=dt, count=int(count)).copy()
