"""Pluggable GPU backends — the parallel CUDA and Metal infrastructures behind one
:class:`~gpusw.Aligner` API.

Everything that touches a GPU is funnelled through a small :class:`Backend` contract so
the engine (:mod:`gpusw.aligner`) stays device-agnostic: it encodes/uploads sequences,
launches the ``sw_cross`` / ``sw_pairs`` kernels, and partitions per-query top-k without
naming CUDA or Metal anywhere.

Two backends implement the contract over the **same** runtime-compiled, score-only,
one-thread-per-pair Gotoh kernel, so their outputs are bit-exact with each other and with
the pure-NumPy oracle:

* :class:`~gpusw.backends.cuda.CudaBackend` — NVIDIA via CuPy/NVRTC (:mod:`gpusw._compile`,
  :mod:`gpusw.kernel`).
* :class:`~gpusw.backends.metal.MetalBackend` — Apple Silicon via PyObjC/Metal
  (:mod:`gpusw._metal`, :mod:`gpusw.metal_kernel`).

``resolve_backend("auto")`` prefers CUDA, then Metal, mirroring "use whatever GPU this
box has". Each backend is cheap to construct (no device is touched until the first
scoring call), so an ``Aligner`` can be built and introspected on a machine with no GPU.
"""
from __future__ import annotations

from contextlib import contextmanager

from ..errors import NoGpuError

__all__ = ["Backend", "resolve_backend", "available_backends"]


class Backend:
    """The device-specific operations the engine needs. Implemented per GPU API.

    A backend owns three things: how a host array reaches the device (:meth:`upload`),
    how the two kernels launch (:meth:`run_cross` / :meth:`run_pairs`), and how a scored
    batch becomes host results (:meth:`as_host` / :meth:`topk`). Scores produced by any
    backend are integer-identical for the same scheme.
    """

    name: str = "base"

    #: Maximum threads dispatchable in a *single* kernel launch on this backend. The
    #: engine chunks cross products / pair batches so no one launch exceeds it. CUDA's
    #: 64-bit thread index makes this effectively unbounded; Metal's grid index is 32-bit
    #: (see :class:`~gpusw.backends.metal.MetalBackend`), so it caps below 2**32.
    max_threads_per_launch: int = 1 << 62

    def available(self) -> bool:
        """True iff this backend can actually run on the current machine."""
        raise NotImplementedError

    @contextmanager
    def device_scope(self, device: int):
        """Context manager that binds the chosen device for the enclosed device work."""
        yield

    def upload(self, arr):
        """Move a host (NumPy) array onto the device; return an opaque device handle."""
        raise NotImplementedError

    def run_cross(self, scheme, maxq, dtype, threads,
                  qbuf, qoff, rbuf, roff, qlist_host, nq, nr):
        """Launch ``sw_cross``. Return ``(scores_native, dt, source)`` where
        ``scores_native`` is a backend-native ``(nq, nr)`` array and ``dt`` is the
        launch wall-time (seconds) for the GCUPS figure."""
        raise NotImplementedError

    def run_pairs(self, scheme, maxq, dtype, threads,
                  qbuf, qoff, rbuf, roff, qi_host, rj_host, n):
        """Launch ``sw_pairs``. Return ``(host_scores_int32, dt, source)`` with
        ``host_scores_int32`` a host ``(n,)`` ``int32`` array."""
        raise NotImplementedError

    def as_host(self, scores_native):
        """Materialise a native ``(nq, nr)`` scores array as host ``int32`` NumPy."""
        raise NotImplementedError

    def topk(self, scores_native, kk):
        """Per-row top-``kk`` of a native ``(nq, nr)`` scores array, best-first.

        Return ``(idx_int32, sc_int32)`` host arrays, each ``(nq, kk)``.
        """
        raise NotImplementedError

    def render(self, scheme, maxq, dtype) -> str:
        """Return this backend's generated kernel source for the scheme (no device)."""
        raise NotImplementedError


def resolve_backend(name: str | None) -> Backend:
    """Return a :class:`Backend` for ``name`` (``"auto"`` | ``"cuda"`` | ``"metal"``).

    ``"auto"`` prefers CUDA, then Metal, and raises :class:`~gpusw.NoGpuError` only if
    neither is usable. Construction never touches the device for the explicit names.
    """
    n = (name or "auto").lower()
    if n == "cuda":
        from .cuda import CudaBackend

        return CudaBackend()
    if n == "metal":
        from .metal import MetalBackend

        return MetalBackend()
    if n == "auto":
        from .cuda import CudaBackend
        from .metal import MetalBackend

        cuda = CudaBackend()
        if cuda.available():
            return cuda
        metal = MetalBackend()
        if metal.available():
            return metal
        raise NoGpuError(
            "no GPU backend available: neither CuPy/CUDA (NVIDIA) nor PyObjC/Metal "
            "(Apple Silicon) is usable. Install 'gpusw[cuda13]' / 'gpusw[cuda12]' on "
            "NVIDIA, or 'gpusw[metal]' on macOS — or score on the CPU with "
            "gpusw.cpu_reference_score()."
        )
    raise ValueError(f"unknown backend {name!r}; choose 'auto', 'cuda', or 'metal'")


def available_backends() -> list[str]:
    """Names of the backends usable on this machine right now (may be empty)."""
    from .cuda import CudaBackend
    from .metal import MetalBackend

    return [b.name for b in (CudaBackend(), MetalBackend()) if b.available()]
