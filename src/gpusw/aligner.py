"""The reusable GPU engine: encode/upload references once, then score many query
sets against them with ``score_cross`` / ``score_pairs`` / ``top_k`` — on **either**
GPU backend (NVIDIA/CUDA or Apple/Metal) behind one API.

The device is reached only through a :class:`~gpusw.backends.Backend` (CUDA via
CuPy/NVRTC, Metal via PyObjC), so this engine names no GPU API directly. Constructing an
``Aligner`` is cheap and GPU-free; the backend is resolved (and the kernel compiled) on
the first scoring call, when query/reference lengths — and thus ``MAXQ`` and the
int16/int32 choice — are known.
"""
from __future__ import annotations

import numpy as np

from .backends import resolve_backend
from .encode import Encoded, funnel
from .result import AlignResult
from .scheme import Scheme

__all__ = ["Aligner"]


def _as_scheme(scheme) -> Scheme:
    if isinstance(scheme, Scheme):
        return scheme
    if isinstance(scheme, str):
        return Scheme.preset(scheme)
    raise TypeError(f"scheme must be a Scheme or preset name, got {type(scheme).__name__}")


class Aligner:
    """Compile-and-reuse GPU Smith-Waterman / Needleman-Wunsch scorer for one scheme.

    Parameters
    ----------
    scheme:
        A :class:`~gpusw.Scheme` or a preset name (e.g. ``"dna"``, ``"blosum62"``).
    threads:
        Threads per block (CUDA) / per threadgroup (Metal). Default 128.
    device:
        Device ordinal (CUDA). Metal uses the single default system device.
    backend:
        ``"auto"`` (prefer CUDA, else Metal), ``"cuda"``, or ``"metal"``.
    """

    def __init__(self, scheme="dna", *, threads: int = 128, device: int = 0,
                 backend: str = "auto"):
        self.scheme = _as_scheme(scheme)
        self.threads = int(threads)
        self.device = int(device)
        self._backend_name = backend
        self._backend = None
        self._refs: Encoded | None = None
        self._queries: Encoded | None = None
        self._d_rbuf = self._d_roff = None
        self._d_qbuf = self._d_qoff = None
        self._last_gcups = 0.0
        self._last_source = ""
        self._last_backend_name = ""

    @property
    def _b(self):
        """Resolve (once) and return the active backend."""
        if self._backend is None:
            self._backend = resolve_backend(self._backend_name)
        return self._backend

    # ----------------------------------------------------------------- inputs
    def index(self, refs) -> Aligner:
        """Encode and upload the reference set once; returns ``self`` for chaining."""
        b = self._b
        with b.device_scope(self.device):
            enc = funnel(refs, self.scheme, id_prefix="r")
            self._refs = enc
            self._d_rbuf = b.upload(enc.codes)
            self._d_roff = b.upload(enc.offsets)
        return self

    def set_queries(self, queries) -> Aligner:
        """Encode and upload a query set; returns ``self`` for chaining."""
        b = self._b
        with b.device_scope(self.device):
            enc = funnel(queries, self.scheme, id_prefix="q")
            self._queries = enc
            self._d_qbuf = b.upload(enc.codes)
            self._d_qoff = b.upload(enc.offsets)
        return self

    # ------------------------------------------------------------- internals
    def _resolved_dtype(self) -> str:
        maxq = self._queries.max_len if self._queries else 0
        maxr = self._refs.max_len if self._refs else 0
        return self.scheme.resolve_dtype(max(maxq, 1), max(maxr, 1))

    def _launch_cross(self, qlist_host, nq: int):
        """Score ``qlist`` (query indices) against all indexed refs → native (nq, nr)."""
        nr = self._refs.n
        dtype = self._resolved_dtype()
        maxq = self._queries.max_len
        scores, dt, src = self._b.run_cross(
            self.scheme, maxq, dtype, self.threads,
            self._d_qbuf, self._d_qoff, self._d_rbuf, self._d_roff,
            qlist_host, nq, nr,
        )
        self._last_source = src
        self._last_backend_name = self._b.name
        qlen = int(self._queries.lengths[np.asarray(qlist_host)].sum())
        cells = qlen * int(self._refs.lengths.sum())
        self._last_gcups = cells / dt / 1e9 if dt > 0 else 0.0
        return scores

    # --------------------------------------------------------------- scoring
    def score_cross(self, queries=None, *, return_ids: bool = False,
                    query_batch: int | None = None):
        """All-pairs scores of queries × indexed references.

        Returns an ``(n_queries, n_refs)`` int32 ``np.ndarray`` (or an
        :class:`~gpusw.AlignResult` if ``return_ids=True``). ``query_batch`` chunks the
        queries to bound device memory for very large cross products.
        """
        if self._refs is None:
            raise ValueError("call index(refs) before score_cross()")
        if queries is not None:
            self.set_queries(queries)
        if self._queries is None:
            raise ValueError("no queries: pass queries= or call set_queries()")
        nq, nr = self._queries.n, self._refs.n
        # never request more threads than the backend can index in one launch (Metal's
        # grid index is 32-bit); CUDA's cap is effectively unbounded, so its default
        # single-launch behavior is unchanged.
        cap_rows = max(1, self._b.max_threads_per_launch // max(nr, 1))
        with self._b.device_scope(self.device):
            out = np.empty((nq, nr), dtype=np.int32)
            step = max(1, min(query_batch or nq, cap_rows))
            for b0 in range(0, nq, step):
                qb = np.arange(b0, min(b0 + step, nq), dtype=np.int32)
                scores = self._launch_cross(qb, len(qb))
                out[b0:b0 + len(qb)] = self._b.as_host(scores)
        if return_ids:
            return AlignResult(out, self._queries.ids, self._refs.ids, self.scheme)
        return out

    def score_pairs(self, qi, rj, *, return_ids: bool = False):
        """Score the explicit index pairs ``(qi[k], rj[k])`` → ``(n_pairs,)`` int32."""
        if self._refs is None or self._queries is None:
            raise ValueError("call index(refs) and set_queries(queries) first")
        qi = np.ascontiguousarray(qi, dtype=np.int32)
        rj = np.ascontiguousarray(rj, dtype=np.int32)
        if qi.shape != rj.shape:
            raise ValueError("qi and rj must have the same length")
        n = int(qi.shape[0])
        dtype = self._resolved_dtype()
        with self._b.device_scope(self.device):
            host, dt, src = self._b.run_pairs(
                self.scheme, self._queries.max_len, dtype, self.threads,
                self._d_qbuf, self._d_qoff, self._d_rbuf, self._d_roff, qi, rj, n,
            )
        self._last_source = src
        self._last_backend_name = self._b.name
        cells = int((self._queries.lengths[qi].astype(np.int64)
                     * self._refs.lengths[rj].astype(np.int64)).sum())
        self._last_gcups = cells / dt / 1e9 if dt > 0 else 0.0
        if return_ids:
            qids = [self._queries.ids[i] for i in qi]
            rids = [self._refs.ids[j] for j in rj]
            return AlignResult(host, qids, rids, self.scheme)
        return host

    def top_k(self, queries=None, k: int = 5, *, query_batch: int = 512) -> AlignResult:
        """Per-query top-``k`` references by score (partition on the device/host).

        Memory-bounded: only the per-query top-``k`` is kept on the host, so this
        scales to reference sets too large to materialise the full score matrix.
        """
        if self._refs is None:
            raise ValueError("call index(refs) before top_k()")
        if queries is not None:
            self.set_queries(queries)
        if self._queries is None:
            raise ValueError("no queries: pass queries= or call set_queries()")
        nq, nr = self._queries.n, self._refs.n
        kk = min(k, nr)
        cap_rows = max(1, self._b.max_threads_per_launch // max(nr, 1))
        qbatch = max(1, min(query_batch, cap_rows))
        idx_out = np.empty((nq, kk), dtype=np.int32)
        sc_out = np.empty((nq, kk), dtype=np.int32)
        with self._b.device_scope(self.device):
            for b0 in range(0, nq, qbatch):
                qb = np.arange(b0, min(b0 + qbatch, nq), dtype=np.int32)
                scores = self._launch_cross(qb, len(qb))  # native (len(qb), nr)
                idx, sc = self._b.topk(scores, kk)
                idx_out[b0:b0 + len(qb)] = idx
                sc_out[b0:b0 + len(qb)] = sc
        return AlignResult(None, self._queries.ids, self._refs.ids, self.scheme,
                           topk_idx=idx_out, topk_scores=sc_out)

    # ------------------------------------------------------------ introspection
    @property
    def backend(self) -> str:
        """Name of the active backend (``"cuda"`` or ``"metal"``); resolves ``"auto"``."""
        return self._b.name

    @property
    def query_ids(self) -> list[str]:
        return list(self._queries.ids) if self._queries else []

    @property
    def reference_ids(self) -> list[str]:
        return list(self._refs.ids) if self._refs else []

    @property
    def gcups(self) -> float:
        """Throughput (giga cell-updates per second) of the most recent launch."""
        return self._last_gcups

    def _intro_maxq(self) -> int:
        return self._queries.max_len if self._queries else 256

    def _intro_dtype(self) -> str:
        return self._resolved_dtype() if (self._queries and self._refs) else "int32"

    def cuda_source(self) -> str:
        """The generated **CUDA** C++ source for this scheme (NVIDIA backend)."""
        if self._last_source and self._last_backend_name == "cuda":
            return self._last_source
        from .backends.cuda import CudaBackend

        return CudaBackend().render(self.scheme, self._intro_maxq(), self._intro_dtype())

    def metal_source(self) -> str:
        """The generated **Metal Shading Language** source for this scheme (Apple backend)."""
        if self._last_source and self._last_backend_name == "metal":
            return self._last_source
        from .backends.metal import MetalBackend

        return MetalBackend().render(self.scheme, self._intro_maxq(), self._intro_dtype())

    def kernel_source(self) -> str:
        """The kernel source last launched, else rendered for the active backend."""
        if self._last_source:
            return self._last_source
        return self._b.render(self.scheme, self._intro_maxq(), self._intro_dtype())
