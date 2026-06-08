"""The reusable GPU engine: encode/upload references once, then score many query
sets against them with ``score_cross`` / ``score_pairs`` / ``top_k``.

CuPy is imported lazily through :mod:`gpusw._compile`; constructing an ``Aligner`` is
cheap, and the kernel is compiled on first launch (when query/reference lengths — and
thus ``MAXQ`` and the int16/int32 choice — are known).
"""
from __future__ import annotations

import time

import numpy as np

from . import _compile
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
        CUDA block size (threads per block). Default 128.
    device:
        CUDA device ordinal.
    """

    def __init__(self, scheme="dna", *, threads: int = 128, device: int = 0):
        self.scheme = _as_scheme(scheme)
        self.threads = int(threads)
        self.device = int(device)
        self._refs: Encoded | None = None
        self._queries: Encoded | None = None
        self._d_rbuf = self._d_roff = None
        self._d_qbuf = self._d_qoff = None
        self._last_gcups = 0.0
        self._last_source = ""

    # ----------------------------------------------------------------- inputs
    def index(self, refs) -> Aligner:
        """Encode and upload the reference set once; returns ``self`` for chaining."""
        cp = _compile.require_cupy()
        with cp.cuda.Device(self.device):
            enc = funnel(refs, self.scheme, id_prefix="r")
            self._refs = enc
            self._d_rbuf = cp.asarray(enc.codes)
            self._d_roff = cp.asarray(enc.offsets)
        return self

    def set_queries(self, queries) -> Aligner:
        """Encode and upload a query set; returns ``self`` for chaining."""
        cp = _compile.require_cupy()
        with cp.cuda.Device(self.device):
            enc = funnel(queries, self.scheme, id_prefix="q")
            self._queries = enc
            self._d_qbuf = cp.asarray(enc.codes)
            self._d_qoff = cp.asarray(enc.offsets)
        return self

    # ------------------------------------------------------------- internals
    def _resolved_dtype(self) -> str:
        maxq = self._queries.max_len if self._queries else 0
        maxr = self._refs.max_len if self._refs else 0
        return self.scheme.resolve_dtype(max(maxq, 1), max(maxr, 1))

    def _store_dtype(self, cp, dtype: str):
        return cp.int16 if dtype == "int16" else cp.int32

    def _launch_cross(self, cp, qlist_dev, nq: int):
        """Score ``qlist`` (query indices) against all indexed refs → cp (nq, nr)."""
        nr = self._refs.n
        dtype = self._resolved_dtype()
        maxq = self._queries.max_len
        _, fcross, _, src = _compile.get_module(self.scheme, maxq, dtype)
        self._last_source = src
        sub = _compile.get_sub(self.scheme)
        out = cp.empty(nq * nr, dtype=self._store_dtype(cp, dtype))
        total = nq * nr
        thr = self.threads
        grid = ((total + thr - 1) // thr,)
        cp.cuda.runtime.deviceSynchronize()
        t0 = time.perf_counter()
        fcross(grid, (thr,),
               (self._d_qbuf, self._d_qoff, self._d_rbuf, self._d_roff,
                qlist_dev, np.int32(nq), np.int32(nr), out, sub))
        cp.cuda.runtime.deviceSynchronize()
        dt = time.perf_counter() - t0
        qlen = int(self._queries.lengths[cp.asnumpy(qlist_dev)].sum())
        cells = qlen * int(self._refs.lengths.sum())
        self._last_gcups = cells / dt / 1e9 if dt > 0 else 0.0
        return out.reshape(nq, nr)

    # --------------------------------------------------------------- scoring
    def score_cross(self, queries=None, *, return_ids: bool = False,
                    query_batch: int | None = None):
        """All-pairs scores of queries × indexed references.

        Returns an ``(n_queries, n_refs)`` int32 ``np.ndarray`` (or an
        :class:`~gpusw.AlignResult` if ``return_ids=True``). ``query_batch`` chunks the
        queries to bound device memory for very large cross products.
        """
        cp = _compile.require_cupy()
        if self._refs is None:
            raise ValueError("call index(refs) before score_cross()")
        if queries is not None:
            self.set_queries(queries)
        if self._queries is None:
            raise ValueError("no queries: pass queries= or call set_queries()")
        nq, nr = self._queries.n, self._refs.n
        with cp.cuda.Device(self.device):
            out = np.empty((nq, nr), dtype=np.int32)
            step = query_batch or nq
            for b0 in range(0, nq, max(step, 1)):
                qb = np.arange(b0, min(b0 + step, nq), dtype=np.int32)
                dev = self._launch_cross(cp, cp.asarray(qb), len(qb))
                out[b0:b0 + len(qb)] = cp.asnumpy(dev).astype(np.int32)
        if return_ids:
            return AlignResult(out, self._queries.ids, self._refs.ids, self.scheme)
        return out

    def score_pairs(self, qi, rj, *, return_ids: bool = False):
        """Score the explicit index pairs ``(qi[k], rj[k])`` → ``(n_pairs,)`` int32."""
        cp = _compile.require_cupy()
        if self._refs is None or self._queries is None:
            raise ValueError("call index(refs) and set_queries(queries) first")
        qi = np.ascontiguousarray(qi, dtype=np.int32)
        rj = np.ascontiguousarray(rj, dtype=np.int32)
        if qi.shape != rj.shape:
            raise ValueError("qi and rj must have the same length")
        n = int(qi.shape[0])
        dtype = self._resolved_dtype()
        _, _, fpairs, src = _compile.get_module(self.scheme, self._queries.max_len, dtype)
        self._last_source = src
        sub = _compile.get_sub(self.scheme)
        with cp.cuda.Device(self.device):
            out = cp.empty(n, dtype=self._store_dtype(cp, dtype))
            thr = self.threads
            grid = ((n + thr - 1) // thr,)
            cp.cuda.runtime.deviceSynchronize()
            t0 = time.perf_counter()
            fpairs(grid, (thr,),
                   (self._d_qbuf, self._d_qoff, self._d_rbuf, self._d_roff,
                    cp.asarray(qi), cp.asarray(rj), np.int32(n), out, sub))
            cp.cuda.runtime.deviceSynchronize()
            dt = time.perf_counter() - t0
            cells = int((self._queries.lengths[qi].astype(np.int64)
                         * self._refs.lengths[rj].astype(np.int64)).sum())
            self._last_gcups = cells / dt / 1e9 if dt > 0 else 0.0
            host = cp.asnumpy(out).astype(np.int32)
        if return_ids:
            qids = [self._queries.ids[i] for i in qi]
            rids = [self._refs.ids[j] for j in rj]
            return AlignResult(host, qids, rids, self.scheme)
        return host

    def top_k(self, k: int = 5, *, queries=None, query_batch: int = 512) -> AlignResult:
        """Per-query top-``k`` references by score (argpartition on the GPU).

        Memory-bounded: only the per-query top-``k`` is kept on the host, so this
        scales to reference sets too large to materialise the full score matrix.
        """
        cp = _compile.require_cupy()
        if self._refs is None:
            raise ValueError("call index(refs) before top_k()")
        if queries is not None:
            self.set_queries(queries)
        if self._queries is None:
            raise ValueError("no queries: pass queries= or call set_queries()")
        nq, nr = self._queries.n, self._refs.n
        kk = min(k, nr)
        idx_out = np.empty((nq, kk), dtype=np.int32)
        sc_out = np.empty((nq, kk), dtype=np.int32)
        with cp.cuda.Device(self.device):
            for b0 in range(0, nq, query_batch):
                qb = np.arange(b0, min(b0 + query_batch, nq), dtype=np.int32)
                dev = self._launch_cross(cp, cp.asarray(qb), len(qb))  # (len(qb), nr)
                part = cp.argpartition(-dev, kk - 1, axis=1)[:, :kk] if kk < nr else \
                    cp.broadcast_to(cp.arange(nr), (len(qb), nr)).copy()
                psc = cp.take_along_axis(dev, part, axis=1)
                order = cp.argsort(-psc, axis=1, kind="stable")
                part = cp.take_along_axis(part, order, axis=1)
                psc = cp.take_along_axis(psc, order, axis=1)
                idx_out[b0:b0 + len(qb)] = cp.asnumpy(part).astype(np.int32)
                sc_out[b0:b0 + len(qb)] = cp.asnumpy(psc).astype(np.int32)
        return AlignResult(None, self._queries.ids, self._refs.ids, self.scheme,
                           topk_idx=idx_out, topk_scores=sc_out)

    # ------------------------------------------------------------ introspection
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

    def cuda_source(self) -> str:
        """The generated CUDA source of the most recently launched kernel."""
        if self._last_source:
            return self._last_source
        maxq = self._queries.max_len if self._queries else 256
        from .kernel import render_source
        return render_source(self.scheme.module_fields(), _compile.bucket_maxq(maxq),
                             self._resolved_dtype() if self._queries and self._refs else "int32")
