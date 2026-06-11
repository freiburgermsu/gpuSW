"""CUDA backend — NVIDIA GPUs via CuPy/NVRTC.

A thin adapter over :mod:`gpusw._compile` (the RawModule/NVRTC cache + CuPy boundary)
and :mod:`gpusw.kernel` (the CUDA C++ generator). Every device operation here is exactly
the CuPy call the engine used before backends existed, so the CUDA path is unchanged:
same module/sub caches, same ``deviceSynchronize``-bracketed timing, same on-device
``argpartition`` top-k.
"""
from __future__ import annotations

import time
from contextlib import contextmanager

import numpy as np

from . import Backend


class CudaBackend(Backend):
    """One-thread-per-pair Smith-Waterman/NW on NVIDIA via runtime-compiled CUDA."""

    name = "cuda"

    def available(self) -> bool:
        from .. import _compile

        return _compile.gpu_available()

    @contextmanager
    def device_scope(self, device: int):
        from .. import _compile

        cp = _compile.require_cupy()
        with cp.cuda.Device(device):
            yield

    def upload(self, arr):
        from .. import _compile

        cp = _compile.require_cupy()
        return cp.asarray(arr)

    @staticmethod
    def _store(cp, dtype):
        return cp.int16 if dtype == "int16" else cp.int32

    def run_cross(self, scheme, maxq, dtype, threads,
                  qbuf, qoff, rbuf, roff, qlist_host, nq, nr):
        from .. import _compile

        cp = _compile.require_cupy()
        _, fcross, _, src = _compile.get_module(scheme, maxq, dtype)
        sub = _compile.get_sub(scheme)
        qlist_dev = cp.asarray(np.ascontiguousarray(qlist_host, dtype=np.int32))
        out = cp.empty(nq * nr, dtype=self._store(cp, dtype))
        total = nq * nr
        thr = int(threads)
        grid = ((total + thr - 1) // thr,)
        cp.cuda.runtime.deviceSynchronize()
        t0 = time.perf_counter()
        fcross(grid, (thr,),
               (qbuf, qoff, rbuf, roff, qlist_dev,
                np.int32(nq), np.int32(nr), out, sub))
        cp.cuda.runtime.deviceSynchronize()
        dt = time.perf_counter() - t0
        return out.reshape(nq, nr), dt, src

    def run_pairs(self, scheme, maxq, dtype, threads,
                  qbuf, qoff, rbuf, roff, qi_host, rj_host, n):
        from .. import _compile

        cp = _compile.require_cupy()
        _, _, fpairs, src = _compile.get_module(scheme, maxq, dtype)
        sub = _compile.get_sub(scheme)
        out = cp.empty(n, dtype=self._store(cp, dtype))
        thr = int(threads)
        grid = ((n + thr - 1) // thr,)
        cp.cuda.runtime.deviceSynchronize()
        t0 = time.perf_counter()
        fpairs(grid, (thr,),
               (qbuf, qoff, rbuf, roff,
                cp.asarray(qi_host), cp.asarray(rj_host), np.int32(n), out, sub))
        cp.cuda.runtime.deviceSynchronize()
        dt = time.perf_counter() - t0
        return cp.asnumpy(out).astype(np.int32), dt, src

    def as_host(self, scores_native):
        from .. import _compile

        cp = _compile.require_cupy()
        return cp.asnumpy(scores_native).astype(np.int32)

    def topk(self, scores_native, kk):
        from .. import _compile

        cp = _compile.require_cupy()
        nq, nr = scores_native.shape
        if kk < nr:
            part = cp.argpartition(-scores_native, kk - 1, axis=1)[:, :kk]
        else:
            part = cp.broadcast_to(cp.arange(nr), (nq, nr)).copy()
        psc = cp.take_along_axis(scores_native, part, axis=1)
        order = cp.argsort(-psc, axis=1, kind="stable")
        part = cp.take_along_axis(part, order, axis=1)
        psc = cp.take_along_axis(psc, order, axis=1)
        return cp.asnumpy(part).astype(np.int32), cp.asnumpy(psc).astype(np.int32)

    def render(self, scheme, maxq, dtype) -> str:
        from .._compile import bucket_maxq
        from ..kernel import render_source

        return render_source(scheme.module_fields(), bucket_maxq(maxq), dtype)
