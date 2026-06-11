"""Metal backend — Apple-Silicon GPUs via PyObjC/Metal.

The structural twin of :mod:`gpusw.backends.cuda`, over :mod:`gpusw._metal` (the
MTLComputePipelineState cache + PyObjC boundary) and :mod:`gpusw.metal_kernel` (the MSL
generator). It launches the **same** Gotoh kernel, so scores are bit-exact with the CUDA
backend and the CPU oracle.

Two honest, GPU-API-level differences from CUDA:

* **Dispatch.** CUDA's ``grid×block`` launch becomes ``dispatchThreadgroups`` with a
  ceil-divided grid and the same in-kernel ``if (t >= total) return;`` bound — identical
  coverage, no over-run.
* **Top-k.** CuPy offers an on-device ``argpartition``; Metal has no such primitive, so on
  unified memory we read the (already host-visible) scores back and partition with NumPy.
  Same result, and cheap because device memory *is* host memory on Apple Silicon.
"""
from __future__ import annotations

import time
from contextlib import contextmanager

import numpy as np

from . import Backend


class MetalBackend(Backend):
    """One-thread-per-pair Smith-Waterman/NW on Apple Silicon via runtime-compiled MSL."""

    name = "metal"

    # MSL's [[thread_position_in_grid]] is a 32-bit uint, so a single dispatch must stay
    # below 2**32 threads or the flat index wraps (silently leaving high-index outputs
    # unwritten). 2**31 keeps the whole grid — including the ceil-rounded last threadgroup
    # — comfortably inside uint range; the engine and run_pairs chunk to honor it.
    max_threads_per_launch = 1 << 31

    def available(self) -> bool:
        from .. import _metal

        return _metal.metal_available()

    @contextmanager
    def device_scope(self, device: int):
        from .. import _metal

        _metal.device()  # initialise the cached device/queue; Metal has one default device
        yield

    def upload(self, arr):
        from .. import _metal

        return _metal.to_buffer(np.ascontiguousarray(arr))

    @staticmethod
    def _store_np(dtype):
        return np.int16 if dtype == "int16" else np.int32

    def _dispatch(self, pso, buffers: dict, scalars: dict, total: int, threads: int) -> float:
        """Encode + run one kernel; return GPU wall-time (s).

        ``buffers`` maps binding index → ``MTLBuffer``; ``scalars`` maps index → a small
        ``np.int32`` value passed by value via ``setBytes``.
        """
        from .. import _metal

        Metal = _metal.require_metal()
        queue = _metal.command_queue()
        cb = queue.commandBuffer()
        enc = cb.computeCommandEncoder()
        enc.setComputePipelineState_(pso)
        for idx, buf in buffers.items():
            enc.setBuffer_offset_atIndex_(buf, 0, idx)
        for idx, val in scalars.items():
            v = np.asarray(val, dtype=np.int32)
            enc.setBytes_length_atIndex_(v.tobytes(), int(v.nbytes), idx)
        tpt = min(int(threads), int(pso.maxTotalThreadsPerThreadgroup()))
        tpt = max(tpt, 1)
        groups = (int(total) + tpt - 1) // tpt
        t0 = time.perf_counter()
        enc.dispatchThreadgroups_threadsPerThreadgroup_(
            Metal.MTLSizeMake(groups, 1, 1), Metal.MTLSizeMake(tpt, 1, 1)
        )
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        dt = time.perf_counter() - t0
        if cb.status() != Metal.MTLCommandBufferStatusCompleted:
            from ..errors import GpuSWError

            raise GpuSWError(
                f"Metal command buffer did not complete (status={cb.status()}): {cb.error()}"
            )
        return dt

    def run_cross(self, scheme, maxq, dtype, threads,
                  qbuf, qoff, rbuf, roff, qlist_host, nq, nr):
        from .. import _metal

        pso_cross, _, src = _metal.get_pipeline(scheme, maxq, dtype)
        sub = _metal.get_sub(scheme)
        qlist_buf = _metal.to_buffer(np.ascontiguousarray(qlist_host, dtype=np.int32))
        store_np = self._store_np(dtype)
        total = nq * nr
        out = _metal.empty_buffer(total * np.dtype(store_np).itemsize)
        dt = self._dispatch(
            pso_cross,
            buffers={0: qbuf, 1: qoff, 2: rbuf, 3: roff, 4: qlist_buf, 7: out, 8: sub},
            scalars={5: np.int32(nq), 6: np.int32(nr)},
            total=total, threads=threads,
        )
        scores = _metal.buffer_to_numpy(out, total, store_np).astype(np.int32).reshape(nq, nr)
        return scores, dt, src

    def run_pairs(self, scheme, maxq, dtype, threads,
                  qbuf, qoff, rbuf, roff, qi_host, rj_host, n):
        from .. import _metal

        _, pso_pairs, src = _metal.get_pipeline(scheme, maxq, dtype)
        sub = _metal.get_sub(scheme)
        qi = np.ascontiguousarray(qi_host, dtype=np.int32)
        rj = np.ascontiguousarray(rj_host, dtype=np.int32)
        store_np = self._store_np(dtype)
        itemsize = np.dtype(store_np).itemsize
        n = int(n)
        cap = self.max_threads_per_launch
        host = np.empty(n, dtype=np.int32)
        total_dt = 0.0
        # chunk so a single dispatch never exceeds the 32-bit grid limit (one iteration
        # for the usual n <= cap case)
        for s0 in range(0, n, cap):
            s1 = min(s0 + cap, n)
            m = s1 - s0
            qi_buf = _metal.to_buffer(qi[s0:s1])
            rj_buf = _metal.to_buffer(rj[s0:s1])
            out = _metal.empty_buffer(m * itemsize)
            total_dt += self._dispatch(
                pso_pairs,
                buffers={0: qbuf, 1: qoff, 2: rbuf, 3: roff, 4: qi_buf, 5: rj_buf, 7: out, 8: sub},
                scalars={6: np.int32(m)},
                total=m, threads=threads,
            )
            host[s0:s1] = _metal.buffer_to_numpy(out, m, store_np).astype(np.int32)
        return host, total_dt, src

    def as_host(self, scores_native):
        # run_cross already returns a host int32 array on unified memory.
        return np.ascontiguousarray(scores_native, dtype=np.int32)

    def topk(self, scores_native, kk):
        scores = scores_native
        nq, nr = scores.shape
        if kk < nr:
            part = np.argpartition(-scores, kk - 1, axis=1)[:, :kk]
        else:
            part = np.broadcast_to(np.arange(nr), (nq, nr)).copy()
        psc = np.take_along_axis(scores, part, axis=1)
        order = np.argsort(-psc, axis=1, kind="stable")
        part = np.take_along_axis(part, order, axis=1)
        psc = np.take_along_axis(psc, order, axis=1)
        return part.astype(np.int32), psc.astype(np.int32)

    def render(self, scheme, maxq, dtype) -> str:
        from .._compile import bucket_maxq
        from ..metal_kernel import render_source

        return render_source(scheme.module_fields(), bucket_maxq(maxq), dtype)
