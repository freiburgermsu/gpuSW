#!/usr/bin/env python
"""Benchmark gpusw throughput (giga cell-updates per second) on synthetic data.

    python benchmarks/bench_gcups.py --queries 64 --qlen 350 --refs 4000 --rlen 1500

Reports wall time and GCUPS for an exhaustive cross-product, the headline metric for
Smith-Waterman engines (cells = sum(query_len) * sum(ref_len)).
"""
from __future__ import annotations

import argparse
import time

import numpy as np

import gpusw
from gpusw import Aligner, Scheme


def rand_seqs(n, length, alphabet, rng, jitter=0.15):
    a = np.array(list(alphabet))
    out = []
    for _ in range(n):
        L = max(1, int(length * (1 + rng.uniform(-jitter, jitter))))
        out.append("".join(rng.choice(a, L)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", type=int, default=64)
    ap.add_argument("--qlen", type=int, default=350)
    ap.add_argument("--refs", type=int, default=4000)
    ap.add_argument("--rlen", type=int, default=1500)
    ap.add_argument("--scheme", default="dna")
    ap.add_argument("--mode", default=None)
    ap.add_argument("--alphabet", default="ACGT")
    ap.add_argument("--repeats", type=int, default=2)
    args = ap.parse_args()

    if not gpusw.gpu_available():
        raise SystemExit("no GPU/CuPy available")
    import cupy as cp

    rng = np.random.default_rng(0)
    queries = rand_seqs(args.queries, args.qlen, args.alphabet, rng)
    refs = rand_seqs(args.refs, args.rlen, args.alphabet, rng)
    scheme = Scheme.preset(args.scheme)
    if args.mode:
        scheme = scheme.replace(mode=args.mode)

    al = Aligner(scheme).index(refs).set_queries(queries)
    cells = sum(len(q) for q in queries) * sum(len(r) for r in refs)
    print(f"device : {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
    print(f"problem: {len(queries)} queries (~{args.qlen}bp) x {len(refs)} refs "
          f"(~{args.rlen}bp) = {cells/1e12:.3f} T cells | scheme={scheme.mode}")

    best = 0.0
    for _ in range(args.repeats):
        cp.cuda.runtime.deviceSynchronize()
        t0 = time.perf_counter()
        al.score_cross(query_batch=16)
        cp.cuda.runtime.deviceSynchronize()
        dt = time.perf_counter() - t0
        g = cells / dt / 1e9
        best = max(best, g)
        print(f"  wall={dt:6.2f}s  {g:6.1f} GCUPS")
    print(f"best   : {best:.1f} GCUPS")


if __name__ == "__main__":
    main()
