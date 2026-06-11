#!/usr/bin/env python
"""metal_16s_align.py — on-box Apple-Silicon (Metal) validation + benchmark of gpusw,
mirroring the original CUDA 16S driver (``examples/legacy_16s_gpu_align.py``) on the same
EmilyKin ASV data, but through the installable, generalized ``gpusw`` API.

Data (all read-only, from the EmilyKin study repo — nothing here is recomputed at scale):
  * Queries:    16S rRNA ASVs           (EmilyKin ``asvs.fasta``, 3,950 seqs, 236–561 bp)
  * References: MAG 16S sequences       (EmilyKin ``mag_16s.fasta``, 343 seqs, ~1,500 bp)

Scheme: local Smith-Waterman, match +2 / mismatch −3 / gap_open −5 / gap_extend −2 — the
**same** scheme the EmilyKin ``edlib → biopython`` CPU pipeline used
(``bvbrc_alignment_hits/edlib_biopython_hits.py``), so every Metal score here is directly
comparable to that pipeline's ``align_score`` values, and bit-exact with both the package
CPU oracle and the CUDA kernel.

What it does, exactly mirroring how the CUDA tier was developed and verified:
  1. Correctness vs **Biopython** on N random real (ASV, ref) pairs  → expect N/N exact.
  2. **Exhaustive** Metal cross product of all ASVs × all refs        → wall time + GCUPS.
  3. Per-ASV **top-k** on Metal vs a Biopython top-k on a sample      → overlap + top-1.
Writes the metrics to ``docs/metal_align_stats_16s.json``.

Run (from the repo root, in a venv with ``pip install -e '.[metal,test]'``):
  python examples/metal_16s_align.py \
      --asvs ../EmilyKin/asvs.fasta --refs ../EmilyKin/mag_16s.fasta
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

import gpusw
from gpusw import Aligner, schemes
from gpusw.encode import read_fasta

# the EmilyKin CPU pipeline scheme (edlib_biopython_hits.py): local SW 2/-3/-5/-2
SCHEME = schemes.DNA
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
_DEFAULT_ASVS = os.path.normpath(os.path.join(_REPO, "..", "EmilyKin", "asvs.fasta"))
_DEFAULT_REFS = os.path.normpath(os.path.join(_REPO, "..", "EmilyKin", "mag_16s.fasta"))
_DEFAULT_OUT = os.path.join(_REPO, "docs", "metal_align_stats_16s.json")


def _biopython_aligner():
    from Bio import Align

    a = Align.PairwiseAligner()
    a.mode = "local"
    a.match_score = SCHEME.match
    a.mismatch_score = SCHEME.mismatch
    a.open_gap_score = SCHEME.gap_open
    a.extend_gap_score = SCHEME.gap_extend
    return a


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--asvs", default=_DEFAULT_ASVS, help="ASV (query) FASTA")
    ap.add_argument("--refs", default=_DEFAULT_REFS, help="reference FASTA (MAG 16S)")
    ap.add_argument("--out", default=_DEFAULT_OUT, help="stats JSON output path")
    ap.add_argument("--pairs", type=int, default=300, help="random pairs for the Biopython correctness check")
    ap.add_argument("--topk", type=int, default=5, help="top-k references kept per ASV")
    ap.add_argument("--topk-sample", type=int, default=25, dest="topk_sample",
                    help="ASVs to validate top-k against Biopython (0 = skip)")
    ap.add_argument("--limit", type=int, default=0, help="restrict to the first N ASVs (smoke test)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not gpusw.metal_available():
        raise SystemExit("Metal GPU not available; this driver requires Apple Silicon + 'gpusw[metal]'.")

    t0 = time.perf_counter()
    rng = np.random.default_rng(args.seed)

    print(f"[load] {args.asvs}\n[load] {args.refs}", flush=True)
    asv_pairs = read_fasta(args.asvs)
    ref_pairs = read_fasta(args.refs)
    if args.limit:
        asv_pairs = asv_pairs[:args.limit]
    asv_ids = [i for i, _ in asv_pairs]
    asv_seqs = [s for _, s in asv_pairs]
    ref_ids = [i for i, _ in ref_pairs]
    ref_seqs = [s for _, s in ref_pairs]
    nq, nr = len(asv_seqs), len(ref_seqs)
    maxq = max(len(s) for s in asv_seqs)
    print(f"[load] {nq:,} ASVs | {nr:,} refs | max ASV len {maxq} | scheme {SCHEME.mode} "
          f"match{SCHEME.match}/mm{SCHEME.mismatch}/go{SCHEME.gap_open}/ge{SCHEME.gap_extend}", flush=True)

    al = Aligner(SCHEME, backend="metal").index(ref_seqs).set_queries(asv_seqs)
    print(f"[metal] backend={al.backend} | device={gpusw._metal.device().name()} | "
          f"dtype={al._resolved_dtype()}", flush=True)

    # ---- 1. correctness vs Biopython on random real pairs --------------------------
    aln = _biopython_aligner()
    npc = min(args.pairs, nq * nr)
    qi = rng.integers(0, nq, npc).astype(np.int32)
    rj = rng.integers(0, nr, npc).astype(np.int32)
    metal_pair = gpusw.align_pairs(asv_seqs, ref_seqs,
                                   pairs=list(zip(qi.tolist(), rj.tolist(), strict=True)),
                                   scheme=SCHEME, backend="metal", return_ids=False)
    bio = np.array([int(round(aln.score(asv_seqs[qi[k]], ref_seqs[rj[k]]))) for k in range(npc)])
    diff = np.abs(metal_pair.astype(np.int64) - bio)
    exact = int((diff == 0).sum())
    print(f"[correctness] {npc} random pairs vs Biopython: exact={exact}/{npc} "
          f"max_abs_diff={int(diff.max())} mean_abs_diff={diff.mean():.4f}", flush=True)

    # ---- 2. exhaustive Metal cross product (all ASVs x all refs) --------------------
    cells = int(sum(len(s) for s in asv_seqs)) * int(sum(len(s) for s in ref_seqs))
    ts = time.perf_counter()
    scores = al.score_cross(return_ids=False)          # (nq, nr) int32, on Metal
    wall = time.perf_counter() - ts
    gcups = cells / wall / 1e9 if wall > 0 else 0.0
    print(f"[exhaustive] {nq:,} ASVs x {nr:,} refs = {nq*nr:,} pairs in {wall:.2f}s "
          f"=> {gcups:.1f} GCUPS (cells={cells/1e12:.3f}T)", flush=True)

    # ---- 3. per-ASV top-k on Metal vs Biopython truth on a sample -------------------
    topk = min(args.topk, nr)
    metal_idx = np.argsort(-scores, axis=1, kind="stable")[:, :topk]
    top1_match = capt = sample_n = 0
    if args.topk_sample:
        sample = rng.choice(nq, min(args.topk_sample, nq), replace=False)
        for a in sample:
            bio_row = np.array([int(round(aln.score(asv_seqs[a], ref_seqs[j]))) for j in range(nr)])
            bio_top = set(np.argsort(-bio_row, kind="stable")[:topk].tolist())
            m_top = metal_idx[a].tolist()
            capt += len(bio_top & set(m_top))
            top1_match += int(m_top[0] == int(np.argmax(bio_row)))
            sample_n += 1
        print(f"[validate] top-{topk} vs Biopython over {sample_n} sampled ASVs: "
              f"overlap {capt}/{sample_n*topk} ({100*capt/(sample_n*topk):.1f}%) | "
              f"top-1 match {top1_match}/{sample_n}", flush=True)

    # best hit per ASV (for a quick human-readable head)
    best_j = metal_idx[:, 0]
    best_sc = scores[np.arange(nq), best_j]
    print("\n[head] ASV -> best MAG ref (Metal):")
    for a in range(min(5, nq)):
        print(f"  {asv_ids[a][:12]}  ->  {ref_ids[int(best_j[a])][:28]:28}  score={int(best_sc[a])}")

    stats = {
        "machine": "Apple M4 (10-core GPU, 16 GB unified)",
        "device": gpusw._metal.device().name(),
        "backend": "metal",
        "kernel": "custom runtime-MSL affine-local-SW, 1 thread/pair (gpusw)",
        "scheme": {"mode": SCHEME.mode, "match": SCHEME.match, "mismatch": SCHEME.mismatch,
                   "gap_open": SCHEME.gap_open, "gap_extend": SCHEME.gap_extend,
                   "note": "identical to EmilyKin edlib_biopython_hits.py CPU pipeline"},
        "n_asvs": nq, "n_refs": nr, "max_asv_len": maxq, "dtype": al._resolved_dtype(),
        "correctness_pairs": npc,
        "correctness_exact_match": f"{exact}/{npc}",
        "correctness_max_abs_diff": int(diff.max()),
        "exhaustive_pairs": nq * nr,
        "exhaustive_cells_T": round(cells / 1e12, 4),
        "exhaustive_wall_s": round(wall, 3),
        "exhaustive_gcups": round(gcups, 1),
        "topk": topk,
        "topk_sample_asvs": sample_n,
        "topk_overlap_pct": round(100 * capt / (sample_n * topk), 1) if sample_n else None,
        "topk_top1_match": f"{top1_match}/{sample_n}" if sample_n else None,
        "total_wall_s": round(time.perf_counter() - t0, 1),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(stats, fh, indent=2)
    print("\n[stats]", json.dumps(stats, indent=2), flush=True)
    print(f"[wrote] {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
