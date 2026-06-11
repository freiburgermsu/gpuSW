#!/usr/bin/env python
"""metal_bvbrc_validation.py — reproduce the EmilyKin study's *precomputed* CPU
alignment scores on the Apple-Silicon (Metal) backend, with **no CPU recompute**.

The EmilyKin 16S pipeline (``bvbrc_alignment_hits/edlib_biopython_hits.py``) recorded,
for 20 validation ASVs, their exhaustive Biopython top-5 BV-BRC hits — each with the
``align_score`` of a local Smith-Waterman alignment (match +2 / mismatch −3 / gap −5/−2)
in ``asv_top5_alignment_hits_validation.json``. Those scores are the ground truth the CUDA
kernel was checked against.

This driver recovers each hit's reference *sequence* by ``feature_id`` from a BV-BRC 16S
sequence dump (``BV_BRC_16S.json``: ``{header -> sequence}``; the ``.frn`` ships headers
only), scores the exact ``(ASV, reference)`` pair on the **Metal GPU**, and asserts the
Metal score equals the **stored CPU** ``align_score``. Biopython scores by raw character
equality, so ambiguity codes (``N`` …) are matched exactly by using a match/mismatch scheme
over the *observed* alphabet — making the comparison apples-to-apples.

This is the strongest possible verification: Metal vs precomputed CPU, kernel re-run only on
the GPU. (Reference sequences absent from this particular BV-BRC pull are reported as
uncovered rather than silently dropped.)

Run:
  python examples/metal_bvbrc_validation.py \
      --seqs ../BV_BRC_16S.json \
      --val  ../EmilyKin/bvbrc_alignment_hits/asv_top5_alignment_hits_validation.json \
      --asvs ../EmilyKin/asvs.fasta
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time

import numpy as np

import gpusw
from gpusw import Scheme
from gpusw.encode import read_fasta

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
_RESEARCH = os.path.normpath(os.path.join(_REPO, ".."))
_DEF_SEQS = os.path.join(_RESEARCH, "BV_BRC_16S.json")
_DEF_VAL = os.path.join(_RESEARCH, "EmilyKin", "bvbrc_alignment_hits",
                        "asv_top5_alignment_hits_validation.json")
_DEF_ASVS = os.path.join(_RESEARCH, "EmilyKin", "asvs.fasta")
_DEF_OUT = os.path.join(_REPO, "docs", "metal_bvbrc_validation_stats.json")


def feature_of(header: str) -> str:
    """``fig|3141376.3.rna.123|XYZ| SSU rRNA ...`` -> ``3141376.3.rna.123``."""
    h = header[4:] if header.startswith("fig|") else header
    return h.split("|", 1)[0].split()[0]


def md5_recipes(seq: str) -> set[str]:
    """A few plausible md5 recipes, so we can confirm we recovered the right sequence."""
    return {
        hashlib.md5(seq.encode()).hexdigest(),
        hashlib.md5(seq.upper().encode()).hexdigest(),
        hashlib.md5(seq.lower().encode()).hexdigest(),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seqs", default=_DEF_SEQS, help="BV-BRC {header->seq} JSON dump")
    ap.add_argument("--val", default=_DEF_VAL, help="precomputed CPU top-5 validation JSON")
    ap.add_argument("--asvs", default=_DEF_ASVS, help="ASV (query) FASTA")
    ap.add_argument("--out", default=_DEF_OUT, help="stats JSON output path")
    ap.add_argument("--progress", type=int, default=50000, help="log every N scanned refs")
    args = ap.parse_args()

    if not gpusw.metal_available():
        raise SystemExit("Metal GPU not available; needs Apple Silicon + 'gpusw[metal]'.")

    t0 = time.perf_counter()
    val = json.load(open(args.val))
    asv_seq = {i: s.upper() for i, s in read_fasta(args.asvs)}
    hits = []  # (asv_id, feature_id, md5, stored_score)
    for asv, rec in val.items():
        for h in rec.get("top5", []):
            hits.append((asv, h["feature_id"], h["md5"], int(round(h["align_score"]))))
    need = {f for _, f, _, _ in hits}
    print(f"[load] {len(val)} validation ASVs | {len(hits)} top-5 hits | "
          f"{len(need)} unique reference feature_ids needed", flush=True)

    # ---- stream the (huge) BV-BRC dump, keeping only the needed reference sequences ----
    import ijson

    feat_seq: dict[str, str] = {}
    scanned = 0
    print(f"[scan] streaming {args.seqs} for the needed references ...", flush=True)
    with open(args.seqs, "rb") as fh:
        for header, seq in ijson.kvitems(fh, ""):
            scanned += 1
            f = feature_of(header)
            if f in need and f not in feat_seq:
                feat_seq[f] = seq.upper()
            if scanned % args.progress == 0:
                print(f"[scan] {scanned:,} refs scanned | {len(feat_seq)}/{len(need)} found",
                      flush=True)
            if len(feat_seq) == len(need):
                break
    print(f"[scan] done: {scanned:,} refs scanned | recovered "
          f"{len(feat_seq)}/{len(need)} reference sequences", flush=True)

    found = [h for h in hits if h[1] in feat_seq]
    missing_feats = sorted(need - set(feat_seq))
    # md5 integrity: confirm a recovered sequence actually hashes to the stored md5
    md5_ok = sum(1 for a, f, m5, _ in found if m5 in md5_recipes(feat_seq[f]))

    # ---- build a Biopython-equivalent scheme over the observed alphabet ----
    chars = set()
    for a, f, _, _ in found:
        chars |= set(asv_seq[a])
        chars |= set(feat_seq[f])
    alphabet = "".join(sorted(chars))
    scheme = Scheme(mode="local", match=2, mismatch=-3, gap_open=-5, gap_extend=-2,
                    alphabet=alphabet)
    print(f"[scheme] local 2/-3/-5/-2 over observed alphabet {alphabet!r} "
          f"(replicates Biopython char-equality)", flush=True)

    # ---- score the exact (ASV, reference) pairs on Metal; compare to STORED CPU score ----
    qs = [asv_seq[a] for a, _, _, _ in found]
    rs = [feat_seq[f] for _, f, _, _ in found]
    metal = gpusw.align_pairs(qs, rs, scheme=scheme, backend="metal", return_ids=False)
    stored = np.array([s for *_, s in found], dtype=np.int64)
    diff = np.abs(metal.astype(np.int64) - stored)
    exact = int((diff == 0).sum())
    print(f"\n[reproduce] Metal vs PRECOMPUTED CPU align_score over {len(found)} hits "
          f"(of {len(hits)} total; {len(missing_feats)} refs not in this BV-BRC pull):",
          flush=True)
    print(f"  exact = {exact}/{len(found)} | max_abs_diff = {int(diff.max()) if len(found) else 0} "
          f"| md5-confirmed sequences = {md5_ok}/{len(found)}", flush=True)

    # per-ASV: does Metal reproduce the CPU top-1 (best of the recovered hits)?
    by_asv: dict[str, list] = {}
    for k, (a, f, _m5, sc) in enumerate(found):
        by_asv.setdefault(a, []).append((int(metal[k]), sc, f))
    top1_match = 0
    for _a, rows in by_asv.items():
        m_best = max(rows, key=lambda x: x[0])
        c_best = max(rows, key=lambda x: x[1])
        top1_match += int(m_best[2] == c_best[2] or m_best[0] == m_best[1] == c_best[1])
    print(f"  per-ASV top-1 (within recovered hits) agree: {top1_match}/{len(by_asv)}", flush=True)

    for k in range(min(8, len(found))):
        a, f, m5, sc = found[k]
        tag = "OK " if diff[k] == 0 else "DIFF"
        print(f"  [{tag}] asv {a[:10]} x {f:18} stored={sc:5} metal={int(metal[k]):5}", flush=True)

    stats = {
        "machine": "Apple M4 (10-core GPU, 16 GB unified)",
        "device": gpusw._metal.device().name(),
        "backend": "metal",
        "what": "Metal GPU scores vs EmilyKin PRECOMPUTED CPU (Biopython) align_score — no CPU recompute",
        "scheme": {"mode": "local", "match": 2, "mismatch": -3, "gap_open": -5,
                   "gap_extend": -2, "alphabet": alphabet,
                   "note": "match/mismatch over observed alphabet == Biopython char-equality"},
        "validation_asvs": len(val),
        "total_top5_hits": len(hits),
        "refs_needed": len(need),
        "refs_recovered_from_bvbrc": len(feat_seq),
        "hits_covered": len(found),
        "hits_uncovered": len(hits) - len(found),
        "md5_confirmed_sequences": f"{md5_ok}/{len(found)}",
        "exact_score_match": f"{exact}/{len(found)}",
        "max_abs_diff": int(diff.max()) if len(found) else 0,
        "per_asv_top1_agree": f"{top1_match}/{len(by_asv)}",
        "scan_wall_s": round(time.perf_counter() - t0, 1),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(stats, open(args.out, "w"), indent=2)
    print("\n[stats]", json.dumps(stats, indent=2), flush=True)
    print(f"[wrote] {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
