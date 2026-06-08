#!/usr/bin/env python
"""Exhaustive DNA local Smith-Waterman: score every query against every reference and
keep the top hits per query. This is the generalized form of the original 16S ASV ->
reference mapping that gpusw was extracted from.

    python examples/dna_exhaustive.py
"""
from __future__ import annotations

import gpusw
from gpusw import Aligner

# In practice: pass FASTA paths, e.g. Aligner("dna").index("refs.fasta")
references = {
    "ref_ecoli": "ACGTACGTTGCAGTCAGTCAGGGACGTACGTACGTAGCTAGCTAGCATCGATCGTAGCTAGT",
    "ref_bsub":  "TTGCAGTCAGTCAGGGTTTTACGTACGTACGTAGCTAGCTAGCATCGATCGTAGCTAGTACG",
    "ref_paer":  "GGGGCCCCGGGGCCCCACGTACGTACGTAGCTAGCTAGCATCGATCGTAGCTAGTACGTACG",
}
queries = {
    "asv_1": "ACGTACGTACGTAGCTAGCTAGCATCGATCG",   # interior of ref_ecoli/ref_bsub
    "asv_2": "GGGGCCCCGGGGCCCCACGTACGT",            # start of ref_paer
}

if not gpusw.gpu_available():
    raise SystemExit("This example needs a CUDA GPU + CuPy (pip install gpusw[cuda13]).")

al = Aligner("dna").index(references)        # encode + upload references once
result = al.top_k(queries, k=2)              # GPU exhaustive + per-query top-k

print(f"throughput: {al.gcups:.2f} GCUPS")
for query_id, hits in result.top_k(2):
    print(f"\n{query_id}:")
    for rank, (ref_id, score) in enumerate(hits, 1):
        print(f"  {rank}. {ref_id:12} score={score}")

# Cross product (full score matrix) if you want every cell:
full = al.score_cross(return_ids=True)
print("\nfull score matrix:\n", full.to_dataframe().pivot(
    index="query", columns="reference", values="score"))
