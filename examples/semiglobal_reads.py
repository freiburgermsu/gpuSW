#!/usr/bin/env python
"""Semiglobal ("glocal") read mapping: short reads should align end-to-end inside longer
references without paying for the reference overhang.

Free *query* end gaps let reference flanks hang off for free; this is the overlap/read-
mapping regime. Compare local vs semiglobal to see the difference.

    python examples/semiglobal_reads.py
"""
from __future__ import annotations

import gpusw
from gpusw import Scheme, align_score, cpu_reference_score

reference = "TTTTTTACGTACGTACGTGGGGGG"   # the read sits in the middle
reads = {
    "read_exact":   "ACGTACGTACGT",
    "read_1mm":     "ACGTACTTACGT",        # one mismatch
}

# Free query end gaps (reference flanks free), reference fully consumed off.
semi = Scheme(mode="semiglobal", free_end_gaps_query=True, free_end_gaps_ref=False)

# Works on CPU with no GPU, too:
print("CPU oracle (semiglobal):")
for name, read in reads.items():
    print(f"  {name:12} score={cpu_reference_score(read, reference, semi)}")

if gpusw.gpu_available():
    res = align_score(reads, [reference], scheme=semi)
    print("\nGPU semiglobal scores:", res.scores.ravel().tolist())
    res_local = align_score(reads, [reference], scheme="dna")  # local for contrast
    print("GPU local scores:     ", res_local.scores.ravel().tolist())
