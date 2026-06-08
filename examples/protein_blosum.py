#!/usr/bin/env python
"""Protein local alignment with a BLOSUM matrix.

Demonstrates substitution-matrix scoring, a non-DNA alphabet, and overriding the gap
costs on a preset.

    python examples/protein_blosum.py
"""
from __future__ import annotations

import gpusw
from gpusw import Matrix, Scheme, align_score

queries = {
    "kinase_motif": "GAGKTLLI",
    "random":       "WWWWCCCC",
}
references = {
    "ref_a": "MKTAYIAKQRGAGKTLLIVTSDEER",
    "ref_b": "GAGKSLLIQACDEFGHIKLMNPQRS",
    "ref_c": "WWWWCCCCYYYYHHHHPPPPGGGGA",
}

if not gpusw.gpu_available():
    raise SystemExit("This example needs a CUDA GPU + CuPy (pip install gpusw[cuda13]).")

# Preset (BLOSUM62, local, gap -11/-1) ...
res = align_score(queries, references, scheme="blosum62")
print("BLOSUM62 local best hits:")
for q, ref, score in res.best():
    print(f"  {q:14} -> {ref:6} score={score}")

# ... or a fully explicit scheme with a different matrix / gaps.
scheme = Scheme(mode="local", matrix=Matrix.named("BLOSUM80"), gap_open=-10, gap_extend=-1)
res2 = align_score(queries, references, scheme=scheme)
print("\nBLOSUM80 local scores:\n", res2.to_dataframe())
