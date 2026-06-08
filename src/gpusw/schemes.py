"""Ready-made :class:`~gpusw.Scheme` presets for common alignment problems.

Access by attribute (``gpusw.schemes.BLOSUM62``) or by name on the high-level
helpers / :meth:`Scheme.preset` (``align_score(q, r, scheme="blosum62")``).
"""
from __future__ import annotations

from .matrix import Matrix
from .scheme import Scheme

__all__ = [
    "DNA",
    "DNA_GLOBAL",
    "DNA_SEMIGLOBAL",
    "BLASTN",
    "EDNAFULL",
    "BLOSUM62",
    "BLOSUM62_GLOBAL",
    "PROTEIN_SEMIGLOBAL",
]

# DNA: the original prFBA scheme (match +2 / mismatch -3, affine -5/-2, local).
# Bit-exact with the validated 16S kernel.
DNA = Scheme(mode="local", match=2, mismatch=-3, gap_open=-5, gap_extend=-2)
DNA_GLOBAL = DNA.replace(mode="global")
DNA_SEMIGLOBAL = DNA.replace(mode="semiglobal")  # overlap (both ends free)

# NCBI blastn-style nucleotide defaults.
BLASTN = Scheme(mode="local", match=2, mismatch=-3, gap_open=-5, gap_extend=-2)

# EMBOSS-style full nucleotide alignment with the EDNAFULL (NUC.4.4) matrix.
EDNAFULL = Scheme(
    mode="global", matrix=Matrix.named("EDNAFULL"), gap_open=-10, gap_extend=-1
)

# Protein: BLAST defaults (BLOSUM62, gap open -11 / extend -1).
BLOSUM62 = Scheme(mode="local", matrix=Matrix.named("BLOSUM62"), gap_open=-11, gap_extend=-1)
BLOSUM62_GLOBAL = BLOSUM62.replace(mode="global")
PROTEIN_SEMIGLOBAL = BLOSUM62.replace(mode="semiglobal")
