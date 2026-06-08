"""Pure-NumPy affine-Gotoh reference aligner — the bit-exactness *oracle*.

This module has **no CuPy and no GPU dependency**: it is the ground truth the CUDA
kernel is validated against, and it lets users score/verify on a CPU-only machine.
It implements the exact same recurrence, gap convention, substitution-table indexing
and free-end-gap handling as the kernel, so ``gpu_score == cpu_reference_score`` holds
to the integer.

Convention (verified bit-exact against Biopython ``PairwiseAligner``):

* rows = reference, cols = query
* first gap residue costs ``gap_open``, each subsequent costs ``gap_extend``
* ``local`` clamps H≥0 and returns the running max
* ``global`` returns the corner cell
* ``semiglobal`` frees terminal gaps: ``free_end_gaps_query`` zeroes the query-side
  boundary (col 0) and lets the answer come from the last column; ``free_end_gaps_ref``
  does the same for the reference (row 0 / last row)
"""
from __future__ import annotations

import numpy as np

from .encode import build_lut

__all__ = ["cpu_reference_score", "cpu_reference_matrix"]

_NEG = -(1 << 30)


def _encode(seq: str, scheme, lut) -> np.ndarray:
    s = seq.upper() if scheme.case_insensitive else seq
    raw = np.frombuffer(s.encode("ascii", "ignore"), dtype=np.uint8)
    return lut[raw].astype(np.int64)


def _score_codes(qc: np.ndarray, rc: np.ndarray, sub: np.ndarray, scheme) -> int:
    """Score one (query-codes, ref-codes) pair against ``sub`` (NS×NS int)."""
    n = int(qc.shape[0])  # query columns
    m = int(rc.shape[0])  # reference rows
    go, ge = scheme.gap_open, scheme.gap_extend
    mode = scheme.mode
    local = mode == "local"
    free_q = scheme.free_end_gaps_query
    free_r = scheme.free_end_gaps_ref

    # empty-sequence edges (kept bit-identical with the kernel)
    if n == 0 or m == 0:
        if local or (n == 0 and m == 0):
            return 0
        if n == 0:  # query empty: align the m reference residues to gaps
            return 0 if (mode == "semiglobal" and free_q) else go + (m - 1) * ge
        return 0 if (mode == "semiglobal" and free_r) else go + (n - 1) * ge

    Hp = np.empty(n + 1, dtype=np.int64)
    F = np.empty(n + 1, dtype=np.int64)
    Hp[0] = 0
    for j in range(1, n + 1):
        if local:
            Hp[j] = 0
        elif mode == "global":
            Hp[j] = go + (j - 1) * ge
        else:
            Hp[j] = 0 if free_r else go + (j - 1) * ge
    F[:] = _NEG

    best = 0
    corner = endcol = endrow = _NEG
    for i in range(1, m + 1):
        a = int(rc[i - 1])
        subrow = sub[a]
        if local:
            Hleft = 0
        elif mode == "global":
            Hleft = go + (i - 1) * ge
        else:
            Hleft = 0 if free_q else go + (i - 1) * ge
        diag = int(Hp[0])
        Hp[0] = Hleft
        Eleft = _NEG
        for j in range(1, n + 1):
            b = int(qc[j - 1])
            Fj = Hp[j] + go
            fe = F[j] + ge
            if fe > Fj:
                Fj = fe
            Ej = Hleft + go
            ee = Eleft + ge
            if ee > Ej:
                Ej = ee
            Hij = diag + int(subrow[b])
            if Ej > Hij:
                Hij = Ej
            if Fj > Hij:
                Hij = Fj
            if local and Hij < 0:
                Hij = 0
            if Hij > best:
                best = Hij
            if not local:
                if i == m and j == n:
                    corner = Hij
                if free_q and j == n and Hij > endcol:
                    endcol = Hij
                if free_r and i == m and Hij > endrow:
                    endrow = Hij
            diag = int(Hp[j])
            Hp[j] = Hij
            F[j] = Fj
            Hleft = Hij
            Eleft = Ej

    if local:
        return int(best)
    if mode == "global":
        return int(corner)
    res = corner
    if free_q:
        res = max(res, endcol)
    if free_r:
        res = max(res, endrow)
    if free_q and free_r:  # both ends free -> the empty alignment (score 0) is allowed
        res = max(res, 0)
    return int(res)


def cpu_reference_score(query: str, ref: str, scheme) -> int:
    """Optimal alignment score of ``query`` vs ``ref`` under ``scheme`` (CPU oracle)."""
    lut, _ = build_lut(scheme)
    sub = scheme.substitution_table().astype(np.int64)
    qc = _encode(query, scheme, lut)
    rc = _encode(ref, scheme, lut)
    return _score_codes(qc, rc, sub, scheme)


def cpu_reference_matrix(queries, refs, scheme) -> np.ndarray:
    """All-pairs score matrix (``len(queries) × len(refs)``) via the CPU oracle.

    Intended for small inputs / tests — it is a straightforward Python implementation,
    not the fast path.
    """
    lut, _ = build_lut(scheme)
    sub = scheme.substitution_table().astype(np.int64)
    qcs = [_encode(q, scheme, lut) for q in queries]
    rcs = [_encode(r, scheme, lut) for r in refs]
    out = np.empty((len(qcs), len(rcs)), dtype=np.int64)
    for qi, qc in enumerate(qcs):
        for rj, rc in enumerate(rcs):
            out[qi, rj] = _score_codes(qc, rc, sub, scheme)
    return out
