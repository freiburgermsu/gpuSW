"""Anchor the pure-NumPy oracle to Biopython across modes and matrices.

This pins the gap convention, the local/global/semiglobal semantics, and the
substitution-matrix scoring that the GPU kernel is then validated against.
"""
import numpy as np
import pytest

from gpusw import Matrix, Scheme
from gpusw.reference import cpu_reference_score

pytest.importorskip("Bio")
from Bio import Align  # noqa: E402
from Bio.Align import substitution_matrices  # noqa: E402

_BIO_NAME = {"EDNAFULL": "NUC.4.4"}


def bio_aligner(scheme):
    a = Align.PairwiseAligner()
    a.mode = "local" if scheme.mode == "local" else "global"
    a.open_gap_score = scheme.gap_open
    a.extend_gap_score = scheme.gap_extend
    if scheme.matrix is not None:
        name = _BIO_NAME.get(scheme.matrix.name, scheme.matrix.name)
        a.substitution_matrix = substitution_matrices.load(name)
    else:
        a.match_score = scheme.match
        a.mismatch_score = scheme.mismatch
    if scheme.mode == "semiglobal":
        a.target_end_open_gap_score = 0 if scheme.free_end_gaps_ref else scheme.gap_open
        a.target_end_extend_gap_score = 0 if scheme.free_end_gaps_ref else scheme.gap_extend
        a.query_end_open_gap_score = 0 if scheme.free_end_gaps_query else scheme.gap_open
        a.query_end_extend_gap_score = 0 if scheme.free_end_gaps_query else scheme.gap_extend
    return a


def bio_score(query, ref, scheme):
    # target = reference, query = query (matters for semiglobal end gaps)
    return int(round(bio_aligner(scheme).score(ref, query)))


def rand_seqs(rng, n, lo, hi, alphabet):
    a = np.array(list(alphabet))
    return ["".join(rng.choice(a, int(rng.integers(lo, hi)))) for _ in range(n)]


DNA_MM = [
    Scheme(mode="local"),
    Scheme(mode="global"),
    Scheme(mode="semiglobal", free_end_gaps_query=False, free_end_gaps_ref=False),
    Scheme(mode="semiglobal", free_end_gaps_query=True, free_end_gaps_ref=False),
    Scheme(mode="semiglobal", free_end_gaps_query=False, free_end_gaps_ref=True),
    Scheme(mode="semiglobal", free_end_gaps_query=True, free_end_gaps_ref=True),
]


@pytest.mark.parametrize("scheme", DNA_MM, ids=lambda s: f"{s.mode}-{s.free_end_gaps_query}{s.free_end_gaps_ref}")
def test_dna_matches_biopython(scheme):
    rng = np.random.default_rng(123)
    for _ in range(80):
        q = rand_seqs(rng, 1, 3, 16, "ACGT")[0]
        r = rand_seqs(rng, 1, 3, 18, "ACGT")[0]
        assert cpu_reference_score(q, r, scheme) == bio_score(q, r, scheme)


PROT = "ARNDCQEGHILKMFPSTWYV"


@pytest.mark.parametrize("mode", ["local", "global"])
def test_blosum62_matches_biopython(mode):
    scheme = Scheme(mode=mode, matrix=Matrix.named("BLOSUM62"), gap_open=-11, gap_extend=-1)
    rng = np.random.default_rng(7)
    for _ in range(60):
        q = rand_seqs(rng, 1, 4, 18, PROT)[0]
        r = rand_seqs(rng, 1, 4, 20, PROT)[0]
        assert cpu_reference_score(q, r, scheme) == bio_score(q, r, scheme)


def test_ednafull_global_matches_biopython():
    scheme = Scheme(mode="global", matrix=Matrix.named("EDNAFULL"), gap_open=-10, gap_extend=-1)
    rng = np.random.default_rng(99)
    for _ in range(60):
        q = rand_seqs(rng, 1, 4, 16, "ACGT")[0]
        r = rand_seqs(rng, 1, 4, 18, "ACGT")[0]
        assert cpu_reference_score(q, r, scheme) == bio_score(q, r, scheme)


def test_known_anchors():
    assert cpu_reference_score("ACGT", "ACT", Scheme(mode="local")) == 4
    assert cpu_reference_score("ACGTAC", "ACAC", Scheme(mode="global")) == 1
    # overlap: ACGT inside TTACGTTT (free query end gaps) -> 4 matches * 2 = 8
    overlap = Scheme(mode="semiglobal", free_end_gaps_query=True, free_end_gaps_ref=False)
    assert cpu_reference_score("ACGT", "TTACGTTT", overlap) == 8
