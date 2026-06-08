"""GPU kernel == CPU oracle, to the integer, across modes / matrices / dtypes / edges.

These tests require a GPU and are skipped automatically otherwise (see conftest.py).
"""
import numpy as np
import pytest

from gpusw import Matrix, Scheme, align_score
from gpusw._compile import get_module
from gpusw.reference import cpu_reference_matrix

pytestmark = pytest.mark.gpu


def rand_seqs(rng, n, lo, hi, alphabet):
    a = np.array(list(alphabet))
    return ["".join(rng.choice(a, int(rng.integers(lo, hi)))) for _ in range(n)]


SCHEMES = {
    "dna_local": (Scheme(mode="local"), "ACGTN"),
    "dna_global": (Scheme(mode="global"), "ACGT"),
    "dna_semi_qr": (Scheme(mode="semiglobal"), "ACGT"),
    "dna_semi_q": (Scheme(mode="semiglobal", free_end_gaps_query=True, free_end_gaps_ref=False), "ACGT"),
    "dna_semi_r": (Scheme(mode="semiglobal", free_end_gaps_query=False, free_end_gaps_ref=True), "ACGT"),
    "ednafull_global": (Scheme(mode="global", matrix=Matrix.named("EDNAFULL"), gap_open=-10, gap_extend=-1), "ACGTN"),
    "blosum62_local": (Scheme(mode="local", matrix=Matrix.named("BLOSUM62"), gap_open=-11, gap_extend=-1), "ARNDCQEGHILKMFPSTWYV"),
    "blosum62_global": (Scheme(mode="global", matrix=Matrix.named("BLOSUM62"), gap_open=-11, gap_extend=-1), "ARNDCQEGHILKMFPSTWYVX"),
    "protein_semiglobal": (Scheme(mode="semiglobal", matrix=Matrix.named("BLOSUM62"), gap_open=-11, gap_extend=-1), "ARNDCQEGHILKMFPSTWYV"),
}


@pytest.mark.parametrize("name", list(SCHEMES))
def test_gpu_equals_oracle(name):
    scheme, alpha = SCHEMES[name]
    rng = np.random.default_rng(hash(name) % 2**32)
    queries = rand_seqs(rng, 6, 5, 28, alpha)
    refs = rand_seqs(rng, 9, 5, 32, alpha)
    gpu = align_score(queries, refs, scheme=scheme, return_ids=False)
    cpu = cpu_reference_matrix(queries, refs, scheme)
    assert np.array_equal(gpu.astype(np.int64), cpu)


@pytest.mark.parametrize("name", list(SCHEMES))
def test_gpu_equals_oracle_edges(name):
    scheme, alpha = SCHEMES[name]
    c0 = alpha[0]
    queries = ["", c0, c0 * 30, "".join(alpha[:4]), alpha[:1] * 1]
    refs = ["", c0, "".join(reversed(alpha)) * 2, c0 * 25]
    gpu = align_score(queries, refs, scheme=scheme, return_ids=False)
    cpu = cpu_reference_matrix(queries, refs, scheme)
    assert np.array_equal(gpu.astype(np.int64), cpu)


def test_int32_promotion_bitexact():
    # large match score forces a score > int16 range on SHORT sequences (so the
    # per-thread DP buffers stay small); auto must promote to int32 and stay exact.
    scheme = Scheme(mode="local", match=1000, mismatch=-3, gap_open=-5, gap_extend=-2)
    q = "ACGT" * 15  # 60 bp -> perfect score 60_000 > 32767
    r = "ACGT" * 15
    assert scheme.resolve_dtype(len(q), len(r)) == "int32"
    gpu = align_score([q], [r], scheme=scheme, return_ids=False)
    assert gpu.dtype == np.int32
    assert int(gpu[0, 0]) == 1000 * len(q)
    # and it equals the oracle
    cpu = cpu_reference_matrix([q], [r], scheme)
    assert np.array_equal(gpu.astype(np.int64), cpu)


def test_module_cache_identity():
    s = Scheme(mode="local")
    a = get_module(s, 300, "int16")
    b = get_module(s, 300, "int16")
    assert a is b  # same key -> cached module object
    c = get_module(s, 300, "int32")
    assert c is not a  # different dtype -> different module


def test_dna_default_regression():
    # the original prFBA scheme reproduces a known perfect-match / partial score
    s = Scheme()  # local match2/mm-3/go-5/ge-2
    assert int(align_score(["ACGTACGTAC"], ["ACGTACGTAC"], scheme=s, return_ids=False)[0, 0]) == 20
    assert int(align_score(["ACGTACGTAC"], ["TTACGTACGTACTT"], scheme=s, return_ids=False)[0, 0]) == 20
