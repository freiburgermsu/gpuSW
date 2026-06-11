"""Metal kernel == CPU oracle, to the integer, across modes / matrices / dtypes / edges.

The Apple-Silicon twin of ``test_gpu_bitexact.py``: identical assertions, ``backend=
"metal"``. These require a Metal GPU + PyObjC and are skipped automatically otherwise
(see conftest.py). Because the oracle is the *same* ground truth the CUDA path is checked
against, passing here means the Metal scores are integer-identical to the CUDA scores too.
"""
import numpy as np
import pytest

from gpusw import Aligner, Matrix, Scheme, align_pairs, align_score
from gpusw._metal import get_pipeline
from gpusw.reference import cpu_reference_matrix, cpu_reference_score

pytestmark = pytest.mark.metal

BACKEND = "metal"


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
def test_metal_equals_oracle(name):
    scheme, alpha = SCHEMES[name]
    rng = np.random.default_rng(hash(name) % 2**32)
    queries = rand_seqs(rng, 6, 5, 28, alpha)
    refs = rand_seqs(rng, 9, 5, 32, alpha)
    gpu = align_score(queries, refs, scheme=scheme, backend=BACKEND, return_ids=False)
    cpu = cpu_reference_matrix(queries, refs, scheme)
    assert np.array_equal(gpu.astype(np.int64), cpu)


@pytest.mark.parametrize("name", list(SCHEMES))
def test_metal_equals_oracle_edges(name):
    scheme, alpha = SCHEMES[name]
    c0 = alpha[0]
    queries = ["", c0, c0 * 30, "".join(alpha[:4]), alpha[:1] * 1]
    refs = ["", c0, "".join(reversed(alpha)) * 2, c0 * 25]
    gpu = align_score(queries, refs, scheme=scheme, backend=BACKEND, return_ids=False)
    cpu = cpu_reference_matrix(queries, refs, scheme)
    assert np.array_equal(gpu.astype(np.int64), cpu)


@pytest.mark.parametrize("name", list(SCHEMES))
def test_metal_pairs_equals_oracle(name):
    """The separate sw_pairs kernel must be bit-exact too (1:1 and explicit pairs)."""
    scheme, alpha = SCHEMES[name]
    rng = np.random.default_rng((hash(name) ^ 0x5151) % 2**32)
    queries = rand_seqs(rng, 12, 5, 28, alpha)
    refs = rand_seqs(rng, 12, 5, 32, alpha)
    gpu = align_pairs(queries, refs, scheme=scheme, backend=BACKEND, return_ids=False)  # 1:1 zip
    cpu = [cpu_reference_score(q, r, scheme) for q, r in zip(queries, refs, strict=True)]
    assert gpu.astype(np.int64).tolist() == cpu


@pytest.mark.parametrize("name", list(SCHEMES))
def test_metal_pairs_equals_oracle_edges(name):
    scheme, alpha = SCHEMES[name]
    c0 = alpha[0]
    qs = ["", c0, c0 * 30, "".join(alpha[:4])]
    rs = ["", c0, "".join(reversed(alpha)) * 2, c0 * 25]
    pairs = [(i, j) for i in range(len(qs)) for j in range(len(rs))]
    gpu = align_pairs(qs, rs, pairs=pairs, scheme=scheme, backend=BACKEND, return_ids=False)
    cpu = [cpu_reference_score(qs[i], rs[j], scheme) for i, j in pairs]
    assert gpu.astype(np.int64).tolist() == cpu


def test_int32_promotion_pairs_bitexact():
    scheme = Scheme(mode="local", match=1000, mismatch=-3, gap_open=-5, gap_extend=-2)
    qs = ["ACGT" * 15, "ACGTAC" * 5, "AAAA"]
    rs = ["ACGT" * 15, "ACGTAC" * 5, "TTTT", "ACGT" * 15]
    pairs = [(0, 0), (1, 1), (2, 2), (0, 3)]
    res = align_pairs(qs, rs, pairs=pairs, scheme=scheme, backend=BACKEND, return_ids=False)
    assert res.dtype == np.int32
    cpu = cpu_reference_matrix(qs, rs, scheme)
    assert res.tolist() == [int(cpu[i, j]) for i, j in pairs]


def test_int32_promotion_bitexact():
    # large match score forces a score > int16 range on SHORT sequences; auto must
    # promote to int32 and stay exact (same envelope logic as CUDA).
    scheme = Scheme(mode="local", match=1000, mismatch=-3, gap_open=-5, gap_extend=-2)
    q = "ACGT" * 15  # 60 bp -> perfect score 60_000 > 32767
    r = "ACGT" * 15
    assert scheme.resolve_dtype(len(q), len(r)) == "int32"
    gpu = align_score([q], [r], scheme=scheme, backend=BACKEND, return_ids=False)
    assert gpu.dtype == np.int32
    assert int(gpu[0, 0]) == 1000 * len(q)
    cpu = cpu_reference_matrix([q], [r], scheme)
    assert np.array_equal(gpu.astype(np.int64), cpu)


def test_pipeline_cache_identity():
    s = Scheme(mode="local")
    a = get_pipeline(s, 300, "int16")
    b = get_pipeline(s, 300, "int16")
    assert a is b  # same key -> cached pipeline objects
    c = get_pipeline(s, 300, "int32")
    assert c is not a  # different dtype -> different pipeline


def test_dna_default_regression():
    # the original prFBA scheme reproduces a known perfect-match / partial score
    s = Scheme()  # local match2/mm-3/go-5/ge-2
    assert int(align_score(["ACGTACGTAC"], ["ACGTACGTAC"], scheme=s, backend=BACKEND, return_ids=False)[0, 0]) == 20
    assert int(align_score(["ACGTACGTAC"], ["TTACGTACGTACTT"], scheme=s, backend=BACKEND, return_ids=False)[0, 0]) == 20


def test_backend_is_metal():
    al = Aligner("dna", backend=BACKEND).index(["ACGT"]).set_queries(["ACGT"])
    al.score_cross()
    assert al.backend == "metal"
    assert "kernel void sw_cross" in al.metal_source()


def test_launch_chunking_under_thread_cap(monkeypatch):
    """A tiny per-launch thread cap forces multi-chunk cross/pairs/top_k; results must
    still equal the oracle — guards the 32-bit Metal grid-index limit fix."""
    from gpusw.backends.metal import MetalBackend
    monkeypatch.setattr(MetalBackend, "max_threads_per_launch", 64)
    scheme = Scheme()  # dna local
    rng = np.random.default_rng(7)
    queries = rand_seqs(rng, 9, 6, 20, "ACGT")
    refs = rand_seqs(rng, 11, 6, 24, "ACGT")           # 9*11 = 99 > 64 -> multiple chunks
    al = Aligner(scheme, backend="metal").index(refs).set_queries(queries)
    cpu = cpu_reference_matrix(queries, refs, scheme)
    assert np.array_equal(al.score_cross(return_ids=False).astype(np.int64), cpu)  # cross chunks
    pairs = [(i, j) for i in range(len(queries)) for j in range(len(refs))]        # 99 pairs
    gp = align_pairs(queries, refs, pairs=pairs, scheme=scheme, backend="metal", return_ids=False)
    assert gp.astype(np.int64).tolist() == [cpu_reference_score(queries[i], refs[j], scheme)
                                            for i, j in pairs]                      # pairs chunk
    # top_k chunks too; compare the per-query top-3 *scores* (tie-ordering of equal scores
    # may differ between argpartition and a full stable sort, but the scores must match)
    tk_scores = [[s for _, s in hits] for _, hits in al.top_k(k=3).top_k(3)]
    full_scores = [[s for _, s in hits] for _, hits in al.score_cross(return_ids=True).top_k(3)]
    assert tk_scores == full_scores


def test_localmem_cap_message_is_runnable(monkeypatch):
    """The 'query too long' error must advertise a length that actually compiles
    (128-bucket aware), with one bucket beyond raising."""
    import re

    from gpusw import _metal
    from gpusw.errors import GpuSWError
    monkeypatch.setattr(_metal, "_LOCALMEM_BUDGET", 4096)  # small budget -> cheap small cap
    scheme = Scheme()
    with pytest.raises(GpuSWError) as ei:
        _metal.get_pipeline(scheme, 100000, "int32")
    cap = int(re.search(r"~(\d+) residues", str(ei.value)).group(1))
    assert cap % 128 == 0
    _metal.get_pipeline(scheme, cap, "int32")              # advertised cap must compile
    with pytest.raises(GpuSWError):                        # one bucket beyond must raise
        _metal.get_pipeline(scheme, cap + 128, "int32")


def test_realistic_16s_lengths_bitexact():
    """16S-scale lengths (≈350–561 bp ASV queries vs ≈1500 bp refs) — the regime the
    EmilyKin ASV mapping runs in — must stay bit-exact and exercise the int16 path."""
    rng = np.random.default_rng(20260611)
    queries = rand_seqs(rng, 5, 350, 562, "ACGT")   # ASV-sized
    refs = rand_seqs(rng, 8, 1400, 1551, "ACGT")    # 16S-reference-sized
    scheme = Scheme()  # the EmilyKin CPU pipeline scheme: local 2/-3/-5/-2
    assert scheme.resolve_dtype(561, 1550) == "int16"
    gpu = align_score(queries, refs, scheme=scheme, backend=BACKEND, return_ids=False)
    cpu = cpu_reference_matrix(queries, refs, scheme)
    assert np.array_equal(gpu.astype(np.int64), cpu)
