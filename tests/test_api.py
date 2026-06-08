"""High-level API behavior on the GPU: shapes, ids, top_k, pairs, introspection."""
import numpy as np
import pytest

from gpusw import Aligner, align_pairs, align_score

pytestmark = pytest.mark.gpu

QUERIES = {"q_a": "ACGTACGTACGT", "q_b": "TTTTGGGG", "q_c": "ACGGACGT"}
REFS = {"r1": "ACGTACGTACGT", "r2": "ACGTAAAA", "r3": "GGGGTTTT", "r4": "TACGGACGTA"}


def test_align_score_shapes_and_ids():
    res = align_score(QUERIES, REFS, scheme="dna")
    assert res.scores.shape == (3, 4)
    assert res.scores.dtype == np.int32
    assert res.query_ids == ["q_a", "q_b", "q_c"]
    assert res.reference_ids == ["r1", "r2", "r3", "r4"]


def test_return_ids_false_is_ndarray():
    arr = align_score(QUERIES, REFS, scheme="dna", return_ids=False)
    assert isinstance(arr, np.ndarray) and arr.shape == (3, 4)


def test_top_k_matches_full_sort():
    al = Aligner("dna").index(REFS).set_queries(QUERIES)
    full = al.score_cross(return_ids=True)
    tk = al.top_k(k=2)
    assert tk.top_k(2) == full.top_k(2)


def test_topk_memory_bounded_result():
    al = Aligner("dna").index(REFS)
    tk = al.top_k(queries=QUERIES, k=2)
    assert tk.scores is None  # only top-k kept
    assert tk.topk_idx.shape == (3, 2)


def test_align_pairs_zip():
    res = align_pairs(["ACGT", "TTTT"], ["ACGT", "TTTA"], scheme="dna")
    assert res.scores.tolist() == [8, 6]


def test_align_pairs_explicit():
    res = align_pairs(list(QUERIES.values()), list(REFS.values()),
                      pairs=[(0, 0), (0, 3), (2, 3)], scheme="dna", return_ids=False)
    assert res.shape == (3,)


def test_align_pairs_length_mismatch_raises():
    with pytest.raises(ValueError):
        align_pairs(["ACGT", "TT"], ["ACGT"], scheme="dna")


def test_gcups_and_source():
    al = Aligner("dna").index(REFS).set_queries(QUERIES)
    al.score_cross()
    assert al.gcups > 0
    src = al.cuda_source()
    assert "sw_cross" in src and "#define MODE" in src


def test_one_call_modes():
    g = align_score(["AC"], ["TTACTT"], scheme="dna", mode="global", return_ids=False)
    s = align_score(["AC"], ["TTACTT"], scheme="dna", mode="semiglobal", return_ids=False)
    assert s[0, 0] == 4 and g[0, 0] < s[0, 0]
