"""Scheme validation, the substitution table, and the int16/int32 overflow envelope."""
import numpy as np
import pytest

from gpusw import Matrix, Scheme
from gpusw.errors import OverflowRiskError, SchemeError


def test_defaults_match_original_dna():
    s = Scheme()
    assert (s.mode, s.match, s.mismatch, s.gap_open, s.gap_extend) == ("local", 2, -3, -5, -2)
    assert s.eff_alphabet == "ACGT" and s.ns == 5


def test_invalid_fields_raise():
    with pytest.raises(SchemeError):
        Scheme(mode="banana")
    with pytest.raises(SchemeError):
        Scheme(gap_open=5)  # must be <= 0
    with pytest.raises(SchemeError):
        Scheme(unknown="sometimes")


def test_substitution_table_diag_and_sentinel():
    t = Scheme(match=2, mismatch=-3).substitution_table()
    assert t.shape == (5, 5)
    assert np.array_equal(np.diag(t)[:4], [2, 2, 2, 2])
    assert (t[4, :] == -3).all()  # mismatch sentinel row


def test_matrix_scheme_table():
    s = Scheme(mode="local", matrix=Matrix.named("BLOSUM62"), gap_open=-11, gap_extend=-1)
    t = s.substitution_table()
    assert t.shape == (len(s.eff_alphabet) + 1,) * 2


def test_dtype_auto_small_is_int16():
    assert Scheme(match=2, mismatch=-3).resolve_dtype(500, 1500) == "int16"


def test_dtype_auto_large_promotes_to_int32():
    # 20000 * 2 = 40000 > 30000 -> must promote
    assert Scheme(match=2, mismatch=-3).resolve_dtype(20000, 20000) == "int32"


def test_dtype_forced_int16_overflow_raises():
    with pytest.raises(OverflowRiskError):
        Scheme(match=2, mismatch=-3, dtype="int16").resolve_dtype(20000, 20000)


def test_dtype_forced_int32_always():
    assert Scheme(dtype="int32").resolve_dtype(10, 10) == "int32"


def test_global_negative_lower_bound_promotes():
    # global accumulates large negatives along the boundary ladder
    s = Scheme(mode="global", gap_open=-5, gap_extend=-2)
    assert s.resolve_dtype(20000, 20000) == "int32"


def test_table_hash_distinguishes_unknown_policy():
    a = Scheme(match=2, mismatch=-3, unknown="mismatch").table_hash()
    b = Scheme(match=2, mismatch=-3, unknown="zero").table_hash()
    assert a != b
