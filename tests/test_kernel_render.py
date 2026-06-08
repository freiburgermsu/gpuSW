"""Kernel source generation: deterministic, scheme-sensitive, and MAXQ bucketing."""
from gpusw import Scheme
from gpusw._compile import bucket_maxq
from gpusw.kernel import neg_inf_for, render_source


def _src(scheme, maxq=256, dtype="int16"):
    return render_source(scheme.module_fields(), maxq, dtype)


def test_deterministic():
    s = Scheme(mode="local")
    assert _src(s) == _src(s)


def test_mode_changes_source():
    a = _src(Scheme(mode="local"))
    b = _src(Scheme(mode="global"))
    c = _src(Scheme(mode="semiglobal"))
    assert a != b != c and a != c
    assert "#define MODE (0)" in a
    assert "#define MODE (1)" in b
    assert "#define MODE (2)" in c


def test_gap_and_ns_baked():
    s = Scheme(match=2, mismatch=-3, gap_open=-5, gap_extend=-2)
    src = _src(s)
    assert "#define GOPEN (-5)" in src
    assert "#define GEXT (-2)" in src
    assert "#define NS (5)" in src


def test_dtype_typedef_and_neginf():
    assert "typedef short hstore_t;" in _src(Scheme(), dtype="int16")
    assert "typedef int hstore_t;" in _src(Scheme(), dtype="int32")
    assert neg_inf_for("int16") == -32000
    assert neg_inf_for("int32") == -(1 << 28)


def test_free_flags_baked():
    s = Scheme(mode="semiglobal", free_end_gaps_query=True, free_end_gaps_ref=False)
    src = _src(s)
    assert "#define FREE_Q (1)" in src
    assert "#define FREE_R (0)" in src


def test_both_kernels_present():
    src = _src(Scheme())
    assert "sw_cross" in src and "sw_pairs" in src
    assert "const int* __restrict__ SUB" in src


def test_bucket_maxq():
    assert bucket_maxq(1) == 128
    assert bucket_maxq(128) == 128
    assert bucket_maxq(129) == 256
    assert bucket_maxq(561) == 640
