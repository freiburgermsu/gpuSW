"""MSL kernel source generation: deterministic, scheme-sensitive, MAXQ bucketing.

The Metal twin of ``test_kernel_render.py``. These exercise only the string generator
(:mod:`gpusw.metal_kernel`), so they need no GPU and run anywhere (incl. CI on Linux).
"""
from gpusw import Scheme
from gpusw._compile import bucket_maxq
from gpusw.kernel import neg_inf_for
from gpusw.metal_kernel import render_source


def _src(scheme, maxq=256, dtype="int16"):
    return render_source(scheme.module_fields(), maxq, dtype)


def test_is_msl_not_cuda():
    src = _src(Scheme())
    assert "#include <metal_stdlib>" in src
    assert "using namespace metal;" in src
    assert "kernel void sw_cross" in src and "kernel void sw_pairs" in src
    assert "[[thread_position_in_grid]]" in src
    assert "device const uchar*" in src
    # the CUDA-isms must NOT leak into the Metal source
    assert "__global__" not in src and "__restrict__" not in src and "blockIdx" not in src


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
    # NEGINF is shared with the CUDA generator (same sentinels)
    assert f"#define NEGINF ({neg_inf_for('int16')})" in _src(Scheme(), dtype="int16")
    assert f"#define NEGINF ({neg_inf_for('int32')})" in _src(Scheme(), dtype="int32")


def test_free_flags_baked():
    s = Scheme(mode="semiglobal", free_end_gaps_query=True, free_end_gaps_ref=False)
    src = _src(s)
    assert "#define FREE_Q (1)" in src
    assert "#define FREE_R (0)" in src


def test_buffer_bindings_present():
    src = _src(Scheme())
    # the launch side (backends/metal.py) depends on these exact binding indices
    assert "[[buffer(0)]]" in src and "[[buffer(8)]]" in src
    assert "device const int*   SUB    [[buffer(8)]]" in src


def test_maxq_baked_via_bucket():
    assert "#define MAXQ (640)" in _src(Scheme(), maxq=bucket_maxq(561))
    assert "#define MAXQ (128)" in _src(Scheme(), maxq=bucket_maxq(1))
