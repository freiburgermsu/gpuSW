"""Generate the Metal Shading Language (MSL) source for a given scheme, compiled
at runtime by the Metal framework (no Xcode, no offline ``metal``/``metallib``).

This is the Apple-Silicon sibling of :mod:`gpusw.kernel` (the CUDA generator): it
emits the **same** affine-gap (Gotoh) score-only dynamic program, one GPU thread per
``(query, reference)`` pair, so a Metal launch is **bit-exact** with both the CUDA
kernel and the pure-NumPy oracle. The query is the inner DP buffer (two ``MAXQ``-sized
per-thread arrays); the reference is the outer loop.

As with CUDA, the generator is *always-matrix*: scoring is ``s = SUB[a * NS + b]``
where ``SUB`` is a small ``NS×NS`` table uploaded once. Match/mismatch schemes simply
build a diagonal ``SUB``. ``NS``, the gap costs, the alignment mode, the free-end-gap
flags, ``MAXQ`` and the DP element type are baked into the source as
``#define``/``typedef``, so each distinct scheme compiles to its own specialised,
branch-free kernel.

The only differences from the CUDA source are mechanical and inherent to MSL:
``unsigned char`` → ``uchar``; pointer parameters carry the ``device``/``constant``
address space; the entry points are ``kernel void`` taking ``[[buffer(n)]]`` bindings
and the global thread index via ``[[thread_position_in_grid]]``. The DP recurrence,
the mode/free-end-gap branches, the empty-sequence edges and the int16↔int32 store
behaviour are character-for-character the same as :func:`gpusw.kernel.render_source`.
"""
from __future__ import annotations

from .kernel import neg_inf_for

__all__ = ["render_source"]

_MODE_CODE = {"local": 0, "global": 1, "semiglobal": 2}

# Buffer-binding indices — the Python launch side (gpusw._metal) must match these.
#   sw_cross:  0 qbuf 1 qoff 2 rbuf 3 roff 4 qlist 5 nq 6 nr 7 out 8 SUB
#   sw_pairs:  0 qbuf 1 qoff 2 rbuf 3 roff 4 pair_qi 5 pair_rj 6 npair 7 out 8 SUB


def render_source(module_fields: dict, maxq: int, dtype: str) -> str:
    """Return the Metal Shading Language source for one compiled library.

    Parameters
    ----------
    module_fields:
        Output of :meth:`gpusw.Scheme.module_fields` — ``mode``, ``gap_open``,
        ``gap_extend``, ``free_q``, ``free_r``, ``ns``. Identical to the CUDA path.
    maxq:
        Maximum query length (sizes the per-thread DP buffers). Validated by callers.
    dtype:
        ``"int16"`` or ``"int32"`` — the DP/output element type.
    """
    mode = module_fields["mode"]
    mode_code = _MODE_CODE[mode]
    store_t = "short" if dtype == "int16" else "int"
    neg = neg_inf_for(dtype)

    return f"""
#include <metal_stdlib>
using namespace metal;

#define MAXQ ({int(maxq)})
#define NS ({int(module_fields["ns"])})
#define GOPEN ({int(module_fields["gap_open"])})
#define GEXT ({int(module_fields["gap_extend"])})
#define NEGINF ({int(neg)})
#define MODE ({mode_code})           /* 0=local 1=global 2=semiglobal */
#define FREE_Q ({1 if module_fields["free_q"] else 0})
#define FREE_R ({1 if module_fields["free_r"] else 0})
typedef {store_t} hstore_t;

inline int sw_score(
    device const uchar* q, int lq,
    device const uchar* r, int lr,
    device const int* SUB)
{{
    if (lq > MAXQ) lq = MAXQ;

#if MODE == 0
    if (lq == 0 || lr == 0) return 0;
#else
    if (lq == 0 || lr == 0) {{
        if (lq == 0 && lr == 0) return 0;
        if (lq == 0) {{
#if MODE == 2
            if (FREE_Q) return 0;
#endif
            return GOPEN + (lr - 1) * GEXT;
        }} else {{
#if MODE == 2
            if (FREE_R) return 0;
#endif
            return GOPEN + (lq - 1) * GEXT;
        }}
    }}
#endif

    hstore_t Hp[MAXQ + 1];   /* previous DP row H(i-1, .) */
    hstore_t F[MAXQ + 1];    /* previous DP row F(i-1, .) (gap in query) */

    Hp[0] = 0;
    for (int j = 1; j <= lq; ++j) {{
#if MODE == 0
        Hp[j] = 0;
#elif MODE == 1
        Hp[j] = (hstore_t)(GOPEN + (j - 1) * GEXT);
#else
        Hp[j] = (hstore_t)(FREE_R ? 0 : GOPEN + (j - 1) * GEXT);
#endif
        F[j] = (hstore_t)NEGINF;
    }}
    F[0] = (hstore_t)NEGINF;

    int best = 0;                                  /* local running max */
    int corner = NEGINF, endcol = NEGINF, endrow = NEGINF;

    for (int i = 1; i <= lr; ++i) {{
        uchar a = r[i - 1];
        device const int* subrow = SUB + (int)a * NS;
#if MODE == 0
        int Hleft = 0;
#elif MODE == 1
        int Hleft = GOPEN + (i - 1) * GEXT;
#else
        int Hleft = FREE_Q ? 0 : GOPEN + (i - 1) * GEXT;
#endif
        int diag = Hp[0];
        Hp[0] = (hstore_t)Hleft;
        int Eleft = NEGINF;
        for (int j = 1; j <= lq; ++j) {{
            uchar b = q[j - 1];
            int Fj = Hp[j] + GOPEN; int fe = F[j] + GEXT;   if (fe > Fj) Fj = fe;
            int Ej = Hleft + GOPEN; int ee = Eleft + GEXT;  if (ee > Ej) Ej = ee;
            int Hij = diag + subrow[(int)b];
            if (Ej > Hij) Hij = Ej;
            if (Fj > Hij) Hij = Fj;
#if MODE == 0
            if (Hij < 0) Hij = 0;
            if (Hij > best) best = Hij;
#else
            if (i == lr && j == lq) corner = Hij;
#if FREE_Q
            if (j == lq && Hij > endcol) endcol = Hij;
#endif
#if FREE_R
            if (i == lr && Hij > endrow) endrow = Hij;
#endif
#endif
            diag = Hp[j];
            Hp[j] = (hstore_t)Hij; F[j] = (hstore_t)Fj;
            Hleft = Hij; Eleft = Ej;
        }}
    }}

#if MODE == 0
    return best;
#elif MODE == 1
    return corner;
#else
    int res = corner;
#if FREE_Q
    if (endcol > res) res = endcol;
#endif
#if FREE_R
    if (endrow > res) res = endrow;
#endif
#if FREE_Q && FREE_R
    if (res < 0) res = 0;   /* both ends free: the empty alignment scores 0 */
#endif
    return res;
#endif
}}

/* cross product: score qlist[0..nq) x refs[0..nr)  -> out[nq*nr] (row-major nq x nr) */
kernel void sw_cross(
    device const uchar* qbuf   [[buffer(0)]],
    device const int*   qoff   [[buffer(1)]],
    device const uchar* rbuf   [[buffer(2)]],
    device const int*   roff   [[buffer(3)]],
    device const int*   qlist  [[buffer(4)]],
    constant int&       nq     [[buffer(5)]],
    constant int&       nr     [[buffer(6)]],
    device hstore_t*    out    [[buffer(7)]],
    device const int*   SUB    [[buffer(8)]],
    uint gid [[thread_position_in_grid]])
{{
    long t = (long)gid;
    long total = (long)nq * nr;
    if (t >= total) return;
    int qi = qlist[(int)(t / nr)];
    int rj = (int)(t % nr);
    int qs = qoff[qi], rs = roff[rj];
    out[t] = (hstore_t)sw_score(qbuf + qs, qoff[qi + 1] - qs,
                                rbuf + rs, roff[rj + 1] - rs, SUB);
}}

/* arbitrary pairs: score (pair_qi[k], pair_rj[k]) -> out[k] */
kernel void sw_pairs(
    device const uchar* qbuf     [[buffer(0)]],
    device const int*   qoff     [[buffer(1)]],
    device const uchar* rbuf     [[buffer(2)]],
    device const int*   roff     [[buffer(3)]],
    device const int*   pair_qi  [[buffer(4)]],
    device const int*   pair_rj  [[buffer(5)]],
    constant int&       npair    [[buffer(6)]],
    device hstore_t*    out      [[buffer(7)]],
    device const int*   SUB      [[buffer(8)]],
    uint gid [[thread_position_in_grid]])
{{
    long k = (long)gid;
    if (k >= npair) return;
    int qi = pair_qi[k], rj = pair_rj[k];
    int qs = qoff[qi], rs = roff[rj];
    out[k] = (hstore_t)sw_score(qbuf + qs, qoff[qi + 1] - qs,
                                rbuf + rs, roff[rj + 1] - rs, SUB);
}}
"""
