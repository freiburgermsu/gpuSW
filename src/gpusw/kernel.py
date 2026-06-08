"""Generate the CUDA C++ source for a given scheme, compiled at runtime by
CuPy/NVRTC (no ``nvcc``, no CUDA toolkit).

One CUDA thread scores one ``(query, reference)`` pair with affine-gap (Gotoh)
dynamic programming, score-only. The query is the inner DP buffer (kept short so
two ``MAXQ``-sized per-thread arrays stay small); the reference is the outer loop.

The generator is *always-matrix*: scoring is ``s = SUB[a * NS + b]`` where ``SUB``
is a small ``NS×NS`` table uploaded once. Match/mismatch schemes simply build a
diagonal ``SUB``. ``NS``, the gap costs, the alignment mode, the free-end-gap flags,
``MAXQ`` and the DP element type are baked into the source as ``#define``/``typedef``,
so each distinct scheme compiles to its own specialised, branch-free kernel.
"""
from __future__ import annotations

__all__ = ["render_source", "neg_inf_for"]

_MODE_CODE = {"local": 0, "global": 1, "semiglobal": 2}


def neg_inf_for(dtype: str) -> int:
    """Sentinel "−∞" that fits the DP store type and stays below any real score."""
    return -32000 if dtype == "int16" else -(1 << 28)


def render_source(module_fields: dict, maxq: int, dtype: str) -> str:
    """Return the CUDA C++ source for one compiled module.

    Parameters
    ----------
    module_fields:
        Output of :meth:`gpusw.Scheme.module_fields` — ``mode``, ``gap_open``,
        ``gap_extend``, ``free_q``, ``free_r``, ``ns``.
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
extern "C" {{
#define MAXQ ({int(maxq)})
#define NS ({int(module_fields["ns"])})
#define GOPEN ({int(module_fields["gap_open"])})
#define GEXT ({int(module_fields["gap_extend"])})
#define NEGINF ({int(neg)})
#define MODE ({mode_code})           /* 0=local 1=global 2=semiglobal */
#define FREE_Q ({1 if module_fields["free_q"] else 0})
#define FREE_R ({1 if module_fields["free_r"] else 0})
typedef {store_t} hstore_t;

__device__ __forceinline__ int sw_score(
    const unsigned char* q, int lq,
    const unsigned char* r, int lr,
    const int* __restrict__ SUB)
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
    #pragma unroll 1
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
        unsigned char a = r[i - 1];
        const int* subrow = SUB + (int)a * NS;
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
        #pragma unroll 1
        for (int j = 1; j <= lq; ++j) {{
            unsigned char b = q[j - 1];
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
__global__ void sw_cross(
    const unsigned char* qbuf, const int* qoff,
    const unsigned char* rbuf, const int* roff,
    const int* qlist, int nq, int nr,
    hstore_t* out, const int* __restrict__ SUB)
{{
    long t = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long total = (long)nq * nr;
    if (t >= total) return;
    int qi = qlist[(int)(t / nr)];
    int rj = (int)(t % nr);
    int qs = qoff[qi], rs = roff[rj];
    out[t] = (hstore_t)sw_score(qbuf + qs, qoff[qi + 1] - qs,
                                rbuf + rs, roff[rj + 1] - rs, SUB);
}}

/* arbitrary pairs: score (pair_qi[k], pair_rj[k]) -> out[k] */
__global__ void sw_pairs(
    const unsigned char* qbuf, const int* qoff,
    const unsigned char* rbuf, const int* roff,
    const int* pair_qi, const int* pair_rj, int npair,
    hstore_t* out, const int* __restrict__ SUB)
{{
    long k = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= npair) return;
    int qi = pair_qi[k], rj = pair_rj[k];
    int qs = qoff[qi], rs = roff[rj];
    out[k] = (hstore_t)sw_score(qbuf + qs, qoff[qi + 1] - qs,
                                rbuf + rs, roff[rj + 1] - rs, SUB);
}}
}}  /* extern "C" */
"""
