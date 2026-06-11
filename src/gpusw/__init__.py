"""gpusw — GPU-accelerated Smith-Waterman / Needleman-Wunsch alignment *scoring*.

A custom CUDA kernel compiled at runtime with CuPy/NVRTC (no ``nvcc``, no CUDA
toolkit) scores one ``(query, reference)`` pair per thread with affine-gap dynamic
programming. Generalized over the scoring scheme (match/mismatch *or* a full
substitution matrix), the alignment mode (local / global / semiglobal), the alphabet,
and the integer dtype — while staying bit-exact with the included pure-NumPy oracle.

Quick start
-----------
>>> import gpusw
>>> res = gpusw.align_score(["ACGTACGT", "TTTT"], ["ACGTACGT", "ACGT"], scheme="dna")
>>> res.scores            # (2, 2) int32 cross-product           # doctest: +SKIP
>>> res.top_k(1)          # best reference per query             # doctest: +SKIP

No GPU? The oracle still works everywhere:
>>> gpusw.cpu_reference_score("ACGTACGT", "ACGT", gpusw.schemes.DNA)
8
"""
from __future__ import annotations

import numpy as np

from . import schemes
from ._compile import gpu_available
from ._metal import metal_available
from ._version import __version__
from .aligner import Aligner
from .backends import available_backends
from .errors import (
    EncodeError,
    GpuSWError,
    NoGpuError,
    OverflowRiskError,
    SchemeError,
)
from .matrix import Matrix
from .matrix import available as available_matrices
from .reference import cpu_reference_matrix, cpu_reference_score
from .result import AlignResult
from .scheme import Scheme

__all__ = [
    "align_score",
    "align_pairs",
    "Aligner",
    "Scheme",
    "Matrix",
    "AlignResult",
    "schemes",
    "cpu_reference_score",
    "cpu_reference_matrix",
    "available_matrices",
    "gpu_available",
    "metal_available",
    "available_backends",
    "GpuSWError",
    "NoGpuError",
    "SchemeError",
    "OverflowRiskError",
    "EncodeError",
    "__version__",
]


def _build_scheme(
    scheme,
    *,
    mode=None,
    match=None,
    mismatch=None,
    gap_open=None,
    gap_extend=None,
    matrix=None,
    alphabet=None,
    unknown=None,
    case_insensitive=None,
    dtype=None,
    free_end_gaps_query=None,
    free_end_gaps_ref=None,
) -> Scheme:
    """Resolve a base scheme (instance or preset name) and apply non-None overrides."""
    base = scheme if isinstance(scheme, Scheme) else Scheme.preset(scheme)
    if isinstance(matrix, str):
        matrix = Matrix.named(matrix)
    overrides = {
        k: v
        for k, v in dict(
            mode=mode,
            match=match,
            mismatch=mismatch,
            gap_open=gap_open,
            gap_extend=gap_extend,
            matrix=matrix,
            alphabet=alphabet,
            unknown=unknown,
            case_insensitive=case_insensitive,
            dtype=dtype,
            free_end_gaps_query=free_end_gaps_query,
            free_end_gaps_ref=free_end_gaps_ref,
        ).items()
        if v is not None
    }
    return base.replace(**overrides) if overrides else base


def align_score(
    queries,
    refs,
    *,
    scheme="dna",
    mode=None,
    match=None,
    mismatch=None,
    gap_open=None,
    gap_extend=None,
    matrix=None,
    alphabet=None,
    unknown=None,
    dtype=None,
    threads=128,
    backend="auto",
    return_ids=True,
    query_batch=None,
):
    """All-pairs alignment scores of every query against every reference.

    ``queries`` and ``refs`` are funnelled by :func:`gpusw.encode.funnel` — lists of
    strings, dicts, ``(id, seq)`` iterables, FASTA paths/text, or pre-encoded buffers.
    ``backend`` selects the GPU: ``"auto"`` (prefer CUDA, else Metal), ``"cuda"``, or
    ``"metal"``.

    Returns an :class:`AlignResult` (``.scores`` is ``(n_queries, n_refs)`` int32) by
    default, or a bare ``np.ndarray`` if ``return_ids=False``.
    """
    sch = _build_scheme(
        scheme, mode=mode, match=match, mismatch=mismatch, gap_open=gap_open,
        gap_extend=gap_extend, matrix=matrix, alphabet=alphabet, unknown=unknown,
        dtype=dtype,
    )
    al = Aligner(sch, threads=threads, backend=backend).index(refs).set_queries(queries)
    return al.score_cross(return_ids=return_ids, query_batch=query_batch)


def align_pairs(
    queries,
    refs,
    *,
    pairs=None,
    scheme="dna",
    mode=None,
    match=None,
    mismatch=None,
    gap_open=None,
    gap_extend=None,
    matrix=None,
    alphabet=None,
    unknown=None,
    dtype=None,
    threads=128,
    backend="auto",
    return_ids=True,
):
    """Score specific query/reference pairs.

    With ``pairs=None`` the queries and references are zipped 1:1 (and must be the same
    length). Otherwise ``pairs`` is a list of ``(query_index, reference_index)`` tuples.
    ``backend`` selects the GPU (``"auto"`` | ``"cuda"`` | ``"metal"``). Returns an
    :class:`AlignResult` (``.scores`` is ``(n_pairs,)``) or a bare array.
    """
    sch = _build_scheme(
        scheme, mode=mode, match=match, mismatch=mismatch, gap_open=gap_open,
        gap_extend=gap_extend, matrix=matrix, alphabet=alphabet, unknown=unknown,
        dtype=dtype,
    )
    al = Aligner(sch, threads=threads, backend=backend).index(refs).set_queries(queries)
    nq, nr = al._queries.n, al._refs.n
    if pairs is None:
        if nq != nr:
            raise ValueError(
                f"pairs=None requires len(queries)==len(refs) for 1:1 scoring "
                f"(got {nq} vs {nr}); pass pairs=[(qi, rj), ...]"
            )
        qi = np.arange(nq, dtype=np.int32)
        rj = np.arange(nr, dtype=np.int32)
    else:
        qi = np.fromiter((p[0] for p in pairs), dtype=np.int32, count=len(pairs))
        rj = np.fromiter((p[1] for p in pairs), dtype=np.int32, count=len(pairs))
    return al.score_pairs(qi, rj, return_ids=return_ids)
