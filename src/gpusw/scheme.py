"""The :class:`Scheme` — the single object that funnels *every* alignment-scoring
parameter (mode, scoring, alphabet, gap costs, overflow policy) into a frozen,
hashable form the kernel generator and the CPU oracle both read.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from .errors import OverflowRiskError, SchemeError
from .matrix import Matrix

__all__ = ["Scheme"]

MODES = ("local", "global", "semiglobal")
UNKNOWN_POLICIES = ("mismatch", "zero", "error")
DTYPES = ("auto", "int16", "int32")

# int16 is used only when the worst-case score range fits comfortably inside this
# margin (a fixed sentinel of -30000 must also stay representable).
_INT16_SAFE = 30000


@dataclass(frozen=True)
class Scheme:
    """A complete, hashable specification of an alignment-scoring problem.

    Parameters
    ----------
    mode:
        ``"local"`` (Smith-Waterman), ``"global"`` (Needleman-Wunsch), or
        ``"semiglobal"`` (free end gaps per the ``free_end_gaps_*`` flags).
    match, mismatch:
        Used only when ``matrix is None`` (the fast match/mismatch path).
    gap_open:
        Cost of the **first** residue of a gap (Biopython ``open_gap_score``).
    gap_extend:
        Cost of each **subsequent** gap residue (Biopython ``extend_gap_score``).
    matrix:
        Optional :class:`~gpusw.Matrix`; overrides ``match``/``mismatch`` and fixes
        the alphabet.
    alphabet:
        Residue order for the match/mismatch path (defaults to ``"ACGT"``). When a
        ``matrix`` is given its own alphabet is authoritative.
    free_end_gaps_query, free_end_gaps_ref:
        Semiglobal only. ``free_end_gaps_query`` makes terminal gaps in the *query*
        free; ``free_end_gaps_ref`` does the same for the *reference*.
    unknown:
        Out-of-alphabet symbol policy: ``"mismatch"`` | ``"zero"`` | ``"error"``.
    case_insensitive:
        Fold input case before encoding (default ``True``).
    dtype:
        DP/output element type: ``"auto"`` (promote int16→int32 when a score could
        overflow) | ``"int16"`` | ``"int32"``.
    """

    mode: str = "local"
    match: int = 2
    mismatch: int = -3
    gap_open: int = -5
    gap_extend: int = -2
    matrix: Matrix | None = None
    alphabet: str | None = None
    free_end_gaps_query: bool = True
    free_end_gaps_ref: bool = True
    unknown: str = "mismatch"
    case_insensitive: bool = True
    dtype: str = "auto"

    # derived (set in __post_init__)
    eff_alphabet: str = field(default="ACGT", repr=False)

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise SchemeError(f"mode must be one of {MODES}, got {self.mode!r}")
        if self.unknown not in UNKNOWN_POLICIES:
            raise SchemeError(f"unknown must be one of {UNKNOWN_POLICIES}")
        if self.dtype not in DTYPES:
            raise SchemeError(f"dtype must be one of {DTYPES}")
        for nm in ("match", "mismatch", "gap_open", "gap_extend"):
            v = getattr(self, nm)
            if int(v) != v:
                raise SchemeError(f"{nm} must be an integer, got {v!r}")
            object.__setattr__(self, nm, int(v))
        if self.gap_open > 0 or self.gap_extend > 0:
            raise SchemeError("gap_open and gap_extend must be <= 0")
        alpha = self.matrix.alphabet if self.matrix is not None else (self.alphabet or "ACGT")
        alpha = alpha.upper() if self.case_insensitive else alpha
        if len(set(alpha)) != len(alpha):
            raise SchemeError(f"alphabet {alpha!r} has duplicate symbols")
        object.__setattr__(self, "eff_alphabet", alpha)

    # ------------------------------------------------------------------ presets
    @classmethod
    def preset(cls, name: str) -> Scheme:
        from . import schemes

        try:
            return getattr(schemes, name.upper())
        except AttributeError as exc:
            avail = ", ".join(schemes.__all__)
            raise SchemeError(f"unknown preset {name!r}; available: {avail}") from exc

    def replace(self, **kw) -> Scheme:
        """Return a copy with fields overridden."""
        return replace(self, **kw)

    # --------------------------------------------------------- scoring extremes
    def _score_extremes(self) -> tuple[int, int]:
        """(max positive per-cell substitution score, min substitution score)."""
        if self.matrix is not None:
            return max(self.matrix.max_score, 0), self.matrix.min_score
        return max(self.match, 0), min(self.mismatch, self.match)

    def resolve_dtype(self, max_q_len: int, max_r_len: int) -> str:
        """Pick ``"int16"`` or ``"int32"`` given the longest query/reference.

        Raises :class:`OverflowRiskError` if ``dtype="int16"`` is forced but the
        worst-case score magnitude would not fit.
        """
        if self.dtype == "int32":
            return "int32"
        max_pos, min_sub = self._score_extremes()
        diag = min(max_q_len, max_r_len)
        upper = diag * max(max_pos, 1)
        if self.mode == "local":
            lower = 0
        else:
            longest = max(max_q_len, max_r_len)
            lower = min(self.gap_open, self.gap_extend, min_sub, 0) * longest
        fits = (upper <= _INT16_SAFE) and (lower >= -_INT16_SAFE)
        if self.dtype == "int16":
            if not fits:
                raise OverflowRiskError(
                    f"dtype='int16' unsafe: worst-case score range "
                    f"[{lower}, {upper}] exceeds +/-{_INT16_SAFE}. "
                    "Use dtype='auto' or 'int32'."
                )
            return "int16"
        return "int16" if fits else "int32"

    # ------------------------------------------------------------- codegen view
    @property
    def ns(self) -> int:
        """Side length of the (sentinel-padded) substitution table the kernel uses."""
        return len(self.eff_alphabet) + 1

    def substitution_table(self) -> np.ndarray:
        """The ``(NS, NS)`` int32 substitution table the kernel/oracle index with.

        gpusw is *always-matrix*: the match/mismatch parameters simply build a
        diagonal matrix. The final row/col is the out-of-alphabet sentinel, filled
        per the ``unknown`` policy. ``table[a, b]`` scores ref code ``a`` vs query
        code ``b``.
        """
        if self.matrix is not None:
            base = self.matrix
        else:
            base = Matrix.from_match_mismatch(self.match, self.mismatch, self.eff_alphabet)
        _, arr = base.packed(self.unknown)
        return arr

    def table_hash(self) -> str:
        """Stable digest of the substitution table (keys the cached device upload)."""
        import hashlib

        h = hashlib.blake2b(digest_size=12)
        h.update(self.substitution_table().tobytes())
        return h.hexdigest()

    def module_fields(self) -> dict:
        """Source-affecting fields for the RawModule cache (no maxq/dtype/arch).

        Substitution *content* is data (uploaded separately), so it is NOT here —
        only the table side length ``ns`` matters to the generated source.
        """
        return {
            "mode": self.mode,
            "gap_open": self.gap_open,
            "gap_extend": self.gap_extend,
            "free_q": bool(self.free_end_gaps_query),
            "free_r": bool(self.free_end_gaps_ref),
            "ns": self.ns,
        }
