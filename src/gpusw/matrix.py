"""Substitution matrices — funnel arbitrary scoring tables into a discrete,
kernel-ready ``(K, K)`` integer array over a fixed alphabet.

A :class:`Matrix` pairs a canonical ``alphabet`` (the residue order that defines
integer codes ``0..K-1``) with an integer score table. The kernel indexes it as
``s = SUB[a * NS + b]`` where ``a``/``b`` are encoded residue codes and ``NS`` is
the (sentinel-padded) side length.

Named matrices (BLOSUM/PAM/EDNAFULL) ship as package data in NCBI text format and
are loaded lazily via :func:`importlib.resources`.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from .errors import SchemeError

__all__ = ["Matrix"]


def _parse_ncbi(text: str) -> tuple[str, np.ndarray]:
    """Parse an NCBI/BLAST-format substitution matrix → (alphabet, square int array)."""
    rows: list[list[int]] = []
    cols: list[str] | None = None
    row_syms: list[str] = []
    for raw in text.splitlines():
        line = raw.rstrip()
        if not line or line.lstrip().startswith(("#", "//")):
            continue
        toks = line.split()
        if cols is None:
            # header row: the column symbols
            cols = toks
            continue
        sym, vals = toks[0], toks[1:]
        if len(vals) != len(cols):
            raise SchemeError(
                f"matrix row {sym!r} has {len(vals)} values, expected {len(cols)}"
            )
        row_syms.append(sym)
        rows.append([int(round(float(v))) for v in vals])
    if cols is None or not rows:
        raise SchemeError("could not parse a substitution matrix from the given text")
    if row_syms != cols:
        raise SchemeError(
            f"matrix row order {row_syms} != column order {cols}; non-square matrices "
            "are not supported"
        )
    return "".join(cols), np.asarray(rows, dtype=np.int32)


@dataclass(frozen=True)
class Matrix:
    """A discrete substitution matrix over a fixed alphabet.

    Attributes
    ----------
    alphabet:
        Canonical residue order. Symbol ``alphabet[k]`` has integer code ``k``.
    array:
        ``(K, K)`` int32 score table, ``array[a, b]`` = score of aligning code ``a``
        with code ``b``. Symmetric for the standard biological matrices.
    name:
        Optional label (set for named/package matrices).
    """

    alphabet: str
    array: np.ndarray
    name: str | None = None

    def __post_init__(self) -> None:
        arr = np.ascontiguousarray(self.array, dtype=np.int32)
        object.__setattr__(self, "array", arr)
        k = len(self.alphabet)
        if arr.shape != (k, k):
            raise SchemeError(
                f"matrix array shape {arr.shape} does not match alphabet length {k}"
            )
        if len(set(self.alphabet)) != k:
            raise SchemeError(f"alphabet {self.alphabet!r} has duplicate symbols")

    # ------------------------------------------------------------------ builders
    @classmethod
    def from_match_mismatch(
        cls, match: int, mismatch: int, alphabet: str = "ACGT"
    ) -> Matrix:
        """A diagonal match / off-diagonal mismatch matrix over ``alphabet``."""
        k = len(alphabet)
        arr = np.full((k, k), int(mismatch), dtype=np.int32)
        np.fill_diagonal(arr, int(match))
        return cls(alphabet, arr, name=f"match{match}_mismatch{mismatch}")

    @classmethod
    def from_array(cls, alphabet: str, arr) -> Matrix:
        """Wrap an explicit ``(K, K)`` integer array over ``alphabet``."""
        return cls(alphabet, np.asarray(arr, dtype=np.int32))

    @classmethod
    def from_ncbi(cls, text_or_path, name: str | None = None) -> Matrix:
        """Parse an NCBI/BLAST-format matrix from a string or a file path."""
        import os

        text = text_or_path
        if isinstance(text_or_path, (str, bytes, os.PathLike)) and os.path.exists(
            text_or_path
        ):
            with open(text_or_path) as fh:
                text = fh.read()
        alphabet, arr = _parse_ncbi(text)
        return cls(alphabet, arr, name=name)

    @classmethod
    def named(cls, name: str) -> Matrix:
        """Load a bundled named matrix, e.g. ``"BLOSUM62"``, ``"EDNAFULL"``."""
        from importlib.resources import files

        key = name.upper().replace("-", "").replace("_", "")
        try:
            res = files("gpusw.data.matrices").joinpath(f"{key}.txt")
            text = res.read_text()
        except (FileNotFoundError, ModuleNotFoundError) as exc:
            avail = ", ".join(sorted(available()))
            raise SchemeError(
                f"unknown matrix {name!r}; bundled matrices: {avail}"
            ) from exc
        return cls.from_ncbi(text, name=key)

    # -------------------------------------------------------------- kernel/dtype
    def packed(self, unknown: str) -> tuple[str, np.ndarray]:
        """Return ``(alphabet_with_sentinel, (K+1, K+1) int32 array)``.

        The extra final row/col is the *sentinel* used for out-of-alphabet symbols.
        ``unknown`` policy: ``"mismatch"`` → sentinel scores like the worst diagonal
        mismatch (most-negative off-diagonal value, so unknowns never help an
        alignment); ``"zero"`` → neutral 0; the ``"error"`` policy is enforced at
        encode time (here it behaves like ``"mismatch"`` as a safe fallback).
        """
        k = len(self.alphabet)
        out = np.zeros((k + 1, k + 1), dtype=np.int32)
        out[:k, :k] = self.array
        if unknown == "zero":
            fill = 0
        else:  # "mismatch" (and the "error" safety fallback)
            off = self.array[~np.eye(k, dtype=bool)]
            fill = int(off.min()) if off.size else int(self.array.min())
        out[k, :] = fill
        out[:, k] = fill
        return self.alphabet, out

    @property
    def max_score(self) -> int:
        return int(self.array.max())

    @property
    def min_score(self) -> int:
        return int(self.array.min())

    def hash(self) -> str:
        """Stable digest of (alphabet, array) for the compile cache key."""
        h = hashlib.blake2b(digest_size=12)
        h.update(self.alphabet.encode())
        h.update(np.ascontiguousarray(self.array, dtype=np.int32).tobytes())
        return h.hexdigest()

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        nm = self.name or "custom"
        return f"Matrix({nm!r}, alphabet={self.alphabet!r}, K={len(self.alphabet)})"


def available() -> list[str]:
    """List bundled matrix names (without the ``.txt`` suffix)."""
    from importlib.resources import files

    try:
        return [
            p.name[:-4]
            for p in files("gpusw.data.matrices").iterdir()
            if p.name.endswith(".txt")
        ]
    except (FileNotFoundError, ModuleNotFoundError):  # pragma: no cover
        return []
