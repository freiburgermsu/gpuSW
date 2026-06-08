"""The input funnel: turn the many ways users hand over sequences into the single
discrete form the GPU kernel consumes — a concatenated ``uint8`` code buffer plus an
``int32`` offset array.

Accepted input forms (all routed through :func:`funnel`):

* ``list``/``tuple``/``np.ndarray`` of ``str`` → ids ``"q0", "q1", ...``
* ``dict[str, str]`` → ids are the keys (insertion order)
* an iterable of ``(id, seq)`` pairs, or objects exposing ``.id`` and ``.seq``
  (Biopython ``SeqRecord`` works **without importing Biopython**)
* a path to a FASTA file (optionally ``.gz``)
* FASTA *text* (a string beginning with ``>``)
* a single raw sequence ``str``/``bytes``
* a pre-built ``(codes, offsets)`` tuple → fast path, no re-encoding
* an existing :class:`Encoded` (returned as-is when its alphabet matches)
"""
from __future__ import annotations

import gzip
import os
from dataclasses import dataclass

import numpy as np

from .errors import EncodeError

__all__ = ["Encoded", "funnel", "build_lut", "read_fasta"]


def build_lut(scheme) -> tuple[np.ndarray, int]:
    """Return ``(lut[256] uint8, sentinel_code)`` for ``scheme``'s alphabet.

    Alphabet symbol ``alphabet[k]`` maps to code ``k``; every other byte maps to the
    sentinel code ``K = len(alphabet)``.
    """
    alpha = scheme.eff_alphabet
    k = len(alpha)
    lut = np.full(256, k, dtype=np.uint8)
    for code, sym in enumerate(alpha):
        lut[ord(sym)] = code
        if scheme.case_insensitive:
            lut[ord(sym.lower())] = code
            lut[ord(sym.upper())] = code
    return lut, k


@dataclass
class Encoded:
    """Sequences funnelled into discrete kernel-ready form.

    Attributes
    ----------
    codes:
        ``uint8`` array, all sequences concatenated; values in ``[0, K]`` (``K`` is
        the sentinel for out-of-alphabet symbols).
    offsets:
        ``int32`` array of length ``n + 1``; sequence ``i`` is
        ``codes[offsets[i]:offsets[i+1]]``.
    ids:
        sequence identifiers (length ``n``).
    lengths:
        ``int32`` per-sequence lengths (``diff(offsets)``).
    alphabet:
        the effective alphabet the codes were built against (guards reuse).
    """

    codes: np.ndarray
    offsets: np.ndarray
    ids: list[str]
    lengths: np.ndarray
    alphabet: str

    @property
    def n(self) -> int:
        return len(self.ids)

    @property
    def max_len(self) -> int:
        return int(self.lengths.max()) if self.n else 0


# file extensions that mark a string as a *path* (so a typo'd path is an error, not a seq)
_SEQ_EXTS = (".fasta", ".fa", ".fna", ".faa", ".ffn", ".fastq", ".fq", ".txt", ".gz")


def _looks_like_path(s: str) -> bool:
    """Heuristic: does this string denote a file path rather than a raw sequence?"""
    return ("/" in s) or (os.sep in s) or s.lower().endswith(_SEQ_EXTS)


def read_fasta(path_or_text) -> list[tuple[str, str]]:
    """Parse FASTA from a file path (``.gz`` ok) or raw text → ``[(id, seq), ...]``.

    Raises :class:`FileNotFoundError` if a path-like string does not exist (so a
    mistyped filename is a clear error rather than a silently-empty result).
    """
    if isinstance(path_or_text, (str, os.PathLike)) and os.path.exists(path_or_text):
        opener = gzip.open if str(path_or_text).endswith(".gz") else open
        with opener(path_or_text, "rt") as fh:
            text = fh.read()
    else:
        s = str(path_or_text)
        if not s.lstrip().startswith(">") and _looks_like_path(s):
            raise FileNotFoundError(f"FASTA file not found: {path_or_text}")
        text = path_or_text
    recs: list[tuple[str, str]] = []
    cur_id, buf = None, []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line[0] == ">":
            if cur_id is not None:
                recs.append((cur_id, "".join(buf)))
            cur_id = line[1:].split()[0] if len(line) > 1 else f"seq{len(recs)}"
            buf = []
        else:
            buf.append(line)
    if cur_id is not None:
        recs.append((cur_id, "".join(buf)))
    return recs


def _to_pairs(obj, id_prefix: str) -> list[tuple[str, str]]:
    """Normalise any supported input into a list of ``(id, seq)`` pairs."""
    # already (id, seq) pairs?
    if isinstance(obj, dict):
        return [(str(k), str(v)) for k, v in obj.items()]
    if isinstance(obj, (bytes, bytearray)):
        # latin-1 is a lossless 1:1 byte->codepoint map (no bytes dropped); any byte
        # outside the alphabet then maps to the sentinel via the 256-entry LUT.
        return [(f"{id_prefix}0", bytes(obj).decode("latin-1"))]
    if isinstance(obj, str):
        # FASTA file path, FASTA text, mistyped path, or a single raw sequence
        if obj.lstrip().startswith(">") or os.path.exists(obj):
            return read_fasta(obj)
        if _looks_like_path(obj):
            raise EncodeError(f"file not found: {obj!r}")
        return [(f"{id_prefix}0", obj)]
    if isinstance(obj, np.ndarray):
        obj = obj.tolist()
    # iterables: list[str], list[(id,seq)], SeqRecord-likes, generators
    try:
        items = list(obj)
    except TypeError as exc:
        raise EncodeError(f"cannot funnel object of type {type(obj).__name__}") from exc
    pairs: list[tuple[str, str]] = []
    for i, it in enumerate(items):
        if isinstance(it, str):
            pairs.append((f"{id_prefix}{i}", it))
        elif isinstance(it, (tuple, list)) and len(it) == 2:
            pairs.append((str(it[0]), str(it[1])))
        elif hasattr(it, "seq"):  # Biopython SeqRecord and friends
            ident = getattr(it, "id", None) or f"{id_prefix}{i}"
            pairs.append((str(ident), str(it.seq)))
        else:
            raise EncodeError(
                f"item {i} of type {type(it).__name__} is not a sequence, "
                "(id, seq) pair, or .seq-bearing record"
            )
    return pairs


def funnel(obj, scheme, *, id_prefix: str = "q") -> Encoded:
    """Funnel any supported input into an :class:`Encoded` for ``scheme``."""
    if isinstance(obj, Encoded):
        if obj.alphabet != scheme.eff_alphabet:
            raise EncodeError(
                f"Encoded alphabet {obj.alphabet!r} != scheme alphabet "
                f"{scheme.eff_alphabet!r}; re-encode for this scheme"
            )
        return obj
    # pre-encoded (codes, offsets[, ids]) fast path
    if (
        isinstance(obj, tuple)
        and len(obj) in (2, 3)
        and isinstance(obj[0], np.ndarray)
        and isinstance(obj[1], np.ndarray)
        and obj[0].dtype == np.uint8
    ):
        codes = np.ascontiguousarray(obj[0], dtype=np.uint8)
        offsets = np.ascontiguousarray(obj[1], dtype=np.int32)
        if offsets.size == 0:
            raise EncodeError("pre-encoded offsets must have length n+1 (>= 1)")
        if (offsets < 0).any():
            raise EncodeError("pre-encoded offsets must be non-negative")
        if (np.diff(offsets) < 0).any():
            raise EncodeError("pre-encoded offsets must be non-decreasing")
        if int(offsets[-1]) > codes.size:
            raise EncodeError(
                f"pre-encoded offsets[-1]={int(offsets[-1])} exceeds len(codes)="
                f"{codes.size}; offsets must index within the codes buffer"
            )
        n = len(offsets) - 1
        ids = list(obj[2]) if len(obj) == 3 else [f"{id_prefix}{i}" for i in range(n)]
        lengths = np.diff(offsets).astype(np.int32)
        return Encoded(codes, offsets, ids, lengths, scheme.eff_alphabet)

    pairs = _to_pairs(obj, id_prefix)
    lut, sentinel = build_lut(scheme)
    ids = [p[0] for p in pairs]
    # encode each sequence first, then derive lengths/offsets from the ACTUAL encoded
    # arrays — robust to case folding or encoding that changes the byte count.
    enc_list: list[np.ndarray] = []
    for ident, seq in pairs:
        s = seq.upper() if scheme.case_insensitive else seq
        raw = np.frombuffer(s.encode("latin-1", "replace"), dtype=np.uint8)
        enc = lut[raw]
        if scheme.unknown == "error" and enc.size and (enc == sentinel).any():
            bad = bytes(raw[enc == sentinel][:1]).decode("latin-1")
            raise EncodeError(
                f"sequence {ident!r} contains out-of-alphabet symbol {bad!r} "
                f"(alphabet={scheme.eff_alphabet!r}); set unknown='mismatch' or 'zero'"
            )
        enc_list.append(enc)
    lengths = np.fromiter((e.size for e in enc_list), dtype=np.int32, count=len(enc_list))
    offsets = np.zeros(len(enc_list) + 1, dtype=np.int32)
    np.cumsum(lengths, out=offsets[1:])
    codes = (np.concatenate(enc_list) if enc_list else np.zeros(0, np.uint8)).astype(np.uint8)
    return Encoded(codes, offsets, ids, lengths, scheme.eff_alphabet)
