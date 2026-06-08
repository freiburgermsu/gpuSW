"""The :class:`AlignResult` returned by the high-level helpers — scores plus the
query/reference identifiers, with convenience top-k and dataframe views.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["AlignResult"]


@dataclass
class AlignResult:
    """Alignment scores with identifiers attached.

    For a cross product, ``scores`` is ``(n_queries, n_refs)``; for paired scoring it
    is ``(n_pairs,)``. ``top_k`` results carry ``topk_idx``/``topk_scores`` instead of
    a dense matrix.
    """

    scores: np.ndarray | None
    query_ids: list[str]
    reference_ids: list[str]
    scheme: object
    topk_idx: np.ndarray | None = None      # (n_queries, k) reference indices
    topk_scores: np.ndarray | None = None   # (n_queries, k) scores

    # ----------------------------------------------------------------- top-k
    def top_k(self, k: int = 5) -> list[tuple[str, list[tuple[str, int]]]]:
        """Per query, the ``k`` best ``(reference_id, score)`` pairs, score-descending."""
        if self.topk_idx is not None:
            idx, sc = self.topk_idx[:, :k], self.topk_scores[:, :k]
        else:
            if self.scores is None or self.scores.ndim != 2:
                raise ValueError("top_k needs a 2-D cross-product result")
            order = np.argsort(-self.scores, axis=1, kind="stable")[:, :k]
            idx = order
            sc = np.take_along_axis(self.scores, order, axis=1)
        out = []
        for qi, qid in enumerate(self.query_ids):
            hits = [
                (self.reference_ids[int(idx[qi, r])], int(sc[qi, r]))
                for r in range(idx.shape[1])
            ]
            out.append((qid, hits))
        return out

    def best(self) -> list[tuple[str, str, int]]:
        """The single best ``(query_id, reference_id, score)`` per query."""
        return [(q, hits[0][0], hits[0][1]) for q, hits in self.top_k(1)]

    # ------------------------------------------------------------- dataframe
    def to_dataframe(self):
        """Long-format ``pandas.DataFrame`` (requires the ``[pandas]`` extra)."""
        try:
            import pandas as pd  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover
            raise ImportError("to_dataframe() needs pandas: pip install 'gpusw[pandas]'") from exc
        rows = []
        if self.topk_idx is not None:
            for qi, qid in enumerate(self.query_ids):
                for rank in range(self.topk_idx.shape[1]):
                    rj = int(self.topk_idx[qi, rank])
                    rows.append((qid, self.reference_ids[rj], rank + 1,
                                 int(self.topk_scores[qi, rank])))
            return pd.DataFrame(rows, columns=["query", "reference", "rank", "score"])
        s = self.scores
        if s is None:
            raise ValueError("no scores to tabulate")
        if s.ndim == 1:
            for i, v in enumerate(s):
                q = self.query_ids[i] if i < len(self.query_ids) else f"q{i}"
                r = self.reference_ids[i] if i < len(self.reference_ids) else f"r{i}"
                rows.append((q, r, int(v)))
        else:
            for qi, qid in enumerate(self.query_ids):
                for rj, rid in enumerate(self.reference_ids):
                    rows.append((qid, rid, int(s[qi, rj])))
        return pd.DataFrame(rows, columns=["query", "reference", "score"])

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        shape = None if self.scores is None else self.scores.shape
        return (f"AlignResult(n_queries={len(self.query_ids)}, "
                f"n_refs={len(self.reference_ids)}, scores_shape={shape})")
