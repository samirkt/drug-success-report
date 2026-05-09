"""Nearest-approved-drug similarity feature groups.

Two scalar features per candidate, computed against an ex-ante pool of
approved drugs whose trials started strictly earlier than the candidate's
own `earliest_start_date`:

  - tanimoto_nn_similarity: max ECFP4 Tanimoto similarity over the pool
  - molformer_nn_similarity: max MoLFormer cosine similarity over the pool

Each group emits a `_missing` companion (1 when the candidate has no
fingerprint/embedding, or the date-filtered pool is empty).

The pool is built from the dataframe passed to `fit` (i.e. the
inner-training subset in the standard training flow). This conservatively
avoids any test-row-to-test-row leakage and keeps the existing
`fit(inner_df) / transform(test_df)` contract intact.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from .base import register

logger = logging.getLogger(__name__)


_DEFAULT_APPROVED: tuple[str, ...] = ("Approved", "Commercialized")
_ECFP4_BITS = 2048
_EMB_DIM = 768


def _has_fp(v) -> bool:
    if v is None:
        return False
    if isinstance(v, float) and np.isnan(v):
        return False
    try:
        return len(v) == _ECFP4_BITS
    except TypeError:
        return False


def _has_emb(v) -> bool:
    if v is None:
        return False
    if isinstance(v, float) and np.isnan(v):
        return False
    try:
        return len(v) == _EMB_DIM
    except TypeError:
        return False


def _to_ordinal(d) -> Optional[int]:
    """date / pandas Timestamp / numpy datetime64 → ordinal day, or None."""
    if d is None:
        return None
    if isinstance(d, float) and np.isnan(d):
        return None
    if hasattr(d, "toordinal"):
        try:
            return int(d.toordinal())
        except Exception:
            return None
    # numpy / pandas: convert to python date.
    try:
        ts = pd.Timestamp(d)
        if pd.isna(ts):
            return None
        return int(ts.to_pydatetime().toordinal())
    except Exception:
        return None


def _to_bitvect(arr):
    from rdkit.DataStructs import ExplicitBitVect

    bv = ExplicitBitVect(_ECFP4_BITS)
    on_bits = np.flatnonzero(np.asarray(arr).astype(np.int8))
    for b in on_bits.tolist():
        bv.SetBit(int(b))
    return bv


@dataclass
class _ApprovedPool:
    """Approved-outcome rows, sorted by start date ascending.

    `payloads` is parallel to `dates_ordinal` — a list of
    `ExplicitBitVect` (Tanimoto) or a 2D `np.ndarray` of L2-normalized
    embeddings (MolFormer).
    """

    dates_ordinal: np.ndarray
    payloads: object  # list[BitVect] or np.ndarray (N, D)

    def __len__(self) -> int:
        return len(self.dates_ordinal)

    def cut_for(self, query_date_ordinal: int) -> int:
        """Index `i` such that `dates_ordinal[:i]` is strictly less than the query date."""
        return int(np.searchsorted(self.dates_ordinal, query_date_ordinal, side="left"))


def _select_approved(
    df: pd.DataFrame, approved_outcomes: Iterable[str], column: str
) -> pd.DataFrame:
    if "outcome" not in df.columns:
        raise ValueError("nn_similarity: dataframe missing `outcome` column")
    if "earliest_start_date" not in df.columns:
        raise ValueError("nn_similarity: dataframe missing `earliest_start_date` column")
    approved_set = set(approved_outcomes)
    sub = df[df["outcome"].isin(approved_set)].copy()
    if column not in sub.columns:
        raise ValueError(f"nn_similarity: dataframe missing `{column}` column")
    return sub


@register
class TanimotoNNGroup:
    name = "tanimoto_nn"

    def __init__(self, *, approved_outcomes: tuple[str, ...] = _DEFAULT_APPROVED) -> None:
        self.approved_outcomes = tuple(approved_outcomes)
        self._pool: Optional[_ApprovedPool] = None

    def is_available(self, df: pd.DataFrame) -> bool:
        return (
            "ecfp4" in df.columns
            and "outcome" in df.columns
            and "earliest_start_date" in df.columns
        )

    def fit(self, df: pd.DataFrame) -> None:
        sub = _select_approved(df, self.approved_outcomes, column="ecfp4")
        rows: list[tuple[int, object]] = []
        for fp, dt in zip(sub["ecfp4"].values, sub["earliest_start_date"].values):
            if not _has_fp(fp):
                continue
            ordn = _to_ordinal(dt)
            if ordn is None:
                continue
            rows.append((ordn, _to_bitvect(fp)))
        rows.sort(key=lambda x: x[0])
        if rows:
            ords = np.asarray([r[0] for r in rows], dtype=np.int64)
            bvs = [r[1] for r in rows]
        else:
            ords = np.empty(0, dtype=np.int64)
            bvs = []
        self._pool = _ApprovedPool(dates_ordinal=ords, payloads=bvs)
        logger.info(
            "tanimoto_nn: fit on %d rows; approved-with-fp pool=%d",
            len(df),
            len(self._pool),
        )

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        from rdkit.DataStructs import BulkTanimotoSimilarity

        if self._pool is None:
            raise RuntimeError("TanimotoNNGroup.transform called before fit()")
        n = len(df)
        out = np.zeros((n, 2), dtype=np.float32)
        if len(self._pool) == 0:
            out[:, 1] = 1.0
            return out
        ecfp_col = df["ecfp4"].values
        date_col = df["earliest_start_date"].values
        pool_dates = self._pool.dates_ordinal
        pool_bvs: list = self._pool.payloads  # type: ignore[assignment]
        for i in range(n):
            fp = ecfp_col[i]
            if not _has_fp(fp):
                out[i, 0] = 0.0
                out[i, 1] = 1.0
                continue
            ordn = _to_ordinal(date_col[i])
            if ordn is None:
                out[i, 0] = 0.0
                out[i, 1] = 1.0
                continue
            cut = int(np.searchsorted(pool_dates, ordn, side="left"))
            if cut == 0:
                out[i, 0] = 0.0
                out[i, 1] = 1.0
                continue
            query_bv = _to_bitvect(fp)
            sims = BulkTanimotoSimilarity(query_bv, pool_bvs[:cut])
            out[i, 0] = float(max(sims)) if sims else 0.0
            out[i, 1] = 0.0
        return out

    def feature_names(self) -> list[str]:
        return ["tanimoto_nn_similarity", "tanimoto_nn_missing"]


@register
class MolformerNNGroup:
    name = "molformer_nn"

    def __init__(self, *, approved_outcomes: tuple[str, ...] = _DEFAULT_APPROVED) -> None:
        self.approved_outcomes = tuple(approved_outcomes)
        self._pool: Optional[_ApprovedPool] = None

    def is_available(self, df: pd.DataFrame) -> bool:
        return (
            "embedding" in df.columns
            and "outcome" in df.columns
            and "earliest_start_date" in df.columns
        )

    def fit(self, df: pd.DataFrame) -> None:
        sub = _select_approved(df, self.approved_outcomes, column="embedding")
        rows: list[tuple[int, np.ndarray]] = []
        for emb, dt in zip(sub["embedding"].values, sub["earliest_start_date"].values):
            if not _has_emb(emb):
                continue
            ordn = _to_ordinal(dt)
            if ordn is None:
                continue
            rows.append((ordn, np.asarray(emb, dtype=np.float32)))
        rows.sort(key=lambda x: x[0])
        if rows:
            ords = np.asarray([r[0] for r in rows], dtype=np.int64)
            mat = np.vstack([r[1] for r in rows]).astype(np.float32)
            norms = np.linalg.norm(mat, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            mat_norm = mat / norms
        else:
            ords = np.empty(0, dtype=np.int64)
            mat_norm = np.empty((0, _EMB_DIM), dtype=np.float32)
        self._pool = _ApprovedPool(dates_ordinal=ords, payloads=mat_norm)
        logger.info(
            "molformer_nn: fit on %d rows; approved-with-embedding pool=%d",
            len(df),
            len(self._pool),
        )

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if self._pool is None:
            raise RuntimeError("MolformerNNGroup.transform called before fit()")
        n = len(df)
        out = np.zeros((n, 2), dtype=np.float32)
        pool_mat: np.ndarray = self._pool.payloads  # type: ignore[assignment]
        if pool_mat.shape[0] == 0:
            out[:, 1] = 1.0
            return out
        emb_col = df["embedding"].values
        date_col = df["earliest_start_date"].values
        pool_dates = self._pool.dates_ordinal
        for i in range(n):
            emb = emb_col[i]
            if not _has_emb(emb):
                out[i, 0] = 0.0
                out[i, 1] = 1.0
                continue
            ordn = _to_ordinal(date_col[i])
            if ordn is None:
                out[i, 0] = 0.0
                out[i, 1] = 1.0
                continue
            cut = int(np.searchsorted(pool_dates, ordn, side="left"))
            if cut == 0:
                out[i, 0] = 0.0
                out[i, 1] = 1.0
                continue
            q = np.asarray(emb, dtype=np.float32)
            qn = np.linalg.norm(q)
            if qn == 0:
                out[i, 0] = 0.0
                out[i, 1] = 1.0
                continue
            q_unit = q / qn
            sims = pool_mat[:cut] @ q_unit
            out[i, 0] = float(np.max(sims))
            out[i, 1] = 0.0
        return out

    def feature_names(self) -> list[str]:
        return ["molformer_nn_similarity", "molformer_nn_missing"]
