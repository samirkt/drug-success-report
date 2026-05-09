"""Tanimoto-NN (k=5) baseline using ECFP4 fingerprints.

For each test candidate, finds the 5 most similar training candidates
by Tanimoto similarity over ECFP4 bits and predicts the mean of their
binary labels. Test candidates without a fingerprint receive the
training-set base rate.
"""

from __future__ import annotations

import logging
from typing import ClassVar, Optional

import numpy as np
import pandas as pd

from . import register

logger = logging.getLogger(__name__)


_K = 5
_ECFP4_BITS = 2048


def _to_bitvect(arr):
    """Convert a 2048-int numpy array (0/1) to an RDKit ExplicitBitVect."""
    from rdkit.DataStructs import ExplicitBitVect

    bv = ExplicitBitVect(_ECFP4_BITS)
    on_bits = np.flatnonzero(np.asarray(arr).astype(np.int8))
    for b in on_bits.tolist():
        bv.SetBit(int(b))
    return bv


def _has_fp(v) -> bool:
    if v is None:
        return False
    if isinstance(v, float) and np.isnan(v):
        return False
    try:
        return len(v) == _ECFP4_BITS
    except TypeError:
        return False


@register
class TanimotoBaseline:
    name: ClassVar[str] = "tanimoto"

    def __init__(self, *, k: int = _K) -> None:
        self.k = k
        self._train_bvs: list = []
        self._train_y: np.ndarray = np.empty(0, dtype=np.int8)
        self._global_rate: float = 0.0
        self._n_train_with_fp: int = 0
        self._n_test_with_fp: int = 0
        self._n_test_fallback: int = 0

    def fit(self, train_df: pd.DataFrame, y_train: np.ndarray) -> None:
        if "ecfp4" not in train_df.columns:
            raise ValueError("tanimoto baseline requires `ecfp4` column on train_df")
        if len(train_df) != len(y_train):
            raise ValueError("train_df and y_train length mismatch")
        self._global_rate = float(np.mean(y_train)) if len(y_train) else 0.0

        bvs = []
        labels = []
        for v, y in zip(train_df["ecfp4"].values, y_train):
            if not _has_fp(v):
                continue
            bvs.append(_to_bitvect(v))
            labels.append(int(y))
        self._train_bvs = bvs
        self._train_y = np.asarray(labels, dtype=np.int8)
        self._n_train_with_fp = len(bvs)
        logger.info(
            "tanimoto: n_train=%d with_fp=%d global=%.4f k=%d",
            len(train_df),
            self._n_train_with_fp,
            self._global_rate,
            self.k,
        )

    def predict_proba(self, test_df: pd.DataFrame) -> np.ndarray:
        from rdkit.DataStructs import BulkTanimotoSimilarity

        if not self._train_bvs:
            logger.warning("tanimoto: no training fingerprints — returning base rate")
            self._n_test_with_fp = 0
            self._n_test_fallback = len(test_df)
            return np.full(len(test_df), self._global_rate, dtype=np.float64)

        out = np.empty(len(test_df), dtype=np.float64)
        n_with = 0
        n_fb = 0
        k = min(self.k, len(self._train_bvs))
        for i, v in enumerate(test_df["ecfp4"].values):
            if not _has_fp(v):
                out[i] = self._global_rate
                n_fb += 1
                continue
            bv = _to_bitvect(v)
            sims = np.asarray(BulkTanimotoSimilarity(bv, self._train_bvs), dtype=np.float64)
            if k >= len(sims):
                top_idx = np.arange(len(sims))
            else:
                top_idx = np.argpartition(-sims, k - 1)[:k]
            out[i] = float(np.mean(self._train_y[top_idx]))
            n_with += 1
        self._n_test_with_fp = n_with
        self._n_test_fallback = n_fb
        logger.info(
            "tanimoto predict: with_fp=%d fallback=%d",
            n_with,
            n_fb,
        )
        return out

    def metadata(self) -> dict:
        return {
            "k": self.k,
            "n_train_with_fp": self._n_train_with_fp,
            "n_test_with_fp": self._n_test_with_fp,
            "n_test_fallback": self._n_test_fallback,
            "global_rate": self._global_rate,
            "missing_fp_strategy": "global_rate",
        }
