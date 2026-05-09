"""ICD-10 stratum base-rate baseline.

Predicts the mean training-set label for the candidate's stratum, with a
3-char prefix → 1-char chapter → global mean fallback chain. Multi-code
candidates use the first ICD-10 code in `icd10_codes`.
"""

from __future__ import annotations

import logging
from typing import ClassVar, Optional

import numpy as np
import pandas as pd

from . import register

logger = logging.getLogger(__name__)


_MIN_N = 10  # minimum stratum size before falling back to the next level


def _first_code(arr) -> Optional[str]:
    """Return the first ICD-10 code in a candidate's codes array, or None.

    `icd10_codes` is a numpy array of strings; empty arrays + None /
    scalar NaN all map to None.
    """
    if arr is None:
        return None
    if isinstance(arr, float) and np.isnan(arr):
        return None
    try:
        if len(arr) == 0:
            return None
        return str(arr[0])
    except TypeError:
        return None


@register
class StratumBaseline:
    name: ClassVar[str] = "stratum"

    def __init__(self, *, min_n: int = _MIN_N) -> None:
        self.min_n = min_n
        self._stratum_rate: dict[str, tuple[float, int]] = {}
        self._chapter_rate: dict[str, tuple[float, int]] = {}
        self._global_rate: float = 0.0
        self._n_train: int = 0
        self._n_train_with_code: int = 0

    def fit(self, train_df: pd.DataFrame, y_train: np.ndarray) -> None:
        if len(train_df) != len(y_train):
            raise ValueError("train_df and y_train length mismatch")
        self._n_train = len(train_df)
        self._global_rate = float(np.mean(y_train)) if len(y_train) else 0.0

        codes = [_first_code(v) for v in train_df.get("icd10_codes", pd.Series([None] * len(train_df))).values]
        self._n_train_with_code = sum(1 for c in codes if c)

        prefix_buckets: dict[str, list[int]] = {}
        chapter_buckets: dict[str, list[int]] = {}
        for code, y in zip(codes, y_train):
            if not code:
                continue
            prefix_buckets.setdefault(code[:3], []).append(int(y))
            chapter_buckets.setdefault(code[:1], []).append(int(y))

        self._stratum_rate = {
            k: (float(np.mean(v)), len(v)) for k, v in prefix_buckets.items()
        }
        self._chapter_rate = {
            k: (float(np.mean(v)), len(v)) for k, v in chapter_buckets.items()
        }
        logger.info(
            "stratum: n_train=%d with_code=%d strata=%d chapters=%d global=%.4f",
            self._n_train,
            self._n_train_with_code,
            len(self._stratum_rate),
            len(self._chapter_rate),
            self._global_rate,
        )

    def predict_proba(self, test_df: pd.DataFrame) -> np.ndarray:
        out = np.empty(len(test_df), dtype=np.float64)
        n_strat = n_chap = n_glob = 0
        codes_col = test_df.get("icd10_codes", pd.Series([None] * len(test_df))).values
        for i, raw in enumerate(codes_col):
            code = _first_code(raw)
            if code:
                rate_n = self._stratum_rate.get(code[:3])
                if rate_n is not None and rate_n[1] >= self.min_n:
                    out[i] = rate_n[0]
                    n_strat += 1
                    continue
                rate_n = self._chapter_rate.get(code[:1])
                if rate_n is not None and rate_n[1] >= self.min_n:
                    out[i] = rate_n[0]
                    n_chap += 1
                    continue
            out[i] = self._global_rate
            n_glob += 1
        self._predict_counts = (n_strat, n_chap, n_glob)
        logger.info(
            "stratum predict: stratum_hit=%d chapter_fallback=%d global_fallback=%d",
            n_strat,
            n_chap,
            n_glob,
        )
        return out

    def metadata(self) -> dict:
        n_strat, n_chap, n_glob = getattr(self, "_predict_counts", (0, 0, 0))
        return {
            "min_n": self.min_n,
            "n_train": self._n_train,
            "n_train_with_icd10": self._n_train_with_code,
            "n_strata": len(self._stratum_rate),
            "n_chapters": len(self._chapter_rate),
            "global_rate": self._global_rate,
            "n_test_stratum_hit": n_strat,
            "n_test_chapter_fallback": n_chap,
            "n_test_global_fallback": n_glob,
        }
