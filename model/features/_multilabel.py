"""Shared utilities for top-K multi-label encoding of list-typed columns.

Used by `targets`, `pathway`, and `disease` (MeSH tree-prefix) groups.
"""

from __future__ import annotations

from collections import Counter
from typing import Iterable

import numpy as np
import pandas as pd


def iter_lists(values: pd.Series) -> Iterable[list[str]]:
    """Yield a clean list[str] for each row, treating NaN/None as []."""
    for v in values.values:
        if v is None:
            yield []
            continue
        # numpy arrays / pandas-compatible iterables behave fine here;
        # scalar NaN is the only nontrivial case.
        if isinstance(v, float) and np.isnan(v):
            yield []
            continue
        try:
            yield [str(x) for x in v if x is not None]
        except TypeError:
            yield []


class TopKMultiLabel:
    """Multi-hot encoder over the K most frequent tokens in TRAIN.

    Long-tail tokens are collapsed into a single `<prefix>_other_count`
    column (count of out-of-vocabulary tokens for that row, capped). An
    `_missing` indicator flags rows with empty token lists.
    """

    def __init__(self, top_k: int, prefix: str) -> None:
        self.top_k = top_k
        self.prefix = prefix
        self.vocab: list[str] = []
        self._index: dict[str, int] = {}

    def fit(self, values: pd.Series) -> None:
        counts: Counter[str] = Counter()
        for tokens in iter_lists(values):
            counts.update(set(tokens))  # dedup within row
        self.vocab = [t for t, _ in counts.most_common(self.top_k)]
        self._index = {t: i for i, t in enumerate(self.vocab)}

    def transform(self, values: pd.Series) -> np.ndarray:
        n = len(values)
        k = len(self.vocab)
        out = np.zeros((n, k + 2), dtype=np.float32)
        for i, tokens in enumerate(iter_lists(values)):
            if not tokens:
                out[i, k + 1] = 1.0  # _missing
                continue
            other = 0
            for tok in set(tokens):
                idx = self._index.get(tok)
                if idx is None:
                    other += 1
                else:
                    out[i, idx] = 1.0
            out[i, k] = float(other)  # _other_count
        return out

    def feature_names(self) -> list[str]:
        names = [f"{self.prefix}_{tok}" for tok in self.vocab]
        names.append(f"{self.prefix}_other_count")
        names.append(f"{self.prefix}_missing")
        return names
