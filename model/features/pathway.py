"""Reactome pathway multi-hot encoding + n_pathways scalar.

Each candidate carries a list of Reactome pathway IDs (union across all
its drug targets). Multi-hot top-K + a numeric `n_pathways` column +
a `_missing` indicator from `reactome_has_data`.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ._multilabel import TopKMultiLabel
from .base import register

logger = logging.getLogger(__name__)


@register
class PathwayGroup:
    name = "pathway"
    list_column = "reactome_pathway_ids"
    n_column = "reactome_n_pathways"

    def __init__(self, top_k: int = 500) -> None:
        self._enc = TopKMultiLabel(top_k=top_k, prefix="pathway")

    def is_available(self, df: pd.DataFrame) -> bool:
        return self.list_column in df.columns

    def fit(self, df: pd.DataFrame) -> None:
        self._enc.fit(df[self.list_column])
        logger.info(
            "pathway: fit on %d rows; vocab=%d Reactome IDs",
            len(df),
            len(self._enc.vocab),
        )

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        mlb = self._enc.transform(df[self.list_column])
        n_path = (
            df[self.n_column].fillna(0).astype(np.float32).values.reshape(-1, 1)
            if self.n_column in df.columns
            else np.zeros((len(df), 1), dtype=np.float32)
        )
        return np.hstack([mlb, n_path])

    def feature_names(self) -> list[str]:
        return self._enc.feature_names() + ["pathway_n_total"]
