"""Mechanism-of-action multi-hot encoding (OpenTargets free-text MoA).

`opentargets_moa` is a list of free-text MoA descriptions per candidate
(e.g. "Glucocorticoid receptor agonist", "Cyclooxygenase inhibitor").
Vocabulary is long-tail (~913 distinct strings); we multi-hot encode the
top-K most frequent in TRAIN.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ._multilabel import TopKMultiLabel
from .base import register

logger = logging.getLogger(__name__)


@register
class MechanismOfActionGroup:
    name = "moa"
    column = "opentargets_moa"

    def __init__(self, top_k: int = 200) -> None:
        self._enc = TopKMultiLabel(top_k=top_k, prefix="moa")

    def is_available(self, df: pd.DataFrame) -> bool:
        return self.column in df.columns

    def fit(self, df: pd.DataFrame) -> None:
        self._enc.fit(df[self.column])
        logger.info(
            "moa: fit on %d rows; vocab=%d MoA strings",
            len(df),
            len(self._enc.vocab),
        )

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        return self._enc.transform(df[self.column])

    def feature_names(self) -> list[str]:
        return self._enc.feature_names()
