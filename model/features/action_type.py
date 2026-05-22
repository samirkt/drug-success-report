"""Drug action-type multi-hot encoding (OpenTargets action enum).

`opentargets_action_type` is a list of OpenTargets action-type enums per
candidate (INHIBITOR, AGONIST, ANTAGONIST, BLOCKER, DEGRADER, ...).
Vocabulary is small (~24 values), so K=50 is effectively full one-hot.
Kept as TopKMultiLabel for symmetry with `targets` / `pathway` / `disease`.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ._multilabel import TopKMultiLabel
from .base import register

logger = logging.getLogger(__name__)


@register
class ActionTypeGroup:
    name = "action_type"
    column = "opentargets_action_type"

    def __init__(self, top_k: int = 50) -> None:
        self._enc = TopKMultiLabel(top_k=top_k, prefix="action")

    def is_available(self, df: pd.DataFrame) -> bool:
        return self.column in df.columns

    def fit(self, df: pd.DataFrame) -> None:
        self._enc.fit(df[self.column])
        logger.info(
            "action_type: fit on %d rows; vocab=%d action-type values",
            len(df),
            len(self._enc.vocab),
        )

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        return self._enc.transform(df[self.column])

    def feature_names(self) -> list[str]:
        return self._enc.feature_names()
