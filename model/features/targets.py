"""Drug target multi-hot encoding (UniProt accessions).

`drug_targets` is a list of UniProt IDs per candidate (populated by the
TargetsEnrichment stage from the ChEMBL snapshot). We multi-hot encode
the top-K most frequent UniProts in TRAIN.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ._multilabel import TopKMultiLabel
from .base import register

logger = logging.getLogger(__name__)


@register
class DrugTargetsGroup:
    name = "targets"
    column = "drug_targets"

    def __init__(self, top_k: int = 200) -> None:
        self._enc = TopKMultiLabel(top_k=top_k, prefix="target")

    def is_available(self, df: pd.DataFrame) -> bool:
        return self.column in df.columns

    def fit(self, df: pd.DataFrame) -> None:
        self._enc.fit(df[self.column])
        logger.info(
            "targets: fit on %d rows; vocab=%d UniProt IDs",
            len(df),
            len(self._enc.vocab),
        )

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        return self._enc.transform(df[self.column])

    def feature_names(self) -> list[str]:
        return self._enc.feature_names()
