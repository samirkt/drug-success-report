"""MoLFormer molecular embeddings (768d).

Stacks precomputed embeddings; missing rows imputed with the train-set
mean and flagged via a `_missing` indicator.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .base import register, stack_arrays

logger = logging.getLogger(__name__)

EMB_DIM = 768


@register
class EmbeddingsGroup:
    name = "embeddings"

    def __init__(self) -> None:
        self._mean: np.ndarray | None = None

    def is_available(self, df: pd.DataFrame) -> bool:
        return "embedding" in df.columns

    def fit(self, df: pd.DataFrame) -> None:
        mat, missing = stack_arrays(df["embedding"], EMB_DIM, np.float32)
        present = missing == 0
        if present.sum() == 0:
            logger.warning("embeddings: no rows have embeddings; mean defaults to zero")
            self._mean = np.zeros(EMB_DIM, dtype=np.float32)
        else:
            self._mean = mat[present].mean(axis=0).astype(np.float32)
        logger.info(
            "embeddings: fit on %d rows; %d (%.1f%%) have embeddings",
            len(df),
            int(present.sum()),
            100.0 * present.sum() / max(len(df), 1),
        )

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if self._mean is None:
            raise RuntimeError("EmbeddingsGroup.transform called before fit()")
        mat, missing = stack_arrays(df["embedding"], EMB_DIM, np.float32)
        # mean-impute missing rows
        if missing.any():
            mat[missing == 1] = self._mean
        return np.hstack([mat, missing.reshape(-1, 1).astype(np.float32)])

    def feature_names(self) -> list[str]:
        return [f"embedding_{i}" for i in range(EMB_DIM)] + ["embedding_missing"]
