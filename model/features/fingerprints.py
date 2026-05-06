"""Molecular structure fingerprints (ECFP4 + MACCS).

Stacks the two precomputed bit vectors from `fingerprints.parquet` into a
single uint8 matrix. Missing rows (no SMILES) → all-zero + per-group
`_missing` indicator column.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .base import FeatureGroup, register, stack_arrays

logger = logging.getLogger(__name__)

ECFP4_DIM = 2048
MACCS_DIM = 167


@register
class FingerprintsGroup:
    name = "fingerprints"

    def __init__(self) -> None:
        self._dim = ECFP4_DIM + MACCS_DIM
        self._n_features = self._dim + 1  # + missing indicator

    def is_available(self, df: pd.DataFrame) -> bool:
        return "ecfp4" in df.columns and "maccs" in df.columns

    def fit(self, df: pd.DataFrame) -> None:
        coverage = df["ecfp4"].notna().sum()
        logger.info(
            "fingerprints: fit on %d rows; %d (%.1f%%) have fingerprints",
            len(df),
            coverage,
            100.0 * coverage / max(len(df), 1),
        )

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        ecfp, miss_e = stack_arrays(df["ecfp4"], ECFP4_DIM, np.uint8)
        maccs, miss_m = stack_arrays(df["maccs"], MACCS_DIM, np.uint8)
        # `_missing` is true if either bit vector is missing (in practice
        # they're always missing together since they share a SMILES source)
        missing = (miss_e | miss_m).reshape(-1, 1)
        return np.hstack([ecfp, maccs, missing.astype(np.uint8)])

    def feature_names(self) -> list[str]:
        return (
            [f"ecfp4_{i}" for i in range(ECFP4_DIM)]
            + [f"maccs_{i}" for i in range(MACCS_DIM)]
            + ["fingerprints_missing"]
        )
