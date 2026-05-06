"""ADMET predictions — DrugBank-approved-percentile columns.

Sources the canonical column list from `pipeline/admet/admet_columns.py`
(filtered to the `*_drugbank_approved_percentile` siblings) so this module
stays in sync if admet_ai's column set changes upstream.

The published `candidate_detail.parquet` may have these columns 100% null
(if the parquet predates the ADMET enrichment fix). To stay tolerant, we
drop columns that exceed `drop_null_threshold` nulls in TRAIN, median-
impute the rest, and flag rows missing values via per-column indicators.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .base import register

logger = logging.getLogger(__name__)


# Locate the canonical ADMET column list under src/pipeline/admet/.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from pipeline.admet.admet_columns import ADMET_COLUMNS, field_name  # noqa: E402


def _percentile_columns() -> list[str]:
    """Return the DataFrame column names for percentile ADMET features."""
    return [
        field_name(c)
        for c in ADMET_COLUMNS
        if c.endswith("_drugbank_approved_percentile")
    ]


@register
class AdmetGroup:
    name = "admet"

    def __init__(
        self,
        drop_null_threshold: float = 0.95,
        indicator_threshold: float = 0.05,
    ) -> None:
        self._drop_null_threshold = drop_null_threshold
        self._indicator_threshold = indicator_threshold
        self._kept_cols: list[str] = []
        self._indicator_cols: list[str] = []
        self._medians: dict[str, float] = {}
        self._all_candidate_cols: list[str] = _percentile_columns()

    def is_available(self, df: pd.DataFrame) -> bool:
        return any(c in df.columns for c in self._all_candidate_cols)

    def fit(self, df: pd.DataFrame) -> None:
        present = [c for c in self._all_candidate_cols if c in df.columns]
        n = len(df)
        self._kept_cols = []
        self._indicator_cols = []
        self._medians = {}
        for col in present:
            null_rate = df[col].isna().mean() if n else 1.0
            if null_rate > self._drop_null_threshold:
                continue
            self._kept_cols.append(col)
            median = float(df[col].median()) if df[col].notna().any() else 0.0
            self._medians[col] = median
            if null_rate > self._indicator_threshold:
                self._indicator_cols.append(col)
        n_dropped = len(present) - len(self._kept_cols)
        logger.info(
            "admet: fit on %d rows; %d/%d cols kept (%d dropped for >%.0f%% null), %d indicators",
            n,
            len(self._kept_cols),
            len(present),
            n_dropped,
            100.0 * self._drop_null_threshold,
            len(self._indicator_cols),
        )
        if not self._kept_cols:
            logger.warning(
                "admet: no usable columns — feature group will produce zero-width output. "
                "Re-run the pipeline to populate `admet_*_drugbank_approved_percentile` columns."
            )

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not self._kept_cols:
            return np.zeros((len(df), 0), dtype=np.float32)
        n = len(df)
        n_val = len(self._kept_cols)
        n_ind = len(self._indicator_cols)
        out = np.zeros((n, n_val + n_ind), dtype=np.float32)
        for i, col in enumerate(self._kept_cols):
            series = df[col] if col in df.columns else pd.Series([np.nan] * n)
            filled = series.fillna(self._medians[col]).astype(np.float32)
            out[:, i] = filled.values
        for j, col in enumerate(self._indicator_cols):
            series = df[col] if col in df.columns else pd.Series([np.nan] * n)
            out[:, n_val + j] = series.isna().astype(np.float32).values
        return out

    def feature_names(self) -> list[str]:
        return list(self._kept_cols) + [f"{c}_missing" for c in self._indicator_cols]
