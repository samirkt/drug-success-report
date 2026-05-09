"""Target-only XGBoost baseline.

A thin wrapper that calls `train_one_run` with `features.enabled=("targets",)`
so we inherit identical hyperparameters, inner-validation early stopping,
and uncalibrated probabilities from the full model pipeline.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Optional

import numpy as np
import pandas as pd

from ..config import FeatureConfig, ModelingConfig
from ..train import RunResult, train_one_run

logger = logging.getLogger(__name__)


TARGET_ONLY_GROUPS: tuple[str, ...] = ("targets",)


def make_target_only_config(base: ModelingConfig) -> ModelingConfig:
    """Clone `base` with features restricted to the targets group only."""
    fc = replace(base.features, enabled=TARGET_ONLY_GROUPS)
    return replace(base, features=fc)


def run_target_only(
    base: ModelingConfig,
    *,
    df: Optional[pd.DataFrame] = None,
    train_idx: Optional[np.ndarray] = None,
    test_idx: Optional[np.ndarray] = None,
) -> RunResult:
    cfg = make_target_only_config(base)
    logger.info("target_only: training XGBoost on groups=%s", cfg.features.enabled)
    return train_one_run(cfg, df=df, train_idx=train_idx, test_idx=test_idx)
