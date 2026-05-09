"""Baseline models for 1:1 comparison against the full model.

Each baseline implements the same interface (`fit`, `predict_proba`,
`metadata`) and writes artifacts in the same shape as the full model run
so the existing `model/report.py` can render side-by-side comparisons.
"""

from __future__ import annotations

from typing import ClassVar, Protocol, runtime_checkable

import numpy as np
import pandas as pd


BASELINES: dict[str, type["Baseline"]] = {}


@runtime_checkable
class Baseline(Protocol):
    name: ClassVar[str]

    def fit(self, train_df: pd.DataFrame, y_train: np.ndarray) -> None: ...
    def predict_proba(self, test_df: pd.DataFrame) -> np.ndarray: ...
    def metadata(self) -> dict: ...


def register(cls: type[Baseline]) -> type[Baseline]:
    if not getattr(cls, "name", None):
        raise ValueError(f"{cls.__name__} missing class-level `name`")
    if cls.name in BASELINES:
        raise ValueError(f"baseline {cls.name!r} already registered")
    BASELINES[cls.name] = cls
    return cls


def build_baseline(name: str, **kwargs) -> Baseline:
    if name not in BASELINES:
        raise KeyError(f"unknown baseline {name!r}; known: {sorted(BASELINES)}")
    return BASELINES[name](**kwargs)


from . import stratum  # noqa: E402,F401
from . import tanimoto  # noqa: E402,F401
from . import killer_figure  # noqa: E402,F401

# `target_only` is not in the BASELINES registry because it reuses
# `train_one_run` directly rather than implementing the Baseline protocol.
# `runner.run_baselines` dispatches to it by name.
ALL_BASELINES: tuple[str, ...] = ("stratum", "tanimoto", "target_only", "killer_figure")
