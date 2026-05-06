"""FeatureGroup Protocol + registry.

Every feature group implements `is_available`, `fit`, `transform`,
`feature_names`. The registry maps a canonical group name (used in CLI
flags and ablation reports) to a class. Concrete groups register
themselves at import time via `register`.
"""

from __future__ import annotations

import logging
from typing import ClassVar, Protocol, runtime_checkable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

FEATURE_GROUPS: dict[str, type["FeatureGroup"]] = {}


@runtime_checkable
class FeatureGroup(Protocol):
    name: ClassVar[str]

    def is_available(self, df: pd.DataFrame) -> bool: ...
    def fit(self, df: pd.DataFrame) -> None: ...
    def transform(self, df: pd.DataFrame) -> np.ndarray: ...
    def feature_names(self) -> list[str]: ...


def register(cls: type[FeatureGroup]) -> type[FeatureGroup]:
    """Class decorator — register a FeatureGroup under its `name`."""
    if not getattr(cls, "name", None):
        raise ValueError(f"{cls.__name__} missing class-level `name`")
    if cls.name in FEATURE_GROUPS:
        raise ValueError(f"feature group {cls.name!r} already registered")
    FEATURE_GROUPS[cls.name] = cls
    return cls


def build_group(name: str, **kwargs) -> FeatureGroup:
    if name not in FEATURE_GROUPS:
        raise KeyError(f"unknown feature group {name!r}; known: {sorted(FEATURE_GROUPS)}")
    return FEATURE_GROUPS[name](**kwargs)


def stack_arrays(values: pd.Series, dim: int, dtype) -> tuple[np.ndarray, np.ndarray]:
    """Stack a pandas Series of numpy arrays into a 2D matrix.

    Rows with NaN / None are filled with zeros. Returns (matrix, missing_mask).
    """
    n = len(values)
    out = np.zeros((n, dim), dtype=dtype)
    missing = np.zeros(n, dtype=np.int8)
    for i, v in enumerate(values.values):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            missing[i] = 1
            continue
        arr = np.asarray(v)
        if arr.shape[0] != dim:
            logger.warning(
                "stack_arrays: row %d has shape %s, expected (%d,) — treating as missing",
                i,
                arr.shape,
                dim,
            )
            missing[i] = 1
            continue
        out[i] = arr.astype(dtype, copy=False)
    return out, missing
