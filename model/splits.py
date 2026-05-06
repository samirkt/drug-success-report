"""Train/test split — stratified row-level + optional group-aware.

`group_by=None` → standard stratified shuffle split.
`group_by="drug_name"` (or any column) → StratifiedGroupKFold; we take fold
0 of `n_splits = round(1/test_size)` so test_size ≈ 1/k. Guarantees zero
group overlap between train and test.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def split(
    df: pd.DataFrame,
    *,
    test_size: float,
    seed: int,
    group_by: Optional[str] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (train_idx, test_idx) into df."""
    y = df["y"].values
    if group_by is None:
        from sklearn.model_selection import train_test_split

        idx = np.arange(len(df))
        train_idx, test_idx = train_test_split(
            idx, test_size=test_size, stratify=y, random_state=seed,
        )
        logger.info(
            "split: stratified row-level; train=%d test=%d (test_size=%.2f, seed=%d)",
            len(train_idx),
            len(test_idx),
            test_size,
            seed,
        )
        return train_idx, test_idx

    if group_by not in df.columns:
        raise ValueError(f"group_by={group_by!r} not in DataFrame columns")
    from sklearn.model_selection import StratifiedGroupKFold

    n_splits = max(2, round(1.0 / test_size))
    kf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    groups = df[group_by].fillna("__missing__").astype(str).values
    train_idx, test_idx = next(kf.split(np.zeros(len(df)), y, groups))
    train_groups = set(groups[train_idx])
    test_groups = set(groups[test_idx])
    overlap = train_groups & test_groups
    logger.info(
        "split: group-aware (group_by=%s); train=%d test=%d unique-train=%d unique-test=%d overlap=%d",
        group_by,
        len(train_idx),
        len(test_idx),
        len(train_groups),
        len(test_groups),
        len(overlap),
    )
    if overlap:
        logger.warning("split: %d overlapping groups (sklearn issue) — review", len(overlap))
    return train_idx, test_idx
