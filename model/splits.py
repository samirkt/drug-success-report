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


def _year_of(v) -> Optional[int]:
    """Extract year from a date / datetime / pd.Timestamp / None."""
    if v is None:
        return None
    if hasattr(v, "year"):
        return int(v.year)
    return None


def split(
    df: pd.DataFrame,
    *,
    test_size: float,
    seed: int,
    group_by: Optional[str] = None,
    time_split_column: Optional[str] = None,
    time_split_year: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (train_idx, test_idx) into df.

    Modes (mutually exclusive, checked in order):
      - `time_split_year` set: temporal split on `time_split_column`'s year
      - `group_by` set: stratified group k-fold (no group overlap)
      - default: stratified shuffle split by `y`
    """
    y = df["y"].values

    if time_split_year is not None:
        if not time_split_column or time_split_column not in df.columns:
            raise ValueError(
                f"time_split_column={time_split_column!r} not in DataFrame"
            )
        years = df[time_split_column].apply(_year_of).values
        train_mask = np.array([yr is not None and yr <= time_split_year for yr in years])
        test_mask = np.array([yr is not None and yr > time_split_year for yr in years])
        n_skip = len(df) - train_mask.sum() - test_mask.sum()
        train_idx = np.flatnonzero(train_mask)
        test_idx = np.flatnonzero(test_mask)
        train_y = y[train_idx]
        test_y = y[test_idx]
        logger.info(
            "split: temporal on %s; cutoff=%d; train=%d (pos=%d, %.1f%%) test=%d (pos=%d, %.1f%%) skipped_no_year=%d",
            time_split_column,
            time_split_year,
            len(train_idx),
            int(train_y.sum()),
            100.0 * train_y.mean() if len(train_y) else 0.0,
            len(test_idx),
            int(test_y.sum()),
            100.0 * test_y.mean() if len(test_y) else 0.0,
            n_skip,
        )
        if len(train_idx) == 0 or len(test_idx) == 0:
            raise ValueError(
                f"temporal split with cutoff={time_split_year} produced empty "
                f"train ({len(train_idx)}) or test ({len(test_idx)}) — "
                f"check the year distribution of {time_split_column}"
            )
        return train_idx, test_idx

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


def split_with_calibration(
    df: pd.DataFrame,
    *,
    calibration_year: int,
    time_split_column: str = "earliest_start_date",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (train_idx, calib_idx, test_idx) for a three-way temporal slice.

    train = year <= calibration_year - 1
    calibrate = year == calibration_year
    test = year > calibration_year
    Rows with a missing year are skipped from all three slices.
    """
    if not time_split_column or time_split_column not in df.columns:
        raise ValueError(f"time_split_column={time_split_column!r} not in DataFrame")
    y = df["y"].values
    years = df[time_split_column].apply(_year_of).values
    train_mask = np.array([yr is not None and yr <= calibration_year - 1 for yr in years])
    calib_mask = np.array([yr is not None and yr == calibration_year for yr in years])
    test_mask = np.array([yr is not None and yr > calibration_year for yr in years])
    n_skip = len(df) - train_mask.sum() - calib_mask.sum() - test_mask.sum()
    train_idx = np.flatnonzero(train_mask)
    calib_idx = np.flatnonzero(calib_mask)
    test_idx = np.flatnonzero(test_mask)
    logger.info(
        "split: 3-way temporal on %s; calibration_year=%d; "
        "train=%d (pos=%d, %.1f%%) calib=%d (pos=%d, %.1f%%) test=%d (pos=%d, %.1f%%) skipped_no_year=%d",
        time_split_column,
        calibration_year,
        len(train_idx),
        int(y[train_idx].sum()),
        100.0 * y[train_idx].mean() if len(train_idx) else 0.0,
        len(calib_idx),
        int(y[calib_idx].sum()),
        100.0 * y[calib_idx].mean() if len(calib_idx) else 0.0,
        len(test_idx),
        int(y[test_idx].sum()),
        100.0 * y[test_idx].mean() if len(test_idx) else 0.0,
        n_skip,
    )
    if len(train_idx) == 0 or len(calib_idx) == 0 or len(test_idx) == 0:
        raise ValueError(
            f"3-way temporal split with calibration_year={calibration_year} produced empty slice — "
            f"train={len(train_idx)} calib={len(calib_idx)} test={len(test_idx)}; "
            f"check the year distribution of {time_split_column}"
        )
    return train_idx, calib_idx, test_idx
