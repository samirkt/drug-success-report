"""HINT-format conversion for a candidate-joined trial frame.

HINT's loader (`HINT/dataloader.py`) reads positional CSV columns
0,3,6,7,8,9 = nctid, label, icdcodes, drugs, smiless, criteria. Columns
1,2,4,5 (status, why_stop, phase, diseases) are read positionally but
unused at training time, so they're emitted as fillers.

This module is the canonical home for the row-filter + reshape logic.
Both the standalone `scripts/build_hint_dataset.py` CLI and the model
trainer (when running in trial granularity) call :func:`to_hint_frame`
to convert an in-memory DataFrame into HINT's 10-column shape.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

SMILES_COLUMN = "smiles"

HINT_COLUMNS: list[str] = [
    "nctid",       # 0
    "status",      # 1 (filler)
    "why_stop",    # 2 (filler)
    "label",       # 3
    "phase",       # 4 (filler)
    "diseases",    # 5
    "icdcodes",    # 6
    "drugs",       # 7
    "smiless",     # 8
    "criteria",    # 9
]

# Map "1"/"2"/"3"/"4" to the canonical phase string used in trial_detail.parquet.
PHASE_FILTER: dict[str, str] = {
    "1": "Phase 1",
    "2": "Phase 2",
    "3": "Phase 3",
    "4": "Phase 4",
}


def _is_nonempty_list(value: Any) -> bool:
    """True iff `value` is a list-like with at least one truthy entry."""
    if value is None:
        return False
    try:
        seq = list(value)
    except TypeError:
        return False
    return any(seq)


def to_hint_frame(df: pd.DataFrame, *, phase: str = "all") -> pd.DataFrame:
    """Filter + reshape a candidate-joined trial frame into HINT's 10-column format.

    `df` must carry these columns: ``nct_id``, ``trial_phase``,
    ``trial_status``, ``trial_why_stopped``, ``trial_inferred_label``,
    ``trial_eligibility_criteria``, ``drug_name``, ``smiles``,
    ``icd10_codes``, ``indication``.

    Filters (matching the standalone build script):
      - drop rows with null/empty `smiles` (HINT's MPNN needs a parseable SMILES)
      - drop rows with null `trial_inferred_label` (no training signal)
      - drop rows with empty `icd10_codes` (HINT's GRAM encoder is ICD-keyed)
      - drop rows with null/empty `trial_eligibility_criteria` (one of HINT's three encoders)

    `phase` accepts "1", "2", "3", "4", or "all" (default). Anything
    other than "all" restricts to a single `trial_phase` value.
    """
    if phase != "all":
        target = PHASE_FILTER.get(phase)
        if target is None:
            raise ValueError(f"phase={phase!r} not recognized. Use 1, 2, 3, 4, or all.")
        df = df[df["trial_phase"] == target]
        logger.info("HINT phase filter %s (%s only): %d rows", phase, target, len(df))

    n = len(df)
    df = df[df[SMILES_COLUMN].notna() & (df[SMILES_COLUMN].astype(str).str.len() > 0)]
    logger.info("HINT filter: non-null %s -> %d (dropped %d)", SMILES_COLUMN, len(df), n - len(df))

    n = len(df)
    df = df[df["trial_inferred_label"].notna()]
    logger.info("HINT filter: non-null trial_inferred_label -> %d (dropped %d)", len(df), n - len(df))

    n = len(df)
    df = df[df["icd10_codes"].apply(_is_nonempty_list)]
    logger.info("HINT filter: non-empty icd10_codes -> %d (dropped %d)", len(df), n - len(df))

    n = len(df)
    df = df[
        df["trial_eligibility_criteria"].notna()
        & (df["trial_eligibility_criteria"].astype(str).str.len() > 0)
    ]
    logger.info(
        "HINT filter: non-empty trial_eligibility_criteria -> %d (dropped %d)",
        len(df),
        n - len(df),
    )

    out = pd.DataFrame({
        "nctid":    df["nct_id"].astype(str),
        "status":   df["trial_status"].fillna("").astype(str),
        "why_stop": df["trial_why_stopped"].fillna("").astype(str),
        "label":    df["trial_inferred_label"].astype(int),
        "phase":    df["trial_phase"].fillna("").astype(str),
        "diseases": df["indication"].apply(lambda x: str([x] if x else [])),
        "icdcodes": df["icd10_codes"].apply(lambda x: str(list(x))),
        "drugs":    df["drug_name"].apply(lambda x: str([x] if x else [])),
        "smiless":  df[SMILES_COLUMN].apply(lambda x: str([x])),
        "criteria": df["trial_eligibility_criteria"].astype(str),
    })
    return out
