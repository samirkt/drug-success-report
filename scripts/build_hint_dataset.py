"""Assemble a HINT-shaped CSV from candidate_detail.parquet + trial_detail.parquet.

HINT's loader reads positional columns 0,3,6,7,8,9 = nctid, label,
icdcodes, drugs, smiless, criteria. Columns 1,2,4,5 (status, why_stop,
phase, diseases) are read positionally but unused at training time.

Drugs / smiless / icdcodes / diseases get wrapped as single-element
Python lists and stringified, since HINT stores those columns as
``str(list[str])`` (ast.literal_eval at load time).

Rows are dropped when:
    - the configured SMILES column is null  (HINT's MPNN needs a parseable SMILES)
    - trial_inferred_label is null  (no training signal)
    - icd10_codes is null/empty  (HINT's GRAM encoder is ICD-keyed)
    - trial_eligibility_criteria is null/empty  (one of HINT's three encoders)

HINT trains a separate model per phase. Use ``--phase {1,2,3}`` to
emit a phase-specific dataset; ``--phase all`` (default) keeps every
phase mixed (useful for inspection, not for training a per-phase model).

Label semantics (computed during the pipeline run as
``trial_inferred_label``):

    - status TERMINATED/WITHDRAWN/SUSPENDED      -> 0
    - candidate APPROVED/COMMERCIALIZED          -> 1
    - candidate FAILED_PHASE_N:
        * trial.phase  < N                       -> 1  (drug advanced past)
        * trial.phase == N                       -> 0  (drug stopped here)
        * trial.phase  > N                       -> dropped
    - candidate ONGOING/UNKNOWN                  -> dropped

So for ``--phase 2``: a Phase 2 trial with FAILED_PHASE_2 is a 0;
with APPROVED or FAILED_PHASE_3 is a 1. Same shape HINT expects.

Usage:

    # Per-phase HINT datasets
    uv run python scripts/build_hint_dataset.py \\
        --candidates docs/candidate_detail.parquet \\
        --trials     docs/trial_detail.parquet \\
        --phase      2 \\
        --output     docs/features/hint_phase2.csv

    # Mixed-phase dataset (inspection only)
    uv run python scripts/build_hint_dataset.py \\
        --candidates docs/candidate_detail.parquet \\
        --trials     docs/trial_detail.parquet \\
        --output     docs/features/hint_all_phases.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger("build_hint_dataset")

SMILES_COLUMN = "smiles"

HINT_COLUMNS = [
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
_PHASE_FILTER = {
    "1": "Phase 1",
    "2": "Phase 2",
    "3": "Phase 3",
    "4": "Phase 4",
}


def _is_nonempty_list(value) -> bool:
    """True iff `value` is a list-like with at least one truthy entry."""
    if value is None:
        return False
    try:
        seq = list(value)
    except TypeError:
        return False
    return any(seq)


def build(
    candidates_parquet: Path,
    trials_parquet: Path,
    output_csv: Path,
    phase: str = "all",
) -> int:
    try:
        import pandas as pd
    except ImportError as exc:
        raise SystemExit(f"Missing dep ({exc.name}). Install with: pip install pandas") from exc

    if not candidates_parquet.exists():
        raise SystemExit(f"--candidates {candidates_parquet} does not exist.")
    if not trials_parquet.exists():
        raise SystemExit(f"--trials {trials_parquet} does not exist.")

    candidates = pd.read_parquet(candidates_parquet)
    trials = pd.read_parquet(trials_parquet)

    cand_cols_needed = ["candidate_id", "drug_name", SMILES_COLUMN, "icd10_codes", "indication"]
    missing = [c for c in cand_cols_needed if c not in candidates.columns]
    if missing:
        raise SystemExit(
            f"{candidates_parquet} is missing columns {missing}. "
            "Re-run the pipeline with --enable-icd10 and SMILES standardization on."
        )
    trial_cols_needed = [
        "nct_id", "candidate_id", "trial_phase", "trial_status",
        "trial_why_stopped", "trial_eligibility_criteria", "trial_inferred_label",
    ]
    missing = [c for c in trial_cols_needed if c not in trials.columns]
    if missing:
        raise SystemExit(
            f"{trials_parquet} is missing columns {missing}. "
            "Re-run the pipeline against the latest writer."
        )

    df = trials.merge(
        candidates[["candidate_id", "drug_name", SMILES_COLUMN, "icd10_codes", "indication"]],
        on="candidate_id",
        how="left",
        suffixes=("", "_cand"),
    )
    raw_n = len(df)
    logger.info("raw joined rows: %d", raw_n)

    if phase != "all":
        target = _PHASE_FILTER.get(phase)
        if target is None:
            raise SystemExit(f"--phase {phase!r} not recognized. Use 1, 2, 3, 4, or all.")
        df = df[df["trial_phase"] == target]
        logger.info("after --phase %s filter (%s only): %d", phase, target, len(df))

    df = df[df[SMILES_COLUMN].notna() & (df[SMILES_COLUMN].astype(str).str.len() > 0)]
    logger.info("after non-null %s: %d", SMILES_COLUMN, len(df))

    df = df[df["trial_inferred_label"].notna()]
    logger.info("after non-null inferred_label: %d", len(df))

    df = df[df["icd10_codes"].apply(_is_nonempty_list)]
    logger.info("after non-empty icd10_codes: %d", len(df))

    df = df[df["trial_eligibility_criteria"].notna() & (df["trial_eligibility_criteria"].astype(str).str.len() > 0)]
    logger.info("after non-empty eligibility_criteria: %d", len(df))

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

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    logger.info("Wrote %d HINT-shaped rows -> %s", len(out), output_csv)
    return len(out)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(
        description=(
            "Emit a HINT-shaped CSV by joining candidate_detail.parquet "
            "with trial_detail.parquet."
        )
    )
    parser.add_argument("--candidates", type=Path, required=True,
                        help="Path to candidate_detail.parquet from a pipeline run.")
    parser.add_argument("--trials", type=Path, required=True,
                        help="Path to trial_detail.parquet from the same run.")
    parser.add_argument("--output", type=Path, required=True,
                        help="Where to write the HINT-shaped CSV.")
    parser.add_argument(
        "--phase",
        choices=["1", "2", "3", "4", "all"],
        default="all",
        help=(
            "HINT trains separate models per phase. Pass 1/2/3 to emit "
            "a phase-specific dataset (only trials whose trial_phase "
            "matches that phase). Default 'all' keeps every phase "
            "mixed (useful for inspection; not what you train on)."
        ),
    )
    args = parser.parse_args(argv)

    try:
        build(args.candidates, args.trials, args.output, phase=args.phase)
    except SystemExit:
        raise
    except Exception as exc:
        logger.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
