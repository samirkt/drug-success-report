"""Assemble a HINT-shaped CSV from candidate_detail.parquet + trial_detail.parquet.

HINT's loader reads positional columns 0,3,6,7,8,9 = nctid, label,
icdcodes, drugs, smiless, criteria. Columns 1,2,4,5 (status, why_stop,
phase, diseases) are read positionally but unused at training time.

Drugs / smiless / icdcodes / diseases get wrapped as single-element
Python lists and stringified, since HINT stores those columns as
``str(list[str])`` (ast.literal_eval at load time).

Rows are dropped when:
    - smiles_canonical is null  (HINT's MPNN needs a parseable SMILES)
    - trial_inferred_label is null  (no training signal)
    - icd10_codes is null/empty  (HINT's GRAM encoder is ICD-keyed)
    - trial_eligibility_criteria is null/empty  (one of HINT's three encoders)

Usage:

    uv run python scripts/build_hint_dataset.py \\
        --candidates docs/candidate_detail.parquet \\
        --trials     docs/trial_detail.parquet \\
        --output     docs/features/hint_dataset.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger("build_hint_dataset")

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

    cand_cols_needed = ["candidate_id", "drug_name", "smiles_canonical", "icd10_codes", "indication"]
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
        candidates[["candidate_id", "drug_name", "smiles_canonical", "icd10_codes", "indication"]],
        on="candidate_id",
        how="left",
        suffixes=("", "_cand"),
    )
    raw_n = len(df)
    logger.info("raw joined rows: %d", raw_n)

    df = df[df["smiles_canonical"].notna() & (df["smiles_canonical"].astype(str).str.len() > 0)]
    logger.info("after non-null smiles_canonical: %d", len(df))

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
        "smiless":  df["smiles_canonical"].apply(lambda x: str([x])),
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
    args = parser.parse_args(argv)

    try:
        build(args.candidates, args.trials, args.output)
    except SystemExit:
        raise
    except Exception as exc:
        logger.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
