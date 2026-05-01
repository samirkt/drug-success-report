"""Build ECFP4 + MACCS fingerprints for the modeling-eligible candidates.

Reads ``candidate_detail.parquet`` (the per-candidate analytical snapshot
written by the pipeline), filters to rows with a non-null
``smiles_canonical``, and emits a Parquet keyed by ``candidate_id`` with
two structure-derived bitvector fingerprints.

Pure-CPU; only RDKit is required (already a core pipeline dep, so no
optional extra needed).

Usage:

    python scripts/build_fingerprints.py \
        --candidates docs/candidate_detail.parquet \
        --output docs/features/fingerprints.parquet

Output schema (one row per candidate with non-null canonical SMILES):

    candidate_id : str
    ecfp4        : list[uint8]   length 2048  (Morgan radius=2, 2048 bits)
    maccs        : list[uint8]   length 167   (RDKit MACCS keys)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger("build_fingerprints")


def _compute_fingerprints(smiles: str):
    """Return (ecfp4_uint8_list, maccs_uint8_list) or (None, None) on failure."""
    from rdkit import Chem
    from rdkit.Chem import AllChem, MACCSkeys
    import numpy as np

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None
    ecfp4_bv = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    maccs_bv = MACCSkeys.GenMACCSKeys(mol)
    ecfp4 = np.array(ecfp4_bv, dtype=np.uint8).tolist()
    maccs = np.array(maccs_bv, dtype=np.uint8).tolist()
    return ecfp4, maccs


def build(candidates_parquet: Path, output_parquet: Path) -> int:
    try:
        import pandas as pd
    except ImportError as exc:
        raise SystemExit(f"Missing dep: {exc.name}") from exc
    try:
        from rdkit import Chem  # noqa: F401  (probe import)
        from rdkit import RDLogger
    except ImportError as exc:
        raise SystemExit(
            "rdkit is required. Install with: pip install rdkit"
        ) from exc
    RDLogger.DisableLog("rdApp.*")

    if not candidates_parquet.exists():
        raise SystemExit(f"--candidates {candidates_parquet} does not exist.")

    df = pd.read_parquet(candidates_parquet)
    if "smiles_canonical" not in df.columns:
        raise SystemExit(
            f"{candidates_parquet} has no `smiles_canonical` column. Re-run "
            "the pipeline with smiles standardization enabled."
        )
    rows = df[df["smiles_canonical"].notna()][["candidate_id", "smiles_canonical"]]
    rows = rows.reset_index(drop=True)
    n_in = len(rows)
    logger.info(
        "Fingerprinting %d / %d candidates with non-null canonical SMILES",
        n_in, len(df),
    )

    candidate_ids: list[str] = []
    ecfp4_col: list[list[int]] = []
    maccs_col: list[list[int]] = []
    n_failed = 0

    for cid, smiles in zip(rows["candidate_id"], rows["smiles_canonical"]):
        ecfp4, maccs = _compute_fingerprints(smiles)
        if ecfp4 is None:
            # Should not happen post-standardization, but defend against
            # the edge case where a SMILES round-trips through the
            # standardizer in a form RDKit then re-parses differently.
            n_failed += 1
            logger.warning("Failed to fingerprint candidate_id=%s smiles=%r", cid, smiles)
            continue
        candidate_ids.append(cid)
        ecfp4_col.append(ecfp4)
        maccs_col.append(maccs)

    out = pd.DataFrame({
        "candidate_id": candidate_ids,
        "ecfp4": ecfp4_col,
        "maccs": maccs_col,
    })
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_parquet, index=False)
    logger.info(
        "Wrote %d fingerprint rows -> %s (%d failed re-parse)",
        len(out), output_parquet, n_failed,
    )
    return len(out)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    parser = argparse.ArgumentParser(
        description=(
            "Compute ECFP4 (Morgan r=2, 2048 bits) + MACCS (167 bits) "
            "fingerprints for candidates with non-null smiles_canonical."
        )
    )
    parser.add_argument(
        "--candidates",
        type=Path,
        required=True,
        help="Path to candidate_detail.parquet from a pipeline run.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Where to write fingerprints.parquet.",
    )
    args = parser.parse_args(argv)

    try:
        build(candidates_parquet=args.candidates, output_parquet=args.output)
    except SystemExit:
        raise
    except Exception as exc:
        logger.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
