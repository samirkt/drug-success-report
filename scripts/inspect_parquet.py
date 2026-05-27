"""Inspect coverage of molecular fingerprint stores over candidate_detail.

Reports how many candidates in outputs/candidate_detail.parquet have entries
in the per-candidate fingerprint stores under outputs/features/, and breaks
the gap down by attributes that explain it (SMILES presence, standardization
status, highest phase, modality).
"""

from pathlib import Path

import pandas as pd

OUTPUTS = Path("outputs")
CANDIDATE_PATH = OUTPUTS / "candidate_detail.parquet"
FEATURE_STORES = {
    "ecfp4 / maccs": OUTPUTS / "features" / "fingerprints.parquet",
    "molformer": OUTPUTS / "features" / "molformer_embeddings.parquet",
}


def pct(n: int, d: int) -> str:
    return f"{n}/{d} ({n / d:.1%})" if d else f"{n}/0 (n/a)"


def coverage_by(df: pd.DataFrame, group_col: str, has_fp: pd.Series) -> pd.DataFrame:
    grouped = df.assign(_has_fp=has_fp).groupby(group_col, dropna=False)
    out = grouped["_has_fp"].agg(["sum", "count"]).rename(
        columns={"sum": "with_fp", "count": "rows"}
    )
    out["coverage"] = out["with_fp"] / out["rows"]
    return out.sort_values("rows", ascending=False)


def main() -> None:
    df = pd.read_parquet(CANDIDATE_PATH)
    n = len(df)
    print(f"candidate_detail rows: {n}")

    has_smiles = df["smiles_canonical"].fillna("").str.len() > 0
    print(f"with canonical SMILES: {pct(int(has_smiles.sum()), n)}")
    print()

    coverage_by_store: dict[str, pd.Series] = {}
    for label, path in FEATURE_STORES.items():
        if not path.exists():
            print(f"[{label}] MISSING file: {path}")
            print()
            continue

        fp = pd.read_parquet(path, columns=["candidate_id"])
        fp_ids = set(fp["candidate_id"].dropna().unique())
        has_fp = df["candidate_id"].isin(fp_ids)
        coverage_by_store[label] = has_fp

        print(f"[{label}]  source: {path}")
        print(f"  store rows: {len(fp)}  unique ids: {len(fp_ids)}")
        print(f"  candidates with fp: {pct(int(has_fp.sum()), n)}")
        print(
            "  conditional on SMILES: "
            f"{pct(int((has_fp & has_smiles).sum()), int(has_smiles.sum()))}"
        )
        missing_with_smiles = int((~has_fp & has_smiles).sum())
        print(f"  missing fp despite having SMILES: {missing_with_smiles}")

        orphans = fp_ids - set(df["candidate_id"])
        if orphans:
            print(f"  store ids not in candidate_detail: {len(orphans)}")
        print()

    if {"ecfp4 / maccs", "molformer"} <= coverage_by_store.keys():
        ecfp = coverage_by_store["ecfp4 / maccs"]
        mol = coverage_by_store["molformer"]
        print("[overlap]")
        print(f"  both fp + molformer: {pct(int((ecfp & mol).sum()), n)}")
        print(f"  fp only:             {pct(int((ecfp & ~mol).sum()), n)}")
        print(f"  molformer only:      {pct(int((~ecfp & mol).sum()), n)}")
        print(f"  neither:             {pct(int((~ecfp & ~mol).sum()), n)}")
        print()

    if "ecfp4 / maccs" in coverage_by_store:
        has_fp = coverage_by_store["ecfp4 / maccs"]
        for col in ("smiles_standardization_status", "highest_phase", "modality"):
            if col not in df.columns:
                continue
            print(f"[ecfp4 / maccs coverage by {col}]")
            tbl = coverage_by(df, col, has_fp).head(10)
            print(tbl.to_string(formatters={"coverage": "{:.1%}".format}))
            print()

        print("[sample candidates with SMILES but no fingerprint]")
        gap = df.loc[~has_fp & has_smiles, [
            "candidate_id", "drug_name", "smiles_standardization_status",
        ]]
        print(gap.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
