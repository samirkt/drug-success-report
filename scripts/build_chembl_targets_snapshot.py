"""Build a name-keyed ChEMBL drug-target snapshot for the enrichment pipeline.

Reads the full ChEMBL SQLite release and writes a slim indexed SQLite file
that `TargetsEnrichment` can open read-only and look up by normalized drug
name. No UniChem crosswalk, no DrugBank dependency — any candidate whose
name (after `canonicalize_drug_name`) matches a ChEMBL preferred-name or
synonym gets its mechanism-of-action targets.

This script is **not** part of the pipeline — it runs once per ChEMBL
release.

Inputs

    --chembl-sqlite   Path to the ChEMBL SQLite `.db` (e.g. chembl_35.db,
                      ~5 GB uncompressed, from ChEMBL's FTP).

Output

    --out             Path to write the name-keyed snapshot SQLite. One
                      table:

        name_targets(query_norm         TEXT NOT NULL,
                     source_name        TEXT NOT NULL,
                     source_kind        TEXT NOT NULL,   -- 'pref_name' | 'synonym'
                     syn_type           TEXT,
                     chembl_id          TEXT NOT NULL,
                     target_chembl_id   TEXT,
                     target_pref_name   TEXT,
                     target_type        TEXT,
                     uniprot_accession  TEXT,            -- pipe-joined
                     action_type        TEXT,
                     canonical_smiles   TEXT)            -- from molecule_dictionary

        CREATE INDEX idx_query_norm ON name_targets(query_norm);

    PRAGMA user_version is stamped with the ChEMBL release number parsed
    from the input filename (best-effort) so the pipeline can log which
    release it is using.

Download hint (as of 2026)

    ChEMBL SQLite:
      https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/chembl_35_sqlite.tar.gz
"""

from __future__ import annotations

import argparse
import logging
import re
import sqlite3
import sys
from pathlib import Path

# Make `pipeline.drugbank_norm.canonicalize_drug_name` importable from a
# standalone script that lives in scripts/ rather than inside the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from pipeline.drugbank_norm import canonicalize_drug_name  # noqa: E402

logger = logging.getLogger("build_chembl_targets_snapshot")


def parse_release_number(chembl_db_path: Path) -> int:
    """Best-effort parse of the ChEMBL release from its filename (e.g. 35)."""
    m = re.search(r"chembl[_-]?(\d+)", chembl_db_path.name, re.IGNORECASE)
    return int(m.group(1)) if m else 0


# Single streamed query: build the drug-target base (gated by
# `drug_mechanism`), then UNION the preferred-name and synonym rows so each
# output row is (source_name, source_kind, syn_type, target columns…).
# UniProt accessions are pipe-joined at aggregation time to avoid colliding
# with any pref_name that contains a comma.
_CHEMBL_QUERY = """
WITH drug_targets AS (
    SELECT
        dm.molregno                        AS molregno,
        md.chembl_id                       AS chembl_id,
        -- canonical_smiles lives on compound_structures, not
        -- molecule_dictionary. Biologics often have no compound_structures
        -- row at all, so the LEFT JOIN is load-bearing.
        comps.canonical_smiles             AS canonical_smiles,
        td.chembl_id                       AS target_chembl_id,
        td.pref_name                       AS target_pref_name,
        td.target_type                     AS target_type,
        REPLACE(
            GROUP_CONCAT(DISTINCT cs.accession),
            ',', '|'
        )                                  AS uniprot_accession,
        dm.action_type                     AS action_type
    FROM drug_mechanism dm
    JOIN molecule_dictionary md   ON md.molregno = dm.molregno
    LEFT JOIN compound_structures comps ON comps.molregno = dm.molregno
    JOIN target_dictionary   td   ON td.tid      = dm.tid
    LEFT JOIN target_components tc ON tc.tid = td.tid
    LEFT JOIN component_sequences cs ON cs.component_id = tc.component_id
    GROUP BY dm.molregno, md.chembl_id, comps.canonical_smiles,
             td.chembl_id, td.pref_name, td.target_type, dm.action_type
)
SELECT md.pref_name  AS source_name,
       'pref_name'   AS source_kind,
       NULL          AS syn_type,
       dt.chembl_id,
       dt.target_chembl_id,
       dt.target_pref_name,
       dt.target_type,
       dt.uniprot_accession,
       dt.action_type,
       dt.canonical_smiles
FROM drug_targets dt
JOIN molecule_dictionary md ON md.molregno = dt.molregno
WHERE md.pref_name IS NOT NULL AND md.pref_name <> ''

UNION ALL

SELECT ms.synonyms   AS source_name,
       'synonym'     AS source_kind,
       ms.syn_type   AS syn_type,
       dt.chembl_id,
       dt.target_chembl_id,
       dt.target_pref_name,
       dt.target_type,
       dt.uniprot_accession,
       dt.action_type,
       dt.canonical_smiles
FROM drug_targets dt
JOIN molecule_synonyms ms ON ms.molregno = dt.molregno
WHERE ms.synonyms IS NOT NULL AND ms.synonyms <> ''
"""


def build_snapshot(chembl_db: Path, out_db: Path) -> None:
    """Stream ChEMBL drug-target rows, normalize names, write slim SQLite."""
    if out_db.exists():
        out_db.unlink()

    logger.info("Opening ChEMBL SQLite (read-only): %s", chembl_db)
    src = sqlite3.connect(f"file:{chembl_db}?mode=ro", uri=True)
    try:
        logger.info("Creating output snapshot: %s", out_db)
        dst = sqlite3.connect(out_db)
        try:
            dst.execute("PRAGMA journal_mode=OFF")
            dst.execute("PRAGMA synchronous=OFF")
            dst.execute(
                """
                CREATE TABLE name_targets (
                    query_norm          TEXT NOT NULL,
                    source_name         TEXT NOT NULL,
                    source_kind         TEXT NOT NULL,
                    syn_type            TEXT,
                    chembl_id           TEXT NOT NULL,
                    target_chembl_id    TEXT,
                    target_pref_name    TEXT,
                    target_type         TEXT,
                    uniprot_accession   TEXT,
                    action_type         TEXT,
                    canonical_smiles    TEXT
                )
                """
            )

            rows_in = 0
            rows_out = 0
            batch: list[tuple] = []
            BATCH_SIZE = 5000

            cur = src.execute(_CHEMBL_QUERY)
            for (
                source_name,
                source_kind,
                syn_type,
                chembl_id,
                target_chembl_id,
                target_pref_name,
                target_type,
                uniprot_accession,
                action_type,
                canonical_smiles,
            ) in cur:
                rows_in += 1
                query_norm = canonicalize_drug_name(source_name or "")
                if not query_norm:
                    continue
                batch.append(
                    (
                        query_norm,
                        source_name,
                        source_kind,
                        syn_type,
                        chembl_id,
                        target_chembl_id,
                        target_pref_name,
                        target_type,
                        uniprot_accession or "",
                        action_type,
                        canonical_smiles,
                    )
                )
                rows_out += 1
                if len(batch) >= BATCH_SIZE:
                    dst.executemany(
                        "INSERT INTO name_targets VALUES "
                        "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        batch,
                    )
                    batch.clear()
            if batch:
                dst.executemany(
                    "INSERT INTO name_targets VALUES "
                    "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    batch,
                )

            dst.execute("CREATE INDEX idx_query_norm ON name_targets(query_norm)")
            release = parse_release_number(chembl_db)
            dst.execute(f"PRAGMA user_version = {release}")
            dst.commit()

            n_drugs = dst.execute(
                "SELECT COUNT(DISTINCT chembl_id) FROM name_targets"
            ).fetchone()[0]
            n_norms = dst.execute(
                "SELECT COUNT(DISTINCT query_norm) FROM name_targets"
            ).fetchone()[0]
            logger.info(
                "Wrote %d rows covering %d distinct ChEMBL drugs and %d "
                "distinct normalized names (skipped %d unnormalizable rows).",
                rows_out, n_drugs, n_norms, rows_in - rows_out,
            )
            logger.info("ChEMBL release stamp: %d", release)
        finally:
            dst.close()
    finally:
        src.close()


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    parser = argparse.ArgumentParser(
        description="Build a slim name-keyed ChEMBL targets snapshot."
    )
    parser.add_argument(
        "--chembl-sqlite",
        type=Path,
        required=True,
        help="Path to the ChEMBL SQLite release (e.g. chembl_35.db).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output path for the slim targets snapshot SQLite.",
    )
    args = parser.parse_args(argv)

    if not args.chembl_sqlite.exists():
        parser.error(f"--chembl-sqlite path does not exist: {args.chembl_sqlite}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    build_snapshot(args.chembl_sqlite, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
