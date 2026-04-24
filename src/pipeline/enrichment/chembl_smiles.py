"""ChEMBL SMILES fallback — fills `cand.smiles` when DrugBank left it empty.

Reads the same ChEMBL snapshot used by `TargetsEnrichment` (built by
`scripts/build_chembl_targets_snapshot.py`). Lookup is name-keyed via
`canonicalize_drug_name(drug_name_raw)`, matching the normalization the
targets stage uses. Runs **after** `SmilesEnrichment` and only writes to
candidates where `cand.smiles is None`, so DrugBank always wins when it
has a string and ChEMBL fills the biologic-shaped gap (peptides,
antibodies, etc.) that DrugBank's `canonical-smiles` mostly omits.

The stage records two coverage lines on top of the DrugBank stage's
`smiles` record:

- ``smiles_chembl`` — how many candidates ChEMBL filled *that DrugBank
  had left empty*. This is the fallback's own contribution.
- ``smiles_combined`` — how many candidates end up with *any* SMILES
  string after both stages have run. This is the user-visible total
  coverage headline.

Old snapshots built before the `canonical_smiles` column landed still
open cleanly: the stage notes the missing column and degrades to empty
coverage rather than raising.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING

from ..drugbank_norm import canonicalize_drug_name
from ..models import CandidateTable

if TYPE_CHECKING:
    from ..pipeline import PipelineConfig
    from utils.tiered_router import CostLedger

logger = logging.getLogger(__name__)


class ChemblSmilesEnrichment:
    """Fill `Candidate.smiles` from ChEMBL when DrugBank left it empty."""

    name = "smiles_chembl"

    def __init__(self) -> None:
        self._lookup: dict[str, str] | None = None
        self._source_path: Path | None = None
        self._release: int | None = None
        self._has_smiles_column: bool = False

    def is_available(self, config: "PipelineConfig") -> bool:
        if not config.enable_chembl_smiles:
            return False
        path = config.chembl_snapshot_path
        if path is None:
            logger.info(
                "ChemblSmilesEnrichment: skipped — no chembl_snapshot_path configured."
            )
            return False
        if not Path(path).exists():
            logger.warning(
                "ChemblSmilesEnrichment: skipped — chembl_snapshot_path %s not found. "
                "Build one with scripts/build_chembl_targets_snapshot.py.",
                path,
            )
            return False
        self._source_path = Path(path)
        return True

    def _build_lookup(self) -> dict[str, str]:
        if self._lookup is not None:
            return self._lookup
        assert self._source_path is not None  # guarded by is_available

        conn = sqlite3.connect(f"file:{self._source_path}?mode=ro", uri=True)
        try:
            try:
                self._release = conn.execute(
                    "PRAGMA user_version"
                ).fetchone()[0]
            except sqlite3.DatabaseError:
                self._release = None

            columns = {
                row[1] for row in conn.execute("PRAGMA table_info(name_targets)")
            }
            if "canonical_smiles" not in columns:
                logger.warning(
                    "ChemblSmilesEnrichment: snapshot at %s has no "
                    "canonical_smiles column — rebuild it with the latest "
                    "scripts/build_chembl_targets_snapshot.py to enable the "
                    "ChEMBL SMILES fallback. Continuing with empty coverage.",
                    self._source_path,
                )
                self._has_smiles_column = False
                self._lookup = {}
                return self._lookup

            self._has_smiles_column = True
            lookup: dict[str, str] = {}
            cur = conn.execute(
                "SELECT query_norm, canonical_smiles FROM name_targets "
                "WHERE canonical_smiles IS NOT NULL AND canonical_smiles <> ''"
            )
            for query_norm, smiles in cur:
                if not query_norm:
                    continue
                key = str(query_norm).strip()
                if not key:
                    continue
                value = str(smiles).strip()
                if not value:
                    continue
                # First-writer-wins — deterministic without forcing a sort
                # over an entire snapshot that may have tens of thousands
                # of rows per name.
                lookup.setdefault(key, value)
        finally:
            conn.close()

        self._lookup = lookup
        logger.info(
            "ChemblSmilesEnrichment: loaded %d normalized-name SMILES rows "
            "from %s (ChEMBL release %s)",
            len(self._lookup),
            self._source_path,
            self._release if self._release else "unknown",
        )
        return self._lookup

    def run(
        self,
        candidates: CandidateTable,
        *,
        ledger: "CostLedger",
    ) -> CandidateTable:
        lookup = self._build_lookup()
        total = len(candidates.candidates)
        enriched_chembl = 0
        for cand in candidates.candidates:
            if cand.smiles:
                continue
            key = canonicalize_drug_name(cand.drug_name_raw or "")
            if not key:
                continue
            smiles = lookup.get(key)
            if smiles:
                cand.smiles = smiles
                enriched_chembl += 1

        total_with_smiles = sum(1 for c in candidates.candidates if c.smiles)

        pct_chembl = (100.0 * enriched_chembl / total) if total > 0 else 0.0
        pct_combined = (100.0 * total_with_smiles / total) if total > 0 else 0.0
        logger.info(
            "ChemblSmilesEnrichment: filled %d/%d previously-missing SMILES (%.0f%%). "
            "Combined SMILES coverage: %d/%d (%.0f%%).",
            enriched_chembl, total, pct_chembl,
            total_with_smiles, total, pct_combined,
        )
        ledger.record_coverage("smiles_chembl", enriched_chembl, total)
        ledger.record_coverage("smiles_combined", total_with_smiles, total)
        return candidates


__all__ = ["ChemblSmilesEnrichment"]
