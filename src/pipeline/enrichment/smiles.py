"""SMILES enrichment — attaches DrugBank-derived SMILES to candidates.

Reuses `drugbank_norm.load_drugbank_lookup` so the exact same DataFrame
that drives DrugBank ID matching also supplies SMILES. No network calls;
the join is a dict lookup keyed on `drugbank_id` resolved during
clustering.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import pandas as pd

from ..drugbank_norm import load_drugbank_lookup
from ..models import CandidateTable

if TYPE_CHECKING:
    from ..pipeline import PipelineConfig
    from utils.tiered_router import CostLedger

logger = logging.getLogger(__name__)


class SmilesEnrichment:
    """Join DrugBank SMILES onto candidates by `drugbank_id`."""

    name = "smiles"

    def __init__(self) -> None:
        self._lookup: dict[str, str] | None = None
        self._source_path: Path | None = None

    def is_available(self, config: "PipelineConfig") -> bool:
        if not config.enable_smiles:
            return False
        path = config.drugbank_csv_path
        if path is None:
            logger.info(
                "SmilesEnrichment: skipped — no drugbank_csv_path configured."
            )
            return False
        if not Path(path).exists():
            logger.warning(
                "SmilesEnrichment: skipped — drugbank_csv_path %s not found.",
                path,
            )
            return False
        self._source_path = Path(path)
        return True

    def _build_lookup(self) -> dict[str, str]:
        if self._lookup is not None:
            return self._lookup
        assert self._source_path is not None  # guarded by is_available

        best_rows, _ = load_drugbank_lookup(self._source_path)
        if "smiles" not in best_rows.columns:
            logger.warning(
                "SmilesEnrichment: DrugBank CSV lacks `smiles` column — "
                "rerun utils/drugbank_minimizer.py to enable. Continuing "
                "with empty SMILES."
            )
            self._lookup = {}
            return self._lookup

        subset = best_rows[["drug_id", "smiles"]].dropna(subset=["drug_id"])
        lookup: dict[str, str] = {}
        for drug_id, smiles in zip(subset["drug_id"], subset["smiles"]):
            if pd.isna(smiles):
                continue
            s = str(smiles).strip()
            if not s:
                continue
            key = str(drug_id).strip()
            if not key:
                continue
            # Keep first non-empty SMILES per drug_id — load_drugbank_lookup
            # already prefers the most-complete row by `query_name`, so
            # first-writer-wins here picks a deterministic representative.
            lookup.setdefault(key, s)
        self._lookup = lookup
        return lookup

    def run(
        self,
        candidates: CandidateTable,
        *,
        ledger: "CostLedger",
    ) -> CandidateTable:
        lookup = self._build_lookup()
        total = len(candidates.candidates)
        enriched = 0
        for cand in candidates.candidates:
            if not cand.drugbank_id:
                continue
            smiles = lookup.get(cand.drugbank_id)
            if smiles:
                cand.smiles = smiles
                enriched += 1

        pct = (100.0 * enriched / total) if total > 0 else 0.0
        logger.info(
            "SmilesEnrichment: %d/%d (%.0f%%) candidates enriched", enriched, total, pct
        )
        ledger.record_coverage(self.name, enriched, total)
        return candidates


__all__ = ["SmilesEnrichment"]
