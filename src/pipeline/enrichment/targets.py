"""Drug-target enrichment — attaches ChEMBL-derived targets to candidates.

Reads a pre-built SQLite snapshot produced by
`scripts/build_chembl_targets_snapshot.py`. The snapshot is a single
indexed table (`name_targets`) keyed on `query_norm`, the output of
`canonicalize_drug_name` applied to each ChEMBL preferred-name and
synonym. Candidates are matched by running the same normalizer on
`candidate.drug_name_raw`, so lookup is DrugBank-independent: a candidate
with `drugbank_id=None` still gets targets as long as its raw name
normalizes to something ChEMBL knows about.

Common generic names ("insulin", "heparin") can match multiple ChEMBL
molregnos. The stage surfaces the union of all matches, deduped by
UniProt accession and target pref_name. The snapshot is opened read-only;
the SQLite connection is closed right after the lookup dict is built.
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


class TargetsEnrichment:
    """Join ChEMBL-derived drug targets onto candidates by normalized name."""

    name = "targets"

    def __init__(self) -> None:
        # {query_norm: (set[uniprot], set[target_pref_name])}
        self._lookup: dict[str, tuple[set[str], set[str]]] | None = None
        self._source_path: Path | None = None
        self._release: int | None = None

    def is_available(self, config: "PipelineConfig") -> bool:
        if not config.enable_targets:
            return False
        path = config.chembl_snapshot_path
        if path is None:
            logger.info(
                "TargetsEnrichment: skipped — no chembl_snapshot_path configured."
            )
            return False
        if not Path(path).exists():
            logger.warning(
                "TargetsEnrichment: skipped — chembl_snapshot_path %s not found. "
                "Build one with scripts/build_chembl_targets_snapshot.py.",
                path,
            )
            return False
        self._source_path = Path(path)
        return True

    def _build_lookup(self) -> dict[str, tuple[set[str], set[str]]]:
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

            lookup: dict[str, tuple[set[str], set[str]]] = {}
            cur = conn.execute(
                "SELECT query_norm, target_pref_name, uniprot_accession "
                "FROM name_targets"
            )
            for query_norm, pref_name, accession_blob in cur:
                if not query_norm:
                    continue
                key = str(query_norm).strip()
                if not key:
                    continue
                uniprots, names = lookup.setdefault(key, (set(), set()))
                if accession_blob:
                    for acc in str(accession_blob).split("|"):
                        acc = acc.strip()
                        if acc:
                            uniprots.add(acc)
                if pref_name:
                    name = str(pref_name).strip()
                    if name:
                        names.add(name)
        finally:
            conn.close()

        self._lookup = lookup
        logger.info(
            "TargetsEnrichment: loaded %d normalized names from %s "
            "(ChEMBL release %s)",
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
        enriched = 0
        for cand in candidates.candidates:
            key = canonicalize_drug_name(cand.drug_name_raw or "")
            if not key:
                continue
            hit = lookup.get(key)
            if not hit:
                continue
            uniprots, names = hit
            if uniprots:
                cand.drug_targets = sorted(uniprots)
            if names:
                cand.target_names = sorted(names)
            if uniprots or names:
                enriched += 1

        pct = (100.0 * enriched / total) if total > 0 else 0.0
        logger.info(
            "TargetsEnrichment: %d/%d (%.0f%%) candidates enriched",
            enriched, total, pct,
        )
        ledger.record_coverage(self.name, enriched, total)
        return candidates


__all__ = ["TargetsEnrichment"]
