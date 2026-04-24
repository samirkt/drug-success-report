"""OpenTargets enrichment — attaches MoA, targets, pathways, indication phase.

Reads the slim SQLite snapshot produced by
`scripts/build_opentargets_snapshot.py`. Two tables:

- ``name_opentargets_drug`` — name-keyed (``query_norm``) drug-level
  aggregates. For a given candidate (normalized via
  ``canonicalize_drug_name``) we fetch all matched rows, union the
  target symbols, pathways, action types, and collect the first few
  mechanism-of-action strings.
- ``name_opentargets_indications`` — per-drug (chembl_id) indication
  rows with `maxPhaseForIndication`. We match the candidate's
  ``indication`` (or ``mesh_indication``) case-insensitively against
  ``indication_name``; a hit populates
  ``opentargets_indication_max_phase``.

The stage is DrugBank-independent and ChEMBL-independent at runtime —
all joins are name-keyed through the shared normalizer. Coverage is
recorded for each user-visible feature separately so the CLI reports
MoA, targets, pathways, and indication-phase coverage as individual
lines.
"""

from __future__ import annotations

import logging
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

from ..drugbank_norm import canonicalize_drug_name
from ..models import CandidateTable

if TYPE_CHECKING:
    from ..pipeline import PipelineConfig
    from utils.tiered_router import CostLedger

logger = logging.getLogger(__name__)


class OpenTargetsEnrichment:
    """Attach OpenTargets drug annotations by normalized drug name."""

    name = "opentargets"

    def __init__(self) -> None:
        # {query_norm: [drug_row, ...]} — one row per (query_norm, chembl_id)
        self._drug_lookup: dict[str, list[dict]] | None = None
        # {chembl_id: [(indication_name, max_phase), ...]}
        self._indications: dict[str, list[tuple[str, int | None]]] | None = None
        self._source_path: Path | None = None
        self._release: int | None = None

    def is_available(self, config: "PipelineConfig") -> bool:
        if not config.enable_opentargets:
            return False
        path = config.opentargets_snapshot_path
        if path is None:
            logger.info(
                "OpenTargetsEnrichment: skipped — no opentargets_snapshot_path "
                "configured."
            )
            return False
        if not Path(path).exists():
            logger.warning(
                "OpenTargetsEnrichment: skipped — opentargets_snapshot_path "
                "%s not found. Build one with "
                "scripts/build_opentargets_snapshot.py.",
                path,
            )
            return False
        self._source_path = Path(path)
        return True

    def _build_lookups(self) -> tuple[
        dict[str, list[dict]],
        dict[str, list[tuple[str, int | None]]],
    ]:
        if self._drug_lookup is not None and self._indications is not None:
            return self._drug_lookup, self._indications
        assert self._source_path is not None  # guarded by is_available

        conn = sqlite3.connect(f"file:{self._source_path}?mode=ro", uri=True)
        try:
            try:
                self._release = conn.execute(
                    "PRAGMA user_version"
                ).fetchone()[0]
            except sqlite3.DatabaseError:
                self._release = None

            drug_lookup: dict[str, list[dict]] = defaultdict(list)
            cur = conn.execute(
                "SELECT query_norm, chembl_id, drug_name, moa_text, "
                "action_type, target_symbols, target_ensembl, pathways "
                "FROM name_opentargets_drug"
            )
            for (
                query_norm,
                chembl_id,
                drug_name,
                moa_text,
                action_type,
                target_symbols,
                target_ensembl,
                pathways,
            ) in cur:
                if not query_norm:
                    continue
                key = str(query_norm).strip()
                if not key:
                    continue
                drug_lookup[key].append(
                    {
                        "chembl_id": str(chembl_id) if chembl_id else "",
                        "drug_name": drug_name,
                        "moa_text": moa_text,
                        "action_type": action_type,
                        "target_symbols": target_symbols,
                        "target_ensembl": target_ensembl,
                        "pathways": pathways,
                    }
                )

            indications: dict[str, list[tuple[str, int | None]]] = defaultdict(list)
            cur = conn.execute(
                "SELECT chembl_id, indication_name, indication_max_phase "
                "FROM name_opentargets_indications "
                "WHERE indication_name IS NOT NULL AND indication_name <> ''"
            )
            for chembl_id, indication_name, max_phase in cur:
                if not chembl_id or not indication_name:
                    continue
                phase = int(max_phase) if max_phase is not None else None
                indications[str(chembl_id)].append(
                    (str(indication_name), phase)
                )
        finally:
            conn.close()

        self._drug_lookup = dict(drug_lookup)
        self._indications = dict(indications)
        logger.info(
            "OpenTargetsEnrichment: loaded %d normalized drug names, "
            "%d indication rows from %s (OT release %s)",
            len(self._drug_lookup),
            sum(len(v) for v in self._indications.values()),
            self._source_path,
            self._release if self._release else "unknown",
        )
        return self._drug_lookup, self._indications

    def run(
        self,
        candidates: CandidateTable,
        *,
        ledger: "CostLedger",
    ) -> CandidateTable:
        drug_lookup, indications = self._build_lookups()
        total = len(candidates.candidates)
        matched = 0
        n_moa = 0
        n_targets = 0
        n_pathways = 0
        n_phase = 0
        for cand in candidates.candidates:
            key = canonicalize_drug_name(cand.drug_name_raw or "")
            if not key:
                continue
            rows = drug_lookup.get(key)
            if not rows:
                continue
            matched += 1

            moa_seen: list[str] = []
            action_seen: list[str] = []
            symbols_seen: set[str] = set()
            pathways_seen: set[str] = set()
            candidate_chembl_ids: list[str] = []
            for row in rows:
                if row["chembl_id"]:
                    candidate_chembl_ids.append(row["chembl_id"])
                if row["moa_text"]:
                    for m in str(row["moa_text"]).split("|"):
                        m = m.strip()
                        if m and m not in moa_seen:
                            moa_seen.append(m)
                if row["action_type"]:
                    for a in str(row["action_type"]).split("|"):
                        a = a.strip()
                        if a and a not in action_seen:
                            action_seen.append(a)
                if row["target_symbols"]:
                    for s in str(row["target_symbols"]).split("|"):
                        s = s.strip()
                        if s:
                            symbols_seen.add(s)
                if row["pathways"]:
                    for p in str(row["pathways"]).split("|"):
                        p = p.strip()
                        if p:
                            pathways_seen.add(p)

            if moa_seen:
                cand.opentargets_moa = "|".join(moa_seen)
                n_moa += 1
            if action_seen:
                cand.opentargets_action_type = "|".join(action_seen)
            if symbols_seen:
                cand.opentargets_targets = sorted(symbols_seen)
                n_targets += 1
            if pathways_seen:
                cand.opentargets_pathways = sorted(pathways_seen)
                n_pathways += 1

            # Indication matching: gather all indications for every
            # matched chembl_id, then compare case-insensitively against
            # the candidate's indication (then mesh_indication). We do
            # NOT fall back to max-across-all-indications — that would
            # overstate evidence for the specific indication we're
            # analyzing.
            targets_indications: dict[str, int] = {}
            for cid in candidate_chembl_ids:
                for name, phase in indications.get(cid, ()):
                    if phase is None:
                        continue
                    norm = name.strip().lower()
                    if not norm:
                        continue
                    # Keep the max phase seen per indication name so
                    # duplicate rows across chembl_ids don't downgrade.
                    prior = targets_indications.get(norm)
                    if prior is None or phase > prior:
                        targets_indications[norm] = phase
            cand_ind = (cand.indication or "").strip().lower()
            cand_mesh = (getattr(cand, "mesh_indication", "") or "").strip().lower()
            match = targets_indications.get(cand_ind)
            if match is None and cand_mesh:
                match = targets_indications.get(cand_mesh)
            if match is not None:
                cand.opentargets_indication_max_phase = match
                n_phase += 1

        pct_matched = (100.0 * matched / total) if total > 0 else 0.0
        logger.info(
            "OpenTargetsEnrichment: %d/%d drugs matched (%.0f%%). "
            "MoA=%d, Targets=%d, Pathways=%d, Indication-phase=%d.",
            matched, total, pct_matched,
            n_moa, n_targets, n_pathways, n_phase,
        )
        ledger.record_coverage("opentargets_moa", n_moa, total)
        ledger.record_coverage("opentargets_targets", n_targets, total)
        ledger.record_coverage("opentargets_pathways", n_pathways, total)
        ledger.record_coverage("opentargets_indication_phase", n_phase, total)
        return candidates


__all__ = ["OpenTargetsEnrichment"]
