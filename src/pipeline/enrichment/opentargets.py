"""OpenTargets enrichment — attaches MoA, targets, pathways, indication
phase, tractability, LOEUF, and target-disease genetic-association
evidence.

Reads the slim SQLite snapshot produced by
`scripts/build_opentargets_snapshot.py`. Three tables:

- ``name_opentargets_drug`` — name-keyed (``query_norm``) drug-level
  aggregates. For a given candidate (normalized via
  ``canonicalize_drug_name``) we fetch all matched rows, union the
  target symbols, pathways, action types, tractability modalities/
  labels, and take the min LOEUF across the drug's targets.
- ``name_opentargets_indications`` — per-drug (chembl_id) indication
  rows with `(efo_id, indication_name, max_phase)`. We match the
  candidate's ``indication`` (or ``mesh_indication``) case-insensitively
  against ``indication_name``; a hit populates
  ``opentargets_indication_max_phase`` and surfaces the matched
  ``efo_id`` for the genetic-evidence join.
- ``name_opentargets_target_disease_evidence`` — (chembl_id, ensembl_id,
  efo_id, genetic_score) rows for the OT ``genetic_association``
  datatype. Aggregated max-across-targets per (drug, EFO) for
  ``opentargets_genetic_score``; max-across-(target, any-EFO) for the
  ``..._max_any_indication`` fallback.

The stage is DrugBank-independent and ChEMBL-independent at runtime —
all joins are name-keyed through the shared normalizer. Coverage is
recorded for each user-visible feature separately so the CLI reports
MoA, targets, pathways, indication-phase, tractability, LOEUF, and
genetic-score coverage as individual lines. Older snapshots that lack
the tractability/LOEUF columns or the evidence table are tolerated:
the missing features are recorded as zero coverage and Candidates pass
through with the default (empty list / None) values.
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
        # {chembl_id: [(efo_id, indication_name, max_phase), ...]}
        self._indications: (
            dict[str, list[tuple[str | None, str, int | None]]] | None
        ) = None
        # {(chembl_id, efo_id): max(genetic_score across the drug's targets)}
        self._genetic_by_efo: dict[tuple[str, str], float] | None = None
        # {chembl_id: max(genetic_score) across all (target, any-EFO)}
        self._genetic_any: dict[str, float] | None = None
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

    @staticmethod
    def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
        try:
            return {
                row[1] for row in conn.execute(
                    f"PRAGMA table_info({table})"
                )
            }
        except sqlite3.DatabaseError:
            return set()

    def _build_lookups(self) -> tuple[
        dict[str, list[dict]],
        dict[str, list[tuple[str | None, str, int | None]]],
        dict[tuple[str, str], float],
        dict[str, float],
    ]:
        if (
            self._drug_lookup is not None
            and self._indications is not None
            and self._genetic_by_efo is not None
            and self._genetic_any is not None
        ):
            return (
                self._drug_lookup,
                self._indications,
                self._genetic_by_efo,
                self._genetic_any,
            )
        assert self._source_path is not None  # guarded by is_available

        conn = sqlite3.connect(f"file:{self._source_path}?mode=ro", uri=True)
        try:
            try:
                self._release = conn.execute(
                    "PRAGMA user_version"
                ).fetchone()[0]
            except sqlite3.DatabaseError:
                self._release = None

            drug_cols = self._table_columns(conn, "name_opentargets_drug")
            has_tractability = "tractability_modalities" in drug_cols
            has_loeuf = "loeuf_min" in drug_cols

            # Build the SELECT lazily so older snapshots without the
            # new columns still load. Missing columns return None per
            # row and the run loop treats them as absent.
            select_cols = [
                "query_norm", "chembl_id", "drug_name", "moa_text",
                "action_type", "target_symbols", "target_ensembl",
                "pathways",
            ]
            if has_tractability:
                select_cols.extend([
                    "tractability_modalities", "tractability_labels",
                ])
            if has_loeuf:
                select_cols.append("loeuf_min")

            drug_lookup: dict[str, list[dict]] = defaultdict(list)
            cur = conn.execute(
                f"SELECT {', '.join(select_cols)} FROM name_opentargets_drug"
            )
            for row in cur:
                rec = dict(zip(select_cols, row))
                query_norm = rec.get("query_norm")
                if not query_norm:
                    continue
                key = str(query_norm).strip()
                if not key:
                    continue
                drug_lookup[key].append(
                    {
                        "chembl_id": str(rec.get("chembl_id") or ""),
                        "drug_name": rec.get("drug_name"),
                        "moa_text": rec.get("moa_text"),
                        "action_type": rec.get("action_type"),
                        "target_symbols": rec.get("target_symbols"),
                        "target_ensembl": rec.get("target_ensembl"),
                        "pathways": rec.get("pathways"),
                        "tractability_modalities":
                            rec.get("tractability_modalities"),
                        "tractability_labels":
                            rec.get("tractability_labels"),
                        "loeuf_min": rec.get("loeuf_min"),
                    }
                )

            indications: dict[
                str, list[tuple[str | None, str, int | None]]
            ] = defaultdict(list)
            cur = conn.execute(
                "SELECT chembl_id, indication_efo_id, indication_name, "
                "indication_max_phase "
                "FROM name_opentargets_indications "
                "WHERE indication_name IS NOT NULL AND indication_name <> ''"
            )
            for chembl_id, efo_id, indication_name, max_phase in cur:
                if not chembl_id or not indication_name:
                    continue
                phase = int(max_phase) if max_phase is not None else None
                indications[str(chembl_id)].append(
                    (
                        str(efo_id) if efo_id else None,
                        str(indication_name),
                        phase,
                    )
                )

            # Pre-aggregate genetic evidence so the run loop is a pair
            # of dict lookups: by (chembl_id, efo_id) for the matched-
            # indication score, and by chembl_id for the any-indication
            # fallback. Both are "max across the drug's targets".
            genetic_by_efo: dict[tuple[str, str], float] = {}
            genetic_any: dict[str, float] = {}
            try:
                ev_cur = conn.execute(
                    "SELECT chembl_id, efo_id, genetic_score "
                    "FROM name_opentargets_target_disease_evidence"
                )
                for chembl_id, efo_id, score in ev_cur:
                    if not chembl_id or not efo_id or score is None:
                        continue
                    try:
                        s = float(score)
                    except (TypeError, ValueError):
                        continue
                    cid = str(chembl_id)
                    efo = str(efo_id)
                    prior = genetic_by_efo.get((cid, efo))
                    if prior is None or s > prior:
                        genetic_by_efo[(cid, efo)] = s
                    prior_any = genetic_any.get(cid)
                    if prior_any is None or s > prior_any:
                        genetic_any[cid] = s
            except sqlite3.DatabaseError:
                # Older snapshot — evidence table absent. Coverage stays at zero.
                logger.info(
                    "OpenTargetsEnrichment: snapshot has no "
                    "name_opentargets_target_disease_evidence table — "
                    "genetic-score features will all be None. Rebuild "
                    "with scripts/build_opentargets_snapshot.py to enable."
                )
        finally:
            conn.close()

        self._drug_lookup = dict(drug_lookup)
        self._indications = dict(indications)
        self._genetic_by_efo = genetic_by_efo
        self._genetic_any = genetic_any
        logger.info(
            "OpenTargetsEnrichment: loaded %d normalized drug names, "
            "%d indication rows, %d genetic-evidence (chembl_id, efo_id) "
            "pairs from %s (OT release %s)",
            len(self._drug_lookup),
            sum(len(v) for v in self._indications.values()),
            len(self._genetic_by_efo),
            self._source_path,
            self._release if self._release else "unknown",
        )
        return (
            self._drug_lookup,
            self._indications,
            self._genetic_by_efo,
            self._genetic_any,
        )

    def run(
        self,
        candidates: CandidateTable,
        *,
        ledger: "CostLedger",
    ) -> CandidateTable:
        drug_lookup, indications, genetic_by_efo, genetic_any = self._build_lookups()
        total = len(candidates.candidates)
        matched = 0
        n_moa = 0
        n_targets = 0
        n_pathways = 0
        n_phase = 0
        n_tract = 0
        n_loeuf = 0
        n_genetic = 0
        n_genetic_any = 0
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
            modalities_seen: list[str] = []
            labels_seen: list[str] = []
            loeuf_vals: list[float] = []
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
                if row.get("tractability_modalities"):
                    for m in str(row["tractability_modalities"]).split("|"):
                        m = m.strip()
                        if m and m not in modalities_seen:
                            modalities_seen.append(m)
                if row.get("tractability_labels"):
                    for lbl in str(row["tractability_labels"]).split("|"):
                        lbl = lbl.strip()
                        if lbl and lbl not in labels_seen:
                            labels_seen.append(lbl)
                if row.get("loeuf_min") is not None:
                    try:
                        loeuf_vals.append(float(row["loeuf_min"]))
                    except (TypeError, ValueError):
                        pass

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
            if modalities_seen:
                cand.opentargets_tractability_modalities = list(modalities_seen)
                n_tract += 1
            if labels_seen:
                cand.opentargets_tractability_labels = list(labels_seen)
            if loeuf_vals:
                # Min across the drug's per-chembl_id rows — each row
                # already encodes the min across its target set, so the
                # outer min handles drugs with multiple ChEMBL records.
                cand.opentargets_loeuf_min = min(loeuf_vals)
                n_loeuf += 1

            # Indication matching: gather all indications for every
            # matched chembl_id, then compare case-insensitively against
            # the candidate's indication (then mesh_indication). We do
            # NOT fall back to max-across-all-indications — that would
            # overstate evidence for the specific indication we're
            # analyzing.
            #
            # We also retain the matched EFO ID so the genetic-evidence
            # join (keyed on (chembl_id, efo_id)) can use the exact same
            # indication resolution as the indication-phase column.
            phase_by_name: dict[str, int] = {}
            efo_by_name: dict[str, str] = {}
            for cid in candidate_chembl_ids:
                for efo_id, name, phase in indications.get(cid, ()):
                    norm = name.strip().lower()
                    if not norm:
                        continue
                    if phase is not None:
                        prior = phase_by_name.get(norm)
                        if prior is None or phase > prior:
                            phase_by_name[norm] = phase
                    # Keep the first-seen EFO per normalized name; OT
                    # rarely emits multiple EFOs for the same indication
                    # name, and any choice is consistent across runs.
                    if efo_id and norm not in efo_by_name:
                        efo_by_name[norm] = efo_id
            cand_ind = (cand.indication or "").strip().lower()
            cand_mesh = (getattr(cand, "mesh_indication", "") or "").strip().lower()
            phase_match = phase_by_name.get(cand_ind)
            if phase_match is None and cand_mesh:
                phase_match = phase_by_name.get(cand_mesh)
            if phase_match is not None:
                cand.opentargets_indication_max_phase = phase_match
                n_phase += 1
            matched_efo = efo_by_name.get(cand_ind)
            if matched_efo is None and cand_mesh:
                matched_efo = efo_by_name.get(cand_mesh)

            # Genetic evidence (matched indication): max over the
            # candidate's matched ChEMBL IDs at the resolved EFO.
            if matched_efo:
                best_genetic: float | None = None
                for cid in candidate_chembl_ids:
                    s = genetic_by_efo.get((cid, matched_efo))
                    if s is None:
                        continue
                    if best_genetic is None or s > best_genetic:
                        best_genetic = s
                if best_genetic is not None:
                    cand.opentargets_genetic_score = best_genetic
                    n_genetic += 1

            # Target-level genetic evidence fallback: max over the
            # candidate's matched ChEMBL IDs across any EFO. Useful for
            # capturing "are this drug's targets genetics-driven at all"
            # when indication match misses.
            best_any: float | None = None
            for cid in candidate_chembl_ids:
                s = genetic_any.get(cid)
                if s is None:
                    continue
                if best_any is None or s > best_any:
                    best_any = s
            if best_any is not None:
                cand.opentargets_genetic_score_max_any_indication = best_any
                n_genetic_any += 1

        pct_matched = (100.0 * matched / total) if total > 0 else 0.0
        logger.info(
            "OpenTargetsEnrichment: %d/%d drugs matched (%.0f%%). "
            "MoA=%d, Targets=%d, Pathways=%d, Indication-phase=%d, "
            "Tractability=%d, LOEUF=%d, Genetic-score=%d (any-EFO=%d).",
            matched, total, pct_matched,
            n_moa, n_targets, n_pathways, n_phase,
            n_tract, n_loeuf, n_genetic, n_genetic_any,
        )
        ledger.record_coverage("opentargets_moa", n_moa, total)
        ledger.record_coverage("opentargets_targets", n_targets, total)
        ledger.record_coverage("opentargets_pathways", n_pathways, total)
        ledger.record_coverage("opentargets_indication_phase", n_phase, total)
        ledger.record_coverage("opentargets_tractability", n_tract, total)
        ledger.record_coverage("opentargets_loeuf", n_loeuf, total)
        ledger.record_coverage("opentargets_genetic_score", n_genetic, total)
        ledger.record_coverage(
            "opentargets_genetic_score_any", n_genetic_any, total
        )
        return candidates


__all__ = ["OpenTargetsEnrichment"]
