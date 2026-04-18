"""
Stage 2: Candidate Matching / Trial Clustering

Groups redundant trials that represent the same drug-indication program into
deduplicated Candidate records. Single-pass clustering on a
`(drug_key, indication_key)` tuple. Each row resolves to one drug_key from a
priority ladder (DrugBank ID → mesh-list leaf → canonicalized row name) and
one indication_key (mesh-list leaf → normalized text). No alias sets, no
union-find, no transitive closure: two rows share a cluster iff their
resolved keys are equal.
"""

from __future__ import annotations

import dataclasses
import logging
import re
import statistics
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from tqdm import tqdm
from typing import Optional

from ..models import Candidate, CandidateTable, RawTrial, TrialPhase, TrialTable

logger = logging.getLogger(__name__)

PHASE_ORDER = [
    TrialPhase.UNKNOWN,
    TrialPhase.NOT_APPLICABLE,
    TrialPhase.PHASE_1,
    TrialPhase.PHASE_2,
    TrialPhase.PHASE_3,
    TrialPhase.PHASE_4,
]

_SALT_SUFFIXES = re.compile(
    r"\b(hcl|hydrochloride|sodium|potassium|acetate|sulfate|mesylate|tartrate|maleate|fumarate|phosphate)\b"
)


def _normalize(text: str) -> str:
    text = (text or "").lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = _SALT_SUFFIXES.sub("", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Per-row key resolvers
# ---------------------------------------------------------------------------

DrugKey = tuple[str, str]  # ("db" | "mesh" | "name", identifier)


def _resolve_drug_key(
    trial: RawTrial,
    db_norm_to_id: dict[str, str],
    synonym_reverse: dict[str, str],
    canonicalize,
) -> DrugKey:
    """Resolve a trial row to one drug_key. First hit wins.

    1. DrugBank exact lookup on the canonicalized row intervention.
    2. Synonym reverse-map lookup on the canonicalized row intervention.
    3. Mesh-list leaf on the row whose canonicalized form matches the row's
       drug name; try DrugBank + synonym lookups on that leaf.
    4. If step 3 selected a leaf but no DrugBank match, return the leaf as
       a ``("mesh", ...)`` key.
    5. First-word fallback: if ``row_norm`` has >=2 tokens and its first
       token is itself a known DrugBank ``query_norm``, return
       ``("name", first_word)``. This groups unresolved variants (e.g.
       "insulin lispro" + "insulin aspart") into a shared name-tier cluster
       **without** hijacking the ``("db", ...)`` cluster that claims the
       parent drug. The first-word-in-DrugBank guard prevents stopword
       collapse (e.g. "small molecule X" does NOT land in ``("name", "small")``).
    6. Fall back to ``("name", row_norm)``.
    """
    row_norm = canonicalize(trial.intervention)

    db_id = db_norm_to_id.get(row_norm) or synonym_reverse.get(row_norm)
    if db_id:
        return ("db", db_id)

    leaf_norm: Optional[str] = None
    for leaf in trial.mesh_intervention_terms or []:
        cand_norm = canonicalize(leaf)
        if not cand_norm:
            continue
        if _name_matches(row_norm, cand_norm):
            leaf_norm = cand_norm
            break

    if leaf_norm:
        db_id = db_norm_to_id.get(leaf_norm) or synonym_reverse.get(leaf_norm)
        if db_id:
            return ("db", db_id)
        return ("mesh", leaf_norm)

    tokens = row_norm.split()
    if len(tokens) >= 2 and tokens[0] in db_norm_to_id:
        return ("name", tokens[0])

    return ("name", row_norm)


def _name_matches(row_norm: str, leaf_norm: str) -> bool:
    """Check whether a mesh-list leaf belongs to the row's row-level drug.

    Exact canonicalized equality is the strict match. We also accept the
    case where either string appears as a whole token in the other — this
    handles studies whose row-level intervention is a brand name / codename
    while the MeSH leaf is the INN (or vice versa) without pulling in
    unrelated co-listed drugs.
    """
    if not row_norm or not leaf_norm:
        return False
    if row_norm == leaf_norm:
        return True
    row_tokens = set(row_norm.split())
    leaf_tokens = set(leaf_norm.split())
    return leaf_norm in row_tokens or row_norm in leaf_tokens


def _resolve_indication_key(trial: RawTrial) -> str:
    """First mesh-list condition leaf (lowercased) if present, else normalized text."""
    for leaf in trial.mesh_condition_terms or []:
        leaf_norm = (leaf or "").lower().strip()
        if leaf_norm:
            return leaf_norm
    return _normalize(trial.indication)


# ---------------------------------------------------------------------------
# DrugBank resolver loader
# ---------------------------------------------------------------------------


def _load_drugbank_resolvers(
    drugbank_csv_path: Optional[Path],
    synonyms_csv_path: Optional[Path],
) -> tuple[dict[str, str], dict[str, str], callable]:
    """Return (db_norm_to_id, synonym_reverse, canonicalize_fn).

    Wraps the existing helpers in ``pipeline/drugbank_norm.py``. When
    ``drugbank_csv_path`` is None, both maps are empty and every row falls
    back to the ``("name", row_norm)`` tier; this is the correct behaviour
    for offline dev runs with no DrugBank data on hand.
    """
    from ..drugbank_norm import (
        canonicalize_drug_name,
        load_drugbank_lookup,
        load_drugbank_synonyms,
    )

    if drugbank_csv_path is None:
        return {}, {}, canonicalize_drug_name

    _, best_rows_norm = load_drugbank_lookup(drugbank_csv_path)
    db_norm_to_id = dict(
        zip(
            best_rows_norm["query_norm"].astype(str),
            best_rows_norm["drug_id"].astype(str),
        )
    )

    synonym_reverse: dict[str, str] = {}
    if synonyms_csv_path is not None:
        _, synonym_reverse = load_drugbank_synonyms(synonyms_csv_path)

    return db_norm_to_id, synonym_reverse, canonicalize_drug_name


# ---------------------------------------------------------------------------
# Stage
# ---------------------------------------------------------------------------


class CandidateClusteringStage:
    """
    Deduplicates trials into drug-indication candidates via a single-pass
    `(drug_key, indication_key)` grouping.

    Inputs:  TrialTable
    Outputs: CandidateTable
    """

    def __init__(
        self,
        drugbank_csv_path: Optional[Path] = None,
        drugbank_synonyms_csv_path: Optional[Path] = None,
    ):
        """
        Args:
            drugbank_csv_path:
                Path to ``drugbank_approvals.csv``. Required for DrugBank ID
                resolution; ``None`` disables it (every row falls back to
                the canonicalized intervention name).
            drugbank_synonyms_csv_path:
                Path to ``drugbank_synonyms.csv`` (produced by
                ``scripts/build_drugbank_derivatives.py``). When provided,
                codename / brand / INN variants for the same DrugBank ID
                resolve to the same drug_key.
        """
        self.drugbank_csv_path = drugbank_csv_path
        self.drugbank_synonyms_csv_path = drugbank_synonyms_csv_path

    def run(self, trial_table: TrialTable) -> CandidateTable:
        """Cluster trials into candidates. Returns a populated CandidateTable."""
        clusters = self._cluster(trial_table)
        candidates = [
            self._build_candidate(cid, trials)
            for cid, trials in tqdm(
                clusters.items(),
                desc="Building candidates",
                unit="candidate",
                disable=not sys.stderr.isatty(),
            )
        ]
        self._log_summary(candidates)
        return CandidateTable(candidates=candidates)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _cluster(self, trial_table: TrialTable) -> dict[DrugKey, list[RawTrial]]:
        """Group trials by ``(drug_key, indication_key)``. One pass, no re-merge."""
        db_norm_to_id, synonym_reverse, canonicalize = _load_drugbank_resolvers(
            self.drugbank_csv_path,
            self.drugbank_synonyms_csv_path,
        )

        clusters: dict[tuple[DrugKey, str], list[RawTrial]] = {}
        for trial in tqdm(
            trial_table.trials,
            desc="Clustering trials",
            unit="trial",
            disable=not sys.stderr.isatty(),
        ):
            drug_key = _resolve_drug_key(trial, db_norm_to_id, synonym_reverse, canonicalize)
            indication_key = _resolve_indication_key(trial)
            clusters.setdefault((drug_key, indication_key), []).append(trial)
        return clusters

    def _build_candidate(
        self,
        candidate_id,
        trials: list[RawTrial],
    ) -> Candidate:
        """Collapse a cluster of trials into a single Candidate record."""
        first = trials[0]
        drug_name_raw = first.intervention
        drug_name = _normalize(drug_name_raw)
        indication = _normalize(first.indication)
        trial_ids = [t.nct_id for t in trials]
        sponsors = list(dict.fromkeys(t.sponsor for t in trials))
        highest_phase = max(trials, key=lambda t: PHASE_ORDER.index(t.phase)).phase
        start_dates = [t.start_date for t in trials if t.start_date is not None]
        end_dates = [t.completion_date for t in trials if t.completion_date is not None]
        earliest_start_date = min(start_dates) if start_dates else None
        latest_completion_date = max(end_dates) if end_dates else None
        single_arm_p_values = [
            dataclasses.replace(pv, phase=t.phase)
            for t in trials
            if t.is_single_arm
            for pv in t.primary_p_values
        ]

        # Display-only MeSH tags: pick the most frequent mesh-list leaf across
        # the cluster's constituent rows. Post-SQL-filter these are never
        # ancestor terms.
        mesh_drug = _most_common_leaf(trials, "mesh_intervention_terms")
        mesh_indication = _most_common_leaf(trials, "mesh_condition_terms")
        mesh_condition_tree_numbers = list({
            tn for t in trials
            for tn in getattr(t, "mesh_condition_tree_numbers", [])
        })

        drugbank_id: Optional[str] = None
        if isinstance(candidate_id, tuple):
            drug_key = candidate_id[0]
            if isinstance(drug_key, tuple) and drug_key[0] == "db":
                drugbank_id = drug_key[1]

        return Candidate(
            candidate_id=_candidate_id_to_str(candidate_id),
            drug_name=drug_name,
            indication=indication,
            trial_ids=trial_ids,
            highest_phase=highest_phase,
            sponsors=sponsors,
            earliest_start_date=earliest_start_date,
            latest_completion_date=latest_completion_date,
            drug_name_raw=drug_name_raw,
            drugbank_id=drugbank_id,
            single_arm_p_values=single_arm_p_values,
            mesh_indication=mesh_indication,
            mesh_drug=mesh_drug,
            mesh_condition_tree_numbers=mesh_condition_tree_numbers,
        )

    def _log_summary(self, candidates: list[Candidate]) -> None:
        if not candidates:
            logger.info("Clustering produced 0 candidates.")
            return

        trial_counts = [len(c.trial_ids) for c in candidates]
        namespace_counts: Counter[str] = Counter()
        for c in candidates:
            if c.drugbank_id:
                namespace_counts["db"] += 1
            elif c.candidate_id.startswith("mesh:"):
                namespace_counts["mesh"] += 1
            else:
                namespace_counts["name"] += 1

        logger.info(
            "Clustering: %d candidates from %d trials (mean %.1f, median %d trials/candidate); "
            "namespace breakdown db=%d mesh=%d name=%d.",
            len(candidates),
            sum(trial_counts),
            statistics.fmean(trial_counts),
            int(statistics.median(trial_counts)),
            namespace_counts["db"],
            namespace_counts["mesh"],
            namespace_counts["name"],
        )


def _most_common_leaf(trials: list[RawTrial], attr: str) -> Optional[str]:
    counts: Counter[str] = Counter()
    for trial in trials:
        for term in getattr(trial, attr, []) or []:
            normalized = (term or "").lower().strip()
            if normalized:
                counts[normalized] += 1
    if not counts:
        return None
    return counts.most_common(1)[0][0]


def _candidate_id_to_str(candidate_id) -> str:
    """Render a (drug_key, indication_key) tuple into a stable string id."""
    if isinstance(candidate_id, str):
        return candidate_id
    drug_key, indication_key = candidate_id
    namespace, identifier = drug_key
    safe_ind = _sanitize_id_segment(indication_key)
    return f"{namespace}:{identifier}__{safe_ind}"


def _sanitize_id_segment(segment: str) -> str:
    segment = unicodedata.normalize("NFKD", segment or "")
    segment = "".join(ch for ch in segment if not unicodedata.combining(ch))
    segment = segment.replace("|", "/").strip()
    return segment
