"""
Stage 2: Candidate Matching / Trial Clustering

Groups redundant trials that represent the same drug–indication program
into deduplicated Candidate records.
"""

import dataclasses
import logging
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from abc import ABC, abstractmethod
from collections import Counter
from pathlib import Path
from tqdm import tqdm
from typing import Optional

from ..models import Candidate, CandidateTable, RawTrial, TrialPhase, TrialTable

logger = logging.getLogger(__name__)

# Phase ranking: higher index = more advanced
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
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = _SALT_SUFFIXES.sub("", text)
    text = text.strip()
    return text


def _pick_mesh_indication(trials: list) -> Optional[str]:
    """Return the most frequent MeSH condition term across a cluster's trials.

    Returns None when no trials carry MeSH condition terms (api-source fallback).
    """
    counts: Counter = Counter()
    for trial in trials:
        for term in getattr(trial, "mesh_condition_terms", []):
            counts[term.lower().strip()] += 1
    if not counts:
        return None
    return counts.most_common(1)[0][0]


def _pick_mesh_drug(trials: list) -> Optional[str]:
    """Return the most frequent MeSH intervention term across a cluster's trials.

    Returns None when no trials carry MeSH intervention terms (api-source fallback).
    """
    counts: Counter = Counter()
    for trial in trials:
        for term in getattr(trial, "mesh_intervention_terms", []):
            counts[term.lower().strip()] += 1
    if not counts:
        return None
    return counts.most_common(1)[0][0]


# ---------------------------------------------------------------------------
# Union-find (disjoint-set) for alias-based candidate merging
# ---------------------------------------------------------------------------

class UnionFind:
    """Path-compressed union-find with union-by-size over integer indices."""

    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.size = [1] * n

    def find(self, x: int) -> int:
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]
        return True


# ---------------------------------------------------------------------------
# Strategy ABC and concrete implementations
# ---------------------------------------------------------------------------

class ClusteringStrategy(ABC):
    @abstractmethod
    def cluster(self, trial_table: TrialTable) -> dict[str, list[RawTrial]]:
        ...


class StringMatchStrategy(ClusteringStrategy):
    def cluster(self, trial_table: TrialTable) -> dict[str, list[RawTrial]]:
        clusters: dict[str, list[RawTrial]] = {}
        for trial in tqdm(
            trial_table.trials,
            desc="Clustering trials (fuzzy)",
            unit="trial",
            disable=not sys.stderr.isatty(),
        ):
            key = f"{_normalize(trial.intervention)}__{_normalize(trial.indication)}"
            clusters.setdefault(key, []).append(trial)
        return clusters


class EmbeddingsStrategy(ClusteringStrategy):
    def cluster(self, trial_table: TrialTable) -> dict[str, list[RawTrial]]:
        # TODO: implement embedding-based clustering
        raise NotImplementedError("EmbeddingsStrategy is not yet implemented")


class HybridStrategy(ClusteringStrategy):
    def cluster(self, trial_table: TrialTable) -> dict[str, list[RawTrial]]:
        """Cluster using canonical MeSH drug + indication keys.

        Drug key:       sorted MeSH intervention terms (if present) else _normalize(intervention)
        Indication key: sorted MeSH condition terms (if present) else _normalize(indication)

        Sorting ensures multiple MeSH tags produce a deterministic key regardless of
        the order they appear on the trial. Falls back to normalized free-text for
        non-AACT sources where MeSH terms are absent.
        """
        clusters: dict[str, list[RawTrial]] = {}
        for trial in tqdm(
            trial_table.trials,
            desc="Clustering trials (hybrid)",
            unit="trial",
            disable=not sys.stderr.isatty(),
        ):
            mesh_int = getattr(trial, "mesh_intervention_terms", [])
            if mesh_int:
                drug_key = "|".join(sorted(t.lower().strip() for t in mesh_int))
            else:
                drug_key = _normalize(trial.intervention)

            mesh_cond = getattr(trial, "mesh_condition_terms", [])
            if mesh_cond:
                indication_key = "|".join(sorted(t.lower().strip() for t in mesh_cond))
            else:
                indication_key = _normalize(trial.indication)

            clusters.setdefault(f"{drug_key}__{indication_key}", []).append(trial)
        return clusters


# ---------------------------------------------------------------------------
# Stage
# ---------------------------------------------------------------------------

class CandidateClusteringStage:
    """
    Deduplicates trials into drug–indication candidates.

    Inputs:  TrialTable
    Outputs: CandidateTable

    Methods:
    - Embedding-based similarity
    - Fuzzy string matching on drug/indication names
    - LLM adjudication for ambiguous clusters
    """

    def __init__(
        self,
        method: str = "hybrid",
        llm_adjudicate: bool = True,
        drugbank_csv_path: Optional[Path] = None,
        drop_unmatched_drugbank: bool = True,
        drugbank_synonyms_csv_path: Optional[Path] = None,
    ):
        """
        Args:
            method:               primary clustering strategy: "embeddings" | "fuzzy" | "hybrid"
            llm_adjudicate:       whether to run an LLM pass to resolve ambiguous merges
            drugbank_csv_path:    path to drugbank_approvals.csv; if None, DrugBank dedup is skipped
            drop_unmatched_drugbank:
                                  when True, candidates without a DrugBank or drug MeSH match
                                  are removed; when False, unmatched candidates are retained
            drugbank_synonyms_csv_path:
                                  path to drugbank_synonyms.csv (produced by
                                  `scripts/build_drugbank_derivatives.py`). When provided,
                                  clustering uses the synonym reverse-map as a third-level
                                  fallback during DrugBank ID lookup and emits an extra
                                  ("syn", norm) alias tier in the union-find pass so
                                  codename-only and INN-only candidate rows for the same
                                  drug-indication program are merged.
        """
        self.method = method
        self.llm_adjudicate = llm_adjudicate
        self.drugbank_csv_path = drugbank_csv_path
        self.drop_unmatched_drugbank = drop_unmatched_drugbank
        self.drugbank_synonyms_csv_path = drugbank_synonyms_csv_path

    def run(self, trial_table: TrialTable) -> CandidateTable:
        """Cluster trials into candidates. Returns a populated CandidateTable."""
        clusters = self._cluster(trial_table)
        if self.llm_adjudicate:
            clusters = self._adjudicate(clusters)
        candidates = [
            self._build_candidate(cid, trials)
            for cid, trials in tqdm(
                clusters.items(),
                desc="Building candidates",
                unit="candidate",
                disable=not sys.stderr.isatty(),
            )
        ]
        candidates = self._apply_drugbank_dedup(candidates)
        return CandidateTable(candidates=candidates)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _cluster(self, trial_table: TrialTable) -> dict[str, list]:
        """
        Group trials by drug–indication similarity.
        Returns a dict mapping cluster_id → list of RawTrial.
        """
        strategy: ClusteringStrategy = {
            "fuzzy": StringMatchStrategy(),
            "embeddings": StringMatchStrategy(),  # TODO: swap in EmbeddingsStrategy
            "hybrid": HybridStrategy(),
        }[self.method]
        return strategy.cluster(trial_table)

    def _adjudicate(self, clusters: dict[str, list]) -> dict[str, list]:
        """Passthrough — LLM adjudication reserved for a future pass."""
        return clusters

    def _build_candidate(self, candidate_id: str, trials: list) -> Candidate:
        """Collapse a cluster of trials into a single Candidate record."""
        drug_name_raw = trials[0].intervention
        drug_name = _normalize(drug_name_raw)
        indication = _normalize(trials[0].indication)
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
        mesh_indication = _pick_mesh_indication(trials)
        mesh_drug = _pick_mesh_drug(trials)
        mesh_condition_tree_numbers = list({
            tn for t in trials
            for tn in getattr(t, "mesh_condition_tree_numbers", [])
        })
        return Candidate(
            candidate_id=candidate_id,
            drug_name=drug_name,
            indication=indication,
            trial_ids=trial_ids,
            highest_phase=highest_phase,
            sponsors=sponsors,
            earliest_start_date=earliest_start_date,
            latest_completion_date=latest_completion_date,
            drug_name_raw=drug_name_raw,
            single_arm_p_values=single_arm_p_values,
            mesh_indication=mesh_indication,
            mesh_drug=mesh_drug,
            mesh_condition_tree_numbers=mesh_condition_tree_numbers,
        )

    def _apply_drugbank_dedup(self, candidates: list[Candidate]) -> list[Candidate]:
        """Assign DrugBank IDs and re-merge by canonical (drug, indication) key.

        Canonical drug key priority: DrugBank ID > drug MeSH term > normalized name.
        Canonical indication key: mesh_indication if available, else indication.

        Candidates eligible for re-merge: those with a DrugBank ID OR a drug MeSH term.
        Unmatched candidates (neither) are either dropped or retained based on
        `drop_unmatched_drugbank`.

        Prints a formatted coverage table with separate drug-coverage and
        indication-coverage sections.
        """
        if self.drugbank_csv_path is None:
            return candidates

        from ..drugbank_norm import (
            canonicalize_drug_name,
            load_drugbank_lookup,
            load_drugbank_synonyms,
        )
        _, best_rows_norm = load_drugbank_lookup(self.drugbank_csv_path)
        norm_to_drug_id = dict(
            zip(best_rows_norm["query_norm"].astype(str), best_rows_norm["drug_id"].astype(str))
        )

        # Optional synonym reverse-map from the DrugBank XML export.
        # synonym_forward:  drugbank_id -> [synonym_norm, ...]
        # synonym_reverse:  synonym_norm -> drugbank_id
        synonym_forward: dict[str, list[str]] = {}
        synonym_reverse: dict[str, str] = {}
        if self.drugbank_synonyms_csv_path is not None:
            synonym_forward, synonym_reverse = load_drugbank_synonyms(
                self.drugbank_synonyms_csv_path,
            )

        # Matching dominates stage runtime on large candidate sets.
        # Use thread-level parallelism with O(1) dictionary lookups.
        def _match_name(drug_name: str) -> Optional[str]:
            norm = canonicalize_drug_name(drug_name)
            if not norm:
                return None
            exact = norm_to_drug_id.get(norm)
            if exact is not None:
                return exact
            first_word = norm.split(maxsplit=1)[0]
            hit = norm_to_drug_id.get(first_word)
            if hit is not None:
                return hit
            # Third-level fallback: look up in the synonym reverse map, so
            # codename-only candidate rows (e.g. "BMS-986156") resolve to
            # the INN's DrugBank ID (e.g. vopratelimab → DB99999).
            return synonym_reverse.get(norm)

        names = [c.drug_name_raw or c.drug_name for c in candidates]
        workers = min(32, (os.cpu_count() or 1))
        if workers > 1 and len(names) > 1:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                matched_ids = list(
                    tqdm(
                        pool.map(_match_name, names),
                        total=len(names),
                        desc=f"Matching DrugBank IDs ({workers} threads)",
                        unit="candidate",
                        disable=not sys.stderr.isatty(),
                    )
                )
        else:
            matched_ids = [
                _match_name(name)
                for name in tqdm(
                    names,
                    desc="Matching DrugBank IDs",
                    unit="candidate",
                    disable=not sys.stderr.isatty(),
                )
            ]

        for candidate, drugbank_id in zip(candidates, matched_ids):
            candidate.drugbank_id = drugbank_id

        total_trials = sum(len(c.trial_ids) for c in candidates)

        # Drug coverage metrics (trial-level)
        drugbank_trials = sum(len(c.trial_ids) for c in candidates if c.drugbank_id is not None)
        mesh_drug_trials = sum(len(c.trial_ids) for c in candidates if c.mesh_drug is not None)
        union_drug_trials = sum(
            len(c.trial_ids) for c in candidates
            if c.drugbank_id is not None or c.mesh_drug is not None
        )

        # Indication coverage metrics (trial-level)
        mesh_indication_trials = sum(
            len(c.trial_ids) for c in candidates if c.mesh_indication is not None
        )

        logger.info(
            "DrugBank coverage: %d / %d trials (%.1f%%).",
            drugbank_trials,
            total_trials,
            (100.0 * drugbank_trials / total_trials) if total_trials else 0.0,
        )

        # Re-merge candidates eligible by DrugBank OR drug MeSH.
        eligible = [c for c in candidates if c.drugbank_id is not None or c.mesh_drug is not None]
        unmatched = [c for c in candidates if c.drugbank_id is None and c.mesh_drug is None]

        # Union-find over (drug_alias, indication_alias) signatures. Two
        # candidates are merged if they agree on ANY drug alias AND ANY
        # indication alias, drawn from the tagged sets below. Tagging
        # prevents cross-namespace collisions (e.g., a DrugBank ID equal to
        # a MeSH term string would otherwise falsely merge).
        uf = UnionFind(len(eligible))
        sig_to_indices: dict[tuple, list[int]] = {}
        for idx, c in enumerate(eligible):
            drug_aliases = [t for t in (
                ("db", c.drugbank_id),
                ("mesh", c.mesh_drug),
                ("name", c.drug_name),
            ) if t[1]]
            # Synonym alias tier: if this candidate's DrugBank ID appears in
            # the synonym forward map, every synonym of that DrugBank ID
            # becomes a matching key. This unifies a pair where one member
            # is named by an INN and the other by a codename or brand —
            # cases the canonical name/MeSH matches would otherwise miss.
            if c.drugbank_id and synonym_forward:
                for syn in synonym_forward.get(c.drugbank_id, ()):
                    drug_aliases.append(("syn", syn))
            indication_aliases = [t for t in (
                ("mesh_ind", c.mesh_indication),
                ("name_ind", c.indication),
            ) if t[1]]
            for d in drug_aliases:
                for i in indication_aliases:
                    sig_to_indices.setdefault((d, i), []).append(idx)

        n_unions = 0
        for indices in sig_to_indices.values():
            if len(indices) < 2:
                continue
            anchor = indices[0]
            for other in indices[1:]:
                if uf.union(anchor, other):
                    n_unions += 1

        components: dict[int, list[Candidate]] = {}
        for idx, c in enumerate(eligible):
            components.setdefault(uf.find(idx), []).append(c)

        deduped = [
            self._merge_candidates(group) if len(group) > 1 else group[0]
            for group in components.values()
        ]
        candidates_after = len(deduped) + (0 if self.drop_unmatched_drugbank else len(unmatched))
        mean_component_size = (
            len(eligible) / len(components) if components else 0.0
        )
        logger.info(
            "Union-find dedup: %d eligible candidates → %d components "
            "(%d unions applied, mean component size %.2f)",
            len(eligible), len(components), n_unions, mean_component_size,
        )

        self._report_coverage(
            total_trials=total_trials,
            drugbank_trials=drugbank_trials,
            mesh_drug_trials=mesh_drug_trials,
            union_drug_trials=union_drug_trials,
            mesh_indication_trials=mesh_indication_trials,
            candidates_before=len(candidates),
            candidates_after=candidates_after,
        )

        if self.drop_unmatched_drugbank:
            return deduped
        return deduped + unmatched

    def _merge_candidates(self, group: list[Candidate]) -> Candidate:
        """Merge a group of candidates with the same canonical (drug, indication) key into one."""
        highest_phase = max(group, key=lambda c: PHASE_ORDER.index(c.highest_phase)).highest_phase
        trial_ids = list(dict.fromkeys(tid for c in group for tid in c.trial_ids))
        sponsors = list(dict.fromkeys(s for c in group for s in c.sponsors))
        start_dates = [c.earliest_start_date for c in group if c.earliest_start_date is not None]
        end_dates = [c.latest_completion_date for c in group if c.latest_completion_date is not None]
        return Candidate(
            candidate_id=group[0].candidate_id,
            drug_name=group[0].drug_name,
            indication=group[0].indication,
            trial_ids=trial_ids,
            highest_phase=highest_phase,
            sponsors=sponsors,
            earliest_start_date=min(start_dates) if start_dates else None,
            latest_completion_date=max(end_dates) if end_dates else None,
            drug_name_raw=group[0].drug_name_raw,
            drugbank_id=group[0].drugbank_id,
            mesh_indication=group[0].mesh_indication,
            mesh_drug=group[0].mesh_drug,
            mesh_condition_tree_numbers=list({
                tn for c in group for tn in c.mesh_condition_tree_numbers
            }),
        )

    def _report_coverage(
        self,
        total_trials: int,
        drugbank_trials: int,
        mesh_drug_trials: int,
        union_drug_trials: int,
        mesh_indication_trials: int,
        candidates_before: int,
        candidates_after: int,
    ) -> None:
        """Print a formatted deduplication coverage table."""
        def pct(n: int) -> str:
            return f"{100.0 * n / total_trials:.1f}%" if total_trials else "N/A"

        lines = [
            "",
            "  ┌─────────────────────────────────────────────────────────┐",
            "  │              Deduplication Coverage Report              │",
            "  ├─────────────────────────────────────────────────────────┤",
            f"  │  Total trials:               {total_trials:>6}                    │",
            "  ├─────────────────────────────────────────────────────────┤",
            "  │  Drug coverage                                          │",
            f"  │    DrugBank matched:         {drugbank_trials:>6} / {total_trials} ({pct(drugbank_trials):>6})  │",
            f"  │    Drug MeSH matched:        {mesh_drug_trials:>6} / {total_trials} ({pct(mesh_drug_trials):>6})  │",
            f"  │    Union (either):           {union_drug_trials:>6} / {total_trials} ({pct(union_drug_trials):>6})  │",
            "  ├─────────────────────────────────────────────────────────┤",
            "  │  Indication coverage                                    │",
            f"  │    MeSH indication matched:  {mesh_indication_trials:>6} / {total_trials} ({pct(mesh_indication_trials):>6})  │",
            "  ├─────────────────────────────────────────────────────────┤",
            f"  │  Candidates before dedup:    {candidates_before:>6}                    │",
            f"  │  Candidates after dedup:     {candidates_after:>6}                    │",
            "  └─────────────────────────────────────────────────────────┘",
            "",
        ]
        print("\n".join(lines))
        logger.info(
            "Coverage: DrugBank=%d/%d DrugMeSH=%d/%d UnionDrug=%d/%d "
            "MeSHIndication=%d/%d Candidates before=%d after=%d",
            drugbank_trials, total_trials,
            mesh_drug_trials, total_trials,
            union_drug_trials, total_trials,
            mesh_indication_trials, total_trials,
            candidates_before, candidates_after,
        )
