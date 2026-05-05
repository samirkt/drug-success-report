"""NDC-indication-based outcome adjudication stage.

Third adjudicator alongside ``llm_direct`` and ``fda_timeline``. Uses
the local-SQLite-backed ``utils.ndc_lookup`` for FDA label data (no
network) and any pluggable LLM client for the coverage decision.

Outcome resolution (per Candidate):

  APPROVED          -- LLM judges that the trial indication is covered
                       by at least one approved label indication.
  COMMERCIALIZED    -- (only when ``_INCLUDE_COMMERCIALIZED=True``) the
                       APPROVED case plus an active NDC marketing record.
  ONGOING /         -- "Not approved" — drug missing from the FDA label
  FAILED_PHASE_N       database, or LLM judges no label indication
                       covers the trial indication. Mapping is shared
                       with the FDA-timeline stage via
                       ``stages._no_approval.classify_no_approval``.
  UNKNOWN           -- LLM/system error. Not persisted (so retry is free
                       on the next run).
"""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Optional

from ..models import (
    Candidate,
    CandidateOutcome,
    CandidateOutcomeRecord,
    CandidateTable,
    OutcomeTable,
)
from ..ndc import (
    LLMClient,
    NDCAdjudicator,
    NDCVerdict,
    resolve_drug_key,
    synonyms_for_candidate,
)
from ._no_approval import classify_no_approval

logger = logging.getLogger(__name__)


# Internal toggle — flip to True to upgrade APPROVED -> COMMERCIALIZED
# when an active NDC marketing record exists for the matched synonym.
# Intentionally not exposed via CLI / PipelineConfig per design constraint.
_INCLUDE_COMMERCIALIZED = False


DEFAULT_FAILURE_WINDOW_DAYS = 2 * 365


@dataclass
class AdjudicationConfig:
    as_of: Optional[date] = None
    failure_window_days: int = DEFAULT_FAILURE_WINDOW_DAYS
    drugbank_synonyms_csv: Optional[Path] = None


class AdjudicationStage:
    """Pipeline stage that adjudicates each Candidate via the local NDC DB."""

    def __init__(
        self,
        llm_client: LLMClient,
        config: Optional[AdjudicationConfig] = None,
        cache=None,
        workers: int = 1,
    ):
        self.config = config or AdjudicationConfig()
        self.cache = cache
        self.workers = max(1, int(workers))
        self.adjudicator = NDCAdjudicator(
            llm_client=llm_client,
            drugbank_synonyms_csv=self.config.drugbank_synonyms_csv,
        )
        # Serializes get_drug() calls when _INCLUDE_COMMERCIALIZED=True;
        # ndc_lookup uses a module-global SQLite connection.
        self._ndc_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self, candidates: CandidateTable) -> OutcomeTable:
        table = OutcomeTable()
        n = len(candidates.candidates)
        llm = self.adjudicator.llm
        hits_at_start = getattr(llm, "cache_hits", 0)
        misses_at_start = getattr(llm, "cache_misses", 0)
        run_start = time.monotonic()

        logger.info(
            "NDC adjudication: %d candidate(s) to process (workers=%d)",
            n, self.workers,
        )

        drug_lookup = self._prepass(candidates)

        if self.workers <= 1 or n <= 1:
            for i, candidate in enumerate(candidates.candidates, start=1):
                record, elapsed = self._run_one(candidate, drug_lookup)
                self._log_progress(i, n, candidate, record, elapsed)
                table.outcomes[record.candidate_id] = record
                self._cache_outcome(candidate, record)
        else:
            with ThreadPoolExecutor(max_workers=self.workers) as executor:
                future_to_candidate = {
                    executor.submit(self._run_one, candidate, drug_lookup): candidate
                    for candidate in candidates.candidates
                }
                for i, future in enumerate(as_completed(future_to_candidate), start=1):
                    candidate = future_to_candidate[future]
                    record, elapsed = future.result()
                    self._log_progress(i, n, candidate, record, elapsed)
                    table.outcomes[record.candidate_id] = record
                    self._cache_outcome(candidate, record)

        total_elapsed = time.monotonic() - run_start
        if hasattr(llm, "cache_hits"):
            hits = llm.cache_hits - hits_at_start
            misses = llm.cache_misses - misses_at_start
            attempted = hits + misses
            hit_pct = (100.0 * hits / attempted) if attempted else 0.0
            logger.info(
                "NDC adjudication complete in %.1fs. LLM cache: %d hit / %d miss (%.0f%% hit rate).",
                total_elapsed, hits, misses, hit_pct,
            )
        else:
            logger.info("NDC adjudication complete in %.1fs.", total_elapsed)
        return table

    # ------------------------------------------------------------------
    # Pre-pass: one bulk SQLite lookup for every unique drug
    # ------------------------------------------------------------------

    def _prepass(self, candidates: CandidateTable) -> dict[str, dict]:
        """Build ``drug_key -> {matched_synonym, label_indications}``.

        Single batched call to ``utils.ndc_lookup.get_drugs_with_indications_bulk``
        for every unique drug. Drugs with no FDA label hit map to
        ``{"matched_synonym": None, "label_indications": []}``.
        """
        from utils import ndc_lookup

        drug_synonyms: dict[str, list[str]] = {}
        for candidate in candidates.candidates:
            key = resolve_drug_key(candidate)
            if key in drug_synonyms:
                continue
            drug_synonyms[key] = synonyms_for_candidate(
                candidate, self.adjudicator.drugbank_forward
            )

        groups = list(drug_synonyms.values())
        try:
            bulk = ndc_lookup.get_drugs_with_indications_bulk(groups)
        except Exception as e:
            raise RuntimeError(
                f"FDA local DB not available ({e}). "
                "Build it with: python -c 'from utils.ndc_lookup import build_db; build_db()'"
            ) from e

        drug_lookup: dict[str, dict] = {}
        for key, synonyms in drug_synonyms.items():
            matched_synonym = None
            label_indications: list[str] = []
            for syn in synonyms:
                record = bulk.get(syn)
                if record and record.get("indication_count", 0) > 0:
                    matched_synonym = syn
                    label_indications = list(record.get("indications", []))
                    break
            drug_lookup[key] = {
                "matched_synonym": matched_synonym,
                "label_indications": label_indications,
            }

        n_hits = sum(1 for v in drug_lookup.values() if v["label_indications"])
        logger.info(
            "NDC pre-pass: %d unique drug(s); %d had FDA label hits, %d missing.",
            len(drug_lookup), n_hits, len(drug_lookup) - n_hits,
        )
        return drug_lookup

    # ------------------------------------------------------------------
    # Per-candidate adjudication
    # ------------------------------------------------------------------

    def _run_one(
        self, candidate: Candidate, drug_lookup: dict[str, dict]
    ) -> tuple[CandidateOutcomeRecord, float]:
        c_start = time.monotonic()
        try:
            record = self._adjudicate(candidate, drug_lookup)
        except Exception as e:
            logger.exception(
                "NDC adjudication failed for candidate %s: %s",
                candidate.candidate_id, e,
            )
            record = self._error_record(candidate, str(e))
        return record, time.monotonic() - c_start

    def _adjudicate(
        self, candidate: Candidate, drug_lookup: dict[str, dict]
    ) -> CandidateOutcomeRecord:
        # KnowledgeCache short-circuit
        if self.cache is not None:
            from ..knowledge_cache import KnowledgeCache
            key = KnowledgeCache.make_adjudication_key(
                candidate.drug_name, candidate.indication, candidate.highest_phase.value
            )
            cached = self.cache.get_ndc_outcome(key, candidate.candidate_id)
            if cached is not None:
                return cached

        drug_key = resolve_drug_key(candidate)
        lookup = drug_lookup.get(drug_key) or {
            "matched_synonym": None,
            "label_indications": [],
        }
        label_indications = lookup["label_indications"]
        matched_synonym = lookup["matched_synonym"]

        if not label_indications:
            return self._not_approved_record(
                candidate,
                reasoning="No FDA label indications found for any synonym.",
                evidence=["ndc_lookup"],
                confidence=0.9,
            )

        verdict: NDCVerdict = self.adjudicator.match_indication(
            label_indications=label_indications,
            trial_indication=candidate.indication,
            mesh_indication=candidate.mesh_indication,
            matched_synonym=matched_synonym,
        )

        if not verdict.approved:
            return self._not_approved_record(
                candidate,
                reasoning=f"LLM: {verdict.reasoning}",
                evidence=verdict.evidence_sources,
                confidence=verdict.confidence,
            )

        return self._approved_record(candidate, verdict)

    # ------------------------------------------------------------------
    # Verdict assembly
    # ------------------------------------------------------------------

    def _approved_record(
        self, candidate: Candidate, verdict: NDCVerdict
    ) -> CandidateOutcomeRecord:
        outcome = CandidateOutcome.APPROVED
        commercialization_date: Optional[date] = None
        evidence = list(verdict.evidence_sources)
        reasoning = f"LLM matched '{verdict.matched_indication}': {verdict.reasoning}"

        if _INCLUDE_COMMERCIALIZED and verdict.matched_synonym:
            commercial_date = self._check_commercialized(verdict.matched_synonym)
            if commercial_date is not None:
                outcome = CandidateOutcome.COMMERCIALIZED
                commercialization_date = commercial_date
                evidence.append("ndc_commercial")
                reasoning += f"; active NDC marketing since {commercial_date}."

        return CandidateOutcomeRecord(
            candidate_id=candidate.candidate_id,
            outcome=outcome,
            confidence=verdict.confidence,
            reasoning=reasoning,
            evidence_sources=evidence,
            approval_date=None,
            commercialization_date=commercialization_date,
        )

    def _not_approved_record(
        self,
        candidate: Candidate,
        *,
        reasoning: str,
        evidence: list[str],
        confidence: float,
    ) -> CandidateOutcomeRecord:
        outcome = classify_no_approval(
            candidate,
            failure_window_days=self.config.failure_window_days,
            as_of=self.config.as_of,
        )
        return CandidateOutcomeRecord(
            candidate_id=candidate.candidate_id,
            outcome=outcome,
            confidence=confidence if outcome != CandidateOutcome.ONGOING else min(confidence, 0.5),
            reasoning=reasoning,
            evidence_sources=list(evidence),
            approval_date=None,
            commercialization_date=None,
        )

    def _error_record(self, candidate: Candidate, err: str) -> CandidateOutcomeRecord:
        return CandidateOutcomeRecord(
            candidate_id=candidate.candidate_id,
            outcome=CandidateOutcome.UNKNOWN,
            confidence=0.0,
            reasoning=f"Error: {err}",
            evidence_sources=["error"],
            approval_date=None,
            commercialization_date=None,
        )

    def _check_commercialized(self, synonym: str) -> Optional[date]:
        """Stub for the commercialized check; only invoked when
        ``_INCLUDE_COMMERCIALIZED=True``. Returns the marketing-start
        date if the drug has any active NDC record, else None."""
        from utils import ndc_lookup

        with self._ndc_lock:
            try:
                is_commercialized, start_date = ndc_lookup.get_drug(synonym)
            except Exception as e:
                logger.warning("ndc_lookup.get_drug(%r) failed: %s", synonym, e)
                return None

        if not is_commercialized or not start_date:
            return None
        try:
            return date.fromisoformat(start_date[:10])
        except ValueError:
            return None

    # ------------------------------------------------------------------
    # Persistence + logging
    # ------------------------------------------------------------------

    def _cache_outcome(
        self, candidate: Candidate, record: CandidateOutcomeRecord
    ) -> None:
        if self.cache is None:
            return
        if record.outcome == CandidateOutcome.UNKNOWN and "error" in record.evidence_sources:
            return
        from ..knowledge_cache import KnowledgeCache
        key = KnowledgeCache.make_adjudication_key(
            candidate.drug_name, candidate.indication, candidate.highest_phase.value
        )
        self.cache.put_ndc_outcome(key, record)

    def _log_progress(
        self,
        i: int,
        n: int,
        candidate: Candidate,
        record: CandidateOutcomeRecord,
        elapsed: float,
    ) -> None:
        logger.info(
            "[%d/%d] %s | %s -> %s (%.1fs)",
            i, n,
            _truncate(candidate.drug_name, 30),
            _truncate(candidate.indication, 40),
            record.outcome.value,
            elapsed,
        )


def _truncate(s: str, n: int) -> str:
    if not s:
        return ""
    return s if len(s) <= n else s[: n - 1] + "…"
