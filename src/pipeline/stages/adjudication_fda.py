"""FDA-timeline-based outcome adjudication stage.

Alternative to `stages/adjudication.OutcomeAdjudicationStage` that
reconstructs each drug's indication-level FDA approval timeline from
openFDA (ORIG submissions + SUPPL new-indication letters), matches the
trial indication against approved indications, and separately checks
NDC commercial status.

Outcome resolution (per Candidate):

  APPROVED          -- indication matches an approved indication AND
                       approval_date <= as_of; NOT currently marketed OR
                       commercial check skipped.
  COMMERCIALIZED    -- indication matches an approved indication AND
                       currently marketed (active NDC records).
  FAILED_PHASE_3    -- highest_phase >= 3, no approval match, trials
                       are stale. Uses `latest_completion_date` (fallback
                       `earliest_start_date`) as a lower bound on
                       recency — so the 2-year staleness window is
                       conservative.
  FAILED_PHASE_2    -- highest_phase == 2, no approval match, stale.
  FAILED_PHASE_1    -- highest_phase == 1, no approval match, stale.
  ONGOING           -- no approval match AND trials still active within
                       the failure window, OR phase is N/A / Unknown.
  UNKNOWN           -- uncategorizable error (network, parse, etc.).
                       Matches the convention of the direct-LLM stage
                       so the funnel aggregator buckets errors
                       separately from ongoing candidates.
"""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional

from ..models import (
    Candidate,
    CandidateOutcome,
    CandidateOutcomeRecord,
    CandidateTable,
    OutcomeTable,
    TrialPhase,
)
from ..fda import (
    CommercialStatusChecker,
    FDAClient,
    IndicationAdjudicator,
    LLMClient,
    MatchResult,
    TimelineBuilder,
)
from ..fda.timeline import DrugApprovalTimeline, IndicationApprovalEvent

logger = logging.getLogger(__name__)


DEFAULT_FAILURE_WINDOW_DAYS = 2 * 365  # ClinSR's 2-year PTnT threshold


@dataclass
class AdjudicationConfig:
    as_of: Optional[date] = None  # defaults to today
    failure_window_days: int = DEFAULT_FAILURE_WINDOW_DAYS
    check_commercial_status: bool = True


class AdjudicationStage:
    """Pipeline stage that computes outcomes for each Candidate via FDA timelines."""

    def __init__(
        self,
        fda_client: FDAClient,
        llm_client: LLMClient,
        config: Optional[AdjudicationConfig] = None,
        cache=None,  # pipeline.knowledge_cache.KnowledgeCache | None
        workers: int = 1,
    ):
        self.fda = fda_client
        self.adjudicator = IndicationAdjudicator(llm_client)
        self.timeline_builder = TimelineBuilder(self.fda, self.adjudicator)
        self.commercial = CommercialStatusChecker(self.fda)
        self.config = config or AdjudicationConfig()
        self.cache = cache
        self.workers = max(1, int(workers))

        # Cache timelines across candidates that share a drug. Many
        # candidates in a CandidateTable differ only by indication.
        self._timeline_cache: dict[str, DrugApprovalTimeline] = {}
        # Per-drug locks deduplicate concurrent timeline builds for the
        # same drug; the outer lock guards the lock-dict itself.
        self._timeline_locks: dict[str, threading.Lock] = {}
        self._timeline_locks_dict_lock = threading.Lock()

    def run(self, candidates: CandidateTable) -> OutcomeTable:
        table = OutcomeTable()
        n = len(candidates.candidates)
        llm = getattr(self.adjudicator, "llm", None)
        hits_at_start = getattr(llm, "cache_hits", 0)
        misses_at_start = getattr(llm, "cache_misses", 0)
        run_start = time.monotonic()

        logger.info(
            "FDA adjudication: %d candidate(s) to process (workers=%d)",
            n, self.workers,
        )

        if self.workers <= 1 or n <= 1:
            for i, candidate in enumerate(candidates.candidates, start=1):
                record, elapsed = self._run_one(candidate)
                self._log_progress(i, n, candidate, record, elapsed)
                table.outcomes[record.candidate_id] = record
                self._cache_outcome(candidate, record)
        else:
            with ThreadPoolExecutor(max_workers=self.workers) as executor:
                future_to_candidate = {
                    executor.submit(self._run_one, candidate): candidate
                    for candidate in candidates.candidates
                }
                for i, future in enumerate(as_completed(future_to_candidate), start=1):
                    candidate = future_to_candidate[future]
                    record, elapsed = future.result()
                    self._log_progress(i, n, candidate, record, elapsed)
                    table.outcomes[record.candidate_id] = record
                    self._cache_outcome(candidate, record)

        total_elapsed = time.monotonic() - run_start
        if llm is not None and hasattr(llm, "cache_hits"):
            hits = llm.cache_hits - hits_at_start
            misses = llm.cache_misses - misses_at_start
            attempted = hits + misses
            hit_pct = (100.0 * hits / attempted) if attempted else 0.0
            logger.info(
                "FDA adjudication complete in %.1fs. LLM cache: %d hit / %d miss (%.0f%% hit rate).",
                total_elapsed, hits, misses, hit_pct,
            )
        else:
            logger.info("FDA adjudication complete in %.1fs.", total_elapsed)
        return table

    def _run_one(self, candidate: Candidate) -> tuple[CandidateOutcomeRecord, float]:
        """Execute one candidate adjudication; return (record, elapsed_seconds).

        Always returns a record — exceptions are converted to UNKNOWN here so
        a single failing candidate cannot poison the parallel pool.
        """
        c_start = time.monotonic()
        try:
            record = self._adjudicate(candidate)
        except Exception as e:
            logger.exception(
                "Adjudication failed for candidate %s: %s",
                candidate.candidate_id,
                e,
            )
            record = self._error_record(candidate, str(e))
        return record, time.monotonic() - c_start

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

    def _cache_outcome(self, candidate: Candidate, record: CandidateOutcomeRecord) -> None:
        """Persist the outcome to fda_adjudication_cache so the candidate summary
        can show both methods' verdicts side-by-side when both have been run."""
        if self.cache is None:
            return
        from ..knowledge_cache import KnowledgeCache
        key = KnowledgeCache.make_adjudication_key(
            candidate.drug_name,
            candidate.indication,
            candidate.highest_phase.value,
        )
        self.cache.put_fda_outcome(key, record)

    # ---------- per-candidate logic ----------

    def _adjudicate(self, candidate: Candidate) -> CandidateOutcomeRecord:
        evidence: list[str] = []
        reasoning_parts: list[str] = []

        timeline = self._get_timeline(candidate.drug_name)
        if not timeline.applications:
            _append(
                evidence, reasoning_parts,
                kind="fda_lookup",
                summary=f"No FDA applications found for '{candidate.drug_name}'.",
            )
            return self._no_approval_record(candidate, evidence, reasoning_parts)

        _append(
            evidence, reasoning_parts,
            kind="fda_lookup",
            summary=(
                f"Found {len(timeline.applications)} FDA application(s): "
                f"{', '.join(a.application_number for a in timeline.applications)}"
            ),
        )

        approved_indications = [e.indication for e in timeline.events]
        match = self.adjudicator.match(
            trial_indication=candidate.indication,
            mesh_indication=candidate.mesh_indication,
            approved=approved_indications,
        )
        _append(
            evidence, reasoning_parts,
            kind="indication_match",
            summary=f"{match.verdict}: {match.reasoning}",
        )

        if not self._is_match_positive(match):
            return self._no_approval_record(candidate, evidence, reasoning_parts)

        approval_event = self._find_approval_event(match, timeline)
        if approval_event is None:
            approval_date = timeline.earliest_approval_date
            confidence = 0.6
            _append(
                evidence, reasoning_parts,
                kind="temporal_attribution",
                summary=(
                    "Matched indication found in timeline but could not be "
                    "attributed to a specific submission event; used earliest "
                    "approval date as fallback."
                ),
            )
        else:
            approval_date = approval_event.approval_date
            confidence = approval_event.confidence
            _append(
                evidence, reasoning_parts,
                kind="temporal_attribution",
                summary=(
                    f"Attributed to {approval_event.submission_type} submission "
                    f"{approval_event.submission_number} on {approval_date} "
                    f"(source: {approval_event.source})."
                ),
            )

        as_of = self.config.as_of or date.today()
        if approval_date and approval_date > as_of:
            _append(
                evidence, reasoning_parts,
                kind="temporal_attribution",
                summary=(
                    f"Approval date {approval_date} is after as-of {as_of}; "
                    "treating as not-yet-approved."
                ),
            )
            return self._no_approval_record(candidate, evidence, reasoning_parts)

        commercialization_date: Optional[date] = None
        outcome = CandidateOutcome.APPROVED

        if self.config.check_commercial_status:
            app_numbers = [a.application_number for a in timeline.applications]
            commercial = self.commercial.check(app_numbers, as_of=as_of)
            _append(
                evidence, reasoning_parts,
                kind="commercial_status",
                summary=commercial.reasoning,
            )
            if commercial.is_commercialized:
                outcome = CandidateOutcome.COMMERCIALIZED
                commercialization_date = commercial.commercialization_date

        return CandidateOutcomeRecord(
            candidate_id=candidate.candidate_id,
            outcome=outcome,
            confidence=confidence,
            reasoning="; ".join(reasoning_parts),
            evidence_sources=evidence,
            approval_date=approval_date,
            commercialization_date=commercialization_date,
        )

    # ---------- helpers ----------

    def _get_timeline(self, drug_name: str) -> DrugApprovalTimeline:
        key = drug_name.lower().strip()
        cached = self._timeline_cache.get(key)
        if cached is not None:
            return cached
        with self._timeline_locks_dict_lock:
            per_drug_lock = self._timeline_locks.setdefault(key, threading.Lock())
        with per_drug_lock:
            cached = self._timeline_cache.get(key)
            if cached is not None:
                return cached
            timeline = self.timeline_builder.build(drug_name)
            self._timeline_cache[key] = timeline
            return timeline

    def _is_match_positive(self, match: MatchResult) -> bool:
        return match.verdict == "APPROVED"

    def _find_approval_event(
        self, match: MatchResult, timeline: DrugApprovalTimeline
    ) -> Optional[IndicationApprovalEvent]:
        if not match.matched_indication:
            return None
        target_text = _normalize(match.matched_indication.indication_text)
        matches = [
            e
            for e in timeline.events
            if _normalize(e.indication.indication_text) == target_text
        ]
        if not matches:
            return None
        return min(matches, key=lambda e: e.approval_date)

    def _no_approval_record(
        self,
        candidate: Candidate,
        evidence: list[str],
        reasoning_parts: list[str],
    ) -> CandidateOutcomeRecord:
        outcome = self._failure_or_ongoing_outcome(candidate)
        return CandidateOutcomeRecord(
            candidate_id=candidate.candidate_id,
            outcome=outcome,
            confidence=0.7 if outcome != CandidateOutcome.ONGOING else 0.5,
            reasoning="; ".join(reasoning_parts),
            evidence_sources=evidence,
            approval_date=None,
            commercialization_date=None,
        )

    def _failure_or_ongoing_outcome(self, candidate: Candidate) -> CandidateOutcome:
        last_update = candidate.latest_completion_date or candidate.earliest_start_date
        if last_update is None:
            return CandidateOutcome.ONGOING

        as_of = self.config.as_of or date.today()
        if as_of - last_update < timedelta(days=self.config.failure_window_days):
            return CandidateOutcome.ONGOING

        phase = _phase_to_int(candidate.highest_phase)
        if phase >= 3:
            # Phase 4 without an approval match is unusual (would imply
            # the drug is approved elsewhere but we couldn't match its
            # indication). Conservatively bucket with Phase 3 failures.
            return CandidateOutcome.FAILED_PHASE_3
        if phase == 2:
            return CandidateOutcome.FAILED_PHASE_2
        if phase == 1:
            return CandidateOutcome.FAILED_PHASE_1
        return CandidateOutcome.ONGOING

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


# ---------- module-level helpers (unit-testable) ----------

_PHASE_TO_INT: dict[TrialPhase, int] = {
    TrialPhase.PHASE_1: 1,
    TrialPhase.PHASE_2: 2,
    TrialPhase.PHASE_3: 3,
    TrialPhase.PHASE_4: 4,
    TrialPhase.NOT_APPLICABLE: 0,
    TrialPhase.UNKNOWN: 0,
}


def _phase_to_int(phase: TrialPhase) -> int:
    return _PHASE_TO_INT.get(phase, 0)


def _append(
    evidence: list[str], reasoning_parts: list[str], *, kind: str, summary: str
) -> None:
    evidence.append(kind)
    reasoning_parts.append(f"[{kind}] {summary}")


def _normalize(s: str) -> str:
    return " ".join(s.lower().split())


def _truncate(s: str, n: int) -> str:
    if not s:
        return ""
    return s if len(s) <= n else s[: n - 1] + "…"
