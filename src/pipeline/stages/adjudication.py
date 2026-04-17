"""
Stage 3b: Candidate Outcome Adjudication (Tiered)

Three improvements over the original implementation:

1. **Deterministic failure detection**: candidates whose trials are all
   TERMINATED or WITHDRAWN at a known phase are short-circuited to
   FAILED_PHASE_X without any LLM call.
2. **Trial status pass-through**: the LLM prompt now includes per-trial
   status counts (Completed, Terminated, Withdrawn, etc.) so the model
   can ground its reasoning in actual data instead of relying solely on
   its training knowledge.
3. **Tiered Sonnet → Opus routing**: initial adjudication runs on Sonnet;
   LOW-confidence or UNKNOWN results are escalated to Opus.

Runs in parallel with Candidate Attribute Classification (stage 3a).
"""

from __future__ import annotations

import logging
from collections import Counter
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING

from ..models import (
    Candidate,
    CandidateOutcome,
    CandidateOutcomeRecord,
    CandidateTable,
    OutcomeTable,
    RawTrial,
    TrialPhase,
    TrialStatus,
)
from utils.prompt_runner import (
    BATCH_SIZE,
    load_prompts_from_txt,
)
from utils.tiered_router import (
    CostLedger,
    default_escalation_predicate,
    tiered_batch_call,
)

if TYPE_CHECKING:
    from ..knowledge_cache import KnowledgeCache

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).parent / "prompts"
_SYSTEM_PROMPT_PATH = _PROMPTS_DIR / "adjudication_system.txt"
_USER_PROMPT_PATH = _PROMPTS_DIR / "adjudication_user.txt"

_CONFIDENCE_MAP = {"HIGH": 1.0, "MEDIUM": 0.5, "LOW": 0.25}

# Map LLM output strings to CandidateOutcome enum members by name
_OUTCOME_BY_NAME = {member.name: member for member in CandidateOutcome}

# Phase mapping for deterministic failure detection
_PHASE_TO_FAILURE: dict[TrialPhase, CandidateOutcome] = {
    TrialPhase.PHASE_1: CandidateOutcome.FAILED_PHASE_1,
    TrialPhase.PHASE_2: CandidateOutcome.FAILED_PHASE_2,
    TrialPhase.PHASE_3: CandidateOutcome.FAILED_PHASE_3,
}

# Terminal statuses that indicate failure
_TERMINAL_STATUSES = frozenset({TrialStatus.TERMINATED, TrialStatus.WITHDRAWN})


def _parse_date(s: str | None) -> date | None:
    if not s:
        return None
    try:
        return date.fromisoformat(s)
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Deterministic failure detection
# ---------------------------------------------------------------------------


def try_deterministic_failure(
    candidate: Candidate,
) -> CandidateOutcomeRecord | None:
    """
    Check if a candidate can be deterministically classified as failed
    based on trial statuses alone, without an LLM call.

    Returns a CandidateOutcomeRecord if all trials are terminal (TERMINATED
    or WITHDRAWN) at a recognized phase, or None if the candidate needs
    LLM adjudication.

    Requires candidate._raw_trials to be populated by the clustering stage.
    """
    raw_trials: list[RawTrial] | None = getattr(candidate, "_raw_trials", None)
    if not raw_trials:
        return None

    # All trials must be terminal
    if not all(t.status in _TERMINAL_STATUSES for t in raw_trials):
        return None

    # The highest phase must map to a known failure outcome
    failure_outcome = _PHASE_TO_FAILURE.get(candidate.highest_phase)
    if failure_outcome is None:
        return None

    # Build reasoning from status counts
    status_counts = Counter(t.status.value for t in raw_trials)
    status_summary = ", ".join(f"{v} {k}" for k, v in status_counts.items())

    return CandidateOutcomeRecord(
        candidate_id=candidate.candidate_id,
        outcome=failure_outcome,
        confidence=1.0,
        reasoning=(
            f"Deterministic: all {len(raw_trials)} trials are terminal "
            f"({status_summary}) at {candidate.highest_phase.value}."
        ),
        evidence_sources=["trial_status_deterministic"],
    )


# ---------------------------------------------------------------------------
# Parse / default helpers for tiered_batch_call
# ---------------------------------------------------------------------------


def _parse_adjudication(
    data: dict, candidate: Candidate
) -> CandidateOutcomeRecord:
    """Convert a raw LLM response dict into a CandidateOutcomeRecord."""
    outcome_str = data.get("outcome", "UNKNOWN")
    outcome = _OUTCOME_BY_NAME.get(outcome_str, CandidateOutcome.UNKNOWN)
    return CandidateOutcomeRecord(
        candidate_id=candidate.candidate_id,
        outcome=outcome,
        confidence=_CONFIDENCE_MAP.get(data.get("confidence", ""), 0.0),
        reasoning=data.get("reasoning", ""),
        evidence_sources=data.get("evidence_sources", []),
        approval_date=_parse_date(data.get("approval_date")),
        commercialization_date=_parse_date(data.get("commercialization_date")),
    )


def _default_adjudication(candidate: Candidate) -> CandidateOutcomeRecord:
    """Fallback when both Sonnet and Opus fail for a candidate."""
    return CandidateOutcomeRecord(
        candidate_id=candidate.candidate_id,
        outcome=CandidateOutcome.UNKNOWN,
        confidence=0.0,
    )


# ---------------------------------------------------------------------------
# Stage class
# ---------------------------------------------------------------------------


class OutcomeAdjudicationStage:
    """
    Adjudicates the final development outcome for each candidate using
    tiered Sonnet → Opus routing and deterministic failure detection.

    Inputs:  CandidateTable
    Outputs: OutcomeTable
    """

    def __init__(
        self,
        cache: "KnowledgeCache | None" = None,
        ledger: CostLedger | None = None,
    ):
        self.cache = cache
        self.ledger = ledger

    def run(self, candidate_table: CandidateTable) -> OutcomeTable:
        """Adjudicate outcomes for all candidates. Returns a populated OutcomeTable."""
        candidates = candidate_table.candidates
        total = len(candidates)
        outcomes: dict[str, CandidateOutcomeRecord] = {}
        cache_misses: list[tuple[str | None, Candidate]] = []

        logger.info("Adjudication: starting %d candidates", total)

        # Phase 0: deterministic failure detection.
        n_deterministic = 0
        remaining: list[Candidate] = []
        for candidate in candidates:
            det_result = try_deterministic_failure(candidate)
            if det_result is not None:
                outcomes[candidate.candidate_id] = det_result
                n_deterministic += 1
            else:
                remaining.append(candidate)

        if n_deterministic:
            logger.info(
                "Adjudication: %d candidates resolved deterministically",
                n_deterministic,
            )

        # Phase 1: resolve cache hits among remaining
        for candidate in remaining:
            if self.cache is not None:
                from ..knowledge_cache import KnowledgeCache

                key = KnowledgeCache.make_adjudication_key(
                    candidate.drug_name,
                    candidate.indication,
                    candidate.highest_phase.value,
                )
                cached = self.cache.get_outcome(key, candidate.candidate_id)
                if cached is not None:
                    logger.debug(
                        "Adjudication cache hit for %s", candidate.candidate_id
                    )
                    outcomes[candidate.candidate_id] = cached
                    continue
            else:
                key = None
            cache_misses.append((key, candidate))

        n_hits = len(remaining) - len(cache_misses)
        n_misses = len(cache_misses)
        if n_hits:
            logger.info(
                "Adjudication: %d cache hits, %d LLM calls needed",
                n_hits,
                n_misses,
            )
        _free_candidates = n_hits + n_deterministic
        if self.ledger is not None and _free_candidates > 0:
            self.ledger.record_cache_hits(stage="Adjudication", n_hits=_free_candidates)

        # Phase 2: tiered batch LLM calls for cache misses
        system_prompt, user_template = load_prompts_from_txt(
            _SYSTEM_PROMPT_PATH, _USER_PROMPT_PATH
        )

        processed = 0
        for chunk_start in range(0, n_misses, BATCH_SIZE):
            chunk = cache_misses[chunk_start : chunk_start + BATCH_SIZE]
            chunk_candidates = [c for _, c in chunk]

            try:
                records = tiered_batch_call(
                    system_prompt=system_prompt,
                    user_template=user_template,
                    candidates=chunk_candidates,
                    fields_fn=self._adj_fields,
                    parse_fn=_parse_adjudication,
                    default_fn=_default_adjudication,
                    escalation_predicate=default_escalation_predicate,
                    stage_name="Adjudication",
                    ledger=self.ledger,
                )
            except Exception as exc:
                logger.warning(
                    "Adjudication batch failed (candidates %d-%d): %s",
                    chunk_start,
                    chunk_start + len(chunk) - 1,
                    exc,
                )
                records = [_default_adjudication(c) for c in chunk_candidates]

            for (key, candidate), record in zip(chunk, records):
                outcomes[candidate.candidate_id] = record
                if self.cache is not None and key is not None:
                    self.cache.put_outcome(key, record)

            processed += len(chunk)
            n_done = n_deterministic + n_hits + processed
            logger.info(
                "Adjudication: %d/%d (%.0f%%)",
                n_done,
                total,
                100 * n_done / total,
            )

        return OutcomeTable(outcomes=outcomes)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _adj_fields(i: int, c: Candidate) -> list[str]:
        """Build the per-candidate field block for the LLM prompt.

        Includes trial status counts so the LLM can ground its reasoning
        in actual data rather than relying on training knowledge.
        """
        sponsors_str = ", ".join(c.sponsors) if c.sponsors else "Unknown"
        fields = [
            f"Drug name: {c.drug_name_raw}",
            f"Indication: {c.indication}",
            f"Highest phase reached: {c.highest_phase.value}",
            f"Number of trials: {len(c.trial_ids)}",
            f"Sponsors: {sponsors_str}",
            f"Earliest trial start: {c.earliest_start_date or 'Unknown'}",
            f"Latest trial completion: {c.latest_completion_date or 'Unknown'}",
        ]

        # Append trial status breakdown if _raw_trials is available
        raw_trials: list[RawTrial] | None = getattr(c, "_raw_trials", None)
        if raw_trials:
            status_counts = Counter(t.status.value for t in raw_trials)
            status_parts = [f"{v} {k}" for k, v in sorted(status_counts.items())]
            fields.append(f"Trial statuses: {', '.join(status_parts)}")

        return fields
