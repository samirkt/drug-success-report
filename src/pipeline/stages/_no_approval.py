"""Shared no-approval -> CandidateOutcome mapping.

Lifted from `adjudication_fda.AdjudicationStage._failure_or_ongoing_outcome`
so the FDA-timeline and NDC-indication adjudicators agree on how to map a
"not approved" verdict to the funnel-relevant enum.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Optional

from ..models import Candidate, CandidateOutcome, TrialPhase

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


def classify_no_approval(
    candidate: Candidate,
    *,
    failure_window_days: int,
    as_of: Optional[date] = None,
) -> CandidateOutcome:
    """Map a non-approval result to ONGOING / FAILED_PHASE_{1,2,3}.

    Recency is judged from ``latest_update_submitted_date`` — the most
    recent ClinicalTrials.gov submission across the candidate's trials —
    which tracks any sponsor activity (status, results, protocol
    amendments) rather than only completion. ONGOING is reserved for
    candidates whose latest update is within ``failure_window_days`` of
    ``as_of``; everything else (missing date or stale) is failure,
    bucketed by ``highest_phase``. Phase 4 candidates without an approval
    match bucket with phase 3; N/A / Unknown phases bucket with phase 1
    as the most conservative "no evidence of advancement" choice.
    """
    last_update = candidate.latest_update_submitted_date
    as_of = as_of or date.today()

    if last_update is not None and as_of - last_update < timedelta(
        days=failure_window_days
    ):
        return CandidateOutcome.ONGOING

    phase = _phase_to_int(candidate.highest_phase)
    if phase >= 3:
        return CandidateOutcome.FAILED_PHASE_3
    if phase == 2:
        return CandidateOutcome.FAILED_PHASE_2
    return CandidateOutcome.FAILED_PHASE_1
