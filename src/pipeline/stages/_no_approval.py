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

    Returns ONGOING when there are no trial dates, when the latest trial
    activity is within the failure window, or when the highest phase is
    N/A or Unknown. Otherwise returns FAILED_PHASE_<phase>; phase 4
    candidates without an approval match are bucketed with phase 3.
    """
    last_update = candidate.latest_completion_date or candidate.earliest_start_date
    if last_update is None:
        return CandidateOutcome.ONGOING

    as_of = as_of or date.today()
    if as_of - last_update < timedelta(days=failure_window_days):
        return CandidateOutcome.ONGOING

    phase = _phase_to_int(candidate.highest_phase)
    if phase >= 3:
        return CandidateOutcome.FAILED_PHASE_3
    if phase == 2:
        return CandidateOutcome.FAILED_PHASE_2
    if phase == 1:
        return CandidateOutcome.FAILED_PHASE_1
    return CandidateOutcome.ONGOING
