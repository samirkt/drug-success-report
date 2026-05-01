"""Per-trial label inference for HINT (and other per-trial modeling)."""

from __future__ import annotations

from .models import CandidateOutcome, TrialPhase, TrialStatus

_FAILURE_STATUSES = {
    TrialStatus.TERMINATED,
    TrialStatus.WITHDRAWN,
    TrialStatus.SUSPENDED,
}

_FAILED_PHASE_NUM = {
    CandidateOutcome.FAILED_PHASE_1: 1,
    CandidateOutcome.FAILED_PHASE_2: 2,
    CandidateOutcome.FAILED_PHASE_3: 3,
}

_PHASE_NUM = {
    TrialPhase.PHASE_1: 1,
    TrialPhase.PHASE_2: 2,
    TrialPhase.PHASE_3: 3,
    TrialPhase.PHASE_4: 4,
}


def infer_trial_label(
    *,
    candidate_outcome: CandidateOutcome,
    trial_phase: TrialPhase,
    trial_status: TrialStatus,
) -> int | None:
    """Infer a binary 0/1 outcome label for a single trial.

    Returns None when the label can't be determined (ongoing/unknown drug
    outcome, or a phase that the drug never actually reached).
    """
    if trial_status in _FAILURE_STATUSES:
        return 0
    if candidate_outcome in (CandidateOutcome.APPROVED, CandidateOutcome.COMMERCIALIZED):
        return 1
    failed_at = _FAILED_PHASE_NUM.get(candidate_outcome)
    if failed_at is None:
        return None
    phase_num = _PHASE_NUM.get(trial_phase)
    if phase_num is None:
        return None
    if phase_num < failed_at:
        return 1
    if phase_num == failed_at:
        return 0
    return None
