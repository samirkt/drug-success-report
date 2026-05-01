"""Tests for the per-trial label inference rule.

The function is pure: status × candidate outcome × phase -> {0,1,None}.
We exhaust the rule paths so HINT export's filters behave predictably.
"""

from __future__ import annotations

import pytest

from pipeline.models import CandidateOutcome, TrialPhase, TrialStatus
from pipeline.trial_labels import infer_trial_label


@pytest.mark.parametrize("status", [
    TrialStatus.TERMINATED,
    TrialStatus.WITHDRAWN,
    TrialStatus.SUSPENDED,
])
def test_failure_status_overrides_outcome(status):
    # Even an APPROVED candidate with a TERMINATED early trial -> 0
    assert infer_trial_label(
        candidate_outcome=CandidateOutcome.APPROVED,
        trial_phase=TrialPhase.PHASE_2,
        trial_status=status,
    ) == 0


def test_approved_candidate_completed_trial_is_one():
    assert infer_trial_label(
        candidate_outcome=CandidateOutcome.APPROVED,
        trial_phase=TrialPhase.PHASE_2,
        trial_status=TrialStatus.COMPLETED,
    ) == 1


def test_commercialized_candidate_completed_trial_is_one():
    assert infer_trial_label(
        candidate_outcome=CandidateOutcome.COMMERCIALIZED,
        trial_phase=TrialPhase.PHASE_3,
        trial_status=TrialStatus.COMPLETED,
    ) == 1


@pytest.mark.parametrize(("outcome", "phase", "expected"), [
    # FAILED_PHASE_2 -> phase 1 trial succeeded, phase 2 failed, phase 3 dropped
    (CandidateOutcome.FAILED_PHASE_2, TrialPhase.PHASE_1, 1),
    (CandidateOutcome.FAILED_PHASE_2, TrialPhase.PHASE_2, 0),
    (CandidateOutcome.FAILED_PHASE_2, TrialPhase.PHASE_3, None),
    # FAILED_PHASE_3 -> phases 1 and 2 succeeded, phase 3 failed, phase 4 dropped
    (CandidateOutcome.FAILED_PHASE_3, TrialPhase.PHASE_1, 1),
    (CandidateOutcome.FAILED_PHASE_3, TrialPhase.PHASE_2, 1),
    (CandidateOutcome.FAILED_PHASE_3, TrialPhase.PHASE_3, 0),
    # FAILED_PHASE_1 -> only phase 1 is a 0; nothing earlier exists
    (CandidateOutcome.FAILED_PHASE_1, TrialPhase.PHASE_1, 0),
])
def test_failed_phase_n_phase_mapping(outcome, phase, expected):
    assert infer_trial_label(
        candidate_outcome=outcome,
        trial_phase=phase,
        trial_status=TrialStatus.COMPLETED,
    ) == expected


def test_failed_phase_with_unknown_phase_returns_none():
    assert infer_trial_label(
        candidate_outcome=CandidateOutcome.FAILED_PHASE_2,
        trial_phase=TrialPhase.UNKNOWN,
        trial_status=TrialStatus.COMPLETED,
    ) is None


@pytest.mark.parametrize("outcome", [CandidateOutcome.ONGOING, CandidateOutcome.UNKNOWN])
def test_ongoing_or_unknown_outcome_returns_none(outcome):
    assert infer_trial_label(
        candidate_outcome=outcome,
        trial_phase=TrialPhase.PHASE_2,
        trial_status=TrialStatus.COMPLETED,
    ) is None


def test_ongoing_with_failure_status_still_returns_zero():
    # The hard-failed status check happens first, so a TERMINATED trial
    # under an ONGOING candidate still gets 0 — useful signal.
    assert infer_trial_label(
        candidate_outcome=CandidateOutcome.ONGOING,
        trial_phase=TrialPhase.PHASE_2,
        trial_status=TrialStatus.TERMINATED,
    ) == 0
