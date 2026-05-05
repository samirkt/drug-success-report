"""Unit tests for the shared no-approval -> CandidateOutcome classifier."""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from pipeline.models import Candidate, CandidateOutcome, TrialPhase
from pipeline.stages._no_approval import classify_no_approval


def _candidate(
    *,
    phase: TrialPhase = TrialPhase.PHASE_2,
    latest: date | None = None,
    earliest: date | None = None,
) -> Candidate:
    return Candidate(
        candidate_id="C1",
        drug_name="DrugA",
        indication="Some Disease",
        highest_phase=phase,
        latest_completion_date=latest,
        earliest_start_date=earliest,
    )


class TestClassifyNoApproval:
    def test_no_dates_returns_ongoing(self):
        cand = _candidate(phase=TrialPhase.PHASE_3, latest=None, earliest=None)
        assert classify_no_approval(cand, failure_window_days=730) == CandidateOutcome.ONGOING

    def test_recent_activity_returns_ongoing(self):
        as_of = date(2025, 1, 1)
        recent = as_of - timedelta(days=100)
        cand = _candidate(phase=TrialPhase.PHASE_3, latest=recent)
        assert classify_no_approval(
            cand, failure_window_days=730, as_of=as_of
        ) == CandidateOutcome.ONGOING

    def test_stale_phase_1_returns_failed_phase_1(self):
        as_of = date(2025, 1, 1)
        stale = as_of - timedelta(days=900)
        cand = _candidate(phase=TrialPhase.PHASE_1, latest=stale)
        assert classify_no_approval(
            cand, failure_window_days=730, as_of=as_of
        ) == CandidateOutcome.FAILED_PHASE_1

    def test_stale_phase_2_returns_failed_phase_2(self):
        as_of = date(2025, 1, 1)
        stale = as_of - timedelta(days=900)
        cand = _candidate(phase=TrialPhase.PHASE_2, latest=stale)
        assert classify_no_approval(
            cand, failure_window_days=730, as_of=as_of
        ) == CandidateOutcome.FAILED_PHASE_2

    def test_stale_phase_3_returns_failed_phase_3(self):
        as_of = date(2025, 1, 1)
        stale = as_of - timedelta(days=900)
        cand = _candidate(phase=TrialPhase.PHASE_3, latest=stale)
        assert classify_no_approval(
            cand, failure_window_days=730, as_of=as_of
        ) == CandidateOutcome.FAILED_PHASE_3

    def test_stale_phase_4_buckets_with_phase_3(self):
        as_of = date(2025, 1, 1)
        stale = as_of - timedelta(days=900)
        cand = _candidate(phase=TrialPhase.PHASE_4, latest=stale)
        assert classify_no_approval(
            cand, failure_window_days=730, as_of=as_of
        ) == CandidateOutcome.FAILED_PHASE_3

    def test_stale_unknown_phase_returns_ongoing(self):
        as_of = date(2025, 1, 1)
        stale = as_of - timedelta(days=900)
        cand = _candidate(phase=TrialPhase.UNKNOWN, latest=stale)
        assert classify_no_approval(
            cand, failure_window_days=730, as_of=as_of
        ) == CandidateOutcome.ONGOING

    def test_falls_back_to_earliest_start_date(self):
        as_of = date(2025, 1, 1)
        stale = as_of - timedelta(days=900)
        cand = _candidate(phase=TrialPhase.PHASE_2, latest=None, earliest=stale)
        assert classify_no_approval(
            cand, failure_window_days=730, as_of=as_of
        ) == CandidateOutcome.FAILED_PHASE_2

    def test_default_as_of_is_today(self):
        # Should not crash when as_of is not provided
        cand = _candidate(phase=TrialPhase.PHASE_2, latest=date(2000, 1, 1))
        result = classify_no_approval(cand, failure_window_days=730)
        assert result == CandidateOutcome.FAILED_PHASE_2
