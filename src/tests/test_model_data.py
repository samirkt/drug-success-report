"""Tests for model/data.py phase-mode label dispatch.

The cohort logic itself lives in src/pipeline/stages/aggregation.py
(FunnelAggregationStage._phases_observed); these tests exercise the
modeling-side wrappers and assert the contract the modeling code
depends on.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.data import (  # noqa: E402
    _build_trial_index,
    apply_label,
    apply_phase_label,
    compute_phase_sets,
)
from model.config import LabelConfig  # noqa: E402

from pipeline.models import TrialStatus  # noqa: E402
from pipeline.stages.aggregation import FunnelAggregationStage  # noqa: E402


# ---------------------------------------------------------------------------
# Aggregation contract — the canary that fires if _phases_observed changes.
# ---------------------------------------------------------------------------

def test_aggregation_phases_observed_signature():
    """If the private aggregation method's shape changes, modeling labels
    silently break — this test makes that loud."""
    trial_index = {
        "NCT001": ("Phase 1", TrialStatus.COMPLETED, date(2010, 1, 1), date(2011, 1, 1)),
        "NCT002": ("Phase 2", TrialStatus.COMPLETED, date(2012, 1, 1), date(2013, 1, 1)),
    }
    observed, advanced = FunnelAggregationStage._phases_observed(
        trial_ids=["NCT001", "NCT002"],
        trial_index=trial_index,
        outcome="Failed Phase 2",
        approval_date=None,
        commercialization_date=None,
        reference_date=date(2026, 1, 1),
        stale_cutoff_years=2.0,
        back_propagate_approval=False,
        highest_phase="Phase 2",
    )
    assert isinstance(observed, set)
    assert isinstance(advanced, set)
    assert "Phase 1" in observed
    assert "Phase 2" in observed
    assert "Phase 2" in advanced


# ---------------------------------------------------------------------------
# Trial-index builder
# ---------------------------------------------------------------------------

def test_build_trial_index_coerces_dates_and_status():
    df = pd.DataFrame({
        "nct_id": ["NCT001", "NCT002", "NCT003"],
        "trial_phase": ["Phase 1", "Phase 2", "Phase 3"],
        "trial_status": ["Completed", "weird-status-not-in-enum", None],
        "trial_start_date": [pd.Timestamp("2010-01-01"), "2012-01-01", None],
        "trial_completion_date": [pd.Timestamp("2011-01-01"), None, pd.Timestamp("2015-01-01")],
    })
    idx = _build_trial_index(df)

    assert idx["NCT001"][0] == "Phase 1"
    assert idx["NCT001"][1] == TrialStatus.COMPLETED
    assert idx["NCT001"][2] == date(2010, 1, 1)
    assert idx["NCT001"][3] == date(2011, 1, 1)

    # Unknown status → falls back to TrialStatus.UNKNOWN, doesn't raise.
    assert idx["NCT002"][1] == TrialStatus.UNKNOWN
    # String date parsed.
    assert idx["NCT002"][2] == date(2012, 1, 1)

    # Missing status → UNKNOWN.
    assert idx["NCT003"][1] == TrialStatus.UNKNOWN


# ---------------------------------------------------------------------------
# Per-phase labeling
# ---------------------------------------------------------------------------

@pytest.fixture
def trial_index_simple():
    """Three candidates' worth of trials.

    cand_A: Phase 1 + Phase 2 both completed.
    cand_B: Phase 3 only, completed.       (Approved with no P1/P2 evidence.)
    cand_C: ongoing — single trial recently started, no terminal status.
    """
    return _build_trial_index(pd.DataFrame({
        "nct_id":            ["A1", "A2", "B1", "C1"],
        "trial_phase":       ["Phase 1", "Phase 2", "Phase 3", "Phase 2"],
        "trial_status":      ["Completed", "Completed", "Completed", "Recruiting"],
        "trial_start_date":  [pd.Timestamp("2010-01-01"), pd.Timestamp("2012-01-01"),
                              pd.Timestamp("2014-01-01"), pd.Timestamp("2025-06-01")],
        "trial_completion_date": [pd.Timestamp("2011-01-01"), pd.Timestamp("2013-01-01"),
                                  pd.Timestamp("2016-01-01"), None],
    }))


def test_compute_phase_sets_phase1_phase2_advance(trial_index_simple):
    observed, advanced = compute_phase_sets(
        trial_ids=["A1", "A2"],
        outcome="Failed Phase 2",
        approval_date=None,
        commercialization_date=None,
        highest_phase="Phase 2",
        trial_index=trial_index_simple,
        reference_date=date(2026, 1, 1),
    )
    assert "Phase 1" in observed
    assert "Phase 2" in observed
    assert "Phase 2" in advanced
    assert "Phase 3" not in advanced


def test_compute_phase_sets_approved_without_phase1_evidence(trial_index_simple):
    observed, advanced = compute_phase_sets(
        trial_ids=["B1"],
        outcome="Approved",
        approval_date=date(2017, 1, 1),
        commercialization_date=None,
        highest_phase="Phase 3",
        trial_index=trial_index_simple,
        reference_date=date(2026, 1, 1),
    )
    # No Phase 1 trial evidence → Phase 1 NOT in cohort.
    assert "Phase 1" not in observed
    assert "Phase 3" in observed
    assert "Approval" in observed


def test_apply_phase_label_three_candidates(trial_index_simple):
    df = pd.DataFrame([
        {  # cand_A: P1+P2 completed, Failed Phase 2 → P1→P2 y=1, P2→P3 y=0
            "candidate_id": "cand_A",
            "trial_ids": ["A1", "A2"],
            "outcome": "Failed Phase 2",
            "approval_date": None,
            "commercialization_date": None,
            "highest_phase": "Phase 2",
        },
        {  # cand_B: P3 only, Approved → P1→P2 excluded, P3→Approval y=1
            "candidate_id": "cand_B",
            "trial_ids": ["B1"],
            "outcome": "Approved",
            "approval_date": date(2017, 1, 1),
            "commercialization_date": None,
            "highest_phase": "Phase 3",
        },
        {  # cand_C: ongoing → excluded from every cohort
            "candidate_id": "cand_C",
            "trial_ids": ["C1"],
            "outcome": "Ongoing",
            "approval_date": None,
            "commercialization_date": None,
            "highest_phase": "Phase 2",
        },
    ])

    p1 = apply_phase_label(df, from_phase=1, trial_index=trial_index_simple,
                           reference_date=date(2026, 1, 1))
    assert set(p1["candidate_id"]) == {"cand_A"}
    assert int(p1.loc[p1["candidate_id"] == "cand_A", "y"].iloc[0]) == 1

    p2 = apply_phase_label(df, from_phase=2, trial_index=trial_index_simple,
                           reference_date=date(2026, 1, 1))
    assert set(p2["candidate_id"]) == {"cand_A"}
    assert int(p2.loc[p2["candidate_id"] == "cand_A", "y"].iloc[0]) == 0

    p3 = apply_phase_label(df, from_phase=3, trial_index=trial_index_simple,
                           reference_date=date(2026, 1, 1))
    assert set(p3["candidate_id"]) == {"cand_B"}
    assert int(p3.loc[p3["candidate_id"] == "cand_B", "y"].iloc[0]) == 1


def test_apply_phase_label_rejects_invalid_from_phase(trial_index_simple):
    df = pd.DataFrame([{"candidate_id": "x", "trial_ids": [], "outcome": "Ongoing",
                        "approval_date": None, "commercialization_date": None,
                        "highest_phase": "Phase 1"}])
    with pytest.raises(ValueError):
        apply_phase_label(df, from_phase=4, trial_index=trial_index_simple)


# ---------------------------------------------------------------------------
# LabelConfig validation
# ---------------------------------------------------------------------------

def test_labelconfig_phase_requires_from_phase():
    with pytest.raises(ValueError):
        LabelConfig(mode="phase", from_phase=None)
    with pytest.raises(ValueError):
        LabelConfig(mode="phase", from_phase=4)
    # Valid:
    cfg = LabelConfig(mode="phase", from_phase=2)
    assert cfg.from_phase == 2


def test_labelconfig_overall_clears_from_phase():
    cfg = LabelConfig(mode="overall", from_phase=2)
    assert cfg.from_phase is None


# ---------------------------------------------------------------------------
# Overall-mode label dispatch — sanity check that we didn't break it.
# ---------------------------------------------------------------------------

def test_apply_label_overall_mode_unchanged():
    df = pd.DataFrame({
        "candidate_id": ["a", "b", "c", "d", "e"],
        "outcome": ["Approved", "Failed Phase 2", "Ongoing", "Commercialized", "Unknown"],
    })
    cfg = LabelConfig()
    out = apply_label(df, cfg)
    # Ongoing + Unknown dropped; 3 rows remain.
    assert len(out) == 3
    assert int(out["y"].sum()) == 2  # Approved + Commercialized
