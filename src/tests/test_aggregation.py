"""
Tests for Stage 4: Funnel Aggregation (pipeline/stages/aggregation.py)

Test categories:
  PASS NOW   — orchestration wiring with mocks, TRANSITIONS constant
  FAIL NOW   — behavioral contracts for _join, _compute_slice, _transition_rate
               These tests will PASS once the implementation is complete.
"""

from datetime import date
from unittest.mock import MagicMock, call, patch

import pytest

from pipeline.models import (
    AttributeTable,
    Candidate,
    CandidateOutcome,
    CandidateOutcomeRecord,
    CandidateTable,
    FunnelResults,
    FunnelSlice,
    OutcomeTable,
    RawTrial,
    TransitionRate,
    TrialPhase,
    TrialStatus,
    TrialTable,
)
from pipeline.stages.aggregation import FunnelAggregationStage, TRANSITIONS


# ---------------------------------------------------------------------------
# Module-level constants  (PASS NOW)
# ---------------------------------------------------------------------------

class TestTransitionsConstant:
    def test_transitions_is_list(self):
        assert isinstance(TRANSITIONS, list)

    def test_transitions_has_four_entries(self):
        assert len(TRANSITIONS) == 4

    def test_transitions_are_tuples_of_two_strings(self):
        for t in TRANSITIONS:
            assert isinstance(t, tuple)
            assert len(t) == 2
            assert all(isinstance(s, str) for s in t)

    def test_transitions_phase1_to_phase2(self):
        assert ("Phase 1", "Phase 2") in TRANSITIONS

    def test_transitions_phase2_to_phase3(self):
        assert ("Phase 2", "Phase 3") in TRANSITIONS

    def test_transitions_phase3_to_approval(self):
        assert ("Phase 3", "Approval") in TRANSITIONS

    def test_transitions_approval_to_market(self):
        assert ("Approval", "Market") in TRANSITIONS


# ---------------------------------------------------------------------------
# run() orchestration with mocks  (PASS NOW)
# ---------------------------------------------------------------------------

class TestFunnelAggregationStageRunOrchestration:
    """Mock _join and _compute_slice to verify run() assembles FunnelResults correctly."""

    def _make_stage_with_mocks(self, joined_records, slice_result):
        stage = FunnelAggregationStage()
        stage._join = MagicMock(return_value=joined_records)
        stage._compute_slice = MagicMock(return_value=slice_result)
        return stage

    def test_run_returns_funnel_results_type(
        self, sample_candidate_table, sample_attribute_table, sample_outcome_table, sample_funnel_slice
    ):
        stage = self._make_stage_with_mocks([], sample_funnel_slice)
        result = stage.run(sample_candidate_table, sample_attribute_table, sample_outcome_table)
        assert isinstance(result, FunnelResults)

    def test_run_calls_join_with_all_three_tables(
        self, sample_candidate_table, sample_attribute_table, sample_outcome_table, sample_funnel_slice
    ):
        stage = self._make_stage_with_mocks([], sample_funnel_slice)
        stage.run(sample_candidate_table, sample_attribute_table, sample_outcome_table)
        stage._join.assert_called_once_with(
            sample_candidate_table, sample_attribute_table, sample_outcome_table, None
        )

    def test_run_computes_overall_slice_with_none_filters(
        self, sample_candidate_table, sample_attribute_table, sample_outcome_table, sample_funnel_slice
    ):
        stage = self._make_stage_with_mocks([], sample_funnel_slice)
        stage.run(sample_candidate_table, sample_attribute_table, sample_outcome_table)
        calls = stage._compute_slice.call_args_list
        overall_call = [c for c in calls if c.kwargs.get("modality") is None and c.kwargs.get("disease_area") is None]
        assert len(overall_call) >= 1

    def test_run_by_modality_contains_modality_keys_from_records(
        self, sample_candidate_table, sample_attribute_table, sample_outcome_table,
        sample_joined_records, sample_funnel_slice
    ):
        stage = FunnelAggregationStage()
        stage._join = MagicMock(return_value=sample_joined_records)
        stage._compute_slice = MagicMock(return_value=sample_funnel_slice)

        result = stage.run(sample_candidate_table, sample_attribute_table, sample_outcome_table)

        expected_modalities = {r["modality"] for r in sample_joined_records if r["modality"]}
        assert set(result.by_modality.keys()) == expected_modalities

    def test_run_by_disease_area_contains_disease_keys_from_records(
        self, sample_candidate_table, sample_attribute_table, sample_outcome_table,
        sample_joined_records, sample_funnel_slice
    ):
        stage = FunnelAggregationStage()
        stage._join = MagicMock(return_value=sample_joined_records)
        stage._compute_slice = MagicMock(return_value=sample_funnel_slice)

        result = stage.run(sample_candidate_table, sample_attribute_table, sample_outcome_table)

        expected_areas = {r["disease_area"] for r in sample_joined_records if r["disease_area"]}
        assert set(result.by_disease_area.keys()) == expected_areas


# ---------------------------------------------------------------------------
# _join  (FAIL NOW — stub; PASS when implemented)
# ---------------------------------------------------------------------------

class TestJoin:
    def test_join_returns_list(
        self, sample_candidate_table, sample_attribute_table, sample_outcome_table
    ):
        stage = FunnelAggregationStage()
        result = stage._join(sample_candidate_table, sample_attribute_table, sample_outcome_table)
        assert isinstance(result, list)

    def test_join_returns_list_of_dicts(
        self, sample_candidate_table, sample_attribute_table, sample_outcome_table
    ):
        stage = FunnelAggregationStage()
        result = stage._join(sample_candidate_table, sample_attribute_table, sample_outcome_table)
        assert all(isinstance(r, dict) for r in result)

    def test_join_includes_candidate_id_key(
        self, sample_candidate_table, sample_attribute_table, sample_outcome_table
    ):
        stage = FunnelAggregationStage()
        result = stage._join(sample_candidate_table, sample_attribute_table, sample_outcome_table)
        for row in result:
            assert "candidate_id" in row

    def test_join_includes_drug_key(
        self, sample_candidate_table, sample_attribute_table, sample_outcome_table
    ):
        stage = FunnelAggregationStage()
        result = stage._join(sample_candidate_table, sample_attribute_table, sample_outcome_table)
        for row in result:
            assert "drug" in row

    def test_join_includes_modality_key(
        self, sample_candidate_table, sample_attribute_table, sample_outcome_table
    ):
        stage = FunnelAggregationStage()
        result = stage._join(sample_candidate_table, sample_attribute_table, sample_outcome_table)
        for row in result:
            assert "modality" in row

    def test_join_includes_disease_area_key(
        self, sample_candidate_table, sample_attribute_table, sample_outcome_table
    ):
        stage = FunnelAggregationStage()
        result = stage._join(sample_candidate_table, sample_attribute_table, sample_outcome_table)
        for row in result:
            assert "disease_area" in row

    def test_join_includes_highest_phase_key(
        self, sample_candidate_table, sample_attribute_table, sample_outcome_table
    ):
        stage = FunnelAggregationStage()
        result = stage._join(sample_candidate_table, sample_attribute_table, sample_outcome_table)
        for row in result:
            assert "highest_phase" in row

    def test_join_includes_outcome_key(
        self, sample_candidate_table, sample_attribute_table, sample_outcome_table
    ):
        stage = FunnelAggregationStage()
        result = stage._join(sample_candidate_table, sample_attribute_table, sample_outcome_table)
        for row in result:
            assert "outcome" in row

    def test_join_one_row_per_candidate(
        self, sample_candidate_table, sample_attribute_table, sample_outcome_table
    ):
        """Output should have exactly one row per candidate."""
        stage = FunnelAggregationStage()
        result = stage._join(sample_candidate_table, sample_attribute_table, sample_outcome_table)
        assert len(result) == len(sample_candidate_table.candidates)

    def test_join_maps_drug_name_correctly(
        self, single_candidate_table, sample_attribute_table, sample_outcome_table
    ):
        stage = FunnelAggregationStage()
        result = stage._join(single_candidate_table, sample_attribute_table, sample_outcome_table)
        assert result[0]["drug"] == single_candidate_table.candidates[0].drug_name


# ---------------------------------------------------------------------------
# _compute_slice  (FAIL NOW — stub; PASS when implemented)
# ---------------------------------------------------------------------------

class TestComputeSlice:
    def test_compute_slice_returns_funnel_slice(self, sample_joined_records):
        stage = FunnelAggregationStage()
        result = stage._compute_slice(sample_joined_records, modality=None, disease_area=None)
        assert isinstance(result, FunnelSlice)

    def test_compute_slice_overall_counts_all_records(self, sample_joined_records):
        stage = FunnelAggregationStage()
        result = stage._compute_slice(sample_joined_records, modality=None, disease_area=None)
        assert result.candidate_count == len(sample_joined_records)

    def test_compute_slice_modality_filter_counts_only_matching(self, sample_joined_records):
        stage = FunnelAggregationStage()
        result = stage._compute_slice(sample_joined_records, modality="peptide", disease_area=None)
        expected_count = sum(1 for r in sample_joined_records if r["modality"] == "peptide")
        assert result.candidate_count == expected_count

    def test_compute_slice_disease_area_filter_counts_only_matching(self, sample_joined_records):
        stage = FunnelAggregationStage()
        result = stage._compute_slice(sample_joined_records, modality=None, disease_area="oncology")
        expected_count = sum(1 for r in sample_joined_records if r["disease_area"] == "oncology")
        assert result.candidate_count == expected_count

    def test_compute_slice_returns_transitions_list(self, sample_joined_records):
        stage = FunnelAggregationStage()
        result = stage._compute_slice(sample_joined_records, modality=None, disease_area=None)
        assert isinstance(result.transitions, list)

    def test_compute_slice_transitions_count_matches_transitions_constant(self, sample_joined_records):
        stage = FunnelAggregationStage()
        result = stage._compute_slice(sample_joined_records, modality=None, disease_area=None)
        assert len(result.transitions) == len(TRANSITIONS)

    def test_compute_slice_modality_set_on_slice(self, sample_joined_records):
        stage = FunnelAggregationStage()
        result = stage._compute_slice(sample_joined_records, modality="peptide", disease_area=None)
        assert result.modality == "peptide"

    def test_compute_slice_disease_area_set_on_slice(self, sample_joined_records):
        stage = FunnelAggregationStage()
        result = stage._compute_slice(sample_joined_records, modality=None, disease_area="metabolic")
        assert result.disease_area == "metabolic"

    def test_compute_slice_empty_records_returns_zero_count(self):
        stage = FunnelAggregationStage()
        result = stage._compute_slice([], modality=None, disease_area=None)
        assert result.candidate_count == 0


# ---------------------------------------------------------------------------
# _transition_rate  (FAIL NOW — stub; PASS when implemented)
# ---------------------------------------------------------------------------

class TestTransitionRate:
    """Forward-looking cohort semantics.

    A candidate is in the denominator for N→N+1 iff `from_phase` ∈ phases_observed.
    It is in the numerator iff any phase strictly later than N is also observed.
    """

    @staticmethod
    def _record(cid, phases, modality="peptide", disease_area="metabolic",
                outcome="Failed Phase 1", highest_phase="Phase 1",
                approval_date=None, commercialization_date=None):
        return {
            "candidate_id": cid, "drug": cid, "indication": "X",
            "modality": modality, "disease_area": disease_area,
            "highest_phase": highest_phase, "outcome": outcome,
            "approval_date": approval_date, "commercialization_date": commercialization_date,
            "phases_observed": set(phases),
        }

    def test_transition_rate_returns_transition_rate_object(self, sample_joined_records):
        stage = FunnelAggregationStage()
        result = stage._transition_rate(sample_joined_records, "Phase 1", "Phase 2")
        assert isinstance(result, TransitionRate)

    def test_transition_rate_from_phase_matches_input(self, sample_joined_records):
        stage = FunnelAggregationStage()
        result = stage._transition_rate(sample_joined_records, "Phase 1", "Phase 2")
        assert result.from_phase == "Phase 1"

    def test_transition_rate_to_phase_matches_input(self, sample_joined_records):
        stage = FunnelAggregationStage()
        result = stage._transition_rate(sample_joined_records, "Phase 1", "Phase 2")
        assert result.to_phase == "Phase 2"

    def test_transition_rate_in_valid_range(self, sample_joined_records):
        stage = FunnelAggregationStage()
        result = stage._transition_rate(sample_joined_records, "Phase 1", "Phase 2")
        assert 0.0 <= result.rate <= 1.0

    def test_transition_rate_numerator_leq_denominator(self, sample_joined_records):
        stage = FunnelAggregationStage()
        result = stage._transition_rate(sample_joined_records, "Phase 1", "Phase 2")
        assert result.numerator <= result.denominator

    def test_transition_rate_denominator_is_non_negative(self, sample_joined_records):
        stage = FunnelAggregationStage()
        result = stage._transition_rate(sample_joined_records, "Phase 1", "Phase 2")
        assert result.denominator >= 0

    def test_transition_rate_zero_denominator_when_no_cohort_members(self):
        """No candidate observed at from_phase → denominator 0, rate 0.0."""
        records = [self._record("c1", phases={"Phase 1"})]
        stage = FunnelAggregationStage()
        result = stage._transition_rate(records, "Phase 3", "Approval")
        assert result.denominator == 0
        assert result.rate == 0.0

    def test_transition_rate_full_progression_gives_rate_one(self):
        """Every P1-cohort member has a later observation → rate 1.0."""
        records = [
            self._record("c1", phases={"Phase 1", "Phase 2"}),
            self._record("c2", phases={"Phase 1", "Phase 2"}),
        ]
        stage = FunnelAggregationStage()
        result = stage._transition_rate(records, "Phase 1", "Phase 2")
        assert result.denominator == 2
        assert result.numerator == 2
        assert result.rate == pytest.approx(1.0)

    def test_transition_rate_partial_progression(self):
        """3 of 5 P1-cohort members also observed at a later phase → 0.6."""
        records = [
            self._record("c0", phases={"Phase 1", "Phase 2"}),
            self._record("c1", phases={"Phase 1", "Phase 2"}),
            self._record("c2", phases={"Phase 1", "Phase 2"}),
            self._record("c3", phases={"Phase 1"}),
            self._record("c4", phases={"Phase 1"}),
        ]
        stage = FunnelAggregationStage()
        result = stage._transition_rate(records, "Phase 1", "Phase 2")
        assert result.denominator == 5
        assert result.numerator == 3
        assert result.rate == pytest.approx(0.6)

    def test_transition_rate_ongoing_without_terminal_trials_excluded(self):
        """Candidates with no terminal phase observations are not in any cohort."""
        records = [
            self._record("c1", phases={"Phase 1", "Phase 2"}),
            self._record("c2", phases=set(), outcome="Ongoing"),
            self._record("c3", phases=set(), outcome="Unknown"),
        ]
        stage = FunnelAggregationStage()
        result = stage._transition_rate(records, "Phase 1", "Phase 2")
        assert result.denominator == 1
        assert result.numerator == 1
        assert result.rate == pytest.approx(1.0)

    # --- Scenario tests matching the plan (Drugs A / B / C) --------------------

    def test_drug_a_trials_and_approval_counts_across_transitions(self):
        """P1+P2 trials + Approved → P1→P2 success; P2→P3 success (Approval is later); not in P3 cohort."""
        records = [
            self._record(
                "drug_a",
                phases={"Phase 1", "Phase 2", "Approval"},
                outcome="Approved",
                approval_date=date(2022, 1, 1),
            )
        ]
        stage = FunnelAggregationStage()
        p1p2 = stage._transition_rate(records, "Phase 1", "Phase 2")
        p2p3 = stage._transition_rate(records, "Phase 2", "Phase 3")
        p3app = stage._transition_rate(records, "Phase 3", "Approval")
        app_mkt = stage._transition_rate(records, "Approval", "Market")

        assert (p1p2.numerator, p1p2.denominator) == (1, 1)
        assert (p2p3.numerator, p2p3.denominator) == (1, 1)
        assert (p3app.numerator, p3app.denominator) == (0, 0)   # not in P3 cohort
        assert (app_mkt.numerator, app_mkt.denominator) == (0, 1)  # no Market evidence

    def test_drug_b_phase3_only_no_approval(self):
        """P3 trial only, no approval → P3→Approval denom but not numer; absent from P1/P2 cohorts."""
        records = [self._record("drug_b", phases={"Phase 3"}, outcome="Failed Phase 3",
                                highest_phase="Phase 3")]
        stage = FunnelAggregationStage()
        assert stage._transition_rate(records, "Phase 1", "Phase 2").denominator == 0
        assert stage._transition_rate(records, "Phase 2", "Phase 3").denominator == 0
        p3 = stage._transition_rate(records, "Phase 3", "Approval")
        assert p3.denominator == 1
        assert p3.numerator == 0

    def test_drug_c_p1_trial_with_approval(self):
        """P1 trial + Approved → P1→P2 success; not in P2/P3 cohorts."""
        records = [self._record(
            "drug_c", phases={"Phase 1", "Approval"},
            outcome="Approved", approval_date=date(2021, 5, 1),
        )]
        stage = FunnelAggregationStage()
        p1p2 = stage._transition_rate(records, "Phase 1", "Phase 2")
        p2p3 = stage._transition_rate(records, "Phase 2", "Phase 3")
        p3app = stage._transition_rate(records, "Phase 3", "Approval")

        assert (p1p2.numerator, p1p2.denominator) == (1, 1)
        assert (p2p3.numerator, p2p3.denominator) == (0, 0)
        assert (p3app.numerator, p3app.denominator) == (0, 0)

    def test_advancement_via_non_terminal_later_trial(self):
        """COMPLETED P1 + RECRUITING P2 → in P1 cohort, P1→P2 success via advancement."""
        records = [{
            "candidate_id": "c1", "drug": "c1", "indication": "X",
            "modality": "peptide", "disease_area": "metabolic",
            "highest_phase": "Phase 2", "outcome": "Ongoing",
            "approval_date": None, "commercialization_date": None,
            "phases_observed": {"Phase 1"},
            "phases_advanced": {"Phase 1", "Phase 2"},
        }]
        stage = FunnelAggregationStage()
        p1p2 = stage._transition_rate(records, "Phase 1", "Phase 2")
        p2p3 = stage._transition_rate(records, "Phase 2", "Phase 3")
        assert (p1p2.numerator, p1p2.denominator) == (1, 1)
        # Not in P2 cohort — P2 trial wasn't terminal.
        assert p2p3.denominator == 0

    def test_failed_phase_outcome_credits_advancement_not_cohort(self):
        """COMPLETED P1 + FAILED_PHASE_2 verdict → P1→P2 success; P2 not in cohort.

        The adjudicator's FAILED_PHASE_2 verdict evidences the drug reached
        Phase 2 (advancement), but alone is insufficient to place the drug in
        the P2 cohort for downstream denominators.
        """
        records = [{
            "candidate_id": "c1", "drug": "c1", "indication": "X",
            "modality": "peptide", "disease_area": "metabolic",
            "highest_phase": "Phase 2", "outcome": "Failed Phase 2",
            "approval_date": None, "commercialization_date": None,
            "phases_observed": {"Phase 1"},
            "phases_advanced": {"Phase 1", "Phase 2"},
        }]
        stage = FunnelAggregationStage()
        p1p2 = stage._transition_rate(records, "Phase 1", "Phase 2")
        p2p3 = stage._transition_rate(records, "Phase 2", "Phase 3")
        assert (p1p2.numerator, p1p2.denominator) == (1, 1)
        assert p2p3.denominator == 0

    def test_approval_to_market_requires_commercialization_evidence(self):
        """Approval cohort members without Market evidence don't count as success."""
        records = [
            self._record("approved_only", phases={"Phase 3", "Approval"}, outcome="Approved",
                         approval_date=date(2020, 1, 1)),
            self._record("commercialized", phases={"Phase 3", "Approval", "Market"},
                         outcome="Commercialized",
                         approval_date=date(2019, 1, 1),
                         commercialization_date=date(2020, 1, 1)),
        ]
        stage = FunnelAggregationStage()
        result = stage._transition_rate(records, "Approval", "Market")
        assert result.denominator == 2
        assert result.numerator == 1
        assert result.rate == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# _join — date keys  (PASS NOW)
# ---------------------------------------------------------------------------

class TestJoinDateKeys:
    def test_join_includes_approval_date_key(
        self, sample_candidate_table, sample_attribute_table, sample_outcome_table
    ):
        stage = FunnelAggregationStage()
        result = stage._join(sample_candidate_table, sample_attribute_table, sample_outcome_table)
        for row in result:
            assert "approval_date" in row

    def test_join_includes_commercialization_date_key(
        self, sample_candidate_table, sample_attribute_table, sample_outcome_table
    ):
        stage = FunnelAggregationStage()
        result = stage._join(sample_candidate_table, sample_attribute_table, sample_outcome_table)
        for row in result:
            assert "commercialization_date" in row


# ---------------------------------------------------------------------------
# _transition_rate — avg_duration_years  (PASS NOW)
# ---------------------------------------------------------------------------

class TestTransitionRateDuration:
    def _base_record(self, cid, phase, outcome="Ongoing", approval_date=None, commercialization_date=None,
                     phases_observed=None):
        if phases_observed is None:
            phases_observed = set()
        return {
            "candidate_id": cid, "drug": "A", "indication": "X",
            "modality": "peptide", "disease_area": "metabolic",
            "highest_phase": phase, "outcome": outcome,
            "approval_date": approval_date,
            "commercialization_date": commercialization_date,
            "phases_observed": set(phases_observed),
        }

    def test_transition_rate_has_avg_duration_years_attribute(self):
        records = [self._base_record("c1", "Phase 2")]
        stage = FunnelAggregationStage()
        result = stage._transition_rate(records, "Phase 1", "Phase 2")
        assert hasattr(result, "avg_duration_years")

    def test_transition_rate_avg_duration_none_when_no_dates(self):
        records = [self._base_record("c1", "Phase 2")]
        stage = FunnelAggregationStage()
        result = stage._transition_rate(records, "Phase 1", "Phase 2")
        assert result.avg_duration_years is None

    def test_transition_rate_approval_to_market_computes_duration(self):
        records = [
            self._base_record(
                "c1", "Phase 3", outcome="Commercialized",
                approval_date=date(2020, 1, 1),
                commercialization_date=date(2021, 1, 1),
            ),
        ]
        stage = FunnelAggregationStage()
        result = stage._transition_rate(records, "Approval", "Market")
        assert result.avg_duration_years is not None
        assert result.avg_duration_years > 0


# ---------------------------------------------------------------------------
# _join — phases_observed construction
# ---------------------------------------------------------------------------

def _build(cid, trial_ids, modality="peptide", disease_area="metabolic",
           outcome=CandidateOutcome.ONGOING, highest_phase=TrialPhase.PHASE_1,
           approval_date=None, commercialization_date=None):
    from pipeline.models import CandidateAttributes
    cand = Candidate(
        candidate_id=cid, drug_name=cid, indication="X",
        trial_ids=list(trial_ids), highest_phase=highest_phase,
    )
    attrs = CandidateAttributes(
        candidate_id=cid, drug_modality=modality, disease_area=disease_area,
    )
    out = CandidateOutcomeRecord(
        candidate_id=cid, outcome=outcome,
        approval_date=approval_date, commercialization_date=commercialization_date,
    )
    return cand, attrs, out


def _tables(triples, trials):
    candidates = [c for c, _, _ in triples]
    attr_map = {c.candidate_id: a for c, a, _ in triples}
    out_map = {c.candidate_id: o for c, _, o in triples}
    return (
        CandidateTable(candidates=candidates),
        AttributeTable(attributes=attr_map),
        OutcomeTable(outcomes=out_map),
        TrialTable(trials=list(trials)),
    )


class TestJoinPhasesObserved:
    def test_terminal_completed_trial_adds_phase(self):
        triples = [_build("c1", ["N1"])]
        trials = [RawTrial(nct_id="N1", title="", intervention="", indication="",
                           sponsor="", phase=TrialPhase.PHASE_2, status=TrialStatus.COMPLETED)]
        ct, at, ot, tt = _tables(triples, trials)
        records = FunnelAggregationStage()._join(ct, at, ot, tt)
        assert records[0]["phases_observed"] == {"Phase 2"}

    def test_recruiting_trial_does_not_add_phase(self):
        triples = [_build("c1", ["N1"])]
        trials = [RawTrial(nct_id="N1", title="", intervention="", indication="",
                           sponsor="", phase=TrialPhase.PHASE_1, status=TrialStatus.RECRUITING)]
        ct, at, ot, tt = _tables(triples, trials)
        records = FunnelAggregationStage()._join(ct, at, ot, tt)
        assert records[0]["phases_observed"] == set()

    def test_suspended_trial_adds_phase(self):
        triples = [_build("c1", ["N1"])]
        trials = [RawTrial(nct_id="N1", title="", intervention="", indication="",
                           sponsor="", phase=TrialPhase.PHASE_2, status=TrialStatus.SUSPENDED)]
        ct, at, ot, tt = _tables(triples, trials)
        records = FunnelAggregationStage()._join(ct, at, ot, tt)
        assert records[0]["phases_observed"] == {"Phase 2"}

    def test_active_not_recruiting_does_not_add_phase(self):
        triples = [_build("c1", ["N1"])]
        trials = [RawTrial(nct_id="N1", title="", intervention="", indication="",
                           sponsor="", phase=TrialPhase.PHASE_2,
                           status=TrialStatus.ACTIVE_NOT_RECRUITING)]
        ct, at, ot, tt = _tables(triples, trials)
        records = FunnelAggregationStage()._join(ct, at, ot, tt)
        assert records[0]["phases_observed"] == set()

    def test_phase4_terminal_trial_contributes_to_approval_cohort(self):
        triples = [_build("c1", ["N1"])]
        trials = [RawTrial(nct_id="N1", title="", intervention="", indication="",
                           sponsor="", phase=TrialPhase.PHASE_4, status=TrialStatus.COMPLETED)]
        ct, at, ot, tt = _tables(triples, trials)
        records = FunnelAggregationStage()._join(ct, at, ot, tt)
        assert "Approval" in records[0]["phases_observed"]

    def test_approved_outcome_adds_approval_cohort(self):
        triples = [_build("c1", [], outcome=CandidateOutcome.APPROVED,
                          approval_date=date(2020, 1, 1))]
        ct, at, ot, tt = _tables(triples, [])
        records = FunnelAggregationStage()._join(ct, at, ot, tt)
        assert "Approval" in records[0]["phases_observed"]
        assert "Market" not in records[0]["phases_observed"]

    def test_commercialized_outcome_adds_market_cohort(self):
        triples = [_build("c1", [], outcome=CandidateOutcome.COMMERCIALIZED,
                          approval_date=date(2019, 1, 1),
                          commercialization_date=date(2020, 1, 1))]
        ct, at, ot, tt = _tables(triples, [])
        records = FunnelAggregationStage()._join(ct, at, ot, tt)
        assert {"Approval", "Market"} <= records[0]["phases_observed"]

    def test_join_without_trial_table_yields_empty_clinical_phases(self):
        triples = [_build("c1", ["N1"])]
        ct, at, ot, _ = _tables(triples, [])
        records = FunnelAggregationStage()._join(ct, at, ot, None)
        assert records[0]["phases_observed"] == set()
        assert records[0]["phases_advanced"] == set()

    def test_recruiting_trial_counts_as_advancement_only(self):
        """COMPLETED P1 + RECRUITING P2 → P1 in cohort, P2 in advancement only."""
        triples = [_build("c1", ["N1", "N2"])]
        trials = [
            RawTrial(nct_id="N1", title="", intervention="", indication="",
                     sponsor="", phase=TrialPhase.PHASE_1, status=TrialStatus.COMPLETED),
            RawTrial(nct_id="N2", title="", intervention="", indication="",
                     sponsor="", phase=TrialPhase.PHASE_2, status=TrialStatus.RECRUITING),
        ]
        ct, at, ot, tt = _tables(triples, trials)
        records = FunnelAggregationStage()._join(ct, at, ot, tt)
        assert records[0]["phases_observed"] == {"Phase 1"}
        assert records[0]["phases_advanced"] == {"Phase 1", "Phase 2"}

    def test_active_not_recruiting_late_phase_counts_as_advancement(self):
        triples = [_build("c1", ["N1", "N2"])]
        trials = [
            RawTrial(nct_id="N1", title="", intervention="", indication="",
                     sponsor="", phase=TrialPhase.PHASE_1, status=TrialStatus.COMPLETED),
            RawTrial(nct_id="N2", title="", intervention="", indication="",
                     sponsor="", phase=TrialPhase.PHASE_3,
                     status=TrialStatus.ACTIVE_NOT_RECRUITING),
        ]
        ct, at, ot, tt = _tables(triples, trials)
        records = FunnelAggregationStage()._join(ct, at, ot, tt)
        assert records[0]["phases_observed"] == {"Phase 1"}
        assert "Phase 3" in records[0]["phases_advanced"]

    def test_failed_phase_outcome_enriches_advancement_only(self):
        """FAILED_PHASE_2 outcome credits advancement but not cohort membership.

        Without corroborating terminal trial data at Phase 2, the adjudicator
        verdict alone must not place the candidate in the Phase 2 cohort.
        """
        triples = [_build("c1", [], outcome=CandidateOutcome.FAILED_PHASE_2)]
        ct, at, ot, tt = _tables(triples, [])
        records = FunnelAggregationStage()._join(ct, at, ot, tt)
        assert "Phase 2" not in records[0]["phases_observed"]
        assert "Phase 2" in records[0]["phases_advanced"]

    def test_approved_outcome_does_not_backfill_clinical_phases(self):
        """APPROVED outcome without trial data stays out of P1/P2/P3 clinical cohorts."""
        triples = [_build("c1", [], outcome=CandidateOutcome.APPROVED,
                          approval_date=date(2020, 1, 1))]
        ct, at, ot, tt = _tables(triples, [])
        records = FunnelAggregationStage()._join(ct, at, ot, tt)
        assert records[0]["phases_observed"] == {"Approval"}
        assert records[0]["phases_advanced"] == {"Approval"}


# ---------------------------------------------------------------------------
# by_modality_and_disease cross-stratified slices  (PASS NOW)
# ---------------------------------------------------------------------------

class TestByModalityAndDisease:
    """Verify that run() populates by_modality_and_disease with cross-stratified slices."""

    def _make_tables(self, records):
        """Build minimal CandidateTable, AttributeTable, OutcomeTable from flat records."""
        from pipeline.models import (
            Candidate, CandidateAttributes, CandidateOutcomeRecord,
            CandidateOutcome, TrialPhase,
        )
        candidates = []
        attributes = {}
        outcomes = {}
        for r in records:
            cid = r["candidate_id"]
            candidates.append(
                Candidate(
                    candidate_id=cid,
                    drug_name=r["drug"],
                    indication=r["indication"],
                    highest_phase=TrialPhase(r["highest_phase"]),
                )
            )
            if r.get("modality") and r.get("disease_area"):
                attributes[cid] = CandidateAttributes(
                    candidate_id=cid,
                    drug_modality=r["modality"],
                    disease_area=r["disease_area"],
                )
            outcomes[cid] = CandidateOutcomeRecord(
                candidate_id=cid,
                outcome=CandidateOutcome.ONGOING,
            )
        return (
            CandidateTable(candidates=candidates),
            AttributeTable(attributes=attributes),
            OutcomeTable(outcomes=outcomes),
        )

    def test_run_by_modality_and_disease_has_cross_stratified_keys(self):
        records = [
            {"candidate_id": "c1", "drug": "A", "indication": "X",
             "modality": "peptide", "disease_area": "oncology", "highest_phase": "Phase 1"},
            {"candidate_id": "c2", "drug": "B", "indication": "Y",
             "modality": "biologic", "disease_area": "metabolic", "highest_phase": "Phase 2"},
        ]
        ct, at, ot = self._make_tables(records)
        result = FunnelAggregationStage().run(ct, at, ot)
        for key in result.by_modality_and_disease:
            assert isinstance(key, tuple) and len(key) == 2
            assert all(isinstance(s, str) for s in key)

    def test_run_by_modality_and_disease_counts_only_matching_modality_and_disease(self):
        records = [
            {"candidate_id": "c1", "drug": "A", "indication": "X",
             "modality": "peptide", "disease_area": "oncology", "highest_phase": "Phase 1"},
            {"candidate_id": "c2", "drug": "B", "indication": "Y",
             "modality": "peptide", "disease_area": "oncology", "highest_phase": "Phase 2"},
            {"candidate_id": "c3", "drug": "C", "indication": "Z",
             "modality": "biologic", "disease_area": "oncology", "highest_phase": "Phase 1"},
            {"candidate_id": "c4", "drug": "D", "indication": "W",
             "modality": "peptide", "disease_area": "metabolic", "highest_phase": "Phase 1"},
        ]
        ct, at, ot = self._make_tables(records)
        result = FunnelAggregationStage().run(ct, at, ot)
        # Only 2 peptide candidates in oncology (c1, c2)
        assert result.by_modality_and_disease[("peptide", "oncology")].candidate_count == 2
        # Only 1 biologic candidate in oncology (c3)
        assert result.by_modality_and_disease[("biologic", "oncology")].candidate_count == 1
        # Only 1 peptide candidate in metabolic (c4)
        assert result.by_modality_and_disease[("peptide", "metabolic")].candidate_count == 1
