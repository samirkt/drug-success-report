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
    CandidateTable,
    FunnelResults,
    FunnelSlice,
    OutcomeTable,
    TransitionRate,
    TrialPhase,
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
            sample_candidate_table, sample_attribute_table, sample_outcome_table
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

    def test_transition_rate_zero_denominator_gives_zero_rate(self):
        """When no candidates reached the from_phase, rate should be 0.0."""
        stage = FunnelAggregationStage()
        # All records are Phase 1 only — none reached Phase 3
        records = [
            {"candidate_id": "c1", "drug": "A", "indication": "X",
             "modality": "peptide", "disease_area": "metabolic",
             "highest_phase": "Phase 1", "outcome": "Ongoing"},
        ]
        result = stage._transition_rate(records, "Phase 3", "Approval")
        assert result.denominator == 0
        assert result.rate == 0.0

    def test_transition_rate_full_progression_gives_rate_one(self):
        """Candidates that all progressed should give rate = 1.0."""
        records = [
            {"candidate_id": "c1", "drug": "A", "indication": "X",
             "modality": "peptide", "disease_area": "metabolic",
             "highest_phase": "Phase 2", "outcome": "Failed Phase 2"},
            {"candidate_id": "c2", "drug": "B", "indication": "Y",
             "modality": "peptide", "disease_area": "metabolic",
             "highest_phase": "Phase 2", "outcome": "Failed Phase 2"},
        ]
        stage = FunnelAggregationStage()
        result = stage._transition_rate(records, "Phase 1", "Phase 2")
        assert result.rate == pytest.approx(1.0)

    def test_transition_rate_partial_progression(self):
        """3 of 5 resolved candidates progressed from Phase 1 → Phase 2 = 0.6."""
        records = [
            {"candidate_id": f"c{i}", "drug": "A", "indication": "X",
             "modality": "peptide", "disease_area": "metabolic",
             "highest_phase": phase, "outcome": "Failed Phase 2" if phase == "Phase 2" else "Failed Phase 1",
             "approval_date": None, "commercialization_date": None}
            for i, phase in enumerate(
                ["Phase 2", "Phase 2", "Phase 2", "Phase 1", "Phase 1"]
            )
        ]
        stage = FunnelAggregationStage()
        result = stage._transition_rate(records, "Phase 1", "Phase 2")
        assert result.denominator == 5
        assert result.numerator == 3
        assert result.rate == pytest.approx(0.6)

    def test_transition_rate_excludes_ongoing_from_denominator(self):
        """Ongoing/Unknown candidates should not count in numerator or denominator."""
        records = [
            {"candidate_id": "c1", "drug": "A", "indication": "X",
             "modality": "peptide", "disease_area": "metabolic",
             "highest_phase": "Phase 2", "outcome": "Failed Phase 2"},
            {"candidate_id": "c2", "drug": "B", "indication": "Y",
             "modality": "peptide", "disease_area": "metabolic",
             "highest_phase": "Phase 1", "outcome": "Ongoing"},
            {"candidate_id": "c3", "drug": "C", "indication": "Z",
             "modality": "peptide", "disease_area": "metabolic",
             "highest_phase": "Phase 1", "outcome": "Unknown"},
        ]
        stage = FunnelAggregationStage()
        result = stage._transition_rate(records, "Phase 1", "Phase 2")
        # Only c1 is resolved and reached Phase 1+; c2 and c3 are excluded
        assert result.denominator == 1
        assert result.numerator == 1
        assert result.rate == pytest.approx(1.0)


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
    def _base_record(self, cid, phase, outcome="Ongoing", approval_date=None, commercialization_date=None):
        return {
            "candidate_id": cid, "drug": "A", "indication": "X",
            "modality": "peptide", "disease_area": "metabolic",
            "highest_phase": phase, "outcome": outcome,
            "approval_date": approval_date,
            "commercialization_date": commercialization_date,
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
