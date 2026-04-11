"""
Tests for pipeline/models.py

All tests in this file should PASS immediately — they validate enum values,
dataclass defaults, and utility methods that are already implemented.
"""

import pytest

from pipeline.models import (
    AttributeTable,
    Candidate,
    CandidateAttributes,
    CandidateOutcome,
    CandidateOutcomeRecord,
    CandidateTable,
    FunnelResults,
    FunnelSlice,
    OutcomeTable,
    RawTrial,
    ReportOutput,
    TransitionRate,
    TrialPhase,
    TrialStatus,
    TrialTable,
)


# ---------------------------------------------------------------------------
# TrialPhase enum
# ---------------------------------------------------------------------------

class TestTrialPhase:
    def test_phase1_value(self):
        assert TrialPhase.PHASE_1.value == "Phase 1"

    def test_phase2_value(self):
        assert TrialPhase.PHASE_2.value == "Phase 2"

    def test_phase3_value(self):
        assert TrialPhase.PHASE_3.value == "Phase 3"

    def test_phase4_value(self):
        assert TrialPhase.PHASE_4.value == "Phase 4"

    def test_not_applicable_value(self):
        assert TrialPhase.NOT_APPLICABLE.value == "N/A"

    def test_unknown_value(self):
        assert TrialPhase.UNKNOWN.value == "Unknown"

    def test_is_string_enum(self):
        assert isinstance(TrialPhase.PHASE_1, str)
        assert TrialPhase.PHASE_1 == "Phase 1"

    def test_all_members_present(self):
        expected = {"Phase 1", "Phase 2", "Phase 3", "Phase 4", "N/A", "Unknown"}
        assert {p.value for p in TrialPhase} == expected


# ---------------------------------------------------------------------------
# TrialStatus enum
# ---------------------------------------------------------------------------

class TestTrialStatus:
    def test_recruiting_value(self):
        assert TrialStatus.RECRUITING.value == "Recruiting"

    def test_completed_value(self):
        assert TrialStatus.COMPLETED.value == "Completed"

    def test_terminated_value(self):
        assert TrialStatus.TERMINATED.value == "Terminated"

    def test_withdrawn_value(self):
        assert TrialStatus.WITHDRAWN.value == "Withdrawn"

    def test_active_not_recruiting_value(self):
        assert TrialStatus.ACTIVE_NOT_RECRUITING.value == "Active, not recruiting"

    def test_unknown_value(self):
        assert TrialStatus.UNKNOWN.value == "Unknown"

    def test_is_string_enum(self):
        assert isinstance(TrialStatus.COMPLETED, str)

    def test_all_members_present(self):
        expected = {
            "Recruiting", "Completed", "Terminated",
            "Withdrawn", "Active, not recruiting", "Unknown",
        }
        assert {s.value for s in TrialStatus} == expected


# ---------------------------------------------------------------------------
# CandidateOutcome enum
# ---------------------------------------------------------------------------

class TestCandidateOutcome:
    def test_failed_phase1_value(self):
        assert CandidateOutcome.FAILED_PHASE_1.value == "Failed Phase 1"

    def test_failed_phase2_value(self):
        assert CandidateOutcome.FAILED_PHASE_2.value == "Failed Phase 2"

    def test_failed_phase3_value(self):
        assert CandidateOutcome.FAILED_PHASE_3.value == "Failed Phase 3"

    def test_approved_value(self):
        assert CandidateOutcome.APPROVED.value == "Approved"

    def test_commercialized_value(self):
        assert CandidateOutcome.COMMERCIALIZED.value == "Commercialized"

    def test_ongoing_value(self):
        assert CandidateOutcome.ONGOING.value == "Ongoing"

    def test_unknown_value(self):
        assert CandidateOutcome.UNKNOWN.value == "Unknown"

    def test_is_string_enum(self):
        assert isinstance(CandidateOutcome.APPROVED, str)


# ---------------------------------------------------------------------------
# RawTrial dataclass
# ---------------------------------------------------------------------------

class TestRawTrial:
    def test_required_fields(self):
        trial = RawTrial(
            nct_id="NCT001",
            title="A Study",
            intervention="DrugX",
            indication="Diabetes",
            sponsor="Acme",
            phase=TrialPhase.PHASE_1,
            status=TrialStatus.RECRUITING,
        )
        assert trial.nct_id == "NCT001"
        assert trial.title == "A Study"
        assert trial.intervention == "DrugX"
        assert trial.indication == "Diabetes"
        assert trial.sponsor == "Acme"
        assert trial.phase == TrialPhase.PHASE_1
        assert trial.status == TrialStatus.RECRUITING

    def test_raw_data_defaults_to_empty_dict(self):
        trial = RawTrial(
            nct_id="NCT001", title="A Study", intervention="DrugX",
            indication="Diabetes", sponsor="Acme",
            phase=TrialPhase.PHASE_1, status=TrialStatus.RECRUITING,
        )
        assert trial.raw_data == {}

    def test_raw_data_is_independent_per_instance(self):
        t1 = RawTrial(
            nct_id="NCT001", title="A", intervention="X", indication="Y",
            sponsor="Z", phase=TrialPhase.PHASE_1, status=TrialStatus.RECRUITING,
        )
        t2 = RawTrial(
            nct_id="NCT002", title="B", intervention="X", indication="Y",
            sponsor="Z", phase=TrialPhase.PHASE_1, status=TrialStatus.RECRUITING,
        )
        t1.raw_data["key"] = "value"
        assert "key" not in t2.raw_data


# ---------------------------------------------------------------------------
# TrialTable dataclass
# ---------------------------------------------------------------------------

class TestTrialTable:
    def test_empty_by_default(self):
        table = TrialTable()
        assert table.trials == []

    def test_len_empty(self):
        assert len(TrialTable()) == 0

    def test_len_with_trials(self, sample_raw_trial):
        table = TrialTable(trials=[sample_raw_trial])
        assert len(table) == 1

    def test_len_multiple(self, sample_raw_trial, another_raw_trial):
        table = TrialTable(trials=[sample_raw_trial, another_raw_trial])
        assert len(table) == 2

    def test_trials_list_is_independent_per_instance(self):
        t1 = TrialTable()
        t2 = TrialTable()
        t1.trials.append(object())
        assert len(t2.trials) == 0


# ---------------------------------------------------------------------------
# Candidate dataclass
# ---------------------------------------------------------------------------

class TestCandidate:
    def test_required_fields(self):
        c = Candidate(candidate_id="c1", drug_name="DrugA", indication="Diabetes")
        assert c.candidate_id == "c1"
        assert c.drug_name == "DrugA"
        assert c.indication == "Diabetes"

    def test_trial_ids_defaults_to_empty_list(self):
        c = Candidate(candidate_id="c1", drug_name="DrugA", indication="Diabetes")
        assert c.trial_ids == []

    def test_highest_phase_defaults_to_unknown(self):
        c = Candidate(candidate_id="c1", drug_name="DrugA", indication="Diabetes")
        assert c.highest_phase == TrialPhase.UNKNOWN

    def test_sponsors_defaults_to_empty_list(self):
        c = Candidate(candidate_id="c1", drug_name="DrugA", indication="Diabetes")
        assert c.sponsors == []

    def test_mutable_defaults_are_independent(self):
        c1 = Candidate(candidate_id="c1", drug_name="A", indication="X")
        c2 = Candidate(candidate_id="c2", drug_name="B", indication="Y")
        c1.trial_ids.append("NCT001")
        assert c2.trial_ids == []


# ---------------------------------------------------------------------------
# CandidateTable dataclass
# ---------------------------------------------------------------------------

class TestCandidateTable:
    def test_empty_by_default(self):
        assert CandidateTable().candidates == []

    def test_len_empty(self):
        assert len(CandidateTable()) == 0

    def test_len_with_candidates(self, sample_candidate):
        table = CandidateTable(candidates=[sample_candidate])
        assert len(table) == 1


# ---------------------------------------------------------------------------
# CandidateAttributes dataclass
# ---------------------------------------------------------------------------

class TestCandidateAttributes:
    def test_required_fields(self):
        attrs = CandidateAttributes(
            candidate_id="c1",
            drug_modality="peptide",
            disease_area="oncology",
        )
        assert attrs.candidate_id == "c1"
        assert attrs.drug_modality == "peptide"
        assert attrs.disease_area == "oncology"

    def test_confidence_defaults_to_zero(self):
        attrs = CandidateAttributes(
            candidate_id="c1", drug_modality="peptide", disease_area="oncology"
        )
        assert attrs.modality_confidence == 0.0
        assert attrs.disease_confidence == 0.0

    def test_reasoning_defaults_to_empty_string(self):
        attrs = CandidateAttributes(
            candidate_id="c1", drug_modality="peptide", disease_area="oncology"
        )
        assert attrs.reasoning == ""


# ---------------------------------------------------------------------------
# AttributeTable dataclass
# ---------------------------------------------------------------------------

class TestAttributeTable:
    def test_empty_by_default(self):
        assert AttributeTable().attributes == {}

    def test_keyed_by_candidate_id(self, sample_attributes_cand001):
        table = AttributeTable(attributes={"cand_001": sample_attributes_cand001})
        assert "cand_001" in table.attributes
        assert table.attributes["cand_001"].drug_modality == "peptide"


# ---------------------------------------------------------------------------
# CandidateOutcomeRecord dataclass
# ---------------------------------------------------------------------------

class TestCandidateOutcomeRecord:
    def test_required_fields(self):
        record = CandidateOutcomeRecord(
            candidate_id="c1",
            outcome=CandidateOutcome.ONGOING,
        )
        assert record.candidate_id == "c1"
        assert record.outcome == CandidateOutcome.ONGOING

    def test_confidence_defaults_to_zero(self):
        record = CandidateOutcomeRecord(candidate_id="c1", outcome=CandidateOutcome.ONGOING)
        assert record.confidence == 0.0

    def test_reasoning_defaults_to_empty_string(self):
        record = CandidateOutcomeRecord(candidate_id="c1", outcome=CandidateOutcome.ONGOING)
        assert record.reasoning == ""

    def test_evidence_sources_defaults_to_empty_list(self):
        record = CandidateOutcomeRecord(candidate_id="c1", outcome=CandidateOutcome.ONGOING)
        assert record.evidence_sources == []


# ---------------------------------------------------------------------------
# OutcomeTable dataclass
# ---------------------------------------------------------------------------

class TestOutcomeTable:
    def test_empty_by_default(self):
        assert OutcomeTable().outcomes == {}


# ---------------------------------------------------------------------------
# TransitionRate dataclass
# ---------------------------------------------------------------------------

class TestTransitionRate:
    def test_fields(self):
        tr = TransitionRate(
            from_phase="Phase 1",
            to_phase="Phase 2",
            numerator=6,
            denominator=10,
            rate=0.6,
        )
        assert tr.from_phase == "Phase 1"
        assert tr.to_phase == "Phase 2"
        assert tr.numerator == 6
        assert tr.denominator == 10
        assert tr.rate == pytest.approx(0.6)


# ---------------------------------------------------------------------------
# FunnelSlice dataclass
# ---------------------------------------------------------------------------

class TestFunnelSlice:
    def test_fields(self):
        fs = FunnelSlice(modality="peptide", disease_area="oncology", candidate_count=5)
        assert fs.modality == "peptide"
        assert fs.disease_area == "oncology"
        assert fs.candidate_count == 5

    def test_transitions_defaults_to_empty_list(self):
        fs = FunnelSlice(modality=None, disease_area=None, candidate_count=0)
        assert fs.transitions == []

    def test_nullable_modality_and_disease_area(self):
        fs = FunnelSlice(modality=None, disease_area=None, candidate_count=0)
        assert fs.modality is None
        assert fs.disease_area is None


# ---------------------------------------------------------------------------
# FunnelResults dataclass
# ---------------------------------------------------------------------------

class TestFunnelResults:
    def test_default_overall_is_empty_slice(self):
        fr = FunnelResults()
        assert fr.overall.candidate_count == 0
        assert fr.overall.modality is None
        assert fr.overall.disease_area is None

    def test_by_modality_defaults_to_empty_dict(self):
        assert FunnelResults().by_modality == {}

    def test_by_disease_area_defaults_to_empty_dict(self):
        assert FunnelResults().by_disease_area == {}

    def test_overall_instances_are_independent(self):
        fr1 = FunnelResults()
        fr2 = FunnelResults()
        fr1.by_modality["peptide"] = FunnelSlice(modality="peptide", disease_area=None, candidate_count=1)
        assert "peptide" not in fr2.by_modality


# ---------------------------------------------------------------------------
# ReportOutput dataclass
# ---------------------------------------------------------------------------

class TestReportOutput:
    def test_defaults(self):
        report = ReportOutput()
        assert report.summary_text == ""
        assert report.tables == {}
        assert report.figures == {}
        assert report.output_path is None

    def test_custom_values(self):
        report = ReportOutput(
            summary_text="10 candidates analyzed.",
            tables={"summary": [{"drug": "X"}]},
            figures={"chart": b"\x89PNG"},
            output_path="/tmp/report",
        )
        assert report.summary_text == "10 candidates analyzed."
        assert report.output_path == "/tmp/report"
        assert report.figures["chart"] == b"\x89PNG"
