"""
Shared fixtures for all pipeline unit tests.

Provides representative sample data for every stage's input/output types,
so individual test modules can focus on behavior rather than data construction.
"""

from datetime import date

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
# Stage 1: Trial Ingestion
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_raw_trial():
    return RawTrial(
        nct_id="NCT00000001",
        title="Phase 2 Study of DrugA in Type 2 Diabetes",
        intervention="DrugA",
        indication="Type 2 Diabetes",
        sponsor="PharmaCo",
        phase=TrialPhase.PHASE_2,
        status=TrialStatus.COMPLETED,
        start_date=date(2018, 1, 1),
        completion_date=date(2021, 6, 30),
    )


@pytest.fixture
def another_raw_trial():
    return RawTrial(
        nct_id="NCT00000002",
        title="Phase 1 Study of PeptideB in Non-Small Cell Lung Cancer",
        intervention="PeptideB",
        indication="Non-Small Cell Lung Cancer",
        sponsor="OncoBio",
        phase=TrialPhase.PHASE_1,
        status=TrialStatus.TERMINATED,
        raw_data={"primary_outcome": "Safety"},
        start_date=date(2019, 3, 1),
        completion_date=None,
    )


@pytest.fixture
def sample_trial_table(sample_raw_trial, another_raw_trial):
    return TrialTable(trials=[sample_raw_trial, another_raw_trial])


@pytest.fixture
def empty_trial_table():
    return TrialTable(trials=[])


# ---------------------------------------------------------------------------
# Stage 2: Candidate Clustering
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_candidate():
    return Candidate(
        candidate_id="cand_001",
        drug_name="DrugA",
        indication="Type 2 Diabetes",
        trial_ids=["NCT00000001", "NCT00000003"],
        highest_phase=TrialPhase.PHASE_2,
        sponsors=["PharmaCo"],
        earliest_start_date=date(2018, 1, 1),
        latest_completion_date=date(2021, 6, 30),
        drug_name_raw="Lepirudin HCl",
        drugbank_id="DB00001",
    )


@pytest.fixture
def another_candidate():
    return Candidate(
        candidate_id="cand_002",
        drug_name="PeptideB",
        indication="Non-Small Cell Lung Cancer",
        trial_ids=["NCT00000002"],
        highest_phase=TrialPhase.PHASE_1,
        sponsors=["OncoBio"],
        drug_name_raw="Insulin Human",
        drugbank_id="DB00030",
    )


@pytest.fixture
def drugbank_csv_content() -> str:
    """Minimal in-memory DrugBank CSV string for tests (no real file needed)."""
    return (
        "drug_id,query_name,query_norm,modality\n"
        "DB00001,lepirudin,lepirudin,peptide\n"
        "DB00001,lepirudin,lepirudin,\n"           # duplicate query_name — less complete row
        "DB00030,insulin human,insulin human,peptide\n"
        "DB00050,insulin,insulin,small molecule\n"
    )


@pytest.fixture
def sample_candidate_table(sample_candidate, another_candidate):
    return CandidateTable(candidates=[sample_candidate, another_candidate])


@pytest.fixture
def single_candidate_table(sample_candidate):
    return CandidateTable(candidates=[sample_candidate])


@pytest.fixture
def empty_candidate_table():
    return CandidateTable(candidates=[])


# ---------------------------------------------------------------------------
# Stage 3a: Attribute Classification
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_attributes_cand001():
    return CandidateAttributes(
        candidate_id="cand_001",
        drug_modality="peptide",
        disease_area="metabolic",
        modality_confidence=0.92,
        disease_confidence=0.87,
        reasoning="DrugA is a GLP-1 analog — a peptide hormone.",
    )


@pytest.fixture
def sample_attributes_cand002():
    return CandidateAttributes(
        candidate_id="cand_002",
        drug_modality="biologic",
        disease_area="oncology",
        modality_confidence=0.78,
        disease_confidence=0.95,
        reasoning="PeptideB targets PD-L1 in NSCLC.",
    )


@pytest.fixture
def sample_attribute_table(sample_attributes_cand001, sample_attributes_cand002):
    return AttributeTable(
        attributes={
            "cand_001": sample_attributes_cand001,
            "cand_002": sample_attributes_cand002,
        }
    )


# ---------------------------------------------------------------------------
# Stage 3b: Outcome Adjudication
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_outcome_cand001():
    return CandidateOutcomeRecord(
        candidate_id="cand_001",
        outcome=CandidateOutcome.APPROVED,
        confidence=0.88,
        reasoning="Phase 3 completed; FDA approved.",
        evidence_sources=["FDA Orange Book", "NCT00000001"],
        approval_date=date(2022, 4, 15),
        commercialization_date=None,
    )


@pytest.fixture
def sample_outcome_cand002():
    return CandidateOutcomeRecord(
        candidate_id="cand_002",
        outcome=CandidateOutcome.FAILED_PHASE_1,
        confidence=0.75,
        reasoning="Phase 1 terminated due to toxicity.",
        evidence_sources=["NCT00000002"],
    )


@pytest.fixture
def sample_outcome_table(sample_outcome_cand001, sample_outcome_cand002):
    return OutcomeTable(
        outcomes={
            "cand_001": sample_outcome_cand001,
            "cand_002": sample_outcome_cand002,
        }
    )


# ---------------------------------------------------------------------------
# Stage 4: Funnel Aggregation — helper records for _join output
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_joined_records():
    """Flat records as produced by FunnelAggregationStage._join()."""
    return [
        {
            "candidate_id": "cand_001",
            "drug": "DrugA",
            "indication": "Type 2 Diabetes",
            "modality": "peptide",
            "disease_area": "metabolic",
            "highest_phase": "Phase 2",
            "outcome": "Approved",
            "approval_date": date(2022, 4, 15),
            "commercialization_date": None,
            "phases_observed": {"Phase 1", "Phase 2", "Approval"},
            "phases_advanced": {"Phase 1", "Phase 2", "Approval"},
        },
        {
            "candidate_id": "cand_002",
            "drug": "PeptideB",
            "indication": "Non-Small Cell Lung Cancer",
            "modality": "biologic",
            "disease_area": "oncology",
            "highest_phase": "Phase 1",
            "outcome": "Failed Phase 1",
            "approval_date": None,
            "commercialization_date": None,
            "phases_observed": {"Phase 1"},
            "phases_advanced": {"Phase 1"},
        },
        {
            "candidate_id": "cand_003",
            "drug": "PeptideC",
            "indication": "Type 2 Diabetes",
            "modality": "peptide",
            "disease_area": "metabolic",
            "highest_phase": "Phase 3",
            "outcome": "Ongoing",
            "approval_date": None,
            "commercialization_date": None,
            "phases_observed": set(),
            "phases_advanced": set(),
        },
    ]


@pytest.fixture
def sample_funnel_slice():
    return FunnelSlice(
        modality=None,
        disease_area=None,
        candidate_count=3,
        transitions=[
            TransitionRate(from_phase="Phase 1", to_phase="Phase 2", numerator=2, denominator=3, rate=0.667),
            TransitionRate(from_phase="Phase 2", to_phase="Phase 3", numerator=1, denominator=2, rate=0.5),
        ],
    )


@pytest.fixture
def sample_funnel_results(sample_funnel_slice):
    peptide_slice = FunnelSlice(
        modality="peptide",
        disease_area=None,
        candidate_count=2,
        transitions=[
            TransitionRate("Phase 1", "Phase 2", 2, 2, 1.0),
        ],
    )
    metabolic_slice = FunnelSlice(
        modality=None,
        disease_area="metabolic",
        candidate_count=2,
        transitions=[
            TransitionRate("Phase 1", "Phase 2", 2, 2, 1.0),
        ],
    )
    return FunnelResults(
        overall=sample_funnel_slice,
        by_modality={"peptide": peptide_slice, "biologic": sample_funnel_slice},
        by_disease_area={"metabolic": metabolic_slice, "oncology": sample_funnel_slice},
        by_modality_and_disease={
            ("peptide", "metabolic"): peptide_slice,
            ("biologic", "oncology"): sample_funnel_slice,
        },
    )


# ---------------------------------------------------------------------------
# Knowledge Cache
# ---------------------------------------------------------------------------

@pytest.fixture
def knowledge_cache(tmp_path):
    from pipeline.knowledge_cache import KnowledgeCache
    cache = KnowledgeCache(tmp_path / "test_cache.db")
    yield cache
    cache.close()


# ---------------------------------------------------------------------------
# Stage 5: Reporting
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_report_output():
    return ReportOutput(
        summary_text="3 candidates analyzed across 2 modalities.",
        tables={
            "candidate_summary": [{"drug": "DrugA", "phase": "Phase 2"}],
            "funnel_overall": [{"from": "Phase 1", "to": "Phase 2", "rate": 0.667}],
            "funnel_by_modality": [],
            "funnel_by_disease": [],
        },
        figures={
            "modality_breakdown": b"\x89PNG",
            "disease_breakdown": b"\x89PNG",
        },
        output_path=None,
    )
