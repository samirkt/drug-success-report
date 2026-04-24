"""
Data models for each pipeline stage.

These dataclasses define the inputs and outputs that flow between stages.
"""

from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Optional


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class TrialPhase(str, Enum):
    PHASE_1 = "Phase 1"
    PHASE_2 = "Phase 2"
    PHASE_3 = "Phase 3"
    PHASE_4 = "Phase 4"
    NOT_APPLICABLE = "N/A"
    UNKNOWN = "Unknown"


class TrialStatus(str, Enum):
    RECRUITING = "Recruiting"
    COMPLETED = "Completed"
    TERMINATED = "Terminated"
    WITHDRAWN = "Withdrawn"
    ACTIVE_NOT_RECRUITING = "Active, not recruiting"
    SUSPENDED = "Suspended"
    UNKNOWN = "Unknown"


class CandidateOutcome(str, Enum):
    FAILED_PHASE_1 = "Failed Phase 1"
    FAILED_PHASE_2 = "Failed Phase 2"
    FAILED_PHASE_3 = "Failed Phase 3"
    APPROVED = "Approved"
    COMMERCIALIZED = "Commercialized"
    ONGOING = "Ongoing"
    UNKNOWN = "Unknown"


# ---------------------------------------------------------------------------
# Stage 1: Trial Ingestion
# ---------------------------------------------------------------------------

@dataclass
class TrialPValue:
    """Primary-outcome p-value record from a single-arm trial."""
    nct_id: str
    outcome_title: str
    p_value: float
    p_value_description: str | None = None
    statistical_method: str | None = None
    param_type: str | None = None       # e.g. "Mean Difference", "Hazard Ratio"
    param_value: float | None = None
    ci_lower_limit: float | None = None
    ci_upper_limit: float | None = None
    phase: "TrialPhase | None" = None   # populated in clustering from RawTrial.phase

    @property
    def is_significant(self) -> bool:
        """Conventional α=0.05 significance threshold."""
        return self.p_value <= 0.05


@dataclass
class RawTrial:
    """A single trial record fetched from ClinicalTrials.gov."""
    nct_id: str
    title: str
    intervention: str
    indication: str
    sponsor: str
    phase: TrialPhase
    status: TrialStatus
    raw_data: dict = field(default_factory=dict)
    start_date: date | None = None
    completion_date: date | None = None
    is_single_arm: bool = False
    primary_p_values: list[TrialPValue] = field(default_factory=list)
    mesh_condition_terms: list[str] = field(default_factory=list)
    mesh_condition_tree_numbers: list[str] = field(default_factory=list)
    mesh_intervention_terms: list[str] = field(default_factory=list)


@dataclass
class TrialTable:
    """Output of the Trial Ingestion stage."""
    trials: list[RawTrial] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.trials)


# ---------------------------------------------------------------------------
# Stage 2: Candidate Matching / Trial Clustering
# ---------------------------------------------------------------------------

@dataclass
class Candidate:
    """
    A unique drug–indication pair derived by clustering related trials.
    One candidate may aggregate many individual trials.
    """
    candidate_id: str
    drug_name: str
    indication: str
    trial_ids: list[str] = field(default_factory=list)
    highest_phase: TrialPhase = TrialPhase.UNKNOWN
    sponsors: list[str] = field(default_factory=list)
    earliest_start_date: date | None = None
    latest_completion_date: date | None = None
    drug_name_raw: str = ""          # original intervention string from AACT (used for LLM prompts)
    drugbank_id: Optional[str] = None  # matched DrugBank primary ID; None if unmatched
    mesh_indication: Optional[str] = None  # canonical MeSH condition term used in dedup
    mesh_drug: Optional[str] = None  # canonical MeSH intervention term used in dedup
    mesh_condition_tree_numbers: list[str] = field(default_factory=list)
    single_arm_p_values: list[TrialPValue] = field(default_factory=list)
    smiles: Optional[str] = None
    drug_targets: list[str] = field(default_factory=list)   # UniProt accessions
    target_names: list[str] = field(default_factory=list)   # ChEMBL pref_names
    icd10_code: Optional[str] = None
    icd10_description: Optional[str] = None


@dataclass
class CandidateTable:
    """Output of the Candidate Matching stage."""
    candidates: list[Candidate] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.candidates)


# ---------------------------------------------------------------------------
# Stage 3a: Candidate Attribute Classification
# ---------------------------------------------------------------------------

@dataclass
class CandidateAttributes:
    """Modality and disease-area tags for a single candidate."""
    candidate_id: str
    drug_modality: str          # e.g. "peptide", "small molecule", "biologic"
    disease_area: str           # e.g. "oncology", "metabolic", "cardiovascular"
    modality_confidence: float = 0.0
    disease_confidence: float = 0.0
    reasoning: str = ""


@dataclass
class AttributeTable:
    """Output of the Candidate Attribute Classification stage."""
    attributes: dict[str, CandidateAttributes] = field(default_factory=dict)  # keyed by candidate_id


# ---------------------------------------------------------------------------
# Stage 3b: Candidate Outcome Adjudication
# ---------------------------------------------------------------------------

@dataclass
class CandidateOutcomeRecord:
    """Adjudicated development outcome for a single candidate."""
    candidate_id: str
    outcome: CandidateOutcome
    confidence: float = 0.0
    reasoning: str = ""
    evidence_sources: list[str] = field(default_factory=list)
    approval_date: date | None = None
    commercialization_date: date | None = None


@dataclass
class OutcomeTable:
    """Output of the Candidate Outcome Adjudication stage."""
    outcomes: dict[str, CandidateOutcomeRecord] = field(default_factory=dict)  # keyed by candidate_id


# ---------------------------------------------------------------------------
# Stage 4: Funnel Aggregation
# ---------------------------------------------------------------------------

@dataclass
class TransitionRate:
    """Success rate for one phase transition."""
    from_phase: str
    to_phase: str
    numerator: int
    denominator: int
    rate: float
    avg_duration_years: float | None = None


@dataclass
class FunnelSlice:
    """
    Aggregated funnel statistics for one stratification slice
    (e.g. modality="peptide", disease_area="oncology").
    """
    modality: Optional[str]
    disease_area: Optional[str]
    candidate_count: int
    transitions: list[TransitionRate] = field(default_factory=list)


@dataclass
class FunnelResults:
    """Output of the Funnel Aggregation stage."""
    overall: FunnelSlice = field(default_factory=lambda: FunnelSlice(None, None, 0))
    by_modality: dict[str, FunnelSlice] = field(default_factory=dict)
    by_disease_area: dict[str, FunnelSlice] = field(default_factory=dict)
    by_modality_and_disease: dict[tuple[str, str], FunnelSlice] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Stage 5: Automated Report
# ---------------------------------------------------------------------------

@dataclass
class ReportOutput:
    """Final report artifacts produced by the Automated Report stage."""
    summary_text: str = ""
    tables: dict[str, list[dict]] = field(default_factory=dict)   # table_name → rows
    figures: dict[str, bytes] = field(default_factory=dict)        # figure_name → image bytes
    html_figures: dict[str, str] = field(default_factory=dict)    # figure_name → HTML string (e.g. Plotly divs)
    output_path: Optional[str] = None
