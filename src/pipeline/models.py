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
    last_update_submitted_date: date | None = None
    is_single_arm: bool = False
    primary_p_values: list[TrialPValue] = field(default_factory=list)
    mesh_condition_terms: list[str] = field(default_factory=list)
    mesh_condition_tree_numbers: list[str] = field(default_factory=list)
    mesh_intervention_terms: list[str] = field(default_factory=list)
    eligibility_criteria: Optional[str] = None
    why_stopped: Optional[str] = None


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
    latest_update_submitted_date: date | None = None
    drug_name_raw: str = ""          # original intervention string from AACT (used for LLM prompts)
    drugbank_id: Optional[str] = None  # matched DrugBank primary ID; None if unmatched
    mesh_indication: Optional[str] = None  # canonical MeSH condition term used in dedup
    mesh_drug: Optional[str] = None  # canonical MeSH intervention term used in dedup
    mesh_condition_tree_numbers: list[str] = field(default_factory=list)
    single_arm_p_values: list[TrialPValue] = field(default_factory=list)
    smiles: Optional[str] = None
    smiles_canonical: Optional[str] = None
    smiles_standardization_status: Optional[str] = None  # ok | failed_parse | failed_standardize | empty
    drug_targets: list[str] = field(default_factory=list)   # UniProt accessions
    target_names: list[str] = field(default_factory=list)   # ChEMBL pref_names
    icd10_description: Optional[str] = None
    icd10_codes: list[str] = field(default_factory=list)  # full NLM lookup
    # OpenTargets — kept in its own namespace so the ChEMBL-derived
    # `drug_targets`/`target_names` (UniProt + ChEMBL pref_name) do not
    # collide with OT's gene-symbol + Ensembl identifiers. `opentargets_*`
    # fields follow the same "empty default" pattern as the other
    # enrichments so existing tests stay green when OT is off.
    opentargets_moa: Optional[str] = None           # mechanism-of-action text
    opentargets_action_type: Optional[str] = None   # e.g. AGONIST, INHIBITOR
    opentargets_targets: list[str] = field(default_factory=list)   # approvedSymbols
    opentargets_pathways: list[str] = field(default_factory=list)  # Reactome names
    # Matched indication's `maxPhaseForIndication` from OT. Only populated
    # when the candidate's `indication` (or `mesh_indication`) case-
    # insensitively matches an OT indication row for the same drug.
    opentargets_indication_max_phase: Optional[int] = None
    # Reactome — pathway membership joined on UniProt accessions in
    # `drug_targets`. Multi-target candidates aggregate by union across
    # all targets. None when stage skipped; False when target list was
    # empty or no targets found in Reactome; True with populated lists
    # otherwise. `reactome_pathway_names` is parallel-ordered with
    # `reactome_pathway_ids` and is for inspection only —
    # `reactome_n_pathways` is the canonical count feature.
    reactome_pathway_ids: list[str] = field(default_factory=list)
    reactome_pathway_names: list[str] = field(default_factory=list)
    reactome_n_pathways: Optional[int] = None
    reactome_has_data: Optional[bool] = None
    # Hierarchy-aware diagnostics derived from `pathway_hierarchy.parquet`.
    # `n_leaf_pathways` counts pathways in the candidate's set with no
    # *child* also present in the set ("locally leaf") — the more useful
    # diagnostic since the source file is ancestry-expanded. `n_leaf_global`
    # uses the Reactome-wide leaf flag as a sanity check. `mean_depth` /
    # `max_depth` summarize `depth_from_top` over the candidate's pathways.
    # `top_level_pathway_ids` is the sorted union of root ancestors;
    # `leaf_pathway_ids` is the sorted list of locally-leaf pathway IDs.
    reactome_n_top_level_pathways: Optional[int] = None
    reactome_n_leaf_pathways: Optional[int] = None
    reactome_n_leaf_global: Optional[int] = None
    reactome_n_internal_pathways: Optional[int] = None
    reactome_mean_depth: Optional[float] = None
    reactome_max_depth: Optional[int] = None
    reactome_top_level_pathway_ids: list[str] = field(default_factory=list)
    reactome_leaf_pathway_ids: list[str] = field(default_factory=list)
    # ADMET — predicted from canonical SMILES via admet_ai (52 raw
    # properties + 52 DrugBank-approved-percentile siblings). Names
    # mirror admet_ai's column tuple in `pipeline.admet.admet_columns`
    # with hyphens normalized to underscores by `field_name`. None when
    # the enrichment is disabled, admet_ai is missing, or the SMILES
    # failed to predict.
    admet_molecular_weight: Optional[float] = None
    admet_logP: Optional[float] = None
    admet_hydrogen_bond_acceptors: Optional[float] = None
    admet_hydrogen_bond_donors: Optional[float] = None
    admet_Lipinski: Optional[float] = None
    admet_QED: Optional[float] = None
    admet_stereo_centers: Optional[float] = None
    admet_tpsa: Optional[float] = None
    admet_PAINS_alert: Optional[float] = None
    admet_BRENK_alert: Optional[float] = None
    admet_NIH_alert: Optional[float] = None
    admet_AMES: Optional[float] = None
    admet_BBB_Martins: Optional[float] = None
    admet_Bioavailability_Ma: Optional[float] = None
    admet_CYP1A2_Veith: Optional[float] = None
    admet_CYP2C19_Veith: Optional[float] = None
    admet_CYP2C9_Substrate_CarbonMangels: Optional[float] = None
    admet_CYP2C9_Veith: Optional[float] = None
    admet_CYP2D6_Substrate_CarbonMangels: Optional[float] = None
    admet_CYP2D6_Veith: Optional[float] = None
    admet_CYP3A4_Substrate_CarbonMangels: Optional[float] = None
    admet_CYP3A4_Veith: Optional[float] = None
    admet_Carcinogens_Lagunin: Optional[float] = None
    admet_ClinTox: Optional[float] = None
    admet_DILI: Optional[float] = None
    admet_HIA_Hou: Optional[float] = None
    admet_NR_AR_LBD: Optional[float] = None
    admet_NR_AR: Optional[float] = None
    admet_NR_AhR: Optional[float] = None
    admet_NR_Aromatase: Optional[float] = None
    admet_NR_ER_LBD: Optional[float] = None
    admet_NR_ER: Optional[float] = None
    admet_NR_PPAR_gamma: Optional[float] = None
    admet_PAMPA_NCATS: Optional[float] = None
    admet_Pgp_Broccatelli: Optional[float] = None
    admet_SR_ARE: Optional[float] = None
    admet_SR_ATAD5: Optional[float] = None
    admet_SR_HSE: Optional[float] = None
    admet_SR_MMP: Optional[float] = None
    admet_SR_p53: Optional[float] = None
    admet_Skin_Reaction: Optional[float] = None
    admet_hERG: Optional[float] = None
    admet_Caco2_Wang: Optional[float] = None
    admet_Clearance_Hepatocyte_AZ: Optional[float] = None
    admet_Clearance_Microsome_AZ: Optional[float] = None
    admet_Half_Life_Obach: Optional[float] = None
    admet_HydrationFreeEnergy_FreeSolv: Optional[float] = None
    admet_LD50_Zhu: Optional[float] = None
    admet_Lipophilicity_AstraZeneca: Optional[float] = None
    admet_PPBR_AZ: Optional[float] = None
    admet_Solubility_AqSolDB: Optional[float] = None
    admet_VDss_Lombardo: Optional[float] = None
    # DrugBank-approved-set percentile rank for each property above.
    admet_molecular_weight_drugbank_approved_percentile: Optional[float] = None
    admet_logP_drugbank_approved_percentile: Optional[float] = None
    admet_hydrogen_bond_acceptors_drugbank_approved_percentile: Optional[float] = None
    admet_hydrogen_bond_donors_drugbank_approved_percentile: Optional[float] = None
    admet_Lipinski_drugbank_approved_percentile: Optional[float] = None
    admet_QED_drugbank_approved_percentile: Optional[float] = None
    admet_stereo_centers_drugbank_approved_percentile: Optional[float] = None
    admet_tpsa_drugbank_approved_percentile: Optional[float] = None
    admet_PAINS_alert_drugbank_approved_percentile: Optional[float] = None
    admet_BRENK_alert_drugbank_approved_percentile: Optional[float] = None
    admet_NIH_alert_drugbank_approved_percentile: Optional[float] = None
    admet_AMES_drugbank_approved_percentile: Optional[float] = None
    admet_BBB_Martins_drugbank_approved_percentile: Optional[float] = None
    admet_Bioavailability_Ma_drugbank_approved_percentile: Optional[float] = None
    admet_CYP1A2_Veith_drugbank_approved_percentile: Optional[float] = None
    admet_CYP2C19_Veith_drugbank_approved_percentile: Optional[float] = None
    admet_CYP2C9_Substrate_CarbonMangels_drugbank_approved_percentile: Optional[float] = None
    admet_CYP2C9_Veith_drugbank_approved_percentile: Optional[float] = None
    admet_CYP2D6_Substrate_CarbonMangels_drugbank_approved_percentile: Optional[float] = None
    admet_CYP2D6_Veith_drugbank_approved_percentile: Optional[float] = None
    admet_CYP3A4_Substrate_CarbonMangels_drugbank_approved_percentile: Optional[float] = None
    admet_CYP3A4_Veith_drugbank_approved_percentile: Optional[float] = None
    admet_Carcinogens_Lagunin_drugbank_approved_percentile: Optional[float] = None
    admet_ClinTox_drugbank_approved_percentile: Optional[float] = None
    admet_DILI_drugbank_approved_percentile: Optional[float] = None
    admet_HIA_Hou_drugbank_approved_percentile: Optional[float] = None
    admet_NR_AR_LBD_drugbank_approved_percentile: Optional[float] = None
    admet_NR_AR_drugbank_approved_percentile: Optional[float] = None
    admet_NR_AhR_drugbank_approved_percentile: Optional[float] = None
    admet_NR_Aromatase_drugbank_approved_percentile: Optional[float] = None
    admet_NR_ER_LBD_drugbank_approved_percentile: Optional[float] = None
    admet_NR_ER_drugbank_approved_percentile: Optional[float] = None
    admet_NR_PPAR_gamma_drugbank_approved_percentile: Optional[float] = None
    admet_PAMPA_NCATS_drugbank_approved_percentile: Optional[float] = None
    admet_Pgp_Broccatelli_drugbank_approved_percentile: Optional[float] = None
    admet_SR_ARE_drugbank_approved_percentile: Optional[float] = None
    admet_SR_ATAD5_drugbank_approved_percentile: Optional[float] = None
    admet_SR_HSE_drugbank_approved_percentile: Optional[float] = None
    admet_SR_MMP_drugbank_approved_percentile: Optional[float] = None
    admet_SR_p53_drugbank_approved_percentile: Optional[float] = None
    admet_Skin_Reaction_drugbank_approved_percentile: Optional[float] = None
    admet_hERG_drugbank_approved_percentile: Optional[float] = None
    admet_Caco2_Wang_drugbank_approved_percentile: Optional[float] = None
    admet_Clearance_Hepatocyte_AZ_drugbank_approved_percentile: Optional[float] = None
    admet_Clearance_Microsome_AZ_drugbank_approved_percentile: Optional[float] = None
    admet_Half_Life_Obach_drugbank_approved_percentile: Optional[float] = None
    admet_HydrationFreeEnergy_FreeSolv_drugbank_approved_percentile: Optional[float] = None
    admet_LD50_Zhu_drugbank_approved_percentile: Optional[float] = None
    admet_Lipophilicity_AstraZeneca_drugbank_approved_percentile: Optional[float] = None
    admet_PPBR_AZ_drugbank_approved_percentile: Optional[float] = None
    admet_Solubility_AqSolDB_drugbank_approved_percentile: Optional[float] = None
    admet_VDss_Lombardo_drugbank_approved_percentile: Optional[float] = None


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
