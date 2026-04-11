"""Candidate summary table (always included, renders last)."""

from __future__ import annotations

from datetime import date

from .._types import ComponentResult, ReportContext
from ....models import CandidateOutcome, TrialPhase


# ---------------------------------------------------------------------------
# Standalone functions
# ---------------------------------------------------------------------------

def latest_milestone_date(candidate, outcome_record) -> date | None:
    """Return latest past date among completion, approval, and commercialization."""
    today = date.today()
    dates = [
        candidate.latest_completion_date,
        outcome_record.approval_date if outcome_record else None,
        outcome_record.commercialization_date if outcome_record else None,
    ]
    valid_dates = [d for d in dates if d is not None and d < today]
    return max(valid_dates) if valid_dates else None


def best_p_value_by_phase(p_values: list) -> dict:
    """Return the lowest (most significant) p-value per TrialPhase."""
    best: dict = {}
    for pv in p_values:
        if pv.phase is None:
            continue
        if pv.phase not in best or pv.p_value < best[pv.phase]:
            best[pv.phase] = pv.p_value
    return best


def inconsistency_flag(outcome_value: str | None, p_by_phase: dict) -> bool | None:
    """Return True when p-value evidence contradicts the clinical outcome."""
    if not p_by_phase:
        return None
    _FAILED_PHASE_MAP = {
        CandidateOutcome.FAILED_PHASE_1.value: TrialPhase.PHASE_1,
        CandidateOutcome.FAILED_PHASE_2.value: TrialPhase.PHASE_2,
        CandidateOutcome.FAILED_PHASE_3.value: TrialPhase.PHASE_3,
    }
    if outcome_value in _FAILED_PHASE_MAP:
        phase = _FAILED_PHASE_MAP[outcome_value]
        p = p_by_phase.get(phase)
        if p is not None and p <= 0.05:
            return True
    if outcome_value in (CandidateOutcome.APPROVED.value, CandidateOutcome.COMMERCIALIZED.value):
        p = p_by_phase.get(TrialPhase.PHASE_3)
        if p is not None and p > 0.05:
            return True
    return False


def candidate_summary_table(candidate_table, attribute_table, outcome_table, modality_filter=None) -> list[dict]:
    """Build a row-per-candidate summary table."""
    rows = []
    for cand in candidate_table.candidates:
        cid = cand.candidate_id
        attrs = attribute_table.attributes.get(cid)
        if modality_filter is not None:
            if attrs is None or attrs.drug_modality != modality_filter:
                continue
        outcome_rec = outcome_table.outcomes.get(cid)
        outcome_value = outcome_rec.outcome.value if outcome_rec else None
        p_by_phase = best_p_value_by_phase(cand.single_arm_p_values)
        rows.append({
            "drug_name": cand.drug_name,
            "indication": cand.indication,
            "highest_phase": cand.highest_phase.value if hasattr(cand.highest_phase, "value") else str(cand.highest_phase),
            "modality": attrs.drug_modality if attrs else None,
            "disease_area": attrs.disease_area if attrs else None,
            "outcome": outcome_value,
            "latest_milestone_date": latest_milestone_date(cand, outcome_rec),
            "p_value_inconsistent": inconsistency_flag(outcome_value, p_by_phase),
        })
    return rows


# ---------------------------------------------------------------------------
# Component class
# ---------------------------------------------------------------------------

class CandidateSummaryComponent:
    key = "candidate_summary"
    order = 100
    title = "Candidate Summary"

    def should_include(self, ctx: ReportContext) -> bool:
        return True

    def render(self, ctx: ReportContext) -> ComponentResult:
        modality_filter = "peptide" if ctx.peptide_only else None
        summary = candidate_summary_table(
            ctx.candidate_table, ctx.attribute_table, ctx.outcome_table,
            modality_filter=modality_filter,
        )
        return ComponentResult(tables={"candidate_summary": summary})
