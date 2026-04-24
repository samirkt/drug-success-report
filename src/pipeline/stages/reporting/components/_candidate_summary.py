"""Candidate summary table (always included, renders last)."""

from __future__ import annotations

from datetime import date
from typing import Optional

from .._types import ComponentResult, ReportContext
from ....models import CandidateOutcome, CandidateOutcomeRecord, TrialPhase


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


def _lookup_cached_outcomes(
    candidate, cache
) -> tuple[Optional[CandidateOutcomeRecord], Optional[CandidateOutcomeRecord]]:
    """Return (direct-LLM record, FDA-timeline record) from the shared cache.

    Either entry may be None if that method has not been run against this
    KnowledgeCache (or the candidate's phase/drug/indication has changed
    since the last run). Safe to call with cache=None (returns (None, None)).
    """
    if cache is None:
        return None, None
    from ....knowledge_cache import KnowledgeCache
    key = KnowledgeCache.make_adjudication_key(
        candidate.drug_name,
        candidate.indication,
        candidate.highest_phase.value,
    )
    llm_direct = cache.get_outcome(key, candidate.candidate_id)
    fda = cache.get_fda_outcome(key, candidate.candidate_id)
    return llm_direct, fda


def _outcome_value(record: Optional[CandidateOutcomeRecord]) -> Optional[str]:
    return record.outcome.value if record else None


def candidate_summary_table(
    candidate_table,
    attribute_table,
    outcome_table,
    modality_filter=None,
    cache=None,
    adjudication_method: str = "fda_timeline",
) -> list[dict]:
    """Build a row-per-candidate summary table.

    Columns include the active-run outcome plus, when `cache` is supplied,
    the most recent cached outcome from each adjudication method
    (`llm_direct_*`, `fda_timeline_*`) and an `outcomes_agree` flag for
    quick cross-method comparison.
    """
    rows = []
    for cand in candidate_table.candidates:
        cid = cand.candidate_id
        attrs = attribute_table.attributes.get(cid)
        if modality_filter is not None:
            if attrs is None or attrs.drug_modality != modality_filter:
                continue
        outcome_rec = outcome_table.outcomes.get(cid)
        outcome_value = _outcome_value(outcome_rec)

        # If the active run used a given method, its record is authoritative
        # — skip the cache lookup for that side and use the fresh record so
        # the summary never disagrees with the OutcomeTable produced this run.
        llm_cached, fda_cached = _lookup_cached_outcomes(cand, cache)
        if adjudication_method == "llm_direct" and outcome_rec is not None:
            llm_record = outcome_rec
            fda_record = fda_cached
        elif adjudication_method == "fda_timeline" and outcome_rec is not None:
            fda_record = outcome_rec
            llm_record = llm_cached
        else:
            llm_record, fda_record = llm_cached, fda_cached

        llm_outcome = _outcome_value(llm_record)
        fda_outcome = _outcome_value(fda_record)
        if llm_outcome is None or fda_outcome is None:
            outcomes_agree = None
        else:
            outcomes_agree = llm_outcome == fda_outcome

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
            "llm_direct_outcome": llm_outcome,
            "llm_direct_confidence": llm_record.confidence if llm_record else None,
            "llm_direct_reasoning": llm_record.reasoning if llm_record else None,
            "fda_timeline_outcome": fda_outcome,
            "fda_timeline_confidence": fda_record.confidence if fda_record else None,
            "fda_timeline_reasoning": fda_record.reasoning if fda_record else None,
            "fda_approval_date": fda_record.approval_date if fda_record else None,
            "fda_commercialization_date": fda_record.commercialization_date if fda_record else None,
            "outcomes_agree": outcomes_agree,
            # Enrichment columns — empty when the corresponding stage
            # was skipped or produced no match. Pipe-joined for list
            # fields to match the trial_detail / candidate_detail CSV
            # convention.
            "smiles": cand.smiles or None,
            "drug_targets": "|".join(cand.drug_targets) if cand.drug_targets else None,
            "target_names": "|".join(cand.target_names) if cand.target_names else None,
            "opentargets_moa": cand.opentargets_moa,
            "opentargets_action_type": cand.opentargets_action_type,
            "opentargets_targets": (
                "|".join(cand.opentargets_targets)
                if cand.opentargets_targets else None
            ),
            "opentargets_pathways": (
                "|".join(cand.opentargets_pathways)
                if cand.opentargets_pathways else None
            ),
            "opentargets_indication_max_phase": cand.opentargets_indication_max_phase,
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
            cache=ctx.cache,
            adjudication_method=ctx.adjudication_method,
        )
        return ComponentResult(tables={"candidate_summary": summary})
