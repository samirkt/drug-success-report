"""
Stage 4: Funnel Aggregation

Computes phase-transition success rates across the candidate population,
stratified by drug modality and disease area.

Transitions computed:
  Phase 1 → Phase 2
  Phase 2 → Phase 3
  Phase 3 → Approval
  Approval → Market (commercialized)
"""

from collections import defaultdict

from ..models import (
    AttributeTable,
    CandidateTable,
    FunnelResults,
    FunnelSlice,
    OutcomeTable,
    TransitionRate,
    TrialTable,
)


TRANSITIONS = [
    ("Phase 1", "Phase 2"),
    ("Phase 2", "Phase 3"),
    ("Phase 3", "Approval"),
    ("Approval", "Market"),
]

_PHASE_ORDER: dict[str, int] = {
    "Phase 1": 0,
    "Phase 2": 1,
    "Phase 3": 2,
    "Approval": 3,
    "Market": 4,
}


class FunnelAggregationStage:
    """
    Aggregates candidate outcomes into phase-funnel success rates.

    Inputs:  CandidateTable, AttributeTable, OutcomeTable
    Outputs: FunnelResults
    """

    def run(
        self,
        candidate_table: CandidateTable,
        attribute_table: AttributeTable,
        outcome_table: OutcomeTable,
        trial_table: TrialTable | None = None,
    ) -> FunnelResults:
        """Compute overall and stratified funnel statistics."""
        records = self._join(candidate_table, attribute_table, outcome_table)

        overall = self._compute_slice(records, modality=None, disease_area=None, trial_table=trial_table)

        modalities = {r["modality"] for r in records if r["modality"]}
        by_modality = {
            m: self._compute_slice(records, modality=m, disease_area=None, trial_table=trial_table)
            for m in modalities
        }

        disease_areas = {r["disease_area"] for r in records if r["disease_area"]}
        by_disease_area = {
            d: self._compute_slice(records, modality=None, disease_area=d, trial_table=trial_table)
            for d in disease_areas
        }

        by_modality_and_disease = {}
        for m in modalities:
            for d in disease_areas:
                s = self._compute_slice(records, modality=m, disease_area=d, trial_table=trial_table)
                if s.candidate_count > 0:
                    by_modality_and_disease[(m, d)] = s

        return FunnelResults(
            overall=overall,
            by_modality=by_modality,
            by_disease_area=by_disease_area,
            by_modality_and_disease=by_modality_and_disease,
        )

    # ------------------------------------------------------------------
    # Internal helpers  (implementation goes here)
    # ------------------------------------------------------------------

    def _join(
        self,
        candidate_table: CandidateTable,
        attribute_table: AttributeTable,
        outcome_table: OutcomeTable,
    ) -> list[dict]:
        """
        Merge candidates, attributes, and outcomes into a flat list of dicts
        with keys: candidate_id, drug, indication, modality, disease_area,
                   highest_phase, outcome.
        """
        records = []
        for c in candidate_table.candidates:
            attrs = attribute_table.attributes.get(c.candidate_id)
            out   = outcome_table.outcomes.get(c.candidate_id)
            records.append({
                "candidate_id": c.candidate_id,
                "drug":         c.drug_name,
                "indication":   c.indication,
                "modality":     attrs.drug_modality if attrs else None,
                "disease_area": attrs.disease_area  if attrs else None,
                "highest_phase": c.highest_phase.value,
                "outcome":      out.outcome.value if out else None,
                "approval_date": out.approval_date if out else None,
                "commercialization_date": out.commercialization_date if out else None,
                "trial_ids":    c.trial_ids,
            })
        return records

    def _compute_slice(
        self,
        records: list[dict],
        modality: str | None,
        disease_area: str | None,
        trial_table: "TrialTable | None" = None,
    ) -> FunnelSlice:
        """
        Filter records to the given slice and compute each transition rate.
        """
        filtered = records
        if modality is not None:
            filtered = [r for r in filtered if r["modality"] == modality]
        if disease_area is not None:
            filtered = [r for r in filtered if r["disease_area"] == disease_area]

        # Compute per-slice phase durations from trials belonging to this slice
        slice_nct_ids = {nct for r in filtered for nct in r.get("trial_ids", [])}
        phase_avg_duration = self._compute_phase_durations(trial_table, nct_ids=slice_nct_ids)

        transitions = [
            self._transition_rate(filtered, from_phase, to_phase, phase_avg_duration=phase_avg_duration)
            for from_phase, to_phase in TRANSITIONS
        ]
        return FunnelSlice(
            modality=modality,
            disease_area=disease_area,
            candidate_count=len(filtered),
            transitions=transitions,
        )

    def _transition_rate(
        self,
        records: list[dict],
        from_phase: str,
        to_phase: str,
        phase_avg_duration: dict[str, float] | None = None,
    ) -> TransitionRate:
        """
        Compute the success rate for a single phase transition
        from `records` that reached at least `from_phase`.
        """
        def effective_level(record: dict) -> int:
            level = _PHASE_ORDER.get(record["highest_phase"], -1)
            outcome = record.get("outcome")
            if outcome == "Approved":
                level = max(level, 3)
            elif outcome == "Commercialized":
                level = max(level, 4)
            return level

        resolved = [r for r in records if r.get("outcome") not in ("Ongoing", "Unknown")]
        from_level = _PHASE_ORDER[from_phase]
        to_level   = _PHASE_ORDER[to_phase]
        denominator = sum(1 for r in resolved if effective_level(r) >= from_level)
        numerator   = sum(1 for r in resolved if effective_level(r) >= to_level)
        rate = numerator / denominator if denominator > 0 else 0.0

        if from_phase == "Approval":
            durations = []
            for r in records:
                appr = r.get("approval_date")
                comm = r.get("commercialization_date")
                if appr and comm:
                    days = (comm - appr).days
                    if days > 0:
                        durations.append(days / 365.25)
            avg_duration_years = sum(durations) / len(durations) if durations else None
        else:
            avg_duration_years = (phase_avg_duration or {}).get(from_phase)

        return TransitionRate(
            from_phase=from_phase,
            to_phase=to_phase,
            numerator=numerator,
            denominator=denominator,
            rate=rate,
            avg_duration_years=avg_duration_years,
        )

    def _compute_phase_durations(
        self,
        trial_table: "TrialTable | None",
        nct_ids: set[str] | None = None,
    ) -> dict[str, float]:
        """Compute average trial duration in years per phase.

        If nct_ids is provided, only trials with matching nct_id are included.
        """
        buckets: dict[str, list[float]] = defaultdict(list)
        if trial_table is None:
            return {}
        for trial in trial_table.trials:
            if nct_ids is not None and trial.nct_id not in nct_ids:
                continue
            if trial.start_date and trial.completion_date and trial.phase.value in _PHASE_ORDER:
                days = (trial.completion_date - trial.start_date).days
                if days > 0:
                    buckets[trial.phase.value].append(days / 365.25)
        return {phase: sum(v) / len(v) for phase, v in buckets.items() if v}
