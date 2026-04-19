"""
Stage 4: Funnel Aggregation

Computes phase-transition success rates using a forward-looking cohort
methodology aligned with BIO/QLS. Two sets of phase observations are tracked
per candidate:

  * `phases_observed` — the cohort-membership set. A phase is included only
    when there is terminal-status evidence at that phase: a COMPLETED /
    TERMINATED / WITHDRAWN / SUSPENDED trial, a FAILED_PHASE_N adjudicator
    verdict, or an approval / commercialization event. This drives the
    denominator of each transition rate.

  * `phases_advanced` — the advancement-evidence set. A phase is included
    when there is any trial at that phase (regardless of status), plus every
    element of `phases_observed`. A started-but-not-yet-terminal Phase 2
    trial is sufficient evidence of P1→P2 advancement. This drives the
    numerator of each transition rate.

A candidate counts as a success for transition N→N+1 iff `from_phase` is in
`phases_observed` AND `phases_advanced` contains any strictly later cohort.
When approval back-propagation is enabled, missing intermediate phases are
imputed only within the contiguous range bounded by the candidate's earliest
and latest observed phases.

Transitions computed:
  Phase 1 → Phase 2
  Phase 2 → Phase 3
  Phase 3 → Approval
  Approval → Market (commercialized)
"""

from collections import defaultdict
from datetime import date

from ..models import (
    AttributeTable,
    CandidateTable,
    FunnelResults,
    FunnelSlice,
    OutcomeTable,
    TransitionRate,
    TrialStatus,
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

# Trial statuses that count as terminal observation of a phase. A phase is
# considered "observed" for cohort purposes when a trial at that phase has
# reached one of these statuses.
_TERMINAL_TRIAL_STATUSES: frozenset[TrialStatus] = frozenset({
    TrialStatus.COMPLETED,
    TrialStatus.TERMINATED,
    TrialStatus.WITHDRAWN,
    TrialStatus.SUSPENDED,
})

# Non-terminal trial statuses that may be promoted to "effectively terminal"
# once the trial's latest activity date is older than the stale-trial cutoff.
# Addresses pre-FDAAA-2007 records whose status field was never updated.
_STALE_STATUS_CANDIDATES: frozenset[TrialStatus] = frozenset({
    TrialStatus.UNKNOWN,
    TrialStatus.ACTIVE_NOT_RECRUITING,
    TrialStatus.RECRUITING,
})


def _latest_trial_activity(
    trial_ids: list[str],
    trial_index: dict[str, tuple[str, TrialStatus, date | None, date | None]],
) -> date | None:
    """Return the latest activity date (completion else start) across the
    candidate's trials, or None if no trial carries any date."""
    dates: list[date] = []
    for nct in trial_ids:
        entry = trial_index.get(nct)
        if entry is None:
            continue
        _, _, start_date, completion_date = entry
        d = completion_date or start_date
        if d is not None:
            dates.append(d)
    return max(dates) if dates else None


def _is_effectively_terminal(
    status: TrialStatus,
    start_date: date | None,
    completion_date: date | None,
    reference_date: date,
    stale_cutoff_years: float,
) -> bool:
    """Return True if a trial should contribute to cohort-membership at its phase.

    True when status is terminal, or when status is a stale-promotion candidate
    (Unknown / Active not recruiting / Recruiting) and the latest available
    activity date (completion_date else start_date) is at least
    `stale_cutoff_years` in the past relative to `reference_date`.
    """
    if status in _TERMINAL_TRIAL_STATUSES:
        return True
    if status not in _STALE_STATUS_CANDIDATES:
        return False
    latest = completion_date or start_date
    if latest is None:
        return False
    return (reference_date - latest).days >= stale_cutoff_years * 365.25


class FunnelAggregationStage:
    """
    Aggregates candidate outcomes into phase-funnel success rates.

    Inputs:  CandidateTable, AttributeTable, OutcomeTable, TrialTable
    Outputs: FunnelResults
    """

    def __init__(
        self,
        reference_date: date | None = None,
        stale_cutoff_years: float = 2.0,
        back_propagate_approval: bool = True,
    ) -> None:
        """
        Args:
            reference_date:
                Anchor date against which trial staleness is measured.
                `None` (default) resolves to `date.today()` at run time.
            stale_cutoff_years:
                A trial with a non-terminal status is promoted into the
                cohort set when its latest activity date is at least this
                many years before `reference_date`. Default 2.0 (ClinSR).
            back_propagate_approval:
                When True, fills missing intermediate phases in both
                `phases_observed` and `phases_advanced` between the earliest
                and latest currently observed phase for approved /
                commercialized candidates. Set to False to preserve strict
                forward-looking semantics.
        """
        self._reference_date = reference_date
        self._stale_cutoff_years = stale_cutoff_years
        self._back_propagate_approval = back_propagate_approval

    def _resolved_reference_date(self) -> date:
        return self._reference_date or date.today()

    def run(
        self,
        candidate_table: CandidateTable,
        attribute_table: AttributeTable,
        outcome_table: OutcomeTable,
        trial_table: TrialTable | None = None,
    ) -> FunnelResults:
        """Compute overall and stratified funnel statistics."""
        records = self._join(candidate_table, attribute_table, outcome_table, trial_table)

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
    # Internal helpers
    # ------------------------------------------------------------------

    def _join(
        self,
        candidate_table: CandidateTable,
        attribute_table: AttributeTable,
        outcome_table: OutcomeTable,
        trial_table: TrialTable | None = None,
    ) -> list[dict]:
        """
        Merge candidates, attributes, outcomes, and trial-level phase
        observations into a flat list of dicts.

        Adds a `phases_observed: set[str]` key whose members are drawn from
        {"Phase 1","Phase 2","Phase 3","Approval","Market"} — the cohorts this
        candidate belongs to under strict forward-looking semantics.
        """
        trial_index = self._build_trial_index(trial_table)
        reference_date = self._resolved_reference_date()
        stale_cutoff_years = self._stale_cutoff_years

        records = []
        for c in candidate_table.candidates:
            attrs = attribute_table.attributes.get(c.candidate_id)
            out   = outcome_table.outcomes.get(c.candidate_id)
            outcome_value = out.outcome.value if out else None
            approval_date = out.approval_date if out else None
            commercialization_date = out.commercialization_date if out else None

            phases_observed, phases_advanced = self._phases_observed(
                trial_ids=c.trial_ids,
                trial_index=trial_index,
                outcome=outcome_value,
                approval_date=approval_date,
                commercialization_date=commercialization_date,
                reference_date=reference_date,
                stale_cutoff_years=stale_cutoff_years,
                back_propagate_approval=self._back_propagate_approval,
                highest_phase=c.highest_phase.value,
            )

            records.append({
                "candidate_id": c.candidate_id,
                "drug":         c.drug_name,
                "indication":   c.indication,
                "modality":     attrs.drug_modality if attrs else None,
                "disease_area": attrs.disease_area  if attrs else None,
                "highest_phase": c.highest_phase.value,
                "outcome":      outcome_value,
                "approval_date": approval_date,
                "commercialization_date": commercialization_date,
                "trial_ids":    c.trial_ids,
                "phases_observed": phases_observed,
                "phases_advanced": phases_advanced,
            })
        return records

    @staticmethod
    def _build_trial_index(
        trial_table: TrialTable | None,
    ) -> dict[str, tuple[str, TrialStatus, date | None, date | None]]:
        """Map nct_id → (phase_value, status, start_date, completion_date)."""
        if trial_table is None:
            return {}
        return {
            t.nct_id: (t.phase.value, t.status, t.start_date, t.completion_date)
            for t in trial_table.trials
        }

    @staticmethod
    def _phases_observed(
        trial_ids: list[str],
        trial_index: dict[str, tuple[str, TrialStatus, date | None, date | None]],
        outcome: str | None,
        approval_date,
        commercialization_date,
        reference_date: date,
        stale_cutoff_years: float,
        back_propagate_approval: bool = False,
        highest_phase: str | None = None,
    ) -> tuple[set[str], set[str]]:
        """Return (cohort_phases, advancement_phases) for a candidate.

        * Ongoing candidates are OMITTED (return empty sets). A candidate is
          ongoing when outcome="Ongoing" OR its latest trial activity is
          within `stale_cutoff_years` of `reference_date` — i.e. still
          plausibly in development. Approved/Commercialized candidates bypass
          this rule: a known approval overrides recency.
        * Unknown outcomes are treated as a failure at the candidate's
          highest_phase — same effect as a FAILED_PHASE_<highest> adjudicator
          verdict (advancement-only; no cohort padding without corroborating
          terminal trial evidence).
        * cohort_phases require terminal evidence: a trial at that phase with
          a terminal status, or a stale non-terminal trial whose latest
          activity date is at least `stale_cutoff_years` before
          `reference_date`, or an approval / commercialization event.
        * advancement_phases include cohort_phases plus any phase where a
          trial of any status exists — a started-but-not-terminal late-phase
          trial is enough to show the candidate advanced past earlier phases.
        """
        is_approved = (
            outcome in ("Approved", "Commercialized") or approval_date is not None
        )
        if not is_approved:
            if outcome == "Ongoing":
                return set(), set()
            latest_activity = _latest_trial_activity(trial_ids, trial_index)
            if latest_activity is not None and (
                (reference_date - latest_activity).days < stale_cutoff_years * 365.25
            ):
                return set(), set()

        cohort: set[str] = set()
        advancement: set[str] = set()

        for nct in trial_ids:
            entry = trial_index.get(nct)
            if entry is None:
                continue
            phase_value, status, start_date, completion_date = entry
            effectively_terminal = _is_effectively_terminal(
                status, start_date, completion_date,
                reference_date, stale_cutoff_years,
            )
            if phase_value in _PHASE_ORDER:
                advancement.add(phase_value)
                if effectively_terminal:
                    cohort.add(phase_value)
            elif phase_value == "Phase 4":
                # Post-marketing trials imply approval was reached.
                advancement.add("Approval")
                if effectively_terminal:
                    cohort.add("Approval")

        # FAILED_PHASE_N outcomes credit advancement only — a drug with a
        # terminal Phase 1 trial and a FAILED_PHASE_2 verdict is correctly
        # counted as a P1→P2 success. We deliberately do NOT add Phase N to
        # the cohort set from an outcome alone: LLM FAILED_PHASE_N verdicts
        # without terminal trial evidence at Phase N (e.g. active late-phase
        # trials) would otherwise pad the Phase N denominator with
        # uncorroborated failures and deflate the downstream rate.
        _failed_phase_map = {
            "Failed Phase 1": "Phase 1",
            "Failed Phase 2": "Phase 2",
            "Failed Phase 3": "Phase 3",
        }
        failed_phase = _failed_phase_map.get(outcome or "")
        if failed_phase is not None:
            advancement.add(failed_phase)

        # Unknown outcome: same treatment as a FAILED_PHASE_<highest_phase>
        # verdict — credit advancement up to the highest phase so earlier
        # transitions count as successes, but do not pad the cohort set.
        if outcome == "Unknown" and highest_phase in ("Phase 1", "Phase 2", "Phase 3"):
            advancement.add(highest_phase)

        # Approval / Market outcomes — terminal events by definition.
        if is_approved:
            cohort.add("Approval")
            advancement.add("Approval")
        if outcome == "Commercialized" or commercialization_date is not None:
            cohort.add("Market")
            advancement.add("Market")

        # Back-propagation for advanced candidates:
        # impute missing intermediate phases
        if back_propagate_approval and False:
            observed_levels = sorted(
                _PHASE_ORDER[p] for p in cohort if p in _PHASE_ORDER
            )
            if observed_levels:
                low = observed_levels[0]
                high = observed_levels[-1]
                for phase, lvl in _PHASE_ORDER.items():
                    if low <= lvl <= high:
                        cohort.add(phase)
                        advancement.add(phase)

        return cohort, advancement

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
        """Thin wrapper around :func:`transition_rate_from_records`.

        Kept as a method for existing callers and tests; all logic lives in
        the module-level function so reporting components can reuse it
        without instantiating the stage.
        """
        return transition_rate_from_records(
            records, from_phase, to_phase, phase_avg_duration=phase_avg_duration,
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


# ---------------------------------------------------------------------------
# Public module-level API — single source of truth for phase-success rates
# ---------------------------------------------------------------------------

def transition_rate_from_records(
    records: list[dict],
    from_phase: str,
    to_phase: str,
    phase_avg_duration: dict[str, float] | None = None,
) -> TransitionRate:
    """Compute one forward-looking cohort transition rate.

    This is the **single source of truth** for phase-success rates. All
    reporting components that need to (re)compute rates from flat records
    should call this function. Do not re-implement the formula in reporting
    code — reach for this helper or consume `FunnelResults` from the
    aggregation stage.

    Expects each record to carry `phases_observed: set[str]` (terminal
    cohort membership) and optionally `phases_advanced: set[str]`
    (advancement evidence, defaults to `phases_observed` if missing).
    """
    from_level = _PHASE_ORDER[from_phase]
    later_phases = {p for p, lvl in _PHASE_ORDER.items() if lvl > from_level}

    from_cohort = [r for r in records if from_phase in r.get("phases_observed", set())]
    denominator = len(from_cohort)
    numerator = sum(
        1 for r in from_cohort
        if r.get("phases_advanced", r.get("phases_observed", set())) & later_phases
    )
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
