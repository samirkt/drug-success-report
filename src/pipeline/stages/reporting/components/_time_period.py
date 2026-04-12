"""Multi-period LOA comparison."""

import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .._types import ComponentResult, ReportContext
from .. import _compute
from .. import _narrative
from ....models import (
    AttributeTable,
    CandidateTable,
    FunnelSlice,
    OutcomeTable,
)


# ---------------------------------------------------------------------------
# Standalone functions
# ---------------------------------------------------------------------------

def _build_candidate_records(
    candidate_table: CandidateTable,
    attribute_table: AttributeTable,
    outcome_table: OutcomeTable,
) -> tuple[list[dict], int]:
    """Flatten candidates into records suitable for _compute.build_funnel_slice."""
    records: list[dict] = []
    n_excluded = 0
    for c in candidate_table.candidates:
        if c.earliest_start_date is None:
            n_excluded += 1
            continue
        attrs = attribute_table.attributes.get(c.candidate_id)
        out = outcome_table.outcomes.get(c.candidate_id)
        records.append({
            "candidate_id": c.candidate_id,
            "highest_phase": c.highest_phase.value,
            "disease_area": attrs.disease_area if attrs else None,
            "modality": attrs.drug_modality if attrs else None,
            "outcome": out.outcome.value if out else None,
            "year": c.earliest_start_date.year,
        })
    return records, n_excluded


def _resolve_periods(
    records: list[dict], periods: list[tuple[int, int]] | None,
) -> tuple[list[tuple[int, int]], list[str]]:
    """Apply the auto-split fallback when periods is None and build display labels."""
    if periods is None:
        from datetime import date as _date
        years = sorted(r["year"] for r in records)
        min_year = years[0]
        max_year = years[-1]
        decade_start = _date.today().year - 9  # last 10 years including current
        if decade_start <= min_year:
            # All data is within last decade; fall back to median split
            median_year = years[len(years) // 2]
            periods = [(min_year, median_year), (median_year + 1, max_year)]
        else:
            periods = [(min_year, decade_start - 1), (decade_start, max_year)]
    period_labels = [f"{s}\u2013{e}" for s, e in periods]
    return periods, period_labels


def partition_by_time_periods(
    candidate_table: CandidateTable,
    attribute_table: AttributeTable,
    outcome_table: OutcomeTable,
    periods: list[tuple[int, int]] | None = None,
) -> tuple[dict[str, dict[str, FunnelSlice]], list[str], int]:
    """Split candidates into time-period cohorts and compute per-disease funnels.

    Args:
        periods: list of (start_year, end_year) tuples. None = auto-split by median.

    Returns:
        (period_slices, period_labels, n_excluded) where period_slices is
        {label: {disease_area: FunnelSlice}}.
    """
    records, n_excluded = _build_candidate_records(
        candidate_table, attribute_table, outcome_table
    )
    if not records:
        return {}, [], n_excluded

    periods, period_labels = _resolve_periods(records, periods)

    period_slices: dict[str, dict[str, FunnelSlice]] = {}
    for (start_yr, end_yr), label in zip(periods, period_labels):
        cohort = [r for r in records if start_yr <= r["year"] <= end_yr]
        disease_areas = {r["disease_area"] for r in cohort if r["disease_area"]}
        slices: dict[str, FunnelSlice] = {}
        for da in disease_areas:
            slices[da] = _compute.build_funnel_slice(cohort, disease_area=da)
        period_slices[label] = slices

    return period_slices, period_labels, n_excluded


def period_overall_slices(
    candidate_table: CandidateTable,
    attribute_table: AttributeTable,
    outcome_table: OutcomeTable,
    periods: list[tuple[int, int]] | None = None,
) -> tuple[dict[str, FunnelSlice], list[str], int]:
    """Compute one aggregate FunnelSlice per time period (across all disease areas).

    Returns:
        (period_overall, period_labels, n_excluded).
    """
    records, n_excluded = _build_candidate_records(
        candidate_table, attribute_table, outcome_table
    )
    if not records:
        return {}, [], n_excluded

    periods, period_labels = _resolve_periods(records, periods)

    period_overall: dict[str, FunnelSlice] = {}
    for (start_yr, end_yr), label in zip(periods, period_labels):
        cohort = [r for r in records if start_yr <= r["year"] <= end_yr]
        period_overall[label] = _compute.build_funnel_slice(cohort)
    return period_overall, period_labels, n_excluded


def time_period_comparison_chart(
    period_slices: dict[str, dict[str, FunnelSlice]],
    period_labels: list[str],
    n_excluded: int,
) -> bytes:
    """Grouped horizontal bar chart: LOA from Phase 1 by disease area, N bars per period."""
    if not period_labels:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No time-period data", ha="center", transform=ax.transAxes)
        ax.axis("off")
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig)
        return buf.getvalue()

    # Collect all disease areas across all periods
    all_das = sorted({da for slices in period_slices.values() for da in slices})

    # Compute LOA from Phase 1 for each (period, disease_area)
    loa_grid: dict[str, dict[str, float]] = {}
    for label in period_labels:
        loa_grid[label] = {}
        for da in all_das:
            fs = period_slices.get(label, {}).get(da)
            if fs and fs.candidate_count > 0:
                loa_grid[label][da] = _compute.compute_loa(fs).get("Phase 1", 0.0)
            else:
                loa_grid[label][da] = 0.0

    # Sort disease areas by average LOA across periods
    avg_loa = {da: np.mean([loa_grid[l].get(da, 0) for l in period_labels]) for da in all_das}
    all_das = sorted(all_das, key=lambda da: avg_loa[da])

    n_periods = len(period_labels)
    n_das = len(all_das)
    colors = plt.cm.tab10(np.linspace(0, 1, max(n_periods, 2)))

    fig, ax = plt.subplots(figsize=(9, max(4, n_das * 0.5)))
    bar_height = 0.8 / n_periods

    for p_idx, label in enumerate(period_labels):
        y_offsets = [i + (p_idx - n_periods / 2 + 0.5) * bar_height for i in range(n_das)]
        vals = [loa_grid[label].get(da, 0) * 100 for da in all_das]
        bars = ax.barh(y_offsets, vals, height=bar_height, label=label, color=colors[p_idx])
        for y, v in zip(y_offsets, vals):
            if v > 0:
                ax.text(v + 0.3, y, f"{v:.1f}%", va="center", fontsize=7)

    ax.set_yticks(range(n_das))
    ax.set_yticklabels(all_das, fontsize=9)
    ax.set_xlabel("LOA from Phase 1 (%)")
    title = "LOA from Phase 1 by Disease Area and Time Period"
    if n_excluded > 0:
        title += f"\n({n_excluded} candidates excluded \u2014 missing start date)"
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def time_period_transition_chart(
    period_overall: dict[str, FunnelSlice],
    period_labels: list[str],
) -> bytes:
    """Grouped bar chart: per-phase transition rates + LOA, one group per time period.

    The legend shows the total candidate count contributing to each period.
    """
    trans_labels = ["P1\u2192P2", "P2\u2192P3", "P3\u2192Appr", "Appr\u2192Mkt", "LOA\n(Phase 1)"]

    if not period_labels:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No time-period data", ha="center", transform=ax.transAxes)
        ax.axis("off")
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig)
        return buf.getvalue()

    # Per-period rate vectors and sample counts
    period_rates: dict[str, list[float]] = {}
    period_sample_n: dict[str, int] = {}
    for label in period_labels:
        fs = period_overall.get(label)
        if fs is None:
            period_rates[label] = [0.0] * len(trans_labels)
            period_sample_n[label] = 0
            continue
        rates = [t.rate * 100 for t in fs.transitions]
        loa = _compute.compute_loa(fs).get("Phase 1", 0.0) * 100
        rates.append(loa)
        period_rates[label] = rates
        period_sample_n[label] = fs.candidate_count

    x = np.arange(len(trans_labels))
    n_periods = len(period_labels)
    total_width = 0.8
    width = total_width / n_periods
    colors = plt.cm.tab10(np.linspace(0, 1, max(n_periods, 2)))

    fig, ax = plt.subplots(figsize=(11, 5))
    for p_idx, label in enumerate(period_labels):
        offset = (p_idx - (n_periods - 1) / 2) * width
        rates = period_rates[label]
        legend_label = f"{label} (n={period_sample_n[label]:,})"
        bars = ax.bar(x + offset, rates, width, label=legend_label, color=colors[p_idx])
        for bar, r in zip(bars, rates):
            if r > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.6,
                    f"{r:.1f}%", ha="center", va="bottom", fontsize=7,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(trans_labels)
    ax.set_ylabel("Probability of Success (%)")
    ax.set_title("Phase Transition Success Rates and LOA by Time Period")
    ax.legend(fontsize=8)

    all_rates = [r for rates in period_rates.values() for r in rates]
    y_max = max(all_rates) if all_rates else 0
    ax.set_ylim(0, max(y_max * 1.15, 1))

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def time_period_loa_table(
    period_slices: dict[str, dict[str, FunnelSlice]],
    period_labels: list[str],
) -> list[dict]:
    """Build LOA comparison table with dynamic columns per period."""
    all_das = sorted({da for slices in period_slices.values() for da in slices})
    rows: list[dict] = []
    for da in all_das:
        row: dict = {"disease_area": da}
        for label in period_labels:
            fs = period_slices.get(label, {}).get(da)
            if fs and fs.candidate_count > 0:
                loa = _compute.compute_loa(fs).get("Phase 1", 0.0)
                n = sum(t.denominator for t in fs.transitions)
            else:
                loa = None
                n = 0
            safe_label = label.replace("\u2013", "-")
            row[f"loa_{safe_label}"] = loa
            row[f"n_{safe_label}"] = n
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Component class
# ---------------------------------------------------------------------------

class TimePeriodComponent:
    key = "time_period"
    order = 70
    title = "LOA by Time Period"

    def should_include(self, ctx: ReportContext) -> bool:
        return not ctx.peptide_only

    def render(self, ctx: ReportContext) -> ComponentResult:
        period_slices, period_labels, n_excluded = partition_by_time_periods(
            ctx.candidate_table, ctx.attribute_table, ctx.outcome_table,
            periods=ctx.time_periods,
        )
        if not period_labels:
            return ComponentResult()

        period_bio = {
            label: _compute.bio_qls_by_disease_area(slices)
            for label, slices in period_slices.items()
        }
        tp_table = time_period_loa_table(period_bio, period_labels)

        period_overall, _, _ = period_overall_slices(
            ctx.candidate_table, ctx.attribute_table, ctx.outcome_table,
            periods=ctx.time_periods,
        )

        return ComponentResult(
            tables={"time_period_loa": tp_table},
            figures={
                "time_period_transitions": time_period_transition_chart(
                    period_overall, period_labels,
                ),
                "time_period_comparison": time_period_comparison_chart(
                    period_bio, period_labels, n_excluded,
                ),
            },
            export_data={"time_period_loa": tp_table},
            narrative=_narrative.time_period_narrative(period_bio, period_labels, n_excluded),
        )
