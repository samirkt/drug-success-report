"""Disease area breakdown chart + stratified funnel table."""

import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .._types import ComponentResult, ReportContext
from .. import _compute


# ---------------------------------------------------------------------------
# Standalone functions
# ---------------------------------------------------------------------------

def breakdown_chart(slices: dict, label: str) -> bytes:
    """Render a grouped bar chart for a stratified breakdown. Returns image bytes."""
    if not slices:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, f"No {label} data", ha="center", transform=ax.transAxes)
        ax.axis("off")
    else:
        sorted_items = sorted(slices.items(), key=lambda kv: kv[1].candidate_count, reverse=True)
        strata = [k for k, _ in sorted_items]
        counts = [v.candidate_count for _, v in sorted_items]
        fig, ax = plt.subplots(figsize=(max(6, len(strata) * 1.5), 4))
        ax.bar(strata, counts, color="steelblue", alpha=0.8)
        ax.set_xlabel(label)
        ax.set_ylabel("Candidate Count")
        ax.set_title(f"Candidate Count by {label}")
        plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def stratified_funnel_table(slices: dict) -> list[dict]:
    """Flatten stratified FunnelSlice dict into a single table."""
    rows = []
    for stratum, slice_ in slices.items():
        for t in slice_.transitions:
            rows.append({
                "stratum": stratum,
                "from_phase": t.from_phase,
                "to_phase": t.to_phase,
                "numerator": t.numerator,
                "denominator": t.denominator,
                "rate": t.rate,
            })
    return rows


# ---------------------------------------------------------------------------
# Component class
# ---------------------------------------------------------------------------

class DiseaseBreakdownComponent:
    key = "disease_breakdown"
    order = 20
    title = "Phase Transition Success by Disease Area"

    def should_include(self, ctx: ReportContext) -> bool:
        return True

    def render(self, ctx: ReportContext) -> ComponentResult:
        if ctx.peptide_only:
            slices = _compute.peptide_disease_slices(ctx.funnel_results, ctx.attribute_table)
        else:
            slices = ctx.funnel_results.by_disease_area

        return ComponentResult(
            tables={
                "funnel_by_disease": stratified_funnel_table(slices),
            },
            figures={
                "disease_breakdown": breakdown_chart(slices, label="Disease Area"),
            },
        )
