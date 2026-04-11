"""Development timeline stacked bar chart."""

import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .._types import ComponentResult, ReportContext
from .. import _compute
from .. import _narrative
from ....models import FunnelSlice


# ---------------------------------------------------------------------------
# Standalone functions
# ---------------------------------------------------------------------------

def timeline_by_disease_chart(
    slices: dict[str, FunnelSlice],
) -> bytes:
    """Horizontal stacked bar chart of phase durations by disease area. Returns PNG bytes."""
    phase_labels = ["P1\u2192P2", "P2\u2192P3", "P3\u2192Appr", "Appr\u2192Mkt"]
    colors = ["#4472C4", "#ED7D31", "#A5A5A5", "#FFC000"]

    # Extract durations per disease area
    items: list[tuple[str, list[float | None]]] = []
    for name, fs in slices.items():
        durations = [t.avg_duration_years for t in fs.transitions]
        # Skip disease areas with no duration data at all
        if all(d is None for d in durations):
            continue
        items.append((name, durations))

    if not items:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No timeline data available", ha="center", transform=ax.transAxes)
        ax.axis("off")
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig)
        return buf.getvalue()

    # Sort by total P1->Approval duration (sum of first 3 phases)
    def _total_clinical(durs: list[float | None]) -> float:
        return sum(d or 0 for d in durs[:3])

    items.sort(key=lambda x: _total_clinical(x[1]))

    labels = [name for name, _ in items]
    all_durations = [durs for _, durs in items]

    fig, ax = plt.subplots(figsize=(10, max(4, len(labels) * 0.5)))

    for phase_idx in range(4):
        lefts = []
        widths = []
        for durs in all_durations:
            left = sum(durs[j] or 0 for j in range(phase_idx))
            width = durs[phase_idx] or 0
            lefts.append(left)
            widths.append(width)
        ax.barh(
            range(len(labels)), widths, left=lefts,
            color=colors[phase_idx], label=phase_labels[phase_idx],
            height=0.6, edgecolor="white", linewidth=0.5,
        )
        # Annotate duration inside each segment
        for i, (l, w) in enumerate(zip(lefts, widths)):
            if w > 0.3:  # only annotate if segment is wide enough
                ax.text(l + w / 2, i, f"{w:.1f}", ha="center", va="center", fontsize=7, color="white", fontweight="bold")

    # Annotate total P1->Approval at end of bar
    for i, durs in enumerate(all_durations):
        total = _total_clinical(durs)
        full = sum(d or 0 for d in durs)
        ax.text(full + 0.2, i, f"{total:.1f} yr", va="center", fontsize=8)

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Duration (Years)")
    ax.set_title("Phase Transition Durations by Disease Area")
    ax.legend(loc="lower right", fontsize=8)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Component class
# ---------------------------------------------------------------------------

class TimelineComponent:
    key = "timeline"
    order = 60
    title = "Drug Development Timelines"

    def should_include(self, ctx: ReportContext) -> bool:
        return not ctx.peptide_only

    def render(self, ctx: ReportContext) -> ComponentResult:
        bio_qls_slices = _compute.bio_qls_by_disease_area(ctx.funnel_results.by_disease_area)

        return ComponentResult(
            figures={
                "timeline_by_disease": timeline_by_disease_chart(bio_qls_slices),
            },
            export_data={"timelines": bio_qls_slices},
            narrative=_narrative.timeline_narrative(bio_qls_slices, ctx.funnel_results.overall),
        )
