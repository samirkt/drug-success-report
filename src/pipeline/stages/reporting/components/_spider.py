"""Spider/radar charts -- overall + per disease area."""

import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .._types import ComponentResult, ReportContext
from .. import _compute
from .. import _narrative
from ....models import FunnelSlice

LOW_N_THRESHOLD = 20


# ---------------------------------------------------------------------------
# Standalone functions
# ---------------------------------------------------------------------------

def spider_chart(funnel_slice: FunnelSlice, label: str) -> bytes:
    """Render a single radar chart using standard funnel transitions."""
    axis_labels = ["P1->P2", "P2->P3", "P3->approval", "approval->commercial"]
    transition_key_to_index = {
        ("Phase 1", "Phase 2"): 0,
        ("Phase 2", "Phase 3"): 1,
        ("Phase 3", "Approval"): 2,
        ("Approval", "Market"): 3,
        ("Approval", "Commercial"): 3,
    }

    values = [0.0, 0.0, 0.0, 0.0]
    for t in funnel_slice.transitions:
        idx = transition_key_to_index.get((t.from_phase, t.to_phase))
        if idx is not None:
            values[idx] = t.rate

    angles = np.linspace(0, 2 * np.pi, len(axis_labels), endpoint=False)
    closed_angles = np.concatenate([angles, [angles[0]]])
    closed_values = values + [values[0]]

    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw={"polar": True})
    ax.plot(closed_angles, closed_values, color="darkorange", linewidth=2)
    ax.fill(closed_angles, closed_values, color="darkorange", alpha=0.2)
    ax.set_xticks(angles)
    ax.set_xticklabels(axis_labels)
    ax.set_yticks([0.25, 0.50, 0.75, 1.0])
    ax.set_ylim(0, 1)
    n = next(
        (t.denominator for t in funnel_slice.transitions if t.from_phase == "Phase 1"),
        funnel_slice.candidate_count,
    )
    ax.set_title(f"{label}: Success rates (n={n})")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def disease_spider_charts(slices: dict[str, FunnelSlice]) -> dict[str, bytes]:
    """Render one radar chart per disease area using standard funnel transitions."""
    charts: dict[str, bytes] = {}
    for disease_area, funnel_slice in sorted(slices.items()):
        if funnel_slice.candidate_count > LOW_N_THRESHOLD:
            safe_name = disease_area.lower().replace(" ", "_").replace("/", "_")
            charts[f"disease_spider_{safe_name}"] = spider_chart(
                funnel_slice, label=disease_area
            )
    return charts


# ---------------------------------------------------------------------------
# Component class
# ---------------------------------------------------------------------------

class SpiderChartComponent:
    key = "spider"
    order = 10
    title = "Phase Transition Success Rates"

    def should_include(self, ctx: ReportContext) -> bool:
        return True

    def render(self, ctx: ReportContext) -> ComponentResult:
        if ctx.peptide_only:
            funnel = ctx.funnel_results.by_modality.get("peptide", ctx.funnel_results.overall)
            slices = _compute.peptide_disease_slices(ctx.funnel_results, ctx.attribute_table)
        else:
            funnel = ctx.funnel_results.overall
            slices = ctx.funnel_results.by_disease_area

        figures: dict[str, bytes] = {
            "overall_spider": spider_chart(funnel, label="Overall"),
        }
        disease_charts = disease_spider_charts(slices)
        figures.update(disease_charts)

        return ComponentResult(
            figures=figures,
            narrative=_narrative.spider_narrative(funnel),
        )
