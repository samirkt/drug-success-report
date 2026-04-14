"""Likelihood of Approval bar charts + tables."""

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

_ALL_LABEL = "All"
_ALL_COLOR = "#888888"


def _horizontal_loa_chart(
    slices: dict[str, FunnelSlice],
    overall: FunnelSlice | None,
    bar_color: str,
    title: str,
) -> bytes:
    """Shared horizontal-bar implementation for disease / modality LOA charts.

    Always prepends an `All` bar driven by ``overall`` so each stratified
    slice has a visible population baseline for comparison.
    """
    items = [(name, _compute.compute_loa(fs).get("Phase 1", 0.0), fs)
             for name, fs in slices.items()]
    items = [(name, loa, fs) for name, loa, fs in items
             if sum(t.denominator for t in fs.transitions) >= 10]
    items.sort(key=lambda x: x[1])  # ascending (lowest at top, highest at bottom)

    labels = [name for name, _, _ in items]
    loas = [loa for _, loa, _ in items]
    n_totals = [sum(t.denominator for t in fs.transitions) for _, _, fs in items]
    colors = [bar_color] * len(items)

    # Prepend the "All" row at the top of the horizontal list.
    if overall is not None:
        overall_loa = _compute.compute_loa(overall).get("Phase 1", 0.0)
        overall_n = sum(t.denominator for t in overall.transitions)
        labels.append(_ALL_LABEL)
        loas.append(overall_loa)
        n_totals.append(overall_n)
        colors.append(_ALL_COLOR)

    fig, ax = plt.subplots(figsize=(8, max(4, len(labels) * 0.45)))
    ax.barh(range(len(labels)), [l * 100 for l in loas], color=colors, height=0.6)

    for i, (loa, n) in enumerate(zip(loas, n_totals)):
        ax.text(loa * 100 + 0.5, i, f"{loa * 100:.1f}% (n={n})", va="center", fontsize=8)

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Likelihood of Approval from Phase 1 (%)")
    ax.set_title(title)
    ax.set_xlim(0, max(loas) * 100 * 1.3 if loas else 100)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def loa_by_disease_chart(
    slices: dict[str, FunnelSlice],
    overall: FunnelSlice | None = None,
) -> bytes:
    """Horizontal bar chart of LOA from Phase 1, by disease area. Returns PNG bytes."""
    return _horizontal_loa_chart(
        slices, overall, bar_color="#4472C4", title="LOA from Phase 1 by Disease Area",
    )


def loa_by_modality_chart(
    slices: dict[str, FunnelSlice],
    overall: FunnelSlice | None = None,
) -> bytes:
    """Horizontal bar chart of LOA from Phase 1, by modality. Returns PNG bytes."""
    return _horizontal_loa_chart(
        slices, overall, bar_color="#ED7D31", title="LOA from Phase 1 by Modality",
    )


# ---------------------------------------------------------------------------
# Component class
# ---------------------------------------------------------------------------

class LOAComponent:
    key = "loa"
    order = 40
    title = "Likelihood of Approval"

    def should_include(self, ctx: ReportContext) -> bool:
        return not ctx.peptide_only

    def render(self, ctx: ReportContext) -> ComponentResult:
        fr = ctx.funnel_results
        bio_qls_slices = _compute.bio_qls_by_disease_area(fr.by_disease_area)

        loa_disease = _compute.loa_table(bio_qls_slices, fr.overall)
        loa_modality = _compute.loa_table(fr.by_modality, fr.overall)

        return ComponentResult(
            tables={
                "loa_by_disease": loa_disease,
                "loa_by_modality": loa_modality,
            },
            figures={
                "loa_by_disease": loa_by_disease_chart(bio_qls_slices, fr.overall),
                "loa_by_modality": loa_by_modality_chart(fr.by_modality, fr.overall),
            },
            export_data={
                "loa_by_disease": loa_disease,
                "loa_by_modality": loa_modality,
            },
            narrative=_narrative.loa_disease_narrative(bio_qls_slices, fr.overall) + "\n\n" + _narrative.modality_narrative(fr.by_modality, fr.overall),
        )
