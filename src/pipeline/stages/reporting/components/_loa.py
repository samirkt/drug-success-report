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

def loa_by_disease_chart(
    slices: dict[str, FunnelSlice],
    overall: FunnelSlice | None = None,
) -> bytes:
    """Horizontal bar chart of LOA from Phase 1, by disease area. Returns PNG bytes."""
    items = [(name, _compute.compute_loa(fs).get("Phase 1", 0.0), fs)
             for name, fs in slices.items()]
    # Filter out slices with insufficient data
    items = [(name, loa, fs) for name, loa, fs in items
             if sum(t.denominator for t in fs.transitions) >= 10]
    items.sort(key=lambda x: x[1])  # ascending (lowest at top, highest at bottom)

    labels = [name for name, _, _ in items]
    loas = [loa for _, loa, _ in items]
    n_totals = [sum(t.denominator for t in fs.transitions) for _, _, fs in items]

    fig, ax = plt.subplots(figsize=(8, max(4, len(labels) * 0.45)))
    bars = ax.barh(range(len(labels)), [l * 100 for l in loas], color="#4472C4", height=0.6)

    for i, (loa, n) in enumerate(zip(loas, n_totals)):
        ax.text(loa * 100 + 0.5, i, f"{loa * 100:.1f}% (n={n})", va="center", fontsize=8)

    if overall is not None:
        overall_loa = _compute.compute_loa(overall).get("Phase 1", 0.0)
        ax.axvline(overall_loa * 100, color="gray", linestyle="--", linewidth=1, label=f"All: {overall_loa*100:.1f}%")
        ax.legend(fontsize=8)

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Likelihood of Approval from Phase 1 (%)")
    ax.set_title("LOA from Phase 1 by Disease Area")
    ax.set_xlim(0, max(loas) * 100 * 1.3 if loas else 100)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def loa_by_modality_chart(
    slices: dict[str, FunnelSlice],
    overall: FunnelSlice | None = None,
) -> bytes:
    """Horizontal bar chart of LOA from Phase 1, by modality. Returns PNG bytes."""
    items = [(name, _compute.compute_loa(fs).get("Phase 1", 0.0), fs)
             for name, fs in slices.items()]
    # Filter out slices with insufficient data
    items = [(name, loa, fs) for name, loa, fs in items
             if sum(t.denominator for t in fs.transitions) >= 10]
    items.sort(key=lambda x: x[1])

    labels = [name for name, _, _ in items]
    loas = [loa for _, loa, _ in items]
    n_totals = [sum(t.denominator for t in fs.transitions) for _, _, fs in items]

    fig, ax = plt.subplots(figsize=(8, max(4, len(labels) * 0.45)))
    ax.barh(range(len(labels)), [l * 100 for l in loas], color="#ED7D31", height=0.6)

    for i, (loa, n) in enumerate(zip(loas, n_totals)):
        ax.text(loa * 100 + 0.5, i, f"{loa * 100:.1f}% (n={n})", va="center", fontsize=8)

    if overall is not None:
        overall_loa = _compute.compute_loa(overall).get("Phase 1", 0.0)
        ax.axvline(overall_loa * 100, color="gray", linestyle="--", linewidth=1, label=f"All: {overall_loa*100:.1f}%")
        ax.legend(fontsize=8)

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Likelihood of Approval from Phase 1 (%)")
    ax.set_title("LOA from Phase 1 by Modality")
    ax.set_xlim(0, max(loas) * 100 * 1.3 if loas else 100)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


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
