"""Static modality + disease heatmaps (matplotlib)."""

import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .._types import ComponentResult, ReportContext
from .. import _compute
from .. import _narrative
from ....models import FunnelResults, FunnelSlice
from ._disease_breakdown import breakdown_chart, stratified_funnel_table


# ---------------------------------------------------------------------------
# Standalone functions
# ---------------------------------------------------------------------------

def modality_heatmap(
    funnel_results: FunnelResults,
    disease_area: str | None = None,
    title_suffix: str = "",
) -> bytes:
    """Annotated heatmap: rows=modalities, cols=transitions. Returns PNG bytes."""
    modalities, trans_labels, rates, denoms, _, _ = _compute.heatmap_grid(
        funnel_results, disease_area=disease_area
    )

    if len(modalities) == 0:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No data for heatmap", ha="center", transform=ax.transAxes)
        ax.axis("off")
    else:
        fig, ax = plt.subplots(
            figsize=(max(6, len(trans_labels) * 2), max(4, len(modalities) * 0.8))
        )
        im = ax.imshow(rates, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")

        for i in range(len(modalities)):
            for j in range(len(trans_labels)):
                r = rates[i, j]
                n = denoms[i, j]
                if np.isnan(r):
                    text = "\u2014"
                else:
                    text = f"{r * 100:.0f}%\n(n={n})"

                color = "black" if (np.isnan(r) or 0.3 < r < 0.7) else "white"
                ax.text(j, i, text, ha="center", va="center", fontsize=9, color=color)

        ax.set_xticks(range(len(trans_labels)))
        ax.set_xticklabels(trans_labels)
        ax.set_yticks(range(len(modalities)))
        ax.set_yticklabels(modalities)
        fig.colorbar(im, ax=ax, label="Transition Rate", shrink=0.8)
        title = "Phase-Transition Success Rates by Modality"
        if title_suffix:
            title += f" ({title_suffix})"
        ax.set_title(title)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def disease_heatmap(
    funnel_results: FunnelResults,
    modality: str | None = None,
    title_suffix: str = "",
) -> bytes:
    """Annotated heatmap: rows=disease areas, cols=transitions. Returns PNG bytes."""
    disease_areas, trans_labels, rates, denoms, _, _ = _compute.disease_heatmap_grid(
        funnel_results, modality=modality
    )

    if len(disease_areas) == 0:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No data for heatmap", ha="center", transform=ax.transAxes)
        ax.axis("off")
    else:
        fig, ax = plt.subplots(
            figsize=(max(6, len(trans_labels) * 2), max(4, len(disease_areas) * 0.6))
        )
        im = ax.imshow(rates, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")

        for i in range(len(disease_areas)):
            for j in range(len(trans_labels)):
                r = rates[i, j]
                n = denoms[i, j]
                if np.isnan(r):
                    text = "\u2014"
                else:
                    text = f"{r * 100:.0f}%\n(n={n})"

                color = "black" if (np.isnan(r) or 0.3 < r < 0.7) else "white"
                ax.text(j, i, text, ha="center", va="center", fontsize=8, color=color)

        ax.set_xticks(range(len(trans_labels)))
        ax.set_xticklabels(trans_labels)
        ax.set_yticks(range(len(disease_areas)))
        ax.set_yticklabels(disease_areas)
        fig.colorbar(im, ax=ax, label="Transition Rate", shrink=0.8)
        title = "Phase-Transition Success Rates by Disease Area"
        if title_suffix:
            title += f" ({title_suffix})"
        ax.set_title(title)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def phase_success_by_disease_chart(
    slices: dict[str, FunnelSlice],
    from_phase: str,
    to_phase: str,
) -> bytes:
    """Horizontal bar chart of a single phase transition rate by disease area."""
    items = []
    for name, fs in slices.items():
        t = next((t for t in fs.transitions if t.from_phase == from_phase and t.to_phase == to_phase), None)
        if t and t.denominator > 0:
            items.append((name, t.rate, t.denominator))

    items.sort(key=lambda x: x[1])  # ascending

    if not items:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No data", ha="center", transform=ax.transAxes)
        ax.axis("off")
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig)
        return buf.getvalue()

    labels = [name for name, _, _ in items]
    rates = [rate * 100 for _, rate, _ in items]
    ns = [n for _, _, n in items]

    fig, ax = plt.subplots(figsize=(8, max(4, len(labels) * 0.45)))
    ax.barh(range(len(labels)), rates, color="#4472C4", height=0.6)

    for i, (rate, n) in enumerate(zip(rates, ns)):
        ax.text(rate + 0.5, i, f"{rate:.1f}% (n={n})", va="center", fontsize=8)

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Probability of Success (%)")
    ax.set_title(f"{from_phase} \u2192 {to_phase} Transition Success Rates by Disease Area")
    ax.set_xlim(0, max(rates) * 1.2 if rates else 100)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Component class
# ---------------------------------------------------------------------------

class HeatmapComponent:
    key = "heatmaps"
    order = 30
    title = "Phase-Transition Heatmaps"

    def should_include(self, ctx: ReportContext) -> bool:
        return not ctx.peptide_only

    def render(self, ctx: ReportContext) -> ComponentResult:
        fr = ctx.funnel_results

        bio_qls_slices = _compute.bio_qls_by_disease_area(fr.by_disease_area)
        bio_qls_cross = _compute.bio_qls_by_modality_and_disease(fr.by_modality_and_disease)
        bio_qls_funnel = FunnelResults(
            overall=fr.overall,
            by_modality=fr.by_modality,
            by_disease_area=bio_qls_slices,
            by_modality_and_disease=bio_qls_cross,
        )

        figures = {
            "modality_breakdown": breakdown_chart(fr.by_modality, label="Modality"),
            "modality_heatmap": modality_heatmap(fr),
            "disease_heatmap_all": disease_heatmap(fr),
            "disease_heatmap": disease_heatmap(bio_qls_funnel),
            "phase1_by_disease": phase_success_by_disease_chart(bio_qls_slices, "Phase 1", "Phase 2"),
            "phase2_by_disease": phase_success_by_disease_chart(bio_qls_slices, "Phase 2", "Phase 3"),
            "phase3_by_disease": phase_success_by_disease_chart(bio_qls_slices, "Phase 3", "Approval"),
        }
        tables = {
            "funnel_by_modality": stratified_funnel_table(fr.by_modality),
        }

        # Export data for xlsx
        export_data = {
            "heatmap_modality": _compute.heatmap_grid(fr),
            "heatmap_disease": _compute.disease_heatmap_grid(bio_qls_funnel),
            "heatmap_modality_bubble_funnel": bio_qls_funnel,
            "heatmap_disease_bubble_funnel": bio_qls_funnel,
            "bio_qls_slices": bio_qls_slices,
            "bio_qls_funnel": bio_qls_funnel,
        }

        return ComponentResult(
            tables=tables,
            figures=figures,
            export_data=export_data,
            narrative=_narrative.disease_success_narrative(bio_qls_slices, fr.overall),
        )
