"""Oncology vs Non-Oncology comparison."""

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

def oncology_comparison_chart(
    onc: FunnelSlice,
    non_onc: FunnelSlice,
    overall: FunnelSlice,
) -> bytes:
    """Grouped bar chart: Oncology vs Non-Oncology transition rates + LOA. Returns PNG bytes."""
    trans_labels = ["P1\u2192P2", "P2\u2192P3", "P3\u2192Appr", "Appr\u2192Mkt", "LOA\n(Phase 1)"]

    onc_rates = [t.rate * 100 for t in onc.transitions]
    non_onc_rates = [t.rate * 100 for t in non_onc.transitions]
    onc_n = [t.denominator for t in onc.transitions]
    non_onc_n = [t.denominator for t in non_onc.transitions]

    # Append LOA from Phase 1
    onc_loa = _compute.compute_loa(onc).get("Phase 1", 0.0) * 100
    non_onc_loa = _compute.compute_loa(non_onc).get("Phase 1", 0.0) * 100
    onc_rates.append(onc_loa)
    non_onc_rates.append(non_onc_loa)
    onc_n.append(sum(t.denominator for t in onc.transitions))
    non_onc_n.append(sum(t.denominator for t in non_onc.transitions))

    x = np.arange(len(trans_labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - width / 2, onc_rates, width, label="Oncology", color="#C00000")
    bars2 = ax.bar(x + width / 2, non_onc_rates, width, label="Non-Oncology", color="#4472C4")

    for bars, rates, ns in [(bars1, onc_rates, onc_n), (bars2, non_onc_rates, non_onc_n)]:
        for bar, rate, n in zip(bars, rates, ns):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                    f"{rate:.1f}%", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(trans_labels)
    ax.set_ylabel("Probability of Success (%)")
    ax.set_title("Oncology vs Non-Oncology Phase Transition Success Rates and LOA")
    ax.legend()
    ax.set_ylim(0, max(max(onc_rates), max(non_onc_rates)) * 1.15)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Component class
# ---------------------------------------------------------------------------

class OncologyComponent:
    key = "oncology"
    order = 50
    title = "Oncology vs Non-Oncology"

    def should_include(self, ctx: ReportContext) -> bool:
        return not ctx.peptide_only

    def render(self, ctx: ReportContext) -> ComponentResult:
        fr = ctx.funnel_results
        onc_split = _compute.oncology_vs_rest(fr.by_disease_area)
        onc_table = _compute.loa_table(onc_split, fr.overall)

        return ComponentResult(
            tables={"oncology_comparison": onc_table},
            figures={
                "oncology_comparison": oncology_comparison_chart(
                    onc_split.get("Oncology", FunnelSlice(None, "Oncology", 0, [])),
                    onc_split.get("Non-Oncology", FunnelSlice(None, "Non-Oncology", 0, [])),
                    fr.overall,
                ),
            },
            export_data={"oncology_comparison": onc_table},
            narrative=_narrative.oncology_narrative(onc_split, fr.overall),
        )
