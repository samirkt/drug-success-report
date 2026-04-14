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

_TRANSITION_KEYS = [
    ("Phase 1", "Phase 2"),
    ("Phase 2", "Phase 3"),
    ("Phase 3", "Approval"),
    ("Approval", "Market"),
]


def _slice_rate_series(fs: FunnelSlice) -> tuple[list[float], list[int]]:
    """Return (rates_pct_with_loa, denominators_with_total) for a slice.

    Pads missing transitions with zeros so the returned vectors always have
    length 5 (4 transitions + LOA), matching ``trans_labels``.
    """
    by_key = {(t.from_phase, t.to_phase): t for t in fs.transitions}
    rates = [by_key[k].rate * 100 if k in by_key else 0.0 for k in _TRANSITION_KEYS]
    ns = [by_key[k].denominator if k in by_key else 0 for k in _TRANSITION_KEYS]
    rates.append(_compute.compute_loa(fs).get("Phase 1", 0.0) * 100)
    ns.append(sum(t.denominator for t in fs.transitions))
    return rates, ns


def oncology_comparison_chart(
    onc: FunnelSlice,
    non_onc: FunnelSlice,
    overall: FunnelSlice,
) -> bytes:
    """Grouped bar chart: Oncology / Non-Oncology / All transition rates + LOA."""
    trans_labels = ["P1\u2192P2", "P2\u2192P3", "P3\u2192Appr", "Appr\u2192Mkt", "LOA\n(Phase 1)"]

    onc_rates, onc_n = _slice_rate_series(onc)
    non_onc_rates, non_onc_n = _slice_rate_series(non_onc)
    all_rates, all_n = _slice_rate_series(overall)

    series = [
        ("Oncology", onc_rates, onc_n, "#C00000"),
        ("Non-Oncology", non_onc_rates, non_onc_n, "#4472C4"),
        ("All", all_rates, all_n, "#888888"),
    ]

    x = np.arange(len(trans_labels))
    width = 0.28
    offsets = [-width, 0.0, width]

    fig, ax = plt.subplots(figsize=(10, 5))
    all_bars = []
    for (label, rates, _, color), off in zip(series, offsets):
        bars = ax.bar(x + off, rates, width, label=label, color=color)
        all_bars.append((bars, rates))

    for bars, rates in all_bars:
        for bar, rate in zip(bars, rates):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                    f"{rate:.1f}%", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(trans_labels)
    ax.set_ylabel("Probability of Success (%)")
    ax.set_title("Oncology vs Non-Oncology vs All: Phase Transition Success Rates and LOA")
    ax.legend()
    peak = max(rate for _, rates, _, _ in series for rate in rates)
    ax.set_ylim(0, peak * 1.15 if peak > 0 else 1)

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
