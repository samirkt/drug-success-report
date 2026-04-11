"""Interactive Plotly bubble heatmaps."""

import numpy as np

from .._types import ComponentResult, ReportContext
from .. import _compute
from ....models import FunnelResults


# ---------------------------------------------------------------------------
# Standalone functions
# ---------------------------------------------------------------------------

def interactive_bubble_heatmap(funnel_results: FunnelResults) -> str:
    """Plotly interactive bubble heatmap with disease-area dropdown. Returns HTML string."""
    import plotly.graph_objects as go
    import plotly.io as pio

    disease_areas = sorted(funnel_results.by_disease_area.keys())
    all_views: list[tuple[str, str | None]] = [("All", None)] + [
        (da, da) for da in disease_areas
    ]

    # Use "All" view for axis labels (superset of modalities)
    all_modalities, all_trans, *_ = _compute.heatmap_grid(funnel_results, disease_area=None)
    mod_index = {m: i for i, m in enumerate(all_modalities)}

    fig = go.Figure()

    for view_idx, (label, da) in enumerate(all_views):
        modalities, trans_labels, rates, denoms, nums, ci_bounds = _compute.heatmap_grid(
            funnel_results, disease_area=da,
        )

        x_vals, y_vals, sizes, colors, hover_texts = [], [], [], [], []
        for i, mod in enumerate(modalities):
            y_pos = mod_index.get(mod, i)
            for j, tl in enumerate(trans_labels):
                r = rates[i, j]
                n = denoms[i, j]
                k = nums[i, j]
                lo, hi = ci_bounds[i][j]
                if np.isnan(r) or n == 0:
                    continue
                x_vals.append(j)
                y_vals.append(y_pos)
                sizes.append(max(8, min(60, np.sqrt(n) * 5)))
                colors.append(r)
                hover_texts.append(
                    f"<b>{mod}</b> | {tl}<br>"
                    f"Rate: {r * 100:.1f}%<br>"
                    f"Passed: {k} / {n}<br>"
                    f"95% CI: [{lo * 100:.1f}%, {hi * 100:.1f}%]"
                )

        fig.add_trace(go.Scatter(
            x=x_vals,
            y=y_vals,
            mode="markers",
            marker=dict(
                size=sizes,
                color=colors,
                colorscale="RdYlGn",
                cmin=0,
                cmax=1,
                showscale=(view_idx == 0),
                colorbar=dict(title="Rate"),
                line=dict(width=1, color="DarkSlateGrey"),
            ),
            text=hover_texts,
            hoverinfo="text",
            visible=(view_idx == 0),
            name=label,
        ))

    # Dropdown buttons
    n_traces = len(fig.data)
    buttons = []
    for view_idx, (label, _) in enumerate(all_views):
        visibility = [False] * n_traces
        if view_idx < n_traces:
            visibility[view_idx] = True
        buttons.append(dict(label=label, method="update", args=[{"visible": visibility}]))

    fig.update_layout(
        title="Phase-Transition Rates: Bubble Size = Sample Size, Color = Rate",
        xaxis=dict(
            tickmode="array",
            tickvals=list(range(len(all_trans))),
            ticktext=all_trans,
            title="Phase Transition",
        ),
        yaxis=dict(
            tickmode="array",
            tickvals=list(range(len(all_modalities))),
            ticktext=all_modalities,
            title="Modality",
            autorange="reversed",
        ),
        updatemenus=[dict(
            buttons=buttons,
            direction="down",
            showactive=True,
            x=1.15,
            y=1.0,
            xanchor="left",
        )],
        hoverlabel=dict(bgcolor="white"),
        height=max(400, len(all_modalities) * 80 + 150),
        width=700,
    )

    return pio.to_html(fig, full_html=False, include_plotlyjs="cdn")


def interactive_disease_bubble_heatmap(funnel_results: FunnelResults) -> str:
    """Plotly interactive bubble heatmap (disease areas) with modality dropdown. Returns HTML string."""
    import plotly.graph_objects as go
    import plotly.io as pio

    modalities = sorted(funnel_results.by_modality.keys())
    all_views: list[tuple[str, str | None]] = [("All", None)] + [
        (m, m) for m in modalities
    ]

    # Use "All" view for axis labels (superset of disease areas)
    all_disease_areas, all_trans, *_ = _compute.disease_heatmap_grid(funnel_results, modality=None)
    da_index = {da: i for i, da in enumerate(all_disease_areas)}

    fig = go.Figure()

    for view_idx, (label, mod) in enumerate(all_views):
        disease_areas, trans_labels, rates, denoms, nums, ci_bounds = _compute.disease_heatmap_grid(
            funnel_results, modality=mod,
        )

        x_vals, y_vals, sizes, colors, hover_texts = [], [], [], [], []
        for i, da in enumerate(disease_areas):
            y_pos = da_index.get(da, i)
            for j, tl in enumerate(trans_labels):
                r = rates[i, j]
                n = denoms[i, j]
                k = nums[i, j]
                lo, hi = ci_bounds[i][j]
                if np.isnan(r) or n == 0:
                    continue
                x_vals.append(j)
                y_vals.append(y_pos)
                sizes.append(max(8, min(60, np.sqrt(n) * 5)))
                colors.append(r)
                hover_texts.append(
                    f"<b>{da}</b> | {tl}<br>"
                    f"Rate: {r * 100:.1f}%<br>"
                    f"Passed: {k} / {n}<br>"
                    f"95% CI: [{lo * 100:.1f}%, {hi * 100:.1f}%]"
                )

        fig.add_trace(go.Scatter(
            x=x_vals,
            y=y_vals,
            mode="markers",
            marker=dict(
                size=sizes,
                color=colors,
                colorscale="RdYlGn",
                cmin=0,
                cmax=1,
                showscale=(view_idx == 0),
                colorbar=dict(title="Rate"),
                line=dict(width=1, color="DarkSlateGrey"),
            ),
            text=hover_texts,
            hoverinfo="text",
            visible=(view_idx == 0),
            name=label,
        ))

    # Dropdown buttons
    n_traces = len(fig.data)
    buttons = []
    for view_idx, (label, _) in enumerate(all_views):
        visibility = [False] * n_traces
        if view_idx < n_traces:
            visibility[view_idx] = True
        buttons.append(dict(label=label, method="update", args=[{"visible": visibility}]))

    fig.update_layout(
        title="Phase-Transition Rates by Disease Area: Bubble Size = Sample Size, Color = Rate",
        xaxis=dict(
            tickmode="array",
            tickvals=list(range(len(all_trans))),
            ticktext=all_trans,
            title="Phase Transition",
        ),
        yaxis=dict(
            tickmode="array",
            tickvals=list(range(len(all_disease_areas))),
            ticktext=all_disease_areas,
            title="Disease Area",
            autorange="reversed",
        ),
        updatemenus=[dict(
            buttons=buttons,
            direction="down",
            showactive=True,
            x=1.15,
            y=1.0,
            xanchor="left",
        )],
        hoverlabel=dict(bgcolor="white"),
        height=max(400, len(all_disease_areas) * 40 + 150),
        width=700,
    )

    return pio.to_html(fig, full_html=False, include_plotlyjs="cdn")


# ---------------------------------------------------------------------------
# Component class
# ---------------------------------------------------------------------------

class BubbleHeatmapComponent:
    key = "bubble_heatmaps"
    order = 35
    title = "Interactive Bubble Heatmaps"

    def should_include(self, ctx: ReportContext) -> bool:
        return not ctx.peptide_only

    def render(self, ctx: ReportContext) -> ComponentResult:
        fr = ctx.funnel_results
        bio_qls_slices = _compute.bio_qls_by_disease_area(fr.by_disease_area)
        bio_qls_cross = _compute.bio_qls_by_modality_and_disease(fr.by_modality_and_disease)
        bio_qls_funnel = FunnelResults(
            overall=fr.overall, by_modality=fr.by_modality,
            by_disease_area=bio_qls_slices, by_modality_and_disease=bio_qls_cross,
        )

        return ComponentResult(
            html_figures={
                "modality_bubble_heatmap": interactive_bubble_heatmap(bio_qls_funnel),
                "disease_bubble_heatmap": interactive_disease_bubble_heatmap(bio_qls_funnel),
            },
        )
