"""Modality proportion over time."""

import io
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .._types import ComponentResult, ReportContext
from .. import _narrative
from ....models import AttributeTable, CandidateTable


# ---------------------------------------------------------------------------
# Standalone functions
# ---------------------------------------------------------------------------

def modality_proportion_data(
    candidate_table: CandidateTable,
    attribute_table: AttributeTable,
) -> list[dict]:
    """Compute modality proportions per year. Returns flat list of row dicts."""
    year_modality: dict[int, Counter] = {}
    for c in candidate_table.candidates:
        if c.earliest_start_date is None:
            continue
        attrs = attribute_table.attributes.get(c.candidate_id)
        if attrs is None:
            continue
        year = c.earliest_start_date.year
        if year not in year_modality:
            year_modality[year] = Counter()
        year_modality[year][attrs.drug_modality] += 1

    rows: list[dict] = []
    for year in sorted(year_modality):
        total = sum(year_modality[year].values())
        for modality, count in sorted(year_modality[year].items()):
            rows.append({
                "year": year,
                "modality": modality,
                "count": count,
                "pct": count / total if total > 0 else 0.0,
            })
    return rows


def modality_proportion_chart(data: list[dict]) -> bytes:
    """Stacked area chart of modality proportions over time. Returns PNG bytes."""
    if not data:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No modality proportion data", ha="center", transform=ax.transAxes)
        ax.axis("off")
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig)
        return buf.getvalue()

    # Pivot data into year x modality matrix
    years = sorted({d["year"] for d in data})
    # Filter out years with fewer than 10 total candidates
    year_totals = {}
    for d in data:
        year_totals[d["year"]] = year_totals.get(d["year"], 0) + d["count"]
    years = [yr for yr in years if year_totals.get(yr, 0) >= 10]
    if not years:
        # Fall back to showing all years if filtering removes everything
        years = sorted({d["year"] for d in data})
    modalities = sorted({d["modality"] for d in data})
    pct_map: dict[tuple[int, str], float] = {(d["year"], d["modality"]): d["pct"] for d in data}

    fig, ax = plt.subplots(figsize=(10, 5))

    if len(years) >= 4:
        # Stacked area chart
        y_arrays = []
        for mod in modalities:
            y_arrays.append([pct_map.get((yr, mod), 0.0) * 100 for yr in years])
        ax.stackplot(years, *y_arrays, labels=modalities, alpha=0.8)
        ax.set_ylabel("% of Candidates")
    else:
        # Grouped bar chart for sparse data
        x = np.arange(len(years))
        width = 0.8 / len(modalities)
        for i, mod in enumerate(modalities):
            vals = [pct_map.get((yr, mod), 0.0) * 100 for yr in years]
            ax.bar(x + i * width, vals, width, label=mod)
        ax.set_xticks(x + width * len(modalities) / 2)
        ax.set_xticklabels(years)
        ax.set_ylabel("% of Candidates")

    ax.set_xlabel("Year")
    ax.set_title("Drug Modality Composition Over Time")
    min_n = min(year_totals.get(yr, 0) for yr in years)
    max_n = max(year_totals.get(yr, 0) for yr in years)
    ax.text(0.5, -0.12, f"(n per year: {min_n}\u2013{max_n}; years with <10 candidates excluded)",
            transform=ax.transAxes, ha="center", fontsize=8, color="gray")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=7, ncol=1)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def interactive_modality_proportion(data: list[dict]) -> str:
    """Plotly stacked area chart of modality proportions. Returns HTML string."""
    import plotly.graph_objects as go
    import plotly.io as pio

    if not data:
        return "<p>No modality proportion data available.</p>"

    years = sorted({d["year"] for d in data})
    # Filter out years with fewer than 10 total candidates
    year_totals = {}
    for d in data:
        year_totals[d["year"]] = year_totals.get(d["year"], 0) + d["count"]
    years = [yr for yr in years if year_totals.get(yr, 0) >= 10]
    if not years:
        # Fall back to showing all years if filtering removes everything
        years = sorted({d["year"] for d in data})
    modalities = sorted({d["modality"] for d in data})
    pct_map: dict[tuple[int, str], float] = {}
    count_map: dict[tuple[int, str], int] = {}
    for d in data:
        pct_map[(d["year"], d["modality"])] = d["pct"]
        count_map[(d["year"], d["modality"])] = d["count"]

    fig = go.Figure()
    for mod in modalities:
        pcts = [pct_map.get((yr, mod), 0.0) * 100 for yr in years]
        counts = [count_map.get((yr, mod), 0) for yr in years]
        hover = [f"{mod}<br>Year: {yr}<br>{p:.1f}% (n={c})" for yr, p, c in zip(years, pcts, counts)]
        fig.add_trace(go.Scatter(
            x=years, y=pcts, name=mod, stackgroup="one",
            hovertext=hover, hoverinfo="text",
        ))

    fig.update_layout(
        title="Drug Modality Composition Over Time",
        xaxis_title="Year", yaxis_title="% of Candidates",
        height=450, width=800,
        legend=dict(font=dict(size=9)),
    )
    return pio.to_html(fig, full_html=False, include_plotlyjs="cdn")


# ---------------------------------------------------------------------------
# Component class
# ---------------------------------------------------------------------------

class ModalityTrendComponent:
    key = "modality_trend"
    order = 80
    title = "Drug Modality Composition Over Time"

    def should_include(self, ctx: ReportContext) -> bool:
        return not ctx.peptide_only

    def render(self, ctx: ReportContext) -> ComponentResult:
        data = modality_proportion_data(ctx.candidate_table, ctx.attribute_table)
        if not data:
            return ComponentResult()

        return ComponentResult(
            tables={"modality_proportion": data},
            figures={"modality_proportion": modality_proportion_chart(data)},
            html_figures={"modality_proportion": interactive_modality_proportion(data)},
            export_data={"modality_proportion": data},
            narrative=_narrative.modality_trend_narrative(data),
        )
