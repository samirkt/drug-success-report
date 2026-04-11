"""Sponsor concentration analysis."""

import io
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .._types import ComponentResult, ReportContext
from .. import _narrative
from .. import _compute
from ....models import AttributeTable, CandidateTable


# ---------------------------------------------------------------------------
# Standalone functions
# ---------------------------------------------------------------------------

def sponsor_concentration_data(
    candidate_table: CandidateTable,
    attribute_table: AttributeTable,
) -> list[dict]:
    """Per-sponsor summary: candidate count, modalities, disease areas."""
    sponsor_candidates: dict[str, int] = Counter()
    sponsor_modalities: dict[str, set] = defaultdict(set)
    sponsor_diseases: dict[str, set] = defaultdict(set)

    for c in candidate_table.candidates:
        attrs = attribute_table.attributes.get(c.candidate_id)
        for sponsor in c.sponsors:
            if not sponsor:
                continue
            sponsor_candidates[sponsor] += 1
            if attrs:
                sponsor_modalities[sponsor].add(attrs.drug_modality)
                sponsor_diseases[sponsor].add(attrs.disease_area)

    rows: list[dict] = []
    for sponsor, count in sponsor_candidates.most_common():
        rows.append({
            "sponsor": sponsor,
            "candidate_count": count,
            "n_modalities": len(sponsor_modalities.get(sponsor, set())),
            "n_disease_areas": len(sponsor_diseases.get(sponsor, set())),
            "modalities": ", ".join(sorted(sponsor_modalities.get(sponsor, set()))),
            "disease_areas": ", ".join(sorted(sponsor_diseases.get(sponsor, set()))),
        })
    return rows


def sponsor_concentration_chart(sponsor_data: list[dict]) -> bytes:
    """Two-panel chart: top sponsors bar + distribution histogram. Returns PNG bytes."""
    if not sponsor_data:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No sponsor data", ha="center", transform=ax.transAxes)
        ax.axis("off")
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig)
        return buf.getvalue()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, max(5, min(len(sponsor_data), 20) * 0.3)))

    # Left: top 20 sponsors
    top = sponsor_data[:20]
    names = [d["sponsor"][:30] for d in reversed(top)]
    counts = [d["candidate_count"] for d in reversed(top)]
    ax1.barh(range(len(names)), counts, color="#4472C4", height=0.6)
    ax1.set_yticks(range(len(names)))
    ax1.set_yticklabels(names, fontsize=7)
    ax1.set_xlabel("Candidate Count")
    ax1.set_title("Top 20 Sponsors by Candidate Count")
    for i, c in enumerate(counts):
        ax1.text(c + 0.5, i, str(c), va="center", fontsize=7)

    # Right: distribution histogram
    all_counts = [d["candidate_count"] for d in sponsor_data]
    max_count = max(all_counts)
    if max_count > 50:
        bins = [1, 2, 5, 10, 25, 50, 100, max_count + 1]
        bins = [b for b in bins if b <= max_count + 1]
        if bins[-1] < max_count + 1:
            bins.append(max_count + 1)
    else:
        bins = range(1, max_count + 2)
    ax2.hist(all_counts, bins=bins, color="#ED7D31", edgecolor="white")
    ax2.set_xlabel("Candidates per Sponsor")
    ax2.set_ylabel("Number of Sponsors")
    ax2.set_title("Sponsor Concentration Distribution")
    if max_count > 50:
        ax2.set_xscale("log")

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Component class
# ---------------------------------------------------------------------------

class SponsorComponent:
    key = "sponsor"
    order = 90
    title = "Sponsor Concentration"

    def should_include(self, ctx: ReportContext) -> bool:
        return not ctx.peptide_only

    def render(self, ctx: ReportContext) -> ComponentResult:
        data = sponsor_concentration_data(ctx.candidate_table, ctx.attribute_table)
        if not data:
            return ComponentResult()

        stats = _compute.sponsor_summary_stats(ctx.candidate_table)

        return ComponentResult(
            tables={"sponsor_concentration": data},
            figures={"sponsor_concentration": sponsor_concentration_chart(data)},
            export_data={"sponsor_concentration": data},
            narrative=_narrative.sponsor_narrative(data, stats),
        )
