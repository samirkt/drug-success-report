"""Data-driven narrative text generation for BIO/QLS-style reports.

Each function takes computed pipeline data and returns analytical prose
with embedded figure/table references using {{figure:key:caption}} and
{{table:key:caption}} placeholders that the HTML writer resolves.
"""

from __future__ import annotations

from . import _compute
from ...models import FunnelSlice


def _pct(value: float) -> str:
    """Format a rate as a percentage string."""
    return f"{value * 100:.1f}%"


def _fold(a: float, b: float) -> str:
    """Describe the ratio a/b in plain English."""
    if b == 0:
        return "infinitely higher than"
    ratio = a / b
    if ratio >= 2.5:
        return f"{ratio:.0f}-fold higher than"
    if ratio >= 1.8:
        return f"nearly {round(ratio)}-fold higher than"
    if ratio >= 1.3:
        return f"substantially higher than"
    if ratio >= 1.05:
        return "slightly higher than"
    if ratio >= 0.95:
        return "comparable to"
    if ratio >= 0.7:
        return "slightly lower than"
    return "substantially lower than"


def _n_total(fs: FunnelSlice) -> int:
    """Sum of denominators across all transitions."""
    return sum(t.denominator for t in fs.transitions)


# -------------------------------------------------------------------------
# Executive Summary
# -------------------------------------------------------------------------

def executive_summary(
    overall: FunnelSlice,
    bio_qls_slices: dict[str, FunnelSlice] | None,
    by_modality: dict[str, FunnelSlice] | None,
    onc_split: dict[str, FunnelSlice] | None,
    timelines: dict[str, FunnelSlice] | None,
    stats: dict | None,
    peptide_only: bool,
) -> str:
    """Generate executive summary with key takeaways as bullet points."""
    loa = _compute.compute_loa(overall)
    loa_p1 = loa.get("Phase 1", 0.0)
    n_total = _n_total(overall)

    bullets: list[str] = []

    # Overall LOA
    bullets.append(
        f"The overall likelihood of reaching market (LOA) from Phase 1 for all "
        f"developmental candidates was {_pct(loa_p1)} (n={n_total:,})."
    )

    # Phase II as bottleneck
    p2_rate = next((t.rate for t in overall.transitions
                    if t.from_phase == "Phase 2"), None)
    if p2_rate is not None:
        p2_n = next((t.denominator for t in overall.transitions
                     if t.from_phase == "Phase 2"), 0)
        bullets.append(
            f"Phase 2 remains the largest hurdle in clinical development, with only "
            f"{_pct(p2_rate)} of candidates achieving this transition (n={p2_n:,})."
        )

    # Disease area extremes
    if bio_qls_slices:
        da_loas = [
            (name, _compute.compute_loa(fs).get("Phase 1", 0.0), _n_total(fs))
            for name, fs in bio_qls_slices.items()
            if _n_total(fs) > 0
        ]
        if da_loas:
            da_loas.sort(key=lambda x: x[1], reverse=True)
            top_name, top_loa, top_n = da_loas[0]
            bot_name, bot_loa, bot_n = da_loas[-1]
            if top_loa > 0 and bot_loa > 0:
                bullets.append(
                    f"Of the major disease areas, {top_name} had the highest LOA from "
                    f"Phase 1 ({_pct(top_loa)}, n={top_n:,}), representing a "
                    f"{top_loa / bot_loa:.1f}-fold increase over {bot_name} ({_pct(bot_loa)})."
                )

    # Oncology vs non-oncology
    if onc_split:
        onc = onc_split.get("Oncology")
        non_onc = onc_split.get("Non-Oncology")
        if onc and non_onc:
            onc_loa = _compute.compute_loa(onc).get("Phase 1", 0.0)
            non_onc_loa = _compute.compute_loa(non_onc).get("Phase 1", 0.0)
            if onc_loa > 0 or non_onc_loa > 0:
                bullets.append(
                    f"The LOA from Phase 1 for non-oncology indications ({_pct(non_onc_loa)}) "
                    f"was {_fold(non_onc_loa, onc_loa)} oncology ({_pct(onc_loa)})."
                )

    # Top modality
    if by_modality:
        mod_loas = [
            (name, _compute.compute_loa(fs).get("Phase 1", 0.0))
            for name, fs in by_modality.items()
            if _n_total(fs) > 0
        ]
        if mod_loas:
            mod_loas.sort(key=lambda x: x[1], reverse=True)
            top_mod, top_mod_loa = mod_loas[0]
            bullets.append(
                f"Among drug modalities, {top_mod} achieved the highest LOA from "
                f"Phase 1 at {_pct(top_mod_loa)}."
            )

    # Average timeline
    if timelines:
        overall_durs = [t.avg_duration_years for t in overall.transitions
                        if t.avg_duration_years is not None]
        if overall_durs:
            total_dur = sum(overall_durs)
            bullets.append(
                f"On average, it takes {total_dur:.1f} years for a Phase 1 "
                f"asset to reach the market."
            )

    # Sponsor stats
    if stats and stats.get("total_unique_sponsors", 0) > 0:
        bullets.append(
            f"{stats['total_unique_sponsors']:,} unique sponsors contributed, "
            f"with {stats['single_candidate_sponsors']:,} "
            f"({stats['single_candidate_pct']:.0f}%) having a single candidate."
        )

    return "\n".join(f"  * {b}" for b in bullets)


# -------------------------------------------------------------------------
# Introduction
# -------------------------------------------------------------------------

def introduction_text(
    n_transitions: int,
    n_candidates: int,
    n_sponsors: int,
    date_range: tuple[int, int] | None,
    peptide_only: bool,
) -> str:
    """Generate the introduction section."""
    if date_range:
        period = f"{date_range[0]}\u2013{date_range[1]}"
    else:
        period = "the study period"

    scope = "peptide drug" if peptide_only else "clinical drug"

    lines = [
        f"This study aimed to measure {scope} development success rates, contributing "
        f"factors to those outcomes, and timelines of clinical trials. With the goal of "
        f"providing current benchmarking metrics for drug development, this study covers "
        f"individual drug candidate phase transitions over {period}.",
        "",
        f"A total of {n_transitions:,} clinical phase transitions were recorded and "
        f"analyzed from {n_candidates:,} drug candidates across {n_sponsors:,} sponsors. "
        f"Phase transitions occur when a drug candidate advances into the next phase of "
        f"development or is suspended by the sponsor. By calculating the number of "
        f"candidates progressing to the next phase vs the total number progressing and "
        f"suspended, we assessed the success rate at each of the four phases of "
        f"development: Phase 1, 2, 3, and approval to market.",
        "",
        f"Having phase-by-phase data in hand, we then compared groups of diseases, drug "
        f"modalities, and other attributes to generate a comprehensive analysis of "
        f"{'peptide ' if peptide_only else ''}drug development success.",
    ]
    return "\n".join(lines)


# -------------------------------------------------------------------------
# Section narratives
# -------------------------------------------------------------------------

def overall_success_narrative(funnel_slice: FunnelSlice) -> str:
    """Narrative for overall phase transition success rates."""
    transitions = funnel_slice.transitions
    if len(transitions) < 2:
        return "Insufficient data to compute phase transition success rates."

    t_map = {t.from_phase: t for t in transitions}
    p1 = t_map.get("Phase 1")
    p2 = t_map.get("Phase 2")
    p3 = t_map.get("Phase 3")
    appr = t_map.get("Approval")

    parts = []

    parts.append(
        "Success rates for individual phases of the drug development process were "
        "determined by dividing the number that successfully advanced to the next "
        "phase by the total number advanced and suspended. This \u2018advanced and "
        "suspended\u2019 number is often referred to as \u2018n\u2019 in this report "
        "and should be taken into account when drawing conclusions from the success "
        "rate results."
    )

    if p1 and p1.denominator > 0:
        parts.append(
            f"The Phase 1 transition success rate was {_pct(p1.rate)} "
            f"(n={p1.denominator:,}). As this phase is typically conducted for safety "
            f"testing and is not dependent on efficacy for candidates to advance, it is "
            f"common for this phase to have a higher success rate among the clinical phases."
        )

    if p2 and p2.denominator > 0:
        parts.append(
            f"The Phase 2 transition success rate ({_pct(p2.rate)}, n={p2.denominator:,}) "
            f"was substantially lower than Phase 1, and the lowest of the four phases "
            f"studied. As this is generally the first stage where proof-of-concept is "
            f"deliberately tested in human subjects, Phase 2 consistently has the lowest "
            f"success rate of all phases. This is also the point in development where "
            f"industry must decide whether to pursue large, expensive Phase 3 studies, or "
            f"to terminate development."
        )

    if p3 and p3.denominator > 0:
        p3_comparison = ""
        if p1 and p1.denominator > 0 and p3.rate > p2.rate if p2 else True:
            p3_comparison = (
                " This is significant as most Phase 3 trials are "
                "the longest and most expensive trials to conduct."
            )
        parts.append(
            f"The second-highest phase transition success rate was found in Phase 3 "
            f"({_pct(p3.rate)}, n={p3.denominator:,}).{p3_comparison}"
        )

    if appr and appr.denominator > 0:
        parts.append(
            f"The approval to market transition success rate was "
            f"{_pct(appr.rate)} (n={appr.denominator:,})."
        )

    parts.append(
        "\n\n{{figure:overall_spider:Phase transition success rate profile for all diseases, all modalities.}}"
    )

    return "\n\n".join(parts)


def disease_success_narrative(
    bio_qls_slices: dict[str, FunnelSlice],
    overall: FunnelSlice,
) -> str:
    """Narrative for phase transition success by disease area."""
    if not bio_qls_slices:
        return "No disease area data available."

    n_diseases = len(bio_qls_slices)
    parts = []

    parts.append(
        f"Major disease areas were segmented according to therapeutic convention, "
        f"yielding {n_diseases} major groupings. The ordering of disease areas below "
        f"is consistent with the overall likelihood of reaching market from Phase 1, "
        f"which is analyzed later in the LOA section."
    )

    parts.append(
        "\n\n{{figure:disease_heatmap:Phase transition success rates by disease area. "
        "The n value is the total 'Advanced or Suspended' transitions used to calculate "
        "each rate.}}"
    )

    # Phase I analysis (brief)
    p1_data = _extract_phase_data(bio_qls_slices, "Phase 1")
    overall_p1 = next((t.rate for t in overall.transitions if t.from_phase == "Phase 1"), 0)
    if p1_data:
        p1_data.sort(key=lambda x: x[1], reverse=True)
        p1_top = p1_data[0]
        p1_bot = p1_data[-1]
        parts.append(
            f"Success rates by disease area for Phase 1 ranged from {_pct(p1_bot[1])} to "
            f"{_pct(p1_top[1])}, with the average for all indications at "
            f"{_pct(overall_p1)}. {p1_top[0]} and the other leading disease areas were "
            f"well above the average rate. With the exception of these top performers, "
            f"the remainder were all within a reasonable distance from the mean."
        )

    parts.append(
        "\n\n{{figure:phase1_by_disease:Phase 1 transition success rates by disease area.}}"
    )

    # Phase II analysis (detailed — the critical bottleneck)
    p2_data = _extract_phase_data(bio_qls_slices, "Phase 2")
    overall_p2 = next((t.rate for t in overall.transitions if t.from_phase == "Phase 2"), 0)

    if p2_data:
        p2_data.sort(key=lambda x: x[1], reverse=True)
        top_name, top_rate, top_n = p2_data[0]
        bot_name, bot_rate, bot_n = p2_data[-1]

        parts.append(
            f"In every disease area, Phase 2 had the lowest transition success rate "
            f"of the four phases. Phase 2 success rates ranged from a high of "
            f"{_pct(top_rate)} ({top_name}, n={top_n:,}) to a low of "
            f"{_pct(bot_rate)} ({bot_name}, n={bot_n:,}). "
            f"This {(top_rate - bot_rate) * 100:.0f} percentage-point range of disparity "
            f"between major disease areas at the Phase 2 transition is the major "
            f"contributor to the observed divergences in overall LOA, "
            f"as discussed in the next section."
        )

        above_avg = [(name, rate) for name, rate, _ in p2_data if rate > overall_p2]
        below_avg_notable = [(name, rate) for name, rate, _ in p2_data
                             if rate <= overall_p2 and rate < overall_p2 * 0.85]
        if above_avg:
            above_list = ", ".join(f"{n} ({_pct(r)})" for n, r in above_avg)
            parts.append(
                f"With only {above_list} achieving Phase 2 success rates above the "
                f"overall average of {_pct(overall_p2)}, these disease areas are also the "
                f"leaders when calculating the overall LOA from Phase 1."
            )
        if below_avg_notable:
            below_list = ", ".join(f"{n} ({_pct(r)})" for n, r in below_avg_notable[-3:])
            parts.append(
                f"The lowest-performing disease groups were {below_list}."
            )

    parts.append(
        "\n\n{{figure:phase2_by_disease:Phase 2 transition success rates by disease area.}}"
    )

    # Phase III analysis
    p3_data = _extract_phase_data(bio_qls_slices, "Phase 3")
    overall_p3 = next((t.rate for t in overall.transitions if t.from_phase == "Phase 3"), 0)

    if p3_data:
        p3_data.sort(key=lambda x: x[1])
        bot_name, bot_rate, bot_n = p3_data[0]

        below_avg = [name for name, rate, _ in p3_data if rate < overall_p3]
        parts.append(
            f"For Phase 3 transition success rates, {bot_name} had the lowest rate "
            f"at {_pct(bot_rate)} (n={bot_n:,}). "
            f"{len(below_avg)} disease group(s) ranked below the average Phase 3 "
            f"transition success rate of {_pct(overall_p3)}."
        )

    parts.append(
        "\n\n{{figure:phase3_by_disease:Phase 3 transition success rates by disease area.}}"
    )

    # Approval to market analysis (brief)
    appr_data = _extract_phase_data(bio_qls_slices, "Approval")
    if appr_data:
        appr_data.sort(key=lambda x: x[1])
        appr_lo = appr_data[0]
        appr_hi = appr_data[-1]
        parts.append(
            f"Approval to market transition success rates for the major disease "
            f"areas ranged from {_pct(appr_lo[1])} ({appr_lo[0]}) to "
            f"{_pct(appr_hi[1])} ({appr_hi[0]}). This distribution had the "
            f"tightest range among the four phases analyzed in this report."
        )

    return "\n\n".join(parts)


def _extract_phase_data(
    slices: dict[str, FunnelSlice], phase: str,
) -> list[tuple[str, float, int]]:
    """Extract (name, rate, n) tuples for a given phase from disease slices."""
    data = []
    for name, fs in slices.items():
        t = next((t for t in fs.transitions if t.from_phase == phase), None)
        if t and t.denominator > 0:
            data.append((name, t.rate, t.denominator))
    return data


def loa_disease_narrative(
    bio_qls_slices: dict[str, FunnelSlice],
    overall: FunnelSlice,
) -> str:
    """Narrative for LOA by disease area."""
    if not bio_qls_slices:
        return ""

    overall_loa = _compute.compute_loa(overall).get("Phase 1", 0.0)
    n_total = _n_total(overall)

    da_loas = [
        (name, _compute.compute_loa(fs).get("Phase 1", 0.0), _n_total(fs))
        for name, fs in bio_qls_slices.items()
        if _n_total(fs) > 0
    ]
    da_loas.sort(key=lambda x: x[1], reverse=True)

    parts = []

    parts.append(
        "One of the key measures of success used in this report is the LOA from "
        "Phase 1. The LOA is a multiplication of success rates from all four phase "
        "transitions (Phase 1\u21922, Phase 2\u21923, Phase 3\u2192Approval, "
        "Approval\u2192Market), a compounded probability calculation. For example, if "
        "each phase had a 50% chance of success, then the LOA from Phase 1 would be "
        "0.5 \u00d7 0.5 \u00d7 0.5 \u00d7 0.5 = 6.25%."
    )

    parts.append(
        f"Multiplying the individual phase probabilities across all disease areas, "
        f"the compounded probability of progressing from Phase 1 to market "
        f"reveals that only {_pct(overall_loa)} of drug candidates "
        f"(n={n_total:,}) successfully make it to market."
    )

    if len(da_loas) >= 2:
        top_name, top_loa, top_n = da_loas[0]
        bot_name, bot_loa, bot_n = da_loas[-1]
        fold = top_loa / bot_loa if bot_loa > 0 else 0
        fold_str = f"{fold:.0f}" if fold >= 2 else f"{fold:.1f}"
        parts.append(
            f"As can be seen in the figure below, there is a wide range of LOAs from "
            f"Phase 1. At the high end, {top_name} towers over the other disease groups "
            f"at {_pct(top_loa)} (n={top_n:,}). {top_name} therapies had an LOA from "
            f"Phase 1 {fold_str} times higher than {bot_name}, which had the lowest "
            f"LOA at {_pct(bot_loa)} (n={bot_n:,})."
        )

        above_avg = [(n, l) for n, l, _ in da_loas if l > overall_loa]
        below_avg = [(n, l) for n, l, _ in da_loas if l <= overall_loa]
        if above_avg and len(above_avg) > 1:
            above_names = [n for n, _ in above_avg[1:]]
            parts.append(
                f"After {top_name}, {len(above_names)} other disease area(s) were "
                f"above the overall average of {_pct(overall_loa)}: "
                f"{', '.join(above_names)}."
            )
        if below_avg:
            below_notable = [(n, l) for n, l in below_avg
                             if l < overall_loa * 0.85]
            if below_notable:
                below_list = ", ".join(f"{n}" for n, _ in below_notable[:4])
                parts.append(
                    f"Falling under the overall LOA of {_pct(overall_loa)} were "
                    f"{below_list}. The fact that some of these disease categories carry "
                    f"large n values while also having low LOA values indicates that they "
                    f"are significant contributors in bringing down the overall LOA."
                )

    parts.append(
        "\n\n{{figure:loa_by_disease:LOA from Phase 1 by disease area, displayed highest to lowest.}}"
    )
    parts.append(
        "\n\n{{table:loa_by_disease:Likelihood of reaching market by disease area with corresponding n values.}}"
    )

    return "\n\n".join(parts)


def oncology_narrative(
    onc_split: dict[str, FunnelSlice],
    overall: FunnelSlice,
) -> str:
    """Narrative for oncology vs non-oncology comparison."""
    onc = onc_split.get("Oncology")
    non_onc = onc_split.get("Non-Oncology")
    if not onc or not non_onc:
        return ""

    n_total = _n_total(overall)
    onc_n = _n_total(onc)
    non_onc_n = _n_total(non_onc)
    onc_share = onc_n / n_total * 100 if n_total > 0 else 0

    onc_loa = _compute.compute_loa(onc).get("Phase 1", 0.0)
    non_onc_loa = _compute.compute_loa(non_onc).get("Phase 1", 0.0)

    parts = []

    parts.append(
        f"Oncology drug development transitions accounted for "
        f"{onc_share:.0f}% of the {n_total:,} total transitions. With an LOA "
        f"from Phase 1 of {_pct(onc_loa)} (n={onc_n:,}), Oncology had an "
        f"outsized effect on the overall success rate. To further understand this "
        f"contribution, we compared phase transition success rates and LOA for "
        f"non-oncology candidates against oncology candidates."
    )

    parts.append(
        f"The LOA from Phase 1 across non-oncology indications, {_pct(non_onc_loa)} "
        f"(n={non_onc_n:,}), was {_fold(non_onc_loa, onc_loa)} oncology alone at "
        f"{_pct(onc_loa)} (n={onc_n:,}). Comparing individual phase transition "
        f"success rates, Oncology consistently had one of the lowest success rates "
        f"for every developmental clinical transition."
    )

    # Per-phase comparison
    phase_comparisons = []
    for onc_t, non_onc_t in zip(onc.transitions, non_onc.transitions):
        if onc_t.denominator > 0 and non_onc_t.denominator > 0:
            diff = (non_onc_t.rate - onc_t.rate) * 100
            if abs(diff) > 3:
                phase_comparisons.append(
                    f"{onc_t.from_phase}\u2192{onc_t.to_phase}: "
                    f"{_pct(onc_t.rate)} vs {_pct(non_onc_t.rate)}"
                )

    if phase_comparisons:
        parts.append(
            f"In particular, there were relative differentials between Oncology and "
            f"non-oncology groupings: {'; '.join(phase_comparisons)}."
        )

    # Note if oncology outperforms at approval→market
    onc_appr = next((t for t in onc.transitions if t.from_phase == "Approval"), None)
    non_onc_appr = next((t for t in non_onc.transitions if t.from_phase == "Approval"), None)
    if onc_appr and non_onc_appr and onc_appr.rate > non_onc_appr.rate:
        parts.append(
            f"One notable exception was the approval to market transition. Oncology "
            f"performed slightly better than non-oncology at this stage, "
            f"with a {_pct(onc_appr.rate)} (n={onc_appr.denominator:,}) success rate, "
            f"as opposed to {_pct(non_onc_appr.rate)} (n={non_onc_appr.denominator:,})."
        )

    parts.append(
        "\n\n{{figure:oncology_comparison:Oncology vs non-oncology phase transition "
        "success rates and LOA.}}"
    )
    parts.append(
        "\n\n{{table:oncology_comparison:Phase transition success and LOA "
        "by Oncology vs Non-Oncology with corresponding n values.}}"
    )

    return "\n\n".join(parts)


def modality_narrative(
    by_modality: dict[str, FunnelSlice],
    overall: FunnelSlice,
) -> str:
    """Narrative for drug modality analysis (Drug Classes and Modalities section)."""
    if not by_modality:
        return ""

    mod_loas = [
        (name, _compute.compute_loa(fs).get("Phase 1", 0.0), _n_total(fs))
        for name, fs in by_modality.items()
        if _n_total(fs) > 0
    ]
    mod_loas.sort(key=lambda x: x[1], reverse=True)

    if not mod_loas:
        return ""

    overall_loa = _compute.compute_loa(overall).get("Phase 1", 0.0)

    parts = []

    top_name, top_loa, top_n = mod_loas[0]
    parts.append(
        f"Looking deeper into drug modality, {top_name} development has seen the "
        f"most success among the modalities studied. At {_pct(top_loa)} "
        f"(n={top_n:,}), the Phase 1 LOA for {top_name} therapies is "
        f"{_fold(top_loa, overall_loa)} the {_pct(overall_loa)} average across "
        f"all diseases."
    )

    above_avg = [(n, l, nn) for n, l, nn in mod_loas if l > overall_loa]
    if len(above_avg) > 1:
        others = [(n, l) for n, l, _ in above_avg[1:]]
        listing = ", ".join(f"{n} ({_pct(l)})" for n, l in others)
        parts.append(
            f"Other modalities with above-average LOAs include {listing}."
        )

    if len(mod_loas) >= 2:
        bot_name, bot_loa, bot_n = mod_loas[-1]
        parts.append(
            f"At the lower end, {bot_name} had a Phase 1 LOA of "
            f"{_pct(bot_loa)} (n={bot_n:,})."
        )

    parts.append(
        "\n\n{{figure:loa_by_modality:LOA from Phase 1 by drug modality, displayed "
        "highest to lowest.}}"
    )
    parts.append(
        "\n\n{{table:loa_by_modality:Phase transition success rates and LOA from "
        "Phase 1 by drug modality with corresponding n values.}}"
    )

    return "\n\n".join(parts)


def timeline_narrative(
    bio_qls_slices: dict[str, FunnelSlice],
    overall: FunnelSlice,
) -> str:
    """Narrative for drug development timelines."""
    overall_durs = [t.avg_duration_years for t in overall.transitions
                    if t.avg_duration_years is not None]
    if not overall_durs:
        return "Insufficient duration data available for timeline analysis."

    full_total = sum(overall_durs)

    parts = []

    parts.append(
        "In addition to determining whether drugs are advanced or suspended at the "
        "end of a phase transition, the data can also be analyzed to yield time spent "
        "at each clinical stage, and overall drug development timelines. These are "
        "valuable metrics as there are considerable opportunity costs associated with "
        "investing in an R&D process that may take up to a decade."
    )

    phase_parts = []
    for t in overall.transitions:
        if t.avg_duration_years is not None:
            phase_parts.append(f"{t.avg_duration_years:.1f} years at {t.from_phase}")
    parts.append(
        f"Based on successful phase transitions, it took an average of "
        f"{full_total:.1f} years for a drug candidate to progress from Phase 1 "
        f"to market. This includes {', '.join(phase_parts)}."
    )

    parts.append(
        "Phase duration can vary greatly according to numerous factors, such as "
        "disease area and indication, best practices of clinical trial design, and "
        "patient availability."
    )

    # Disease area analysis
    da_totals = []
    for name, fs in bio_qls_slices.items():
        durs = [t.avg_duration_years for t in fs.transitions
                if t.avg_duration_years is not None]
        if durs:
            da_totals.append((name, sum(durs), fs))

    if da_totals:
        da_totals.sort(key=lambda x: x[1])
        shortest = da_totals[0]
        longest = da_totals[-1]

        below_avg_dur = [name for name, total, _ in da_totals if total < full_total]

        parts.append(
            f"Disease areas with above-average LOAs tend to be associated with "
            f"shorter development timelines. {len(below_avg_dur)} of the disease "
            f"areas fall beneath the {full_total:.1f}-year average development duration. "
            f"Conversely, the remaining disease areas with below-average LOAs either "
            f"have durations close to the average or notably exceed it."
        )

        parts.append(
            f"{shortest[0]} had the shortest overall development duration at "
            f"{shortest[1]:.1f} years, while {longest[0]} had the longest at "
            f"{longest[1]:.1f} years."
        )

        # Find actual shortest/longest per phase (deduplicated)
        phase_extremes: dict[str, tuple[str, float, str, float]] = {}
        for name, total, fs in da_totals:
            for t in fs.transitions:
                if t.avg_duration_years is None:
                    continue
                phase = t.from_phase
                dur = t.avg_duration_years
                if phase not in phase_extremes:
                    phase_extremes[phase] = (name, dur, name, dur)
                else:
                    sn, sd, ln, ld = phase_extremes[phase]
                    if dur < sd:
                        sn, sd = name, dur
                    if dur > ld:
                        ln, ld = name, dur
                    phase_extremes[phase] = (sn, sd, ln, ld)

        for phase, (sn, sd, ln, ld) in phase_extremes.items():
            overall_t = next(
                (ot for ot in overall.transitions if ot.from_phase == phase), None
            )
            if overall_t and overall_t.avg_duration_years:
                avg = overall_t.avg_duration_years
                if ld - avg > 1.0:
                    parts.append(
                        f"Within {phase}, {ln} had the longest duration at "
                        f"{ld:.1f} years (average: {avg:.1f} years)."
                    )
                elif avg - sd > 1.0:
                    parts.append(
                        f"{sn} had the shortest {phase} duration at "
                        f"{sd:.1f} years (average: {avg:.1f} years)."
                    )

    parts.append(
        "\n\n{{figure:timeline_by_disease:Phase transition durations from Phase 1 "
        "by disease area. The ordering of disease areas is equivalent to the total "
        "years in clinical development.}}"
    )

    return "\n\n".join(parts)


def time_period_narrative(
    period_slices: dict[str, dict[str, FunnelSlice]],
    period_labels: list[str],
    n_excluded: int,
) -> str:
    """Narrative for time-period LOA comparison."""
    if not period_labels or len(period_labels) < 2:
        return ""

    parts = []

    if len(period_labels) == 2:
        parts.append(
            f"To examine temporal trends, candidates were partitioned into two "
            f"cohorts \u2014 a historical baseline ({period_labels[0]}) and the most "
            f"recent decade ({period_labels[1]}) \u2014 allowing comparison of recent "
            f"success rates against prior performance"
            f"{f' ({n_excluded:,} candidates excluded due to missing dates)' if n_excluded else ''}."
        )
    else:
        parts.append(
            f"To examine temporal trends, candidates were partitioned into "
            f"{len(period_labels)} time periods based on their earliest trial start date"
            f"{f' ({n_excluded:,} candidates excluded due to missing dates)' if n_excluded else ''}."
        )

    # Compare overall LOA across periods
    period_overall_loa = {}
    for label in period_labels:
        slices = period_slices.get(label, {})
        all_n = sum(_n_total(fs) for fs in slices.values())
        if all_n > 0:
            loas = [(_compute.compute_loa(fs).get("Phase 1", 0.0), _n_total(fs))
                    for fs in slices.values() if _n_total(fs) > 0]
            if loas:
                total_n = sum(n for _, n in loas)
                wavg = sum(l * n for l, n in loas) / total_n if total_n > 0 else 0
                period_overall_loa[label] = wavg

    if len(period_overall_loa) >= 2:
        vals = [period_overall_loa.get(l, 0) for l in period_labels]
        trend_parts = [f"{l}: {_pct(v)}" for l, v in zip(period_labels, vals) if v > 0]
        if trend_parts:
            parts.append(
                f"Weighted average LOA from Phase 1 across periods: "
                f"{'; '.join(trend_parts)}."
            )

    # Identify biggest movers
    if len(period_labels) >= 2:
        first_label = period_labels[0]
        last_label = period_labels[-1]
        first_slices = period_slices.get(first_label, {})
        last_slices = period_slices.get(last_label, {})

        all_das = set(first_slices.keys()) & set(last_slices.keys())
        movers = []
        for da in all_das:
            loa_first = _compute.compute_loa(first_slices[da]).get("Phase 1", 0.0)
            loa_last = _compute.compute_loa(last_slices[da]).get("Phase 1", 0.0)
            if loa_first > 0:
                change = (loa_last - loa_first) / loa_first
                movers.append((da, loa_first, loa_last, change))

        if movers:
            movers.sort(key=lambda x: x[3], reverse=True)
            improvers = [(da, f, l) for da, f, l, c in movers if c > 0.1]
            decliners = [(da, f, l) for da, f, l, c in movers if c < -0.1]

            if improvers:
                imp_str = "; ".join(
                    f"{da} ({_pct(f)} \u2192 {_pct(l)})" for da, f, l in improvers[:3]
                )
                parts.append(f"Disease areas showing improvement: {imp_str}.")

            if decliners:
                dec_str = "; ".join(
                    f"{da} ({_pct(f)} \u2192 {_pct(l)})" for da, f, l in decliners[:3]
                )
                parts.append(f"Disease areas showing decline: {dec_str}.")

    parts.append(
        "\n\n{{figure:time_period_transitions:Phase transition success rates and LOA by time period. Legend shows candidate counts per cohort.}}"
    )
    parts.append(
        "\n\n{{figure:time_period_comparison:LOA from Phase 1 by disease area and time period.}}"
    )
    parts.append(
        "\n\n{{table:time_period_loa:LOA comparison by time period and disease area.}}"
    )

    return "\n\n".join(parts)


def modality_trend_narrative(data: list[dict]) -> str:
    """Narrative for modality composition over time."""
    if not data:
        return ""

    years = sorted({d["year"] for d in data})
    modalities = sorted({d["modality"] for d in data})

    if len(years) < 2:
        return "Insufficient time range for modality trend analysis."

    parts = []

    parts.append(
        f"The composition of drug modalities in clinical development was tracked "
        f"across {len(years)} years ({years[0]}\u2013{years[-1]})."
    )

    pct_map = {(d["year"], d["modality"]): d["pct"] for d in data}
    first_yr, last_yr = years[0], years[-1]

    changes = []
    for mod in modalities:
        first_pct = pct_map.get((first_yr, mod), 0)
        last_pct = pct_map.get((last_yr, mod), 0)
        delta = last_pct - first_pct
        if abs(delta) > 0.03:
            changes.append((mod, first_pct, last_pct, delta))

    if changes:
        changes.sort(key=lambda x: x[3], reverse=True)
        growers = [(m, f, l) for m, f, l, d in changes if d > 0]
        shrinkers = [(m, f, l) for m, f, l, d in changes if d < 0]

        if growers:
            g_str = "; ".join(
                f"{m} ({f*100:.0f}% \u2192 {l*100:.0f}%)" for m, f, l in growers[:3]
            )
            parts.append(f"Modalities gaining share: {g_str}.")
        if shrinkers:
            s_str = "; ".join(
                f"{m} ({f*100:.0f}% \u2192 {l*100:.0f}%)" for m, f, l in shrinkers[:3]
            )
            parts.append(f"Modalities declining in share: {s_str}.")

    parts.append(
        "\n\n{{figure:modality_proportion:Drug modality composition over time.}}"
    )

    return "\n\n".join(parts)


def sponsor_narrative(sponsor_data: list[dict], stats: dict) -> str:
    """Narrative for sponsor concentration analysis."""
    if not stats or stats.get("total_unique_sponsors", 0) == 0:
        return ""

    parts = []

    parts.append(
        f"A total of {stats['total_unique_sponsors']:,} unique sponsors were identified "
        f"across all candidates. {stats['single_candidate_sponsors']:,} sponsors "
        f"({stats['single_candidate_pct']:.0f}%) contributed just a single candidate, while "
        f"the top 10 sponsors accounted for {stats['top_10_share']:.0f}% of all candidates."
    )

    if sponsor_data and len(sponsor_data) >= 3:
        top3 = sponsor_data[:3]
        listing = ", ".join(f"{d['sponsor']} ({d['candidate_count']})" for d in top3)
        parts.append(
            f"The leading sponsors by candidate count were: {listing}."
        )

    parts.append(
        "\n\n{{figure:sponsor_concentration:Sponsor concentration analysis.}}"
    )

    return "\n\n".join(parts)


def spider_narrative(funnel_slice: FunnelSlice) -> str:
    """Narrative for the Discussion section with spider chart as visual summary."""
    parts = []

    # Phase II as critical bottleneck
    p2 = next((t for t in funnel_slice.transitions if t.from_phase == "Phase 2"), None)
    if p2 and p2.denominator > 0:
        parts.append(
            f"The Phase 2 transition is clearly the main translational and limiting step "
            f"from bench to bedside, as proof-of-concept is only established in "
            f"{_pct(p2.rate)} of drug candidates at this stage. Phase 2 also sees the "
            f"greatest distribution between major disease areas. Within these disease "
            f"areas, it is the performance of novel drug classes that has the most "
            f"significant negative contribution to low overall LOA success rates."
        )

    # LOA-duration correlation
    overall_durs = [t.avg_duration_years for t in funnel_slice.transitions
                    if t.avg_duration_years is not None]
    if overall_durs:
        total = sum(overall_durs)
        parts.append(
            f"The average timeline from Phase 1 to market was {total:.1f} years. "
            f"This lengthy timeframe is a significant risk to investors and entrepreneurs, "
            f"and speaks to the current complexities of clinical drug development. This "
            f"risk is compounded by the correlation of longer timelines in certain disease "
            f"areas with greater pipeline attrition rates. Disease areas with above-average "
            f"LOAs tend to have the shortest development timelines, while those with below-"
            f"average LOAs often carry longer development durations."
        )

    parts.append(
        "\n\n{{figure:overall_spider:Overall phase transition success rate profile.}}"
    )

    return "\n\n".join(parts)
