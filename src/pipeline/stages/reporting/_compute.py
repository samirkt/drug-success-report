"""Shared pure computation functions for reporting components."""

from __future__ import annotations

import numpy as np

from ...models import (
    FunnelResults,
    FunnelSlice,
    TransitionRate,
)
from ..aggregation import transition_rate_from_records

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TRANSITION_COLS = [
    ("Phase 1", "Phase 2"),
    ("Phase 2", "Phase 3"),
    ("Phase 3", "Approval"),
    ("Approval", "Market"),
]
TRANSITION_LABELS = ["P1\u2192P2", "P2\u2192P3", "P3\u2192Appr", "Appr\u2192Mkt"]
LOW_N_THRESHOLD = 10

REPORTING_PHASE_ORDER: dict[str, int] = {
    "Phase 1": 0, "Phase 2": 1, "Phase 3": 2, "Approval": 3, "Market": 4,
}
REPORTING_TRANSITIONS = [
    ("Phase 1", "Phase 2"), ("Phase 2", "Phase 3"),
    ("Phase 3", "Approval"), ("Approval", "Market"),
]


# ---------------------------------------------------------------------------
# LOA
# ---------------------------------------------------------------------------

def compute_loa(funnel_slice: FunnelSlice) -> dict[str, float]:
    """Compute LOA from each phase as cumulative product of transition rates."""
    rates = [t.rate for t in funnel_slice.transitions]
    loa: dict[str, float] = {}
    labels = ["Phase 1", "Phase 2", "Phase 3", "Approval", "Market"]
    for i, label in enumerate(labels):
        product = 1.0
        for r in rates[i:]:
            product *= r
        loa[label] = product
    return loa


def loa_table(
    slices: dict[str, FunnelSlice],
    overall: FunnelSlice,
) -> list[dict]:
    """Build LOA summary table with n-values per phase."""
    rows: list[dict] = []
    for name, fs in sorted(slices.items(), key=lambda kv: compute_loa(kv[1]).get("Phase 1", 0), reverse=True):
        loa = compute_loa(fs)
        row: dict = {"stratum": name}
        for label in ["Phase 1", "Phase 2", "Phase 3", "Approval", "Market"]:
            short = label.lower().replace(" ", "")
            row[f"loa_{short}"] = loa[label]
        for t in fs.transitions:
            short = t.from_phase.lower().replace(" ", "")
            row[f"n_{short}"] = t.denominator
        rows.append(row)
    # "All indications" row
    loa = compute_loa(overall)
    row = {"stratum": "All indications"}
    for label in ["Phase 1", "Phase 2", "Phase 3", "Approval", "Market"]:
        short = label.lower().replace(" ", "")
        row[f"loa_{short}"] = loa[label]
    for t in overall.transitions:
        short = t.from_phase.lower().replace(" ", "")
        row[f"n_{short}"] = t.denominator
    rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# BIO/QLS rebucketing
# ---------------------------------------------------------------------------

def _merge_transitions(
    bucket_nums: dict,
    bucket_denoms: dict,
    bucket_dur_w: dict,
    bucket_dur_d: dict,
    bucket_key,
) -> list[TransitionRate]:
    """Build merged TransitionRate list from accumulated bucket data."""
    transitions = []
    for from_p, to_p in TRANSITION_COLS:
        n = bucket_nums[bucket_key][(from_p, to_p)]
        d = bucket_denoms[bucket_key][(from_p, to_p)]
        rate = n / d if d > 0 else 0.0
        dw = bucket_dur_w[bucket_key][(from_p, to_p)]
        dd = bucket_dur_d[bucket_key][(from_p, to_p)]
        avg_dur = dw / dd if dd > 0 else None
        transitions.append(TransitionRate(from_p, to_p, n, d, rate, avg_dur))
    return transitions


def bio_qls_by_disease_area(
    by_disease_area: dict[str, FunnelSlice],
) -> dict[str, FunnelSlice]:
    """Collapse granular disease areas into BIO/QLS 2011-2020 categories."""
    from pipeline.stages.classification import to_bio_qls_disease_area

    bucket_nums: dict[str, dict[tuple[str, str], int]] = {}
    bucket_denoms: dict[str, dict[tuple[str, str], int]] = {}
    bucket_dur_w: dict[str, dict[tuple[str, str], float]] = {}
    bucket_dur_d: dict[str, dict[tuple[str, str], int]] = {}
    bucket_counts: dict[str, int] = {}

    for da, fs in by_disease_area.items():
        bucket = to_bio_qls_disease_area(da)
        if bucket not in bucket_nums:
            bucket_nums[bucket] = {k: 0 for k in TRANSITION_COLS}
            bucket_denoms[bucket] = {k: 0 for k in TRANSITION_COLS}
            bucket_dur_w[bucket] = {k: 0.0 for k in TRANSITION_COLS}
            bucket_dur_d[bucket] = {k: 0 for k in TRANSITION_COLS}
            bucket_counts[bucket] = 0
        bucket_counts[bucket] += fs.candidate_count
        t_map = {(t.from_phase, t.to_phase): t for t in fs.transitions}
        for key in TRANSITION_COLS:
            t = t_map.get(key)
            if t:
                bucket_nums[bucket][key] += t.numerator
                bucket_denoms[bucket][key] += t.denominator
                if t.avg_duration_years is not None:
                    bucket_dur_w[bucket][key] += t.avg_duration_years * t.denominator
                    bucket_dur_d[bucket][key] += t.denominator

    result: dict[str, FunnelSlice] = {}
    for bucket in sorted(bucket_nums):
        transitions = _merge_transitions(bucket_nums, bucket_denoms, bucket_dur_w, bucket_dur_d, bucket)
        result[bucket] = FunnelSlice(
            modality=None, disease_area=bucket,
            candidate_count=bucket_counts[bucket], transitions=transitions,
        )
    return result


def bio_qls_by_modality_and_disease(
    by_modality_and_disease: dict[tuple[str, str], FunnelSlice],
) -> dict[tuple[str, str], FunnelSlice]:
    """Collapse (modality, disease_area) cross-stratification into BIO/QLS disease buckets."""
    from pipeline.stages.classification import to_bio_qls_disease_area

    bucket_nums: dict[tuple[str, str], dict[tuple[str, str], int]] = {}
    bucket_denoms: dict[tuple[str, str], dict[tuple[str, str], int]] = {}
    bucket_dur_w: dict[tuple[str, str], dict[tuple[str, str], float]] = {}
    bucket_dur_d: dict[tuple[str, str], dict[tuple[str, str], int]] = {}
    bucket_counts: dict[tuple[str, str], int] = {}

    for (mod, da), fs in by_modality_and_disease.items():
        bucket = to_bio_qls_disease_area(da)
        key = (mod, bucket)
        if key not in bucket_nums:
            bucket_nums[key] = {k: 0 for k in TRANSITION_COLS}
            bucket_denoms[key] = {k: 0 for k in TRANSITION_COLS}
            bucket_dur_w[key] = {k: 0.0 for k in TRANSITION_COLS}
            bucket_dur_d[key] = {k: 0 for k in TRANSITION_COLS}
            bucket_counts[key] = 0
        bucket_counts[key] += fs.candidate_count
        t_map = {(t.from_phase, t.to_phase): t for t in fs.transitions}
        for tk in TRANSITION_COLS:
            t = t_map.get(tk)
            if t:
                bucket_nums[key][tk] += t.numerator
                bucket_denoms[key][tk] += t.denominator
                if t.avg_duration_years is not None:
                    bucket_dur_w[key][tk] += t.avg_duration_years * t.denominator
                    bucket_dur_d[key][tk] += t.denominator

    result: dict[tuple[str, str], FunnelSlice] = {}
    for (mod, bucket) in sorted(bucket_nums):
        transitions = _merge_transitions(bucket_nums, bucket_denoms, bucket_dur_w, bucket_dur_d, (mod, bucket))
        result[(mod, bucket)] = FunnelSlice(
            modality=mod, disease_area=bucket,
            candidate_count=bucket_counts[(mod, bucket)], transitions=transitions,
        )
    return result


def oncology_vs_rest(
    by_disease_area: dict[str, FunnelSlice],
) -> dict[str, FunnelSlice]:
    """Partition disease areas into Oncology vs Non-Oncology."""
    from pipeline.stages.classification import to_bio_qls_disease_area

    bucket_nums: dict[str, dict[tuple[str, str], int]] = {}
    bucket_denoms: dict[str, dict[tuple[str, str], int]] = {}
    bucket_dur_w: dict[str, dict[tuple[str, str], float]] = {}
    bucket_dur_d: dict[str, dict[tuple[str, str], int]] = {}
    bucket_counts: dict[str, int] = {}

    for da, fs in by_disease_area.items():
        bucket = "Oncology" if to_bio_qls_disease_area(da) == "Oncology" else "Non-Oncology"
        if bucket not in bucket_nums:
            bucket_nums[bucket] = {k: 0 for k in TRANSITION_COLS}
            bucket_denoms[bucket] = {k: 0 for k in TRANSITION_COLS}
            bucket_dur_w[bucket] = {k: 0.0 for k in TRANSITION_COLS}
            bucket_dur_d[bucket] = {k: 0 for k in TRANSITION_COLS}
            bucket_counts[bucket] = 0
        bucket_counts[bucket] += fs.candidate_count
        t_map = {(t.from_phase, t.to_phase): t for t in fs.transitions}
        for key in TRANSITION_COLS:
            t = t_map.get(key)
            if t:
                bucket_nums[bucket][key] += t.numerator
                bucket_denoms[bucket][key] += t.denominator
                if t.avg_duration_years is not None:
                    bucket_dur_w[bucket][key] += t.avg_duration_years * t.denominator
                    bucket_dur_d[bucket][key] += t.denominator

    result: dict[str, FunnelSlice] = {}
    for bucket in ["Oncology", "Non-Oncology"]:
        if bucket not in bucket_nums:
            result[bucket] = FunnelSlice(
                modality=None, disease_area=bucket, candidate_count=0,
                transitions=[TransitionRate(f, t, 0, 0, 0.0) for f, t in TRANSITION_COLS],
            )
            continue
        transitions = _merge_transitions(bucket_nums, bucket_denoms, bucket_dur_w, bucket_dur_d, bucket)
        result[bucket] = FunnelSlice(
            modality=None, disease_area=bucket,
            candidate_count=bucket_counts[bucket], transitions=transitions,
        )
    return result


# ---------------------------------------------------------------------------
# Transition rate from records (for time-period analysis)
# ---------------------------------------------------------------------------

# Re-exported as the public name used by reporting components. See
# `pipeline/stages/aggregation.py` for the authoritative implementation —
# phase-success rates are computed in exactly one place.
compute_transition_rate = transition_rate_from_records


def build_funnel_slice(
    records: list[dict], disease_area: str | None = None,
) -> FunnelSlice:
    """Build a FunnelSlice from flat candidate records.

    Records must carry `phases_observed` / `phases_advanced` keys produced
    by `FunnelAggregationStage._join`. Rates are computed via the shared
    `transition_rate_from_records` function in the aggregation module.
    """
    if disease_area is not None:
        records = [r for r in records if r["disease_area"] == disease_area]
    transitions = [
        transition_rate_from_records(records, f, t)
        for f, t in REPORTING_TRANSITIONS
    ]
    return FunnelSlice(
        modality=None, disease_area=disease_area,
        candidate_count=len(records), transitions=transitions,
    )


# ---------------------------------------------------------------------------
# Heatmap grid extraction
# ---------------------------------------------------------------------------

def heatmap_grid(
    funnel_results: FunnelResults,
    disease_area: str | None = None,
) -> tuple[list[str], list[str], np.ndarray, np.ndarray, np.ndarray, list[list[tuple[float, float]]]]:
    """Extract grid data for modality heatmaps (rows=modalities, cols=transitions)."""
    from utils.stats import wilson_ci

    if disease_area is None:
        slices = funnel_results.by_modality
    else:
        slices = {
            mod: fs
            for (mod, da), fs in funnel_results.by_modality_and_disease.items()
            if da == disease_area
        }

    modalities = sorted(slices.keys())
    n_mod = len(modalities)
    n_trans = len(TRANSITION_COLS)

    rates = np.full((n_mod, n_trans), np.nan)
    denoms = np.zeros((n_mod, n_trans), dtype=int)
    nums = np.zeros((n_mod, n_trans), dtype=int)
    ci_bounds: list[list[tuple[float, float]]] = [
        [(0.0, 0.0)] * n_trans for _ in range(n_mod)
    ]

    for i, mod in enumerate(modalities):
        fs = slices[mod]
        t_map = {(t.from_phase, t.to_phase): t for t in fs.transitions}
        for j, key in enumerate(TRANSITION_COLS):
            t = t_map.get(key)
            if t and t.denominator > 0:
                rates[i, j] = t.rate
                denoms[i, j] = t.denominator
                nums[i, j] = t.numerator
                _, lo, hi = wilson_ci(t.numerator, t.denominator)
                ci_bounds[i][j] = (lo, hi)

    return modalities, list(TRANSITION_LABELS), rates, denoms, nums, ci_bounds


def disease_heatmap_grid(
    funnel_results: FunnelResults,
    modality: str | None = None,
) -> tuple[list[str], list[str], np.ndarray, np.ndarray, np.ndarray, list[list[tuple[float, float]]]]:
    """Extract grid data for disease-area heatmaps (rows=disease areas, cols=transitions)."""
    from utils.stats import wilson_ci

    if modality is None:
        slices = funnel_results.by_disease_area
    else:
        slices = {
            da: fs
            for (mod, da), fs in funnel_results.by_modality_and_disease.items()
            if mod == modality
        }

    disease_areas = sorted(slices.keys())
    n_da = len(disease_areas)
    n_trans = len(TRANSITION_COLS)

    rates = np.full((n_da, n_trans), np.nan)
    denoms = np.zeros((n_da, n_trans), dtype=int)
    nums = np.zeros((n_da, n_trans), dtype=int)
    ci_bounds: list[list[tuple[float, float]]] = [
        [(0.0, 0.0)] * n_trans for _ in range(n_da)
    ]

    for i, da in enumerate(disease_areas):
        fs = slices[da]
        t_map = {(t.from_phase, t.to_phase): t for t in fs.transitions}
        for j, key in enumerate(TRANSITION_COLS):
            t = t_map.get(key)
            if t and t.denominator > 0:
                rates[i, j] = t.rate
                denoms[i, j] = t.denominator
                nums[i, j] = t.numerator
                _, lo, hi = wilson_ci(t.numerator, t.denominator)
                ci_bounds[i][j] = (lo, hi)

    # Drop rows with no data
    has_data = denoms.sum(axis=1) > 0
    keep = [i for i, ok in enumerate(has_data) if ok]
    disease_areas = [disease_areas[i] for i in keep]
    rates = rates[keep]
    denoms = denoms[keep]
    nums = nums[keep]
    ci_bounds = [ci_bounds[i] for i in keep]

    return disease_areas, list(TRANSITION_LABELS), rates, denoms, nums, ci_bounds


# ---------------------------------------------------------------------------
# Peptide-only filtering
# ---------------------------------------------------------------------------

def peptide_disease_slices(
    funnel_results: FunnelResults,
    attribute_table,
) -> dict[str, FunnelSlice]:
    """Return per-disease slices counting only peptide candidates."""
    peptide_diseases = {
        attrs.disease_area
        for attrs in attribute_table.attributes.values()
        if attrs.drug_modality == "peptide" and attrs.disease_area
    }
    return {
        d: funnel_results.by_modality_and_disease[("peptide", d)]
        for d in peptide_diseases
        if ("peptide", d) in funnel_results.by_modality_and_disease
    }


# ---------------------------------------------------------------------------
# Sponsor stats
# ---------------------------------------------------------------------------

def sponsor_summary_stats(candidate_table) -> dict:
    """Aggregate sponsor concentration statistics."""
    from collections import Counter
    import statistics

    sponsor_counts: Counter = Counter()
    for c in candidate_table.candidates:
        for sponsor in c.sponsors:
            if sponsor:
                sponsor_counts[sponsor] += 1

    if not sponsor_counts:
        return {
            "total_unique_sponsors": 0,
            "single_candidate_sponsors": 0,
            "single_candidate_pct": 0.0,
            "top_10_share": 0.0,
            "median_candidates_per_sponsor": 0,
        }

    counts = list(sponsor_counts.values())
    total = len(sponsor_counts)
    single = sum(1 for c in counts if c == 1)
    total_candidates = len(candidate_table.candidates)
    top_10 = sum(c for _, c in sponsor_counts.most_common(10))

    return {
        "total_unique_sponsors": total,
        "single_candidate_sponsors": single,
        "single_candidate_pct": single / total * 100 if total > 0 else 0.0,
        "top_10_share": top_10 / total_candidates * 100 if total_candidates > 0 else 0.0,
        "median_candidates_per_sponsor": statistics.median(counts),
    }
