"""Diagnose ICD-10 coverage gaps in the candidate output.

Cross-references the candidate parquet against the on-disk NLM cache to
explain *why* each unmapped candidate is unmapped, and surfaces the
highest-impact unmapped lookup keys so they can be targeted with
preprocessing rules or a crosswalk.

Usage:
    python scripts/diagnose_icd10_coverage.py \\
        --candidates outputs/candidate_detail.parquet \\
        --cache outputs/cache/icd_lookup.sqlite \\
        --top 40
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from collections import Counter
from pathlib import Path

import pandas as pd


def _norm(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return str(value).strip()


def _has_codes(codes) -> bool:
    if codes is None:
        return False
    if isinstance(codes, float) and pd.isna(codes):
        return False
    try:
        return len(codes) > 0
    except TypeError:
        return False


def _load_cache(cache_path: Path) -> dict[str, list[str] | None]:
    conn = sqlite3.connect(str(cache_path))
    try:
        rows = conn.execute("SELECT disease_name, codes_json FROM icd_cache").fetchall()
    finally:
        conn.close()
    cache: dict[str, list[str] | None] = {}
    for name, codes_json in rows:
        cache[name] = json.loads(codes_json) if codes_json is not None else None
    return cache


def _classify(row, cache: dict[str, list[str] | None]) -> str:
    mesh = _norm(row.get("mesh_indication"))
    ind = _norm(row.get("indication"))

    if not mesh and not ind:
        return "no_lookup_key"

    mesh_state = "absent"
    if mesh:
        if mesh in cache:
            mesh_state = "hit" if cache[mesh] else "null"
        else:
            mesh_state = "unqueried"

    ind_state = "absent"
    if ind:
        if ind in cache:
            ind_state = "hit" if cache[ind] else "null"
        else:
            ind_state = "unqueried"

    if mesh_state == "hit" or ind_state == "hit":
        return "covered"  # shouldn't happen for the uncovered subset
    if mesh_state == "unqueried" or ind_state == "unqueried":
        return "unqueried_key"
    if mesh_state == "null" and ind_state == "null":
        return "both_keys_null"
    if mesh_state == "null":
        return "mesh_null_no_indication"
    if ind_state == "null":
        return "indication_null_no_mesh"
    return "other"


# Heuristic substrings worth flagging in unmapped keys. Each match suggests
# a different fix lane: abbreviations need expansion, modifiers need
# stripping, multi-condition phrases need splitting.
_PATTERNS: dict[str, re.Pattern] = {
    "modifier_advanced_metastatic": re.compile(r"\b(advanced|metastatic|recurrent|refractory|relapsed|unresectable|progressive)\b", re.I),
    "modifier_stage_or_grade": re.compile(r"\b(stage\s*[ivx0-9]+|grade\s*[ivx0-9]+|high[- ]?grade|low[- ]?grade)\b", re.I),
    "parenthetical": re.compile(r"[()\[\]]"),
    "multi_condition_separator": re.compile(r"\s+(and|or|with|due to|secondary to|associated with|/|;)\s+", re.I),
    "abbreviation_caps": re.compile(r"\b[A-Z]{3,}\b"),
    "her2_positivity": re.compile(r"\b(her2|er|pr|pd[- ]?l1|kras|egfr)[- ]?(positive|negative|mutant|mutated|wild[- ]?type|\+|\-)\b", re.I),
    "pediatric_adult_marker": re.compile(r"\b(pediatric|paediatric|adult|elderly|geriatric|neonatal|infant)\b", re.I),
    "trailing_qualifier_in": re.compile(r"\sin\s", re.I),
}


def _pattern_hits(text: str) -> list[str]:
    return [name for name, rx in _PATTERNS.items() if rx.search(text)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True, type=Path)
    ap.add_argument("--cache", required=True, type=Path)
    ap.add_argument("--top", type=int, default=40, help="Show top-N unmapped keys")
    ap.add_argument("--out", type=Path, default=None,
                    help="Optional CSV of unmapped keys + candidate counts")
    args = ap.parse_args()

    df = pd.read_parquet(args.candidates)
    cache = _load_cache(args.cache)

    total = len(df)
    df["__covered"] = df["icd10_codes"].map(_has_codes)
    covered = int(df["__covered"].sum())
    uncovered = total - covered

    print(f"Candidates total:        {total:>7}")
    print(f"  with ICD-10 codes:     {covered:>7}  ({100*covered/total:5.1f}%)")
    print(f"  without ICD-10 codes:  {uncovered:>7}  ({100*uncovered/total:5.1f}%)")
    print()
    print(f"Cache entries (NLM lookups recorded): {len(cache)}")
    n_cache_null = sum(1 for v in cache.values() if not v)
    print(f"  positive (≥1 code):  {len(cache) - n_cache_null}")
    print(f"  null (NLM no match): {n_cache_null}")
    print()

    uncov = df[~df["__covered"]].copy()
    uncov["__reason"] = uncov.apply(lambda r: _classify(r, cache), axis=1)
    reason_counts = uncov["__reason"].value_counts()

    print("Why uncovered candidates have no codes:")
    print("-" * 60)
    for reason, count in reason_counts.items():
        pct_of_uncov = 100 * count / uncovered if uncovered else 0
        print(f"  {reason:<32}  {count:>7}  ({pct_of_uncov:5.1f}% of uncovered)")
    print()

    # Rank unmapped lookup keys by candidate-count impact. A key here is
    # whichever string the pipeline *would* have used for the candidate;
    # for diagnostic value we list mesh and indication keys side-by-side.
    mesh_misses = Counter()
    ind_misses = Counter()
    for _, row in uncov.iterrows():
        mesh = _norm(row.get("mesh_indication"))
        ind = _norm(row.get("indication"))
        if mesh and cache.get(mesh) in (None, []):
            # cache.get returns None either for "not in cache" or "in cache
            # as null" — both are treated as a miss here.
            if mesh in cache:
                mesh_misses[mesh] += 1
        if ind and cache.get(ind) in (None, []):
            if ind in cache:
                ind_misses[ind] += 1

    print(f"Top {args.top} unmapped MeSH terms (cached-null) by candidate count:")
    print("-" * 60)
    for key, n in mesh_misses.most_common(args.top):
        hits = ",".join(_pattern_hits(key)) or "—"
        print(f"  {n:>5}  [{hits}]  {key}")
    print()

    print(f"Top {args.top} unmapped raw indications (cached-null) by candidate count:")
    print("-" * 60)
    for key, n in ind_misses.most_common(args.top):
        hits = ",".join(_pattern_hits(key)) or "—"
        print(f"  {n:>5}  [{hits}]  {key}")
    print()

    print("Pattern distribution across unmapped keys (any source):")
    print("-" * 60)
    pat_count = Counter()
    pat_candidates = Counter()
    for key, n in (mesh_misses + ind_misses).items():
        for p in _pattern_hits(key):
            pat_count[p] += 1
            pat_candidates[p] += n
    for p, n_keys in pat_count.most_common():
        print(f"  {p:<34}  {n_keys:>5} keys   {pat_candidates[p]:>6} candidates")

    if args.out is not None:
        rows = []
        for key, n in mesh_misses.most_common():
            rows.append({"source": "mesh", "key": key, "n_candidates": n,
                         "patterns": ",".join(_pattern_hits(key))})
        for key, n in ind_misses.most_common():
            rows.append({"source": "indication", "key": key, "n_candidates": n,
                         "patterns": ",".join(_pattern_hits(key))})
        pd.DataFrame(rows).to_csv(args.out, index=False)
        print(f"\nWrote unmapped-key report: {args.out}")


if __name__ == "__main__":
    main()
