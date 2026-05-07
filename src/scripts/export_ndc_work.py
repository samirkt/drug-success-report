"""Export NDC-adjudication work units for offload to a Colab GPU.

Runs the pipeline through clustering, enrichments, year-range filtering,
and (optional) candidate sampling, then mirrors
``AdjudicationStage._prepass`` to bulk-resolve each unique drug against
the local ``fda.db`` mirror. Each candidate that yields a non-empty list
of FDA label indications is written as one line to
``work_units.jsonl``; candidates with no label hits are skipped (the
local pipeline already short-circuits those without an LLM call).

Output:
  --output       work_units.jsonl    -- one JSON object per work unit
  --candidates-pickle clustered.pkl   -- pickled CandidateTable for the importer

Usage:
    python -m scripts.export_ndc_work \\
        --source aact --keyword peptide \\
        --max-candidates 100 --sample-seed 42 \\
        --year-range 2000-2006 \\
        --output /tmp/work_units.jsonl \\
        --candidates-pickle /tmp/clustered.pkl
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
from pathlib import Path

from pipeline import Pipeline, PipelineConfig
from pipeline.ndc import (
    dedup_indications,
    resolve_drug_key,
    split_long_indications,
    synonyms_for_candidate,
    top_relevant_indications,
)

logger = logging.getLogger(__name__)

_LOG_FORMAT = "%(asctime)s %(levelname)s %(message)s"


def _parse_year_range(raw: str | None) -> tuple[int, int] | None:
    if not raw:
        return None
    try:
        start_s, end_s = raw.split("-", 1)
        return (int(start_s), int(end_s))
    except ValueError as e:
        raise SystemExit(f"Invalid --year-range '{raw}'. Expected START-END.") from e


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", choices=["api", "aact"], default="aact")
    p.add_argument("--keyword", default=None)
    p.add_argument("--mesh", default=None)
    p.add_argument("--max-trials", type=int, default=0,
                   help="Cap raw AACT rows. 0 = unlimited (default).")
    p.add_argument("--max-candidates", type=int, default=None,
                   help="Random-sample N candidates after clustering.")
    p.add_argument("--sample-seed", type=int, default=42)
    p.add_argument("--year-range", default="2000-2006",
                   help="Restrict to candidates whose earliest trial start year "
                        "is in [START, END]. Default: 2000-2006.")
    p.add_argument("--cache-path", default="knowledge_cache.db",
                   help="Knowledge cache (used to skip already-adjudicated "
                        "candidates). Pass empty string to disable.")
    p.add_argument("--drugbank-csv", default=None,
                   help="Path to drugbank_approvals.csv (clustering).")
    p.add_argument("--drugbank-synonyms-csv", default="data/drugbank_synonyms.csv",
                   help="Path to drugbank_synonyms.csv used by the pre-pass to "
                        "build the synonym list per candidate.")
    p.add_argument("--use-ct-cache", action="store_true", default=False,
                   help="Reuse aact_cache.pkl when --source aact. Match "
                        "--max-trials and --keyword/--mesh exactly to the run "
                        "that populated the cache (e.g. via run_pipeline.py / "
                        "execute.sh) for cache hits.")
    p.add_argument("--ct-cache-path", default="aact_cache.pkl",
                   help="Path to AACT pickle cache. Default matches "
                        "run_pipeline.py.")
    p.add_argument("--chembl-snapshot", type=Path, default=None,
                   help="Path to chembl_targets.sqlite. Speeds up the targets "
                        "/ pathway enrichments.")
    p.add_argument("--opentargets-snapshot", type=Path, default=None,
                   help="Path to opentargets_snapshot.sqlite. Speeds up the "
                        "OpenTargets enrichment.")
    p.add_argument("--top-k", type=int, default=20,
                   help="Top-K label indications to send (after dedup + split). "
                        "Default 20 matches the pipeline.")
    p.add_argument("--skip-cached", action="store_true", default=False,
                   help="Skip candidates already in the ndc_adjudication_cache "
                        "of --cache-path. Useful for incremental fill-in.")
    p.add_argument("--output", required=True,
                   help="Output JSONL path for work_units.")
    p.add_argument("--candidates-pickle", required=True,
                   help="Output pickle path for the CandidateTable; the "
                        "importer reads this to reconstruct cache keys.")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format=_LOG_FORMAT)
    args = parse_args()

    ingestion_filters: dict[str, str] = {}
    if args.keyword:
        ingestion_filters["keyword"] = args.keyword
    if args.mesh:
        ingestion_filters["mesh_term"] = args.mesh

    cache_path = args.cache_path or None
    drugbank_csv_path = Path(args.drugbank_csv) if args.drugbank_csv else None
    year_range = _parse_year_range(args.year_range)

    config = PipelineConfig(
        data_source=args.source,
        ingestion_filters=ingestion_filters,
        cache_path=cache_path,
        drugbank_csv_path=drugbank_csv_path,
        drugbank_synonyms_csv=Path(args.drugbank_synonyms_csv),
        max_trials=args.max_trials if args.max_trials > 0 else None,
        max_candidates=args.max_candidates if (args.max_candidates or 0) > 0 else None,
        sample_seed=args.sample_seed,
        candidate_year_range=year_range,
        use_ct_cache=args.use_ct_cache,
        ct_cache_path=args.ct_cache_path,
        chembl_snapshot_path=args.chembl_snapshot,
        opentargets_snapshot_path=args.opentargets_snapshot,
        # adjudication won't run -- we stop before stage 3 -- but the
        # config field is still required to be valid.
        adjudication_method="ndc_indication",
    )

    pipeline = Pipeline(config)

    logger.info("Stage 1 -- Trial Ingestion")
    trial_table = pipeline._run_ingestion()
    logger.info("Stage 2 -- Candidate Clustering")
    candidate_table = pipeline._run_clustering(trial_table)
    candidate_table = pipeline._run_enrichments(candidate_table)
    candidate_table = pipeline._filter_by_year_range(candidate_table)
    candidate_table, trial_table = pipeline._sample_candidates(
        candidate_table, trial_table
    )

    logger.info("Final candidate count: %d", len(candidate_table.candidates))

    # Optional: skip candidates already adjudicated in the NDC cache.
    skip_keys: set[str] = set()
    if args.skip_cached and cache_path:
        from pipeline.knowledge_cache import KnowledgeCache
        cache = KnowledgeCache(cache_path)
        for c in candidate_table.candidates:
            key = KnowledgeCache.make_adjudication_key(
                c.drug_name, c.indication, c.highest_phase.value
            )
            if cache.get_ndc_outcome(key, c.candidate_id) is not None:
                skip_keys.add(c.candidate_id)
        cache.close()
        logger.info("--skip-cached: %d / %d already cached, will be skipped",
                    len(skip_keys), len(candidate_table.candidates))

    # Mirror AdjudicationStage._prepass: build drug_key -> {matched_synonym, label_indications}
    from utils import ndc_lookup
    from pipeline.drugbank_norm import load_drugbank_synonyms

    forward, _reverse = load_drugbank_synonyms(Path(args.drugbank_synonyms_csv))

    drug_synonyms: dict[str, list[str]] = {}
    for cand in candidate_table.candidates:
        key = resolve_drug_key(cand)
        if key in drug_synonyms:
            continue
        drug_synonyms[key] = synonyms_for_candidate(cand, forward)

    logger.info("Bulk fda.db lookup for %d unique drug(s)...", len(drug_synonyms))
    bulk = ndc_lookup.get_drugs_with_indications_bulk(list(drug_synonyms.values()))

    drug_lookup: dict[str, dict] = {}
    for key, synonyms in drug_synonyms.items():
        matched_synonym = None
        label_indications: list[str] = []
        for syn in synonyms:
            record = bulk.get(syn)
            if record and record.get("indication_count", 0) > 0:
                matched_synonym = syn
                label_indications = list(record.get("indications", []))
                break
        drug_lookup[key] = {
            "matched_synonym": matched_synonym,
            "label_indications": label_indications,
        }

    n_hits = sum(1 for v in drug_lookup.values() if v["label_indications"])
    logger.info("Pre-pass: %d / %d unique drugs had FDA label hits",
                n_hits, len(drug_lookup))

    # Write work units.
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_written = 0
    n_no_label = 0
    n_skipped_cached = 0
    with out_path.open("w") as f:
        for cand in candidate_table.candidates:
            if cand.candidate_id in skip_keys:
                n_skipped_cached += 1
                continue
            lookup = drug_lookup.get(resolve_drug_key(cand)) or {
                "matched_synonym": None, "label_indications": [],
            }
            if not lookup["label_indications"]:
                n_no_label += 1
                continue

            split = split_long_indications(lookup["label_indications"])
            deduped = dedup_indications(split)
            relevant = top_relevant_indications(
                cand.indication, cand.mesh_indication, deduped, k=args.top_k,
            )

            work_unit = {
                "candidate_id": cand.candidate_id,
                "drug_name": cand.drug_name,
                "indication": cand.indication,
                "highest_phase": cand.highest_phase.value,
                "matched_synonym": lookup["matched_synonym"],
                "trial_indication": cand.indication,
                "mesh_indication": cand.mesh_indication,
                "label_indications": relevant,
            }
            f.write(json.dumps(work_unit) + "\n")
            n_written += 1

    # Pickle the CandidateTable so the importer can rebuild cache keys.
    pkl_path = Path(args.candidates_pickle)
    pkl_path.parent.mkdir(parents=True, exist_ok=True)
    with pkl_path.open("wb") as f:
        pickle.dump(candidate_table, f)

    logger.info(
        "Export complete: %d work units written; %d skipped (no FDA label); "
        "%d skipped (already cached). CandidateTable -> %s",
        n_written, n_no_label, n_skipped_cached, pkl_path,
    )
    print(f"\nExport complete:")
    print(f"  Work units written:       {n_written} -> {out_path}")
    print(f"  Skipped (no FDA label):   {n_no_label}")
    print(f"  Skipped (already cached): {n_skipped_cached}")
    print(f"  CandidateTable pickle:    {pkl_path}")


if __name__ == "__main__":
    main()
