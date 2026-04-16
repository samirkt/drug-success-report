"""
Entry point for the peptide research pipeline.

Usage:
    python src/run_pipeline.py
    python src/run_pipeline.py --source aact --output ./reports
"""

import argparse
import logging
from pathlib import Path

from pipeline import Pipeline, PipelineConfig

_LOG_FORMAT = "%(asctime)s %(levelname)s %(message)s"
logging.basicConfig(level=logging.INFO, format=_LOG_FORMAT)
logging.getLogger().addHandler(
    logging.FileHandler("pipeline.log", mode="w")
)
logging.getLogger().handlers[-1].setFormatter(logging.Formatter(_LOG_FORMAT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the peptide research pipeline.")
    parser.add_argument("--source", choices=["api", "aact"], default="aact",
                        help="Data source: ClinicalTrials.gov REST API or AACT mirror")
    parser.add_argument("--keyword", default=None,
                        help="Keyword filter for trial ingestion")
    parser.add_argument("--mesh", default=None,
                        help="mesh filter for trial ingestion")
    parser.add_argument("--clustering", choices=["embeddings", "fuzzy", "hybrid"], default="hybrid",
                        help="Clustering strategy for candidate matching")
    parser.add_argument("--output", default="docs",
                        help="Directory to write report outputs (default: docs/)")
    parser.add_argument("--formats", nargs="+", default=["html"],
                        help="Report output formats, e.g. html excel pdf")
    parser.add_argument("--max-trials", type=int, default=5,
                        help="Maximum number of trials to fetch and process")
    parser.add_argument(
        "--cache-path",
        default="knowledge_cache.db",
        help="Path to the knowledge cache SQLite file. Pass empty string to disable.",
    )
    parser.add_argument(
        "--drugbank-csv",
        default=None,
        metavar="PATH",
        help="Path to drugbank_approvals.csv for drug normalization and deduplication.",
    )
    parser.add_argument(
        "--keep-unmatched-drugbank",
        action="store_true",
        default=False,
        help="Keep candidates with no DrugBank match (matched candidates are still deduplicated).",
    )
    parser.add_argument(
        "--all-modalities",
        action="store_true",
        default=False,
        help="Include modality breakdown charts and tables for all drug modalities (default: peptide-only)",
    )
    parser.add_argument(
        "--single-arm",
        action="store_true",
        default=False,
        help="Keep only single-intervention/single-condition trials; drops basket and umbrella trials.",
    )
    parser.add_argument(
        "--use-ct-cache",
        action="store_true",
        default=False,
        help="Reuse a local cache of AACT query results to skip re-hitting the clinical trials DB.",
    )
    parser.add_argument(
        "--ct-cache-path",
        default="aact_cache.pkl",
        help="Path to the AACT fetch cache file (only used when --use-ct-cache is set).",
    )
    parser.add_argument(
        "--year-range",
        default=None,
        metavar="START-END",
        help="Restrict the entire report to candidates whose earliest trial "
             "started in [START, END] (inclusive), e.g. --year-range 2000-2008. "
             "Candidates with no start date are dropped.",
    )
    parser.add_argument(
        "--benchmark",
        default=None,
        metavar="PATH",
        help="Path to labeled benchmark CSV. If provided, prints outcome validation metrics after the pipeline run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    ingestion_filters = {}
    if args.keyword:
        ingestion_filters["keyword"] = args.keyword
    if args.mesh:
        ingestion_filters["mesh_term"] = args.mesh
    cache_path = args.cache_path or None
    drugbank_csv_path = Path(args.drugbank_csv) if args.drugbank_csv else None

    year_range = None
    if args.year_range:
        try:
            start_s, end_s = args.year_range.split("-", 1)
            year_range = (int(start_s), int(end_s))
        except ValueError:
            raise SystemExit(
                f"Invalid --year-range '{args.year_range}'. Expected START-END, e.g. 2000-2008."
            )
        if year_range[0] > year_range[1]:
            raise SystemExit(
                f"Invalid --year-range '{args.year_range}': START must be <= END."
            )

    config = PipelineConfig(
        data_source=args.source,
        ingestion_filters=ingestion_filters,
        clustering_method=args.clustering,
        cache_path=cache_path,
        drugbank_csv_path=drugbank_csv_path,
        drop_unmatched_drugbank=not args.keep_unmatched_drugbank,
        report_output_path=args.output,
        report_formats=args.formats,
        max_trials=args.max_trials if args.max_trials > 0 else None,
        peptide_only_report=not args.all_modalities,
        filter_single_arm=args.single_arm,
        use_ct_cache=args.use_ct_cache,
        ct_cache_path=args.ct_cache_path,
        candidate_year_range=year_range,
    )

    pipeline = Pipeline(config)
    result = pipeline.run()

    print(f"\nPipeline complete.")
    print(f"  Trials ingested:    {len(result.trial_table)}")
    print(f"  Candidates found:   {len(result.candidate_table)}")
    if result.report:
        print(f"  Report tables:      {list(result.report.tables.keys())}")
        if result.report.output_path:
            print(f"  Report written to:  {result.report.output_path}")

    if args.benchmark and result.candidate_table and result.outcome_table:
        # TODO: re-enable once eval/ validation is fully set up in this repo
        print("Warning: --benchmark flag is not yet supported in this repo. Skipping validation.")


if __name__ == "__main__":
    main()
