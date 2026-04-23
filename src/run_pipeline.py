"""
Entry point for the peptide research pipeline.

Usage:
    python src/run_pipeline.py
    python src/run_pipeline.py --source aact --output ./reports
"""

import argparse
import logging
import os
from pathlib import Path

from pipeline import Pipeline, PipelineConfig

_LOG_FORMAT = "%(asctime)s %(levelname)s %(message)s"
logging.basicConfig(level=logging.INFO, format=_LOG_FORMAT)
logging.getLogger().addHandler(
    logging.FileHandler("pipeline.log", mode="w")
)
logging.getLogger().handlers[-1].setFormatter(logging.Formatter(_LOG_FORMAT))


def _resolve_perf_overrides(args: argparse.Namespace) -> dict:
    """CLI flag → env var → omit (let PipelineConfig defaults apply).

    Three knobs share this resolution shape:
      --fda-adjudication-workers / FDA_ADJUDICATION_WORKERS
      --fda-llm-timeout          / FDA_LLM_TIMEOUT
      --fda-http-timeout         / FDA_HTTP_TIMEOUT
    """
    out: dict = {}
    workers = args.fda_adjudication_workers
    if workers is None:
        env_workers = os.getenv("FDA_ADJUDICATION_WORKERS")
        if env_workers:
            workers = int(env_workers)
    if workers is not None:
        out["fda_adjudication_workers"] = workers

    llm_timeout = args.fda_llm_timeout
    if llm_timeout is None:
        env_llm_timeout = os.getenv("FDA_LLM_TIMEOUT")
        if env_llm_timeout:
            llm_timeout = float(env_llm_timeout)
    if llm_timeout is not None:
        out["fda_llm_timeout"] = llm_timeout

    http_timeout = args.fda_http_timeout
    if http_timeout is None:
        env_http_timeout = os.getenv("FDA_HTTP_TIMEOUT")
        if env_http_timeout:
            http_timeout = float(env_http_timeout)
    if http_timeout is not None:
        out["fda_http_timeout"] = http_timeout

    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the peptide research pipeline.")
    parser.add_argument("--source", choices=["api", "aact"], default="aact",
                        help="Data source: ClinicalTrials.gov REST API or AACT mirror")
    parser.add_argument("--keyword", default=None,
                        help="Keyword filter for trial ingestion")
    parser.add_argument("--mesh", default=None,
                        help="mesh filter for trial ingestion")
    parser.add_argument("--output", default="docs",
                        help="Directory to write report outputs (default: docs/)")
    parser.add_argument("--formats", nargs="+", default=["html"],
                        help="Report output formats, e.g. html excel pdf")
    parser.add_argument("--max-trials", type=int, default=5,
                        help="Maximum number of trials to fetch and process")
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=None,
        help="Random-sample this many candidates after clustering (and after "
             "--year-range filtering), preserving each candidate's full trial "
             "set. Unset/<=0 disables sampling. Composes with --max-trials, "
             "which still caps the raw ingestion fetch.",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=42,
        help="RNG seed for --max-candidates. Default 42 for reproducibility.",
    )
    parser.add_argument(
        "--cache-path",
        default="knowledge_cache.db",
        help="Path to the knowledge cache SQLite file. Pass empty string to disable.",
    )
    parser.add_argument(
        "--drop-uncached-candidates",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Dev hack: drop candidates whose adjudication key isn't already "
             "in the knowledge cache (right after clustering/year-range "
             "filtering). Lets you iterate on clustering without burning LLM "
             "credits; outputs only cover previously-adjudicated drugs. "
             "Pass --no-drop-uncached-candidates to force-disable. "
             "When unset, the PipelineConfig default applies.",
    )
    parser.add_argument(
        "--drugbank-csv",
        default=None,
        metavar="PATH",
        help="Path to drugbank_approvals.csv for drug normalization and deduplication.",
    )
    parser.add_argument(
        "--require-drugbank-match",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Keep only candidates that resolve to a DrugBank ID during clustering. "
             "Pass --no-require-drugbank-match to keep unmatched candidates. "
             "When unset, the PipelineConfig default applies.",
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
        default="2000-2006",
        metavar="START-END",
        help="Restrict the entire report to candidates whose earliest trial "
             "started in [START, END] (inclusive), e.g. --year-range 2000-2008. "
             "Candidates with no start date are dropped. Default: 2000-2006.",
    )
    parser.add_argument(
        "--benchmark",
        default=None,
        metavar="PATH",
        help="Path to labeled benchmark CSV. If provided, prints outcome validation metrics after the pipeline run.",
    )
    parser.add_argument(
        "--adjudication-method",
        choices=["fda_timeline", "llm_direct"],
        default="llm_direct",
        help="Outcome adjudication method. 'fda_timeline' reconstructs each drug's "
             "FDA approval timeline from openFDA and matches indications against it "
             "(default). 'llm_direct' asks the LLM directly and uses the KnowledgeCache.",
    )
    parser.add_argument(
        "--fda-cache-dir",
        default=".fda_cache",
        help="Directory for on-disk openFDA response cache (only used when "
             "--adjudication-method=fda_timeline).",
    )
    parser.add_argument(
        "--openfda-api-key",
        default=None,
        help="openFDA API key. Falls back to OPENFDA_API_KEY env var. "
             "Optional: unauthenticated clients get a lower rate limit.",
    )
    parser.add_argument(
        "--fda-llm-backend",
        choices=["openai_compat", "anthropic"],
        default="openai_compat",
        help="FDA-timeline adjudication LLM backend. Default 'openai_compat' "
             "routes to local Ollama at http://localhost:11434/v1 serving "
             "qwen2.5:32b-instruct (tuned for 36GB M3 Max; install with "
             "`ollama pull qwen2.5:32b-instruct`). Pass 'anthropic' to opt "
             "in to Sonnet for this stage. Only affects FDA-timeline "
             "adjudication — direct-LLM adjudication and classification "
             "always use Anthropic.",
    )
    parser.add_argument(
        "--fda-llm-base-url",
        default=None,
        help="Base URL for the OpenAI-compatible endpoint used by FDA "
             "adjudication (e.g. http://localhost:11434/v1 for Ollama). "
             "Falls back to FDA_LLM_BASE_URL env var, then to the Ollama "
             "default. Ignored when --fda-llm-backend=anthropic.",
    )
    parser.add_argument(
        "--fda-llm-model",
        default=None,
        help="Model identifier for the FDA LLM backend. Falls back to "
             "FDA_LLM_MODEL env var, then to 'qwen2.5:32b-instruct' for "
             "openai_compat. For anthropic, overrides the default Sonnet.",
    )
    parser.add_argument(
        "--fda-llm-api-key",
        default=None,
        help="Optional API key for the OpenAI-compatible endpoint. Falls "
             "back to FDA_LLM_API_KEY env var. Not needed for local Ollama.",
    )
    parser.add_argument(
        "--fda-adjudication-workers",
        type=int,
        default=4,
        help="Candidate-level parallelism for FDA-timeline adjudication. "
             "Default 1 (sequential). Set <= your Ollama OLLAMA_NUM_PARALLEL "
             "setting; raising past that just queues at the server. On a 36GB "
             "M3 Max with the default qwen2.5:14b-instruct model, "
             "OLLAMA_NUM_PARALLEL=4 + --fda-adjudication-workers 4 is the "
             "sweet spot. Falls back to FDA_ADJUDICATION_WORKERS env var.",
    )
    parser.add_argument(
        "--fda-llm-timeout",
        type=float,
        default=None,
        help="Per-call LLM timeout in seconds for FDA-timeline adjudication "
             "(default 180). On timeout the candidate is recorded as UNKNOWN "
             "and the run continues. Falls back to FDA_LLM_TIMEOUT env var.",
    )
    parser.add_argument(
        "--fda-http-timeout",
        type=float,
        default=None,
        help="Per-call HTTP timeout in seconds for openFDA / DailyMed fetches "
             "(default 30). Falls back to FDA_HTTP_TIMEOUT env var.",
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
        cache_path=cache_path,
        drugbank_csv_path=drugbank_csv_path,
        **({"require_drugbank_match": args.require_drugbank_match}
           if args.require_drugbank_match is not None else {}),
        report_output_path=args.output,
        report_formats=args.formats,
        max_trials=args.max_trials if args.max_trials > 0 else None,
        max_candidates=args.max_candidates if (args.max_candidates or 0) > 0 else None,
        sample_seed=args.sample_seed,
        peptide_only_report=not args.all_modalities,
        filter_single_arm=args.single_arm,
        use_ct_cache=args.use_ct_cache,
        ct_cache_path=args.ct_cache_path,
        candidate_year_range=year_range,
        adjudication_method=args.adjudication_method,
        fda_cache_dir=Path(args.fda_cache_dir),
        openfda_api_key=args.openfda_api_key,
        fda_llm_backend=args.fda_llm_backend,
        fda_llm_base_url=args.fda_llm_base_url or os.getenv("FDA_LLM_BASE_URL"),
        fda_llm_model=args.fda_llm_model or os.getenv("FDA_LLM_MODEL"),
        fda_llm_api_key=args.fda_llm_api_key or os.getenv("FDA_LLM_API_KEY"),
        **({"drop_uncached_candidates": args.drop_uncached_candidates}
           if args.drop_uncached_candidates is not None else {}),
        **_resolve_perf_overrides(args),
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
