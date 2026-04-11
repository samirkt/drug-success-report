"""
CLI for validating pipeline predictions against the outcome benchmark.

Usage:
    python src/eval/run_validation.py \
        --predictions src/eval/candidate_pairs.csv \
        --benchmark   src/eval/outcome_benchmark.csv
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from validate import (
    evaluate,
    load_benchmark,
    load_benchmark_binary_labels,
    load_predictions_csv,
    print_error_cases,
    print_report,
)


def main() -> None:
    default_benchmark = str(Path(__file__).resolve().parent / "outcome_benchmark.csv")

    parser = argparse.ArgumentParser(
        description="Validate pipeline outcome predictions against the labeled benchmark."
    )
    parser.add_argument(
        "--predictions",
        required=True,
        help="CSV with drug_name, indication, outcome columns (pipeline output)",
    )
    parser.add_argument(
        "--benchmark",
        default=default_benchmark,
        help=f"Path to labeled benchmark CSV (default: {default_benchmark})",
    )
    args = parser.parse_args()

    benchmark, skipped = load_benchmark(args.benchmark)
    if not benchmark:
        sys.exit("Benchmark is empty or all rows were skipped.")

    predictions = load_predictions_csv(args.predictions)
    if not predictions:
        sys.exit("Predictions file is empty or has no recognizable rows.")

    report = evaluate(predictions, benchmark, skipped)
    print_report(report)
    benchmark_rows, _ = load_benchmark_binary_labels(args.benchmark)
    print_error_cases(predictions, benchmark_rows)


if __name__ == "__main__":
    main()
