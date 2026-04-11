"""
Score LLM responses from the Claude web interface against the outcome benchmark.

Reads responses positionally — response N corresponds to row N of candidate_pairs.csv.
Accepts a JSONL file (one JSON object per line) or a JSON array file.

Each response must be a JSON object with at least an "outcome" field, e.g.:
    {"outcome": "FAILED_PHASE_2", "confidence": "HIGH", "reasoning": "..."}

Usage:
    python src/eval/score_responses.py --responses responses.jsonl
    python src/eval/score_responses.py --responses responses.json \\
        --pairs src/eval/candidate_pairs.csv \\
        --benchmark src/eval/outcome_benchmark.csv
"""

import argparse
import csv
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent

sys.path.insert(0, str(_HERE))
from validate import (
    OUTCOME_TO_CLASS,
    evaluate,
    load_benchmark,
    load_benchmark_binary_labels,
    print_error_cases,
    print_report,
    _norm,
)

# The LLM returns enum *names* (e.g. "FAILED_PHASE_2") but OUTCOME_TO_CLASS
# uses enum *values* (e.g. "Failed Phase 2"). Build a name→value map too.
_NAME_TO_VALUE: dict[str, str] = {
    "FAILED_PHASE_1": "Failed Phase 1",
    "FAILED_PHASE_2": "Failed Phase 2",
    "FAILED_PHASE_3": "Failed Phase 3",
    "APPROVED": "Approved",
    "COMMERCIALIZED": "Commercialized",
    "ONGOING": "Ongoing",
    "UNKNOWN": "Unknown",
}


def _load_responses(path: str) -> list[dict]:
    """
    Load a JSONL file (one JSON object per line) or a JSON array file.
    Blank lines are skipped.
    """
    text = Path(path).read_text(encoding="utf-8").strip()
    if not text:
        sys.exit(f"Response file is empty: {path}")

    # Try JSON array first
    if text.startswith("["):
        try:
            data = json.loads(text)
            if not isinstance(data, list):
                sys.exit("Response file looks like JSON but is not an array.")
            return data
        except json.JSONDecodeError as exc:
            sys.exit(f"Failed to parse response file as JSON array: {exc}")

    # JSONL — one object per line
    responses = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            sys.exit(f"JSON parse error on line {lineno}: {exc}\n  {line!r}")
        if not isinstance(obj, dict):
            sys.exit(f"Line {lineno} is not a JSON object: {line!r}")
        responses.append(obj)

    return responses


def _outcome_to_bucket(response: dict) -> str:
    """
    Map an LLM response dict to a validation bucket string.
    Accepts both enum names ("FAILED_PHASE_2") and enum values ("Failed Phase 2").
    Falls back to "investigational" for unrecognised values.
    """
    raw = response.get("outcome", "").strip()
    # Try as enum name first (what the LLM returns)
    value = _NAME_TO_VALUE.get(raw)
    if value is not None:
        return OUTCOME_TO_CLASS.get(value, "investigational")
    # Try as enum value directly
    return OUTCOME_TO_CLASS.get(raw, "investigational")


def main() -> None:
    default_pairs = str(_HERE / "candidate_pairs.csv")
    default_benchmark = str(_HERE / "outcome_benchmark.csv")

    parser = argparse.ArgumentParser(
        description="Score LLM adjudication responses against the outcome benchmark."
    )
    parser.add_argument(
        "--responses",
        required=True,
        help="JSONL or JSON array file of LLM responses (one per candidate pair, in order)",
    )
    parser.add_argument(
        "--pairs",
        default=default_pairs,
        help=f"candidate_pairs.csv used to generate the prompts (default: {default_pairs})",
    )
    parser.add_argument(
        "--benchmark",
        default=default_benchmark,
        help=f"Labeled benchmark CSV (default: {default_benchmark})",
    )
    args = parser.parse_args()

    # Load pairs (to map response index → drug/indication key)
    with open(args.pairs, newline="", encoding="utf-8") as f:
        pairs = list(csv.DictReader(f))
    if not pairs:
        sys.exit("candidate_pairs.csv is empty.")

    # Load benchmark and evaluate
    benchmark, skipped = load_benchmark(args.benchmark)
    if not benchmark:
        sys.exit("Benchmark is empty or all rows were skipped.")

    benchmark_rows, _ = load_benchmark_binary_labels(args.benchmark)

    responses_path = Path(args.responses)
    if responses_path.is_dir():
        response_files = sorted(responses_path.glob("*.txt"))
        if not response_files:
            sys.exit(f"No .txt files found in responses directory: {responses_path}")
    else:
        response_files = [responses_path]

    for response_file in response_files:
        # Load LLM responses
        responses = _load_responses(str(response_file))

        if len(responses) != len(pairs):
            print(
                f"Warning ({response_file.name}): {len(responses)} responses but {len(pairs)} "
                "candidate pairs. Matching by position up to the shorter length.",
                file=sys.stderr,
            )

        # Build predictions dict keyed by (drug_norm, indication_norm)
        predictions: dict[tuple[str, str], str] = {}
        n = min(len(responses), len(pairs))
        for i in range(n):
            row = pairs[i]
            key = (_norm(row["drug_name"]), _norm(row["indication"]))
            bucket = _outcome_to_bucket(responses[i])
            predictions[key] = bucket

        if len(response_files) > 1:
            print(f"\n=== Results for {response_file.name} ===")

        report = evaluate(predictions, benchmark, skipped)
        print_report(report)
        print_error_cases(predictions, benchmark_rows)


if __name__ == "__main__":
    main()
