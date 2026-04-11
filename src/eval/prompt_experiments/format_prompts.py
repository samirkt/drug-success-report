"""
Format drug-indication pairs from candidate_pairs.csv into LLM prompt blocks.

Each block contains the filled adjudication user prompt for one candidate,
prefixed with an index so responses can be matched back positionally.

Usage:
    python src/eval/format_prompts.py
    python src/eval/format_prompts.py --input src/eval/candidate_pairs.csv
    python src/eval/format_prompts.py --output prompts.txt
"""

import argparse
import csv
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent

_USER_TEMPLATE = """\
Drug name: {drug_name}
Indication: {indication}
Highest phase reached: {highest_phase}
Number of trials: {trial_count}
Sponsors: {sponsors}
Earliest trial start: {earliest_start}
Latest trial completion: {latest_completion}"""

_SEPARATOR = "─" * 60


def format_prompt(row: dict, index: int) -> str:
    drug = row.get("drug_name", "").strip() or "Unknown"
    indication = row.get("indication", "").strip() or "Unknown"
    highest_phase = row.get("highest_phase", "").strip() or "Unknown"
    sponsors = row.get("sponsors", "").strip() or "Unknown"

    prompt = _USER_TEMPLATE.format(
        drug_name=drug,
        indication=indication,
        highest_phase=highest_phase,
        trial_count="(see trial record)",
        sponsors=sponsors,
        earliest_start="Unknown",
        latest_completion="Unknown",
    )

    header = f"[{index}] {drug} / {indication}"
    return f"{header}\n{prompt}"


def main() -> None:
    default_input = str(_HERE / "candidate_pairs.csv")

    parser = argparse.ArgumentParser(
        description="Format candidate pairs as adjudication prompts for the Claude web interface."
    )
    parser.add_argument(
        "--input",
        default=default_input,
        help=f"Path to candidate_pairs.csv (default: {default_input})",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Write output to this file instead of stdout",
    )
    args = parser.parse_args()

    with open(args.input, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        sys.exit("candidate_pairs.csv is empty or has no data rows.")

    blocks = []
    for i, row in enumerate(rows, start=1):
        blocks.append(format_prompt(row, i))

    output = ("\n" + _SEPARATOR + "\n\n").join(blocks)
    total_line = f"\n{_SEPARATOR}\nTotal: {len(rows)} prompts\n"

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output + total_line)
        print(f"Wrote {len(rows)} prompts to {args.output}", file=sys.stderr)
    else:
        print(output + total_line)


if __name__ == "__main__":
    main()
