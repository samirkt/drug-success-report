"""
Benchmark labeling CLI for drug-indication pairs.

Usage:
    python src/eval/label_benchmark.py --input candidates.csv --output labeled.csv
"""
import argparse
import csv
import sys
from pathlib import Path

# Allow imports from src/eval/ regardless of working directory
sys.path.insert(0, str(Path(__file__).resolve().parent))

from state import is_resolved, load_state, save_state
from evidence import fetch_approval_evidence, fetch_commercialization

# ---------------------------------------------------------------------------
# Label choices
# ---------------------------------------------------------------------------
COMM_CHOICES = {"p": "positive", "n": "negative", "u": "unknown"}
APPR_CHOICES = {"p": "positive", "n": "negative", "a": "ambiguous"}

_W = 66  # display width


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_input(input_path: str) -> list[dict]:
    with open(input_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        sys.exit("Input CSV is empty.")
    if "drug_name" not in rows[0] or "indication" not in rows[0]:
        sys.exit("Input CSV must have 'drug_name' and 'indication' columns.")
    return rows


def _merge(input_rows: list[dict], state_rows: list[dict]) -> list[dict]:
    """Overlay saved state onto input rows; new rows get blank labels."""
    state_index = {(r["drug_name"], r["indication"]): r for r in state_rows}
    merged = []
    for row in input_rows:
        key = (row["drug_name"], row["indication"])
        if key in state_index:
            merged.append(state_index[key])
        else:
            merged.append({
                "drug_name": row["drug_name"],
                "indication": row["indication"],
                "commercialization_label": "",
                "approval_label": "",
                "commercialization_evidence": "",
                "approval_evidence": "",
                "application_number": "",
                "drugsatfda_overview_url": "",
                "sponsor": "",
                "notes": "",
            })
    return merged


def _print_progress(rows: list[dict]) -> None:
    total = len(rows)
    done = sum(1 for r in rows if is_resolved(r))
    print(f"  Progress: {done}/{total} labeled | {total - done} remaining")


def _prompt_label(prompt: str, choices: dict[str, str]) -> str:
    shortcuts = "  ".join(f"{k}={v}" for k, v in choices.items())
    while True:
        raw = input(f"  {prompt} [{shortcuts}]: ").strip().lower()
        if raw in choices:
            return choices[raw]
        if raw in choices.values():
            return raw
        print(f"  Enter one of: {', '.join(f'{k}={v}' for k, v in choices.items())}")


# ---------------------------------------------------------------------------
# Evidence fetch pass
# ---------------------------------------------------------------------------

def _build_comm_evidence(ev: dict) -> str:
    if not ev["is_commercialized"] and ev["marketing_start_date"] is None:
        return "NDC: not found"
    status = "active" if ev["is_commercialized"] else "inactive"
    start = ev["marketing_start_date"] or "unknown"
    end = ev["marketing_end_date"] or "ongoing"
    return f"NDC: {status}  |  start: {start}  |  end: {end}"


def _fetch_evidence(rows: list[dict], output_path: str, drugbank_csv: str | None = None) -> None:
    unresolved = [r for r in rows if not is_resolved(r)]
    if not unresolved:
        return

    print(f"\nFetching NDC commercialization data for {len(unresolved)} pairs...")
    for row in unresolved:
        if row.get("commercialization_evidence"):
            continue
        ev = fetch_commercialization(row["drug_name"], drugbank_csv=drugbank_csv)
        row["commercialization_evidence"] = _build_comm_evidence(ev)
        row["application_number"] = ev.get("application_number") or ""
        row["drugsatfda_overview_url"] = ev.get("drugsatfda_overview_url") or ""
        row["sponsor"] = ev.get("sponsor") or ""

    save_state(rows, output_path)

    needs_approval_ev = [r for r in unresolved if not r.get("approval_evidence")]
    if needs_approval_ev:
        drug_names = list(dict.fromkeys(r["drug_name"] for r in needs_approval_ev))
        print(f"Fetching OpenFDA label indications for {len(drug_names)} unique drug(s)...")
        approval_ev = fetch_approval_evidence(drug_names)
        for row in needs_approval_ev:
            text = approval_ev.get(row["drug_name"], "")
            row["approval_evidence"] = text or "OpenFDA: no label indications found"

    save_state(rows, output_path)
    print("Evidence fetch complete.")


# ---------------------------------------------------------------------------
# Interactive review pass
# ---------------------------------------------------------------------------

def _run_interactive(rows: list[dict], output_path: str) -> None:
    unresolved = sorted(
        (r for r in rows if not is_resolved(r)),
        key=lambda r: r["drug_name"].lower(),
    )
    total = len(unresolved)
    if not total:
        print("All pairs are already labeled.")
        return

    print(f"\nStarting interactive review for {total} unresolved pair(s).")
    print("Press Ctrl-C to quit. Progress is saved after every row.\n")

    try:
        for idx, row in enumerate(unresolved, 1):
            drug = row["drug_name"]
            indication = row["indication"]

            # Header
            print("\n" + "═" * _W)
            print(f"  [{idx}/{total}]  {drug}  ·  {indication}")
            print("═" * _W)

            # Evidence blocks
            print(f"\n  COMMERCIALIZATION (NDC)")
            print(f"  {row.get('commercialization_evidence', '(none)')}")
            if row.get("sponsor"):
                print(f"  Sponsor: {row['sponsor']}")
            if row.get("drugsatfda_overview_url"):
                url = row["drugsatfda_overview_url"]
                link = f"\033]8;;{url}\033\\{url}\033]8;;\033\\"
                print(f"  Drugs@FDA: {link}")

            print(f"\n  APPROVAL (FDA label indications)")
            print(f"  {row.get('approval_evidence', '(none)')}")

            # Prompts
            print("\n" + "─" * _W)

            if row.get("commercialization_label"):
                print(f"  commercialization: {row['commercialization_label']} (already set)")
            else:
                row["commercialization_label"] = _prompt_label(
                    "commercialization", COMM_CHOICES
                )

            if row.get("approval_label"):
                print(f"  approval: {row['approval_label']} (already set)")
            else:
                row["approval_label"] = _prompt_label(
                    "approval         ", APPR_CHOICES
                )

            notes = input("  notes (optional, Enter to skip): ").strip()
            if notes:
                row["notes"] = notes

            save_state(rows, output_path)
            print()
            _print_progress(rows)

    except KeyboardInterrupt:
        print("\n\nInterrupted. Progress saved.")
        _print_progress(rows)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Label drug–indication pairs with commercialization and approval benchmarks."
    )
    parser.add_argument("--input", required=True, help="CSV with drug_name and indication columns")
    parser.add_argument("--output", required=True, help="Output CSV (created or resumed)")
    parser.add_argument(
        "--drugbank",
        default="../../databases/drugbank/drugbank_approvals.csv",
        help="Path to drugbank_approvals.csv for synonym fallback (default: databases/drugbank/drugbank_approvals.csv)",
    )
    parser.add_argument(
        "--no-drugbank",
        action="store_true",
        help="Disable DrugBank synonym fallback entirely",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Re-fetch NDC/OpenFDA evidence for unresolved rows, ignoring cached results. "
             "Rows already labeled by the user are never modified.",
    )
    args = parser.parse_args()

    drugbank_csv = None if args.no_drugbank else args.drugbank

    input_rows = _load_input(args.input)
    state_rows = load_state(args.output)
    rows = _merge(input_rows, state_rows)

    print(f"\nLoaded {len(rows)} drug–indication pair(s).")
    _print_progress(rows)

    if args.refresh:
        _EVIDENCE_FIELDS = (
            "commercialization_evidence", "approval_evidence",
            "application_number", "drugsatfda_overview_url", "sponsor",
        )
        cleared = 0
        for row in rows:
            if not is_resolved(row):
                for field in _EVIDENCE_FIELDS:
                    row[field] = ""
                cleared += 1
        if cleared:
            print(f"--refresh: cleared evidence for {cleared} unresolved row(s).")

    _fetch_evidence(rows, args.output, drugbank_csv)
    _print_progress(rows)

    _run_interactive(rows, args.output)

    resolved = sum(1 for r in rows if is_resolved(r))
    print(f"\nDone. {resolved}/{len(rows)} pair(s) labeled. Output: {args.output}")


if __name__ == "__main__":
    main()
