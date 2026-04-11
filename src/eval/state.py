import csv
from pathlib import Path

OUTPUT_COLUMNS = [
    "drug_name",
    "indication",
    "commercialization_label",
    "approval_label",
    "commercialization_evidence",
    "approval_evidence",
    "application_number",
    "drugsatfda_overview_url",
    "sponsor",
    "notes",
]


def load_state(output_path: str) -> list[dict]:
    """Read existing output CSV; returns empty list if file does not exist."""
    path = Path(output_path)
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return [dict(row) for row in csv.DictReader(f)]


def save_state(rows: list[dict], output_path: str) -> None:
    """Overwrite output CSV with current rows. Called after every label decision."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def is_resolved(row: dict) -> bool:
    """True when both label columns are set."""
    return bool(row.get("commercialization_label")) and bool(row.get("approval_label"))
