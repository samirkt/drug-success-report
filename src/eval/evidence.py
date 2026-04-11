"""
Thin wrappers over ndc_lookup.py for evidence fetching.
All rate-limiting is handled by ndc_lookup internally.
"""
import re
import sys
from pathlib import Path

import pandas as pd

# TODO: re-enable once ndc_lookup is moved into this repo
# ndc_lookup.py previously lived in the parent project root
# from ndc_lookup import _batch_ndc_lookup, get_drugs_with_indications_bulk

_DRUGSATFDA_URL = (
    "https://www.accessdata.fda.gov/scripts/cder/daf/index.cfm"
    "?event=overview.process&ApplNo={}"
)

_MAX_INDICATION_CHARS = 500

_drugbank_cache: dict[str, pd.DataFrame] = {}


def _load_drugbank_df(csv_path: str) -> pd.DataFrame:
    """Load drugbank_approvals.csv once per path, caching in module-level dict."""
    if csv_path not in _drugbank_cache:
        _drugbank_cache[csv_path] = pd.read_csv(
            csv_path, usecols=["query_name", "query_norm", "drug_id"], low_memory=False
        )
    return _drugbank_cache[csv_path]


def _get_drug_synonyms(drug_name: str, df: pd.DataFrame) -> list[str]:
    """Return all DrugBank query_names sharing the same drug_id, excluding drug_name itself."""
    norm = re.sub(r"\s+", " ", drug_name.lower().strip())
    match = df[df["query_norm"] == norm]
    if match.empty:
        return []
    drug_id = match.iloc[0]["drug_id"]
    all_names = df[df["drug_id"] == drug_id]["query_name"].tolist()
    return [n for n in all_names if n.lower() != drug_name.lower()]


def fetch_commercialization(drug_name: str, drugbank_csv: str | None = None) -> dict:
    """
    Returns NDC commercialization evidence for a single drug.

    Result keys:
        is_commercialized (bool)        — True if any record has no end date (ongoing)
        marketing_start_date (str|None) — earliest start date across all records
        marketing_end_date (str|None)   — latest end date; None means at least one record is ongoing
    """
    ndc_by_drug = _batch_ndc_lookup([drug_name])
    records = ndc_by_drug.get(drug_name, [])

    if not records and drugbank_csv:
        df = _load_drugbank_df(drugbank_csv)
        for syn in _get_drug_synonyms(drug_name, df):
            ndc_by_drug = _batch_ndc_lookup([syn])
            records = ndc_by_drug.get(syn, [])
            if records:
                break

    if not records:
        return {
            "is_commercialized": False,
            "marketing_start_date": None,
            "marketing_end_date": None,
            "application_number": None,
            "drugsatfda_overview_url": "",
            "sponsor": None,
        }

    start_dates = [r.get("marketing_start_date") for r in records if r.get("marketing_start_date")]
    end_dates = [r.get("marketing_end_date") for r in records if r.get("marketing_end_date")]
    ongoing = any(r.get("marketing_end_date") is None for r in records)

    # Extract the first application_number found (top-level, then openfda sub-object)
    app_no = None
    for rec in records:
        app_no = rec.get("application_number")
        if not app_no:
            openfda_list = (rec.get("openfda") or {}).get("application_number") or []
            app_no = openfda_list[0] if openfda_list else None
        if app_no:
            break

    if app_no:
        numeric = re.sub(r'^\D+', '', str(app_no))
        url = _DRUGSATFDA_URL.format(numeric)
    else:
        url = ""

    # Extract unique labeler names; preserve first-seen order
    seen = set()
    sponsors = []
    for rec in records:
        name = rec.get("labeler_name")
        if name and name not in seen:
            seen.add(name)
            sponsors.append(name)

    return {
        "is_commercialized": ongoing,
        "marketing_start_date": min(start_dates, default=None),
        "marketing_end_date": None if ongoing else max(end_dates, default=None),
        "application_number": app_no,
        "drugsatfda_overview_url": url,
        "sponsor": sponsors[0] if sponsors else None,
    }


def fetch_approval_evidence(drug_names: list[str]) -> dict[str, str]:
    """
    Returns a map of drug_name -> truncated indications text from OpenFDA labels.
    Empty string when no label indications are found.

    drug_names should be deduplicated by the caller; this function preserves
    order and passes each name as its own group to get_drugs_with_indications_bulk.
    """
    unique = list(dict.fromkeys(drug_names))
    raw = get_drugs_with_indications_bulk(unique)

    result: dict[str, str] = {}
    for drug_name in unique:
        entry = raw.get(drug_name)
        if entry is None:
            # Fall back to case-insensitive search over returned keys
            entry = next(
                (v for v in raw.values() if v.get("query", "").lower() == drug_name.lower()),
                None,
            )

        if entry and entry.get("indications"):
            result[drug_name] = " | ".join(entry["indications"])
        else:
            result[drug_name] = ""

    return result
