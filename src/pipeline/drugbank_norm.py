"""
DrugBank drug name normalization helper.

Ports the two-level normalization logic from reproduce_clinsr.ipynb into a
standalone, per-name lookup function for use in the clustering stage.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Optional

import pandas as pd

_DROP_WORDS = {
    # placebo / arms / generic trial wording
    "placebo", "sham", "control", "controls", "comparator", "active", "standard", "best",
    "available", "therapy", "bat",
    # formulation / route / presentation
    "injectable", "injection", "iv", "sc", "sq", "po", "oral", "intravenous", "subcutaneous",
    "solution", "powder", "tablet", "capsule", "suspension", "infusion", "bolus",
    # release / dosing descriptors
    "extended", "release", "delayed", "immediate", "sustained", "modified",
    "dose", "dosing", "fixed",
    # regimens/protocol language
    "protocol", "regimen", "arm", "group",
    # common carriers/excipients
    "saline", "electrolyte", "glucose", "amino", "acid", "acids",
}

_ROMAN = {"i": "1", "ii": "2", "iii": "3", "iv": "4", "v": "5"}


def canonicalize_drug_name(name: str) -> str:
    """Normalize a drug name for DrugBank lookup.

    Mirrors the canonicalize_drug / normalize_drug_term logic used in
    reproduce_clinsr.ipynb so that pipeline matches align with the notebook.

    Steps (in order):
      1. Unicode NFKD + strip combining chars; µ → u
      2. Strip parenthetical/bracketed/braced content
      3. Lowercase
      4. Separators (/ , _ ; : - – —) → spaces
      5. Remove quotes
      6. Remove remaining punctuation
      7. Normalize isotope formats (e.g. "177 lu" → "177lu")
      8. Drop non-identity tokens (formulation words, route, placebo, etc.)
      9. Roman numeral normalization (i→1 … v→5)
     10. Collapse whitespace
    """
    s = (name or "").strip()
    if not s:
        return ""

    # 1) Unicode normalize + strip diacritics
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.replace("µ", "u")

    # 2) Drop bracketed/parenthetical descriptors
    s = re.sub(r"\([^)]*\)", " ", s)
    s = re.sub(r"\[[^\]]*\]", " ", s)
    s = re.sub(r"\{[^}]*\}", " ", s)

    # 3) Lowercase
    s = s.lower()

    # 4) Normalize separators to spaces
    s = re.sub(r"[/,_;:]", " ", s)
    s = re.sub(r"[-\u2013\u2014]", " ", s)   # hyphen, en-dash, em-dash

    # 5) Remove quotes
    s = re.sub(r"[\"'\u201c\u201d\u2018\u2019`]", "", s)

    # 6) Remove remaining punctuation
    s = re.sub(r"[^\w\s]", " ", s)

    # 7) Normalize isotope formats: "177 lu" → "177lu"
    s = re.sub(r"\b(\d{1,3})\s*(lu|ga|y|i|tc|in)\b", r"\1\2", s)

    # 8) Drop non-identity tokens
    tokens = [t for t in s.split() if t and t not in _DROP_WORDS]

    # 9) Roman numeral normalization
    tokens = [_ROMAN.get(t, t) for t in tokens]

    # 10) Collapse whitespace
    return " ".join(tokens)


def load_drugbank_lookup(csv_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load drugbank_approvals.csv and build two deduplicated lookup tables.

    For each duplicate key, the row with the most non-null columns is kept.

    Returns:
        best_rows:      deduplicated by query_name (exact); most-complete row wins
        best_rows_norm: deduplicated by query_norm (lowercased); most-complete row wins
    """
    df = pd.read_csv(csv_path, low_memory=False)

    best_rows = (
        df
        .assign(_nn=df.notna().sum(axis=1))
        .sort_values(["query_name", "_nn"], ascending=[True, False])
        .drop_duplicates("query_name", keep="first")
        .drop(columns="_nn")
    )

    best_rows_norm = (
        df
        .assign(_nn=df.notna().sum(axis=1))
        .sort_values(["query_norm", "_nn"], ascending=[True, False])
        .drop_duplicates("query_norm", keep="first")
        .drop(columns="_nn")
    )

    return best_rows, best_rows_norm


def match_drug_name(
    drug_name: str,
    best_rows: pd.DataFrame,
    best_rows_norm: pd.DataFrame,
) -> Optional[str]:
    """Two-level lookup: exact canonical match, then first-word fallback.

    Args:
        drug_name:      the drug name to look up (e.g. "Lepirudin (rDNA origin) HCl")
        best_rows:      DataFrame deduped by query_name
        best_rows_norm: DataFrame deduped by query_norm; primary lookup target

    Returns:
        drug_id string or None if no match found
    """
    norm = canonicalize_drug_name(drug_name)

    # Level 1: exact canonical match against query_norm
    exact = best_rows_norm[best_rows_norm["query_norm"] == norm]
    if not exact.empty:
        return str(exact.iloc[0]["drug_id"])

    # Level 2: first-word fallback
    parts = norm.split()
    if not parts:
        return None
    first_word = parts[0]
    fallback = best_rows_norm[best_rows_norm["query_norm"] == first_word]
    if not fallback.empty:
        return str(fallback.iloc[0]["drug_id"])

    return None
