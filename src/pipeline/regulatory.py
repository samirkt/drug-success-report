"""
Regulatory approval index.

Consumes `data/drugbank_products.csv` (built by
`scripts/build_drugbank_derivatives.py`) and exposes a deterministic lookup
from (drug_name, drugbank_id) to an approval record. Used by the outcome
adjudication stage to credit APPROVED outcomes without depending on LLM
judgement — aligning with ClinSR's use of FDA CDER / Drugs@FDA as the
authoritative signal.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from .drugbank_norm import canonicalize_drug_name

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ApprovalRecord:
    """Deterministic approval metadata for a drug."""

    drugbank_id: str
    drug_name_norm: str
    appl_no: str | None
    approval_date: date | None
    first_marketed_date: date | None
    evidence_source: str  # e.g. "drugs_at_fda:NDA020807"


def _parse_date(value) -> date | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    s = str(value).strip()
    if not s or s.lower() == "nan" or s == "<NA>":
        return None
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


class RegulatoryIndex:
    """Two-key deterministic lookup over the DrugBank-derived approvals table.

    Only drugs with approved=True AND a US marketing date are treated as
    "approved" for pipeline purposes, matching ClinSR's FDA-CDER-centric
    definition. Drugs marked approved in other jurisdictions (e.g. EMA
    only) will still appear in the products table but return None from
    `lookup()` because they should not credit US-registered trials.
    """

    def __init__(self, by_drugbank_id: dict[str, ApprovalRecord],
                 by_name_norm: dict[str, ApprovalRecord]):
        self._by_db_id = by_drugbank_id
        self._by_name = by_name_norm

    @classmethod
    def from_csv(cls, products_csv: Path) -> "RegulatoryIndex":
        if not products_csv.exists():
            logger.warning(
                "Regulatory products CSV not found at %s — RegulatoryIndex "
                "will be empty. Run src/scripts/build_drugbank_derivatives.py.",
                products_csv,
            )
            return cls({}, {})

        df = pd.read_csv(products_csv, low_memory=False, dtype={
            "drugbank_id": "string",
            "drug_name_norm": "string",
            "primary_name": "string",
            "appl_no": "string",
            "approved": "string",
            "approval_date": "string",
            "first_marketed_date": "string",
            "us_country_hit": "string",
        })

        by_db_id: dict[str, ApprovalRecord] = {}
        by_name_norm: dict[str, ApprovalRecord] = {}
        for row in df.itertuples(index=False):
            if str(getattr(row, "approved", "")).lower() != "true":
                continue
            if str(getattr(row, "us_country_hit", "")).lower() != "true":
                continue
            appl_no = getattr(row, "appl_no", None)
            appl_no_str: str | None
            try:
                if appl_no is None or pd.isna(appl_no):
                    appl_no_str = None
                else:
                    appl_no_str = str(appl_no).strip() or None
            except (TypeError, ValueError):
                appl_no_str = str(appl_no).strip() or None

            approval_dt = _parse_date(getattr(row, "approval_date", None))
            first_dt = _parse_date(getattr(row, "first_marketed_date", None))
            # Require at least one populated approval date to avoid dateless
            # "approved" rows promoting the candidate with no temporal anchor.
            if approval_dt is None and first_dt is None:
                continue

            record = ApprovalRecord(
                drugbank_id=str(row.drugbank_id),
                drug_name_norm=str(row.drug_name_norm),
                appl_no=appl_no_str,
                approval_date=approval_dt,
                first_marketed_date=first_dt,
                evidence_source=(
                    f"drugs_at_fda:{appl_no_str}" if appl_no_str
                    else "drugs_at_fda:drugbank_products"
                ),
            )
            by_db_id[record.drugbank_id] = record
            # Name map may collide across DrugBank IDs (rare — same canonical
            # name for distinct products). First writer wins; subsequent
            # collisions are logged once at DEBUG.
            if record.drug_name_norm and record.drug_name_norm not in by_name_norm:
                by_name_norm[record.drug_name_norm] = record

        logger.info(
            "RegulatoryIndex loaded: %d approved drugs (by DrugBank ID), "
            "%d unique name-normalizations",
            len(by_db_id), len(by_name_norm),
        )
        return cls(by_db_id, by_name_norm)

    def lookup(
        self,
        drug_name: str | None = None,
        drugbank_id: str | None = None,
    ) -> Optional[ApprovalRecord]:
        """Return the best ApprovalRecord for a candidate, or None.

        Lookup order:
          1. DrugBank ID (strongest signal)
          2. Canonicalized drug name (normalized via canonicalize_drug_name)
        """
        if drugbank_id and drugbank_id in self._by_db_id:
            return self._by_db_id[drugbank_id]
        if drug_name:
            norm = canonicalize_drug_name(drug_name)
            if norm and norm in self._by_name:
                return self._by_name[norm]
        return None

    def __len__(self) -> int:
        return len(self._by_db_id)
