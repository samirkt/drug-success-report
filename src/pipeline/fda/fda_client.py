"""FDA data access layer.

Wraps openFDA, Drugs@FDA, DailyMed, and NDC directory endpoints.
All network calls are rate-limited and cached on disk to make the
pipeline re-runnable without re-hitting FDA servers.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


def _get_httpx():
    """Lazy import so tests using FakeFDA don't require httpx."""
    import httpx
    return httpx


OPENFDA_BASE = "https://api.fda.gov"
DAILYMED_BASE = "https://dailymed.nlm.nih.gov/dailymed/services/v2"
DRUGS_AT_FDA_BASE = "https://www.accessdata.fda.gov/drugsatfda_docs"


@dataclass
class Submission:
    """A single regulatory submission for an application."""

    application_number: str
    submission_type: str  # "ORIG" or "SUPPL"
    submission_number: int
    submission_class_code: Optional[str]  # e.g. "TYPE 6", "EFFICACY"
    submission_class_description: Optional[str]  # e.g. "New Indication"
    submission_status: Optional[str]  # "AP" = approved
    submission_status_date: Optional[date]
    approval_letter_url: Optional[str] = None

    @property
    def is_new_indication(self) -> bool:
        desc = (self.submission_class_description or "").lower()
        return "new indication" in desc or "efficacy" in desc

    @property
    def is_approved(self) -> bool:
        return self.submission_status == "AP"


@dataclass
class Application:
    """A Drugs@FDA application (NDA, BLA, ANDA)."""

    application_number: str  # e.g. "NDA021436"
    sponsor_name: Optional[str]
    active_ingredients: list[str]
    brand_name: Optional[str]
    submissions: list[Submission] = field(default_factory=list)

    @property
    def original_approval_date(self) -> Optional[date]:
        for sub in self.submissions:
            if (
                sub.submission_type == "ORIG"
                and sub.is_approved
                and sub.submission_status_date
            ):
                return sub.submission_status_date
        return None


@dataclass
class NDCRecord:
    """One NDC directory entry."""

    ndc: str
    application_number: Optional[str]
    marketing_start_date: Optional[date]
    marketing_end_date: Optional[date]
    marketing_category: Optional[str]
    labeler_name: Optional[str]

    def is_currently_marketed(self, as_of: Optional[date] = None) -> bool:
        ref = as_of or date.today()
        if not self.marketing_start_date or self.marketing_start_date > ref:
            return False
        if self.marketing_end_date and self.marketing_end_date < ref:
            return False
        return True


class FDAClient:
    """HTTP client with disk caching and polite rate limiting."""

    def __init__(
        self,
        cache_dir: Path,
        openfda_api_key: Optional[str] = None,
        requests_per_second: float = 2.0,
        timeout: float = 30.0,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.api_key = openfda_api_key
        self._min_interval = 1.0 / requests_per_second
        self._last_request_at = 0.0
        self._throttle_lock = threading.Lock()
        httpx = _get_httpx()
        self._client = httpx.Client(timeout=timeout, follow_redirects=True)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "FDAClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ---------- caching ----------

    def _cache_path(self, key: str) -> Path:
        h = hashlib.sha256(key.encode()).hexdigest()[:32]
        return self.cache_dir / f"{h}.json"

    def _cache_get(self, key: str) -> Optional[Any]:
        p = self._cache_path(key)
        if p.exists():
            try:
                return json.loads(p.read_text())
            except json.JSONDecodeError:
                return None
        return None

    def _cache_set(self, key: str, value: Any) -> None:
        self._cache_path(key).write_text(json.dumps(value))

    def _throttle(self) -> None:
        with self._throttle_lock:
            elapsed = time.monotonic() - self._last_request_at
            if elapsed < self._min_interval:
                time.sleep(self._min_interval - elapsed)
            self._last_request_at = time.monotonic()

    def _get_json(self, url: str, params: Optional[dict] = None) -> Optional[dict]:
        cache_key = f"{url}?{json.dumps(params or {}, sort_keys=True)}"
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        if self.api_key and "api.fda.gov" in url:
            params = {**(params or {}), "api_key": self.api_key}

        self._throttle()
        try:
            r = self._client.get(url, params=params)
            if r.status_code == 404:
                self._cache_set(cache_key, {})
                return {}
            r.raise_for_status()
            data = r.json()
        except (Exception, json.JSONDecodeError) as e:
            logger.warning("GET %s failed: %s", url, e)
            return None

        self._cache_set(cache_key, data)
        return data

    def _get_bytes(self, url: str) -> Optional[bytes]:
        cache_key = f"BIN:{url}"
        p = self._cache_path(cache_key)
        if p.with_suffix(".bin").exists():
            return p.with_suffix(".bin").read_bytes()

        self._throttle()
        try:
            r = self._client.get(url)
            r.raise_for_status()
            data = r.content
        except Exception as e:
            logger.warning("GET %s failed: %s", url, e)
            return None

        p.with_suffix(".bin").write_bytes(data)
        return data

    # ---------- high-level lookups ----------

    def find_applications_by_drug(self, drug_name: str) -> list[Application]:
        """Search openFDA drug/drugsfda endpoint for a drug by name.

        Returns all matching applications (there may be several: the
        innovator's NDA plus generics' ANDAs, or combination products).
        """
        # openFDA search syntax: OR across active_ingredient and brand_name
        query = (
            f'(openfda.generic_name:"{drug_name}" '
            f'openfda.brand_name:"{drug_name}" '
            f'products.active_ingredients.name:"{drug_name}")'
        )
        data = self._get_json(
            f"{OPENFDA_BASE}/drug/drugsfda.json",
            params={"search": query, "limit": 100},
        )
        if not data or "results" not in data:
            return []

        return [self._parse_application(r) for r in data["results"]]

    def _parse_application(self, record: dict) -> Application:
        app_no = record.get("application_number", "")
        sponsor = record.get("sponsor_name")
        products = record.get("products", []) or []
        ingredients: list[str] = []
        brand = None
        for p in products:
            brand = brand or p.get("brand_name")
            for ing in p.get("active_ingredients", []) or []:
                name = ing.get("name")
                if name and name not in ingredients:
                    ingredients.append(name)

        submissions = [self._parse_submission(app_no, s)
                       for s in record.get("submissions", []) or []]

        return Application(
            application_number=app_no,
            sponsor_name=sponsor,
            active_ingredients=ingredients,
            brand_name=brand,
            submissions=submissions,
        )

    def _parse_submission(self, app_no: str, s: dict) -> Submission:
        status_date = None
        raw = s.get("submission_status_date")
        if raw:
            try:
                status_date = datetime.strptime(raw, "%Y%m%d").date()
            except ValueError:
                try:
                    status_date = datetime.strptime(raw, "%Y-%m-%d").date()
                except ValueError:
                    pass

        try:
            sub_no = int(s.get("submission_number", 0))
        except (TypeError, ValueError):
            sub_no = 0

        # Approval letter URL pattern for Drugs@FDA
        # e.g. https://www.accessdata.fda.gov/drugsatfda_docs/appletter/2023/125514Orig1s099ltr.pdf
        # We don't always have enough info to construct it; best-effort.
        letter_url = None
        review_docs = s.get("application_docs", []) or []
        for doc in review_docs:
            if (doc.get("type") or "").lower() == "letter":
                letter_url = doc.get("url")
                break

        return Submission(
            application_number=app_no,
            submission_type=s.get("submission_type", ""),
            submission_number=sub_no,
            submission_class_code=s.get("submission_class_code"),
            submission_class_description=s.get("submission_class_code_description"),
            submission_status=s.get("submission_status"),
            submission_status_date=status_date,
            approval_letter_url=letter_url,
        )

    # ---------- DailyMed (SPL labels) ----------

    def get_current_label_text(self, application_number: str) -> Optional[str]:
        """Fetch current indications-and-usage text from DailyMed."""
        data = self._get_json(
            f"{OPENFDA_BASE}/drug/label.json",
            params={
                "search": f'openfda.application_number:"{application_number}"',
                "limit": 1,
            },
        )
        if not data or not data.get("results"):
            return None
        result = data["results"][0]
        indications = result.get("indications_and_usage", [])
        if isinstance(indications, list):
            return "\n".join(indications)
        return indications or None

    def get_label_history(self, set_id: str) -> list[dict]:
        """List all historical SPL versions for a given SPL set ID.

        Each item has version number, effective date, and SPL UUID
        that can be used to fetch the full XML.
        """
        data = self._get_json(f"{DAILYMED_BASE}/spls/{set_id}/history.json")
        if not data:
            return []
        return data.get("data", [])

    def get_label_version_text(self, spl_id: str) -> Optional[str]:
        """Fetch a specific historical SPL version's indications section."""
        # DailyMed serves XML; for simplicity we use the service's JSON
        # rendering endpoint. Extraction of the indications-and-usage
        # section is handled by the SPL parser module.
        data = self._get_json(f"{DAILYMED_BASE}/spls/{spl_id}.json")
        if not data:
            return None
        # The structured JSON endpoint varies; callers that need the
        # full XML should use get_bytes on the .xml URL instead.
        return json.dumps(data)

    # ---------- approval letters ----------

    def get_approval_letter_pdf(self, submission: Submission) -> Optional[bytes]:
        if not submission.approval_letter_url:
            return None
        return self._get_bytes(submission.approval_letter_url)

    # ---------- NDC / commercial status ----------

    def get_ndc_records(self, application_number: str) -> list[NDCRecord]:
        data = self._get_json(
            f"{OPENFDA_BASE}/drug/ndc.json",
            params={
                "search": f'application_number:"{application_number}"',
                "limit": 100,
            },
        )
        if not data or "results" not in data:
            return []

        records: list[NDCRecord] = []
        for r in data["results"]:
            records.append(
                NDCRecord(
                    ndc=r.get("product_ndc", ""),
                    application_number=r.get("application_number"),
                    marketing_start_date=_parse_date(r.get("marketing_start_date")),
                    marketing_end_date=_parse_date(r.get("marketing_end_date")),
                    marketing_category=r.get("marketing_category"),
                    labeler_name=r.get("labeler_name"),
                )
            )
        return records


def _parse_date(raw: Optional[str]) -> Optional[date]:
    if not raw:
        return None
    for fmt in ("%Y%m%d", "%Y-%m-%d"):
        try:
            return datetime.strptime(raw, fmt).date()
        except ValueError:
            continue
    return None
