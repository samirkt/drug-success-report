"""ICD-10-CM lookup against the NLM Clinical Tables API.

Wraps a single endpoint and provides an on-disk sqlite KV cache so
re-runs over the same indication strings don't re-hit the network.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from pathlib import Path
from typing import Optional

import requests

logger = logging.getLogger(__name__)

_BASE_URL = "https://clinicaltables.nlm.nih.gov/api/icd10cm/v3/search"
_DEFAULT_TIMEOUT = 10.0


def get_icd_from_nih(disease_name: str, *, timeout: float = _DEFAULT_TIMEOUT) -> Optional[list[str]]:
    """Return the list of ICD-10-CM codes matching ``disease_name``.

    Returns None when NLM has no matches. Raises ``requests.RequestException``
    on transport errors so callers can decide whether to log + skip.
    """
    response = requests.get(
        _BASE_URL,
        params={"sf": "code,name", "terms": disease_name},
        timeout=timeout,
    )
    response.raise_for_status()
    data = response.json()  # [count, codes, _, results]
    if not data or data[0] == 0:
        return None
    return list(data[1])


class IcdCache:
    """Tiny sqlite KV cache: indication string -> JSON-encoded list[str] | null."""

    def __init__(self, path: str | Path):
        self._path = str(path)
        self._lock = threading.Lock()
        Path(self._path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS icd_cache ("
                "  disease_name TEXT PRIMARY KEY,"
                "  codes_json   TEXT"
                ")"
            )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._path)

    def get(self, disease_name: str) -> tuple[bool, Optional[list[str]]]:
        """Return (hit, value). ``value`` is None either on miss or on a
        cached negative lookup; ``hit`` distinguishes the two.
        """
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT codes_json FROM icd_cache WHERE disease_name = ?",
                (disease_name,),
            ).fetchone()
        if row is None:
            return False, None
        return True, json.loads(row[0]) if row[0] is not None else None

    def put(self, disease_name: str, codes: Optional[list[str]]) -> None:
        payload = json.dumps(codes) if codes is not None else None
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO icd_cache (disease_name, codes_json) VALUES (?, ?)",
                (disease_name, payload),
            )
            conn.commit()


def get_icd_cached(
    disease_name: str,
    cache: Optional[IcdCache],
    *,
    timeout: float = _DEFAULT_TIMEOUT,
) -> Optional[list[str]]:
    """Cache-aware wrapper around :func:`get_icd_from_nih`."""
    if cache is not None:
        hit, value = cache.get(disease_name)
        if hit:
            return value
    codes = get_icd_from_nih(disease_name, timeout=timeout)
    if cache is not None:
        cache.put(disease_name, codes)
    return codes
