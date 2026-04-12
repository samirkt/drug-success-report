"""Temporary on-disk cache for raw AACT query results.

Avoids re-hitting the ClinicalTrials.gov AACT mirror on repeated pipeline runs
with the same ingestion configuration. Cached payloads are:
  - raw row dicts returned by the main interventional-trials query
  - primary-outcome p-value maps returned by the single-arm p-value query

Keys are SHA-256 hashes of the query parameters. Values are pickled Python
objects. The cache is a single pickle file loaded at init and rewritten on
each put; safe for the single-process pipeline runner.
"""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
import threading
from pathlib import Path
from typing import Any

from .models import TrialPValue

logger = logging.getLogger(__name__)


class AACTCache:
    """Pickle-backed cache of raw AACT fetch results."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._lock = threading.Lock()
        self._store: dict[str, Any] = self._load()

    def _load(self) -> dict[str, Any]:
        if not self._path.exists():
            return {}
        try:
            with self._path.open("rb") as f:
                data = pickle.load(f)
            if not isinstance(data, dict):
                logger.warning("AACT cache at %s is not a dict; ignoring.", self._path)
                return {}
            return data
        except (pickle.UnpicklingError, EOFError, OSError) as exc:
            logger.warning("Could not load AACT cache from %s: %s", self._path, exc)
            return {}

    def _flush(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        with tmp.open("wb") as f:
            pickle.dump(self._store, f, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(self._path)

    @staticmethod
    def make_rows_key(source: str, filters: dict, max_trials: int | None) -> str:
        payload = json.dumps(
            {"kind": "rows", "source": source, "filters": filters, "max_trials": max_trials},
            sort_keys=True, default=str,
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    @staticmethod
    def make_pvalues_key(nct_ids: list[str]) -> str:
        payload = json.dumps({"kind": "pvalues", "nct_ids": sorted(set(nct_ids))})
        return hashlib.sha256(payload.encode()).hexdigest()

    def get_rows(self, key: str) -> list[dict] | None:
        with self._lock:
            return self._store.get(key)

    def put_rows(self, key: str, rows: list[dict]) -> None:
        with self._lock:
            self._store[key] = rows
            self._flush()

    def get_pvalues(self, key: str) -> dict[str, list[TrialPValue]] | None:
        with self._lock:
            return self._store.get(key)

    def put_pvalues(self, key: str, p_values: dict[str, list[TrialPValue]]) -> None:
        with self._lock:
            self._store[key] = p_values
            self._flush()
