"""Standalone SQLite cache for ADMET predictions.

Independent of ``pipeline.knowledge_cache``: this module deliberately
imports nothing from ``pipeline.*`` or ``utils.*`` so the ``pipeline.admet``
package is shippable in isolation. WAL mode + a single connection guarded
by a ``threading.Lock`` matches the rest of the project's SQLite caches.

Schema is keyed by ``(sha256(smiles), model_version)`` so bumping the
``admet_ai`` version invalidates rows cleanly without dropping the file.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import threading
from hashlib import sha256
from pathlib import Path
from typing import Iterable, Optional

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS admet_predictions (
    smiles_sha256   TEXT NOT NULL,
    smiles          TEXT NOT NULL,
    model_version   TEXT NOT NULL,
    prediction_json TEXT NOT NULL,
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (smiles_sha256, model_version)
);
CREATE INDEX IF NOT EXISTS idx_admet_smiles ON admet_predictions(smiles);
"""


class AdmetCache:
    """SQLite-backed cache for ADMET prediction dicts."""

    def __init__(self, path: Path | str):
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    @staticmethod
    def _key(smiles: str) -> str:
        return sha256(smiles.encode("utf-8")).hexdigest()

    def get_many(
        self,
        smiles_list: Iterable[str],
        model_version: str,
    ) -> dict[str, dict[str, Optional[float]]]:
        keys = {self._key(s): s for s in smiles_list if s}
        if not keys:
            return {}
        placeholders = ",".join("?" * len(keys))
        sql = (
            "SELECT smiles_sha256, prediction_json FROM admet_predictions "
            f"WHERE model_version = ? AND smiles_sha256 IN ({placeholders})"
        )
        try:
            with self._lock:
                rows = self._conn.execute(sql, [model_version, *keys.keys()]).fetchall()
        except sqlite3.Error as exc:
            logger.warning("AdmetCache read failed: %s — treating as miss.", exc)
            return {}
        return {keys[k]: json.loads(j) for k, j in rows if k in keys}

    def put_many(
        self,
        items: dict[str, dict[str, Optional[float]]],
        model_version: str,
    ) -> None:
        if not items:
            return
        rows = [
            (self._key(s), s, model_version, json.dumps(pred))
            for s, pred in items.items()
        ]
        try:
            with self._lock:
                self._conn.executemany(
                    "INSERT OR REPLACE INTO admet_predictions "
                    "(smiles_sha256, smiles, model_version, prediction_json) "
                    "VALUES (?, ?, ?, ?)",
                    rows,
                )
                self._conn.commit()
        except sqlite3.Error as exc:
            logger.warning("AdmetCache write failed: %s — skipping persist.", exc)

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.close()
            except sqlite3.Error:
                pass


__all__ = ["AdmetCache"]
