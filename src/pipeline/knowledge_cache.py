"""Persistent SQLite-backed cache for LLM classification and adjudication results."""

import hashlib
import json
import logging
import sqlite3
import threading
from datetime import date, datetime, timezone
from pathlib import Path

from .models import CandidateAttributes, CandidateOutcome, CandidateOutcomeRecord


def _parse_iso_date(raw: str | None) -> date | None:
    if not raw:
        return None
    try:
        return date.fromisoformat(raw)
    except ValueError:
        return None

logger = logging.getLogger(__name__)

_DDL = """
CREATE TABLE IF NOT EXISTS classification_cache (
    cache_key           TEXT PRIMARY KEY,
    drug_modality       TEXT NOT NULL,
    disease_area        TEXT NOT NULL,
    modality_confidence REAL NOT NULL,
    disease_confidence  REAL NOT NULL,
    reasoning           TEXT NOT NULL,
    created_at          TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS adjudication_cache (
    cache_key        TEXT PRIMARY KEY,
    outcome          TEXT NOT NULL,
    confidence       REAL NOT NULL,
    reasoning        TEXT NOT NULL,
    evidence_sources TEXT NOT NULL,
    created_at       TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS llm_json_cache (
    cache_key     TEXT PRIMARY KEY,
    response_json TEXT NOT NULL,
    created_at    TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS fda_adjudication_cache (
    cache_key            TEXT PRIMARY KEY,
    outcome              TEXT NOT NULL,
    confidence           REAL NOT NULL,
    reasoning            TEXT NOT NULL,
    evidence_sources     TEXT NOT NULL,
    approval_date        TEXT,
    commercialization_date TEXT,
    created_at           TEXT NOT NULL
);
"""


class KnowledgeCache:
    """
    Thread-safe SQLite cache for classification and adjudication LLM results.

    Each thread gets its own connection via threading.local(). WAL mode allows
    concurrent reads from the parallel 3a/3b executor threads.
    """

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = str(db_path)
        self._local = threading.local()
        # Initialize schema on the calling thread
        conn = self._conn
        conn.executescript(_DDL)
        conn.commit()

    @property
    def _conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            conn = sqlite3.connect(self._db_path, check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.row_factory = sqlite3.Row
            self._local.conn = conn
        return self._local.conn

    def close(self) -> None:
        if hasattr(self._local, "conn") and self._local.conn is not None:
            self._local.conn.close()
            self._local.conn = None

    # ------------------------------------------------------------------
    # Key construction
    # ------------------------------------------------------------------

    @staticmethod
    def make_classification_key(drug_name: str, indication: str) -> str:
        payload = f"{drug_name.lower()}|{indication.lower()}"
        return hashlib.sha256(payload.encode()).hexdigest()

    @staticmethod
    def make_adjudication_key(drug_name: str, indication: str, highest_phase: str) -> str:
        payload = f"{drug_name.lower()}|{indication.lower()}|{highest_phase.lower()}"
        return hashlib.sha256(payload.encode()).hexdigest()

    # ------------------------------------------------------------------
    # Classification cache
    # ------------------------------------------------------------------

    def get_attributes(self, cache_key: str, candidate_id: str) -> CandidateAttributes | None:
        row = self._conn.execute(
            "SELECT * FROM classification_cache WHERE cache_key = ?", (cache_key,)
        ).fetchone()
        if row is None:
            return None
        return CandidateAttributes(
            candidate_id=candidate_id,
            drug_modality=row["drug_modality"],
            disease_area=row["disease_area"],
            modality_confidence=row["modality_confidence"],
            disease_confidence=row["disease_confidence"],
            reasoning=row["reasoning"],
        )

    def put_attributes(self, cache_key: str, attrs: CandidateAttributes) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            """
            INSERT OR REPLACE INTO classification_cache
                (cache_key, drug_modality, disease_area, modality_confidence,
                 disease_confidence, reasoning, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                cache_key,
                attrs.drug_modality,
                attrs.disease_area,
                attrs.modality_confidence,
                attrs.disease_confidence,
                attrs.reasoning,
                now,
            ),
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Migrations
    # ------------------------------------------------------------------

    def migrate_disease_areas(self) -> dict[str, int]:
        """Normalize all disease_area values in the classification cache.

        Applies the same normalize_disease_area() logic used at classification
        time to fix historical entries that predate the rigid taxonomy.

        Returns a dict of {old_value: count} for entries that were updated.
        """
        from pipeline.stages.classification import normalize_disease_area

        rows = self._conn.execute(
            "SELECT cache_key, disease_area FROM classification_cache"
        ).fetchall()

        updated: dict[str, int] = {}
        for row in rows:
            old = row["disease_area"]
            new = normalize_disease_area(old)
            if new != old:
                self._conn.execute(
                    "UPDATE classification_cache SET disease_area = ? WHERE cache_key = ?",
                    (new, row["cache_key"]),
                )
                updated[old] = updated.get(old, 0) + 1

        self._conn.commit()
        total = sum(updated.values())
        logger.info(
            "migrate_disease_areas: updated %d / %d rows (%d distinct old values)",
            total, len(rows), len(updated),
        )
        for old_val, count in sorted(updated.items(), key=lambda kv: -kv[1]):
            logger.info("  %r → %r  (%d rows)", old_val, normalize_disease_area(old_val), count)

        return updated

    # ------------------------------------------------------------------
    # Adjudication cache
    # ------------------------------------------------------------------

    def get_outcome(self, cache_key: str, candidate_id: str) -> CandidateOutcomeRecord | None:
        row = self._conn.execute(
            "SELECT * FROM adjudication_cache WHERE cache_key = ?", (cache_key,)
        ).fetchone()
        if row is None:
            return None
        return CandidateOutcomeRecord(
            candidate_id=candidate_id,
            outcome=CandidateOutcome(row["outcome"]),
            confidence=row["confidence"],
            reasoning=row["reasoning"],
            evidence_sources=json.loads(row["evidence_sources"]),
        )

    def put_outcome(self, cache_key: str, record: CandidateOutcomeRecord) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            """
            INSERT OR REPLACE INTO adjudication_cache
                (cache_key, outcome, confidence, reasoning, evidence_sources, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                cache_key,
                record.outcome.value,
                record.confidence,
                record.reasoning,
                json.dumps(record.evidence_sources),
                now,
            ),
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Generic LLM-JSON cache (used by stages/adjudication_fda.py for
    # indication extract/match calls that return arbitrary JSON payloads)
    # ------------------------------------------------------------------

    @staticmethod
    def make_llm_json_key(*parts: str) -> str:
        payload = "|".join(p.lower() for p in parts)
        return hashlib.sha256(payload.encode()).hexdigest()

    def get_llm_json(self, cache_key: str) -> dict | None:
        row = self._conn.execute(
            "SELECT response_json FROM llm_json_cache WHERE cache_key = ?",
            (cache_key,),
        ).fetchone()
        if row is None:
            return None
        try:
            return json.loads(row["response_json"])
        except json.JSONDecodeError:
            return None

    def put_llm_json(self, cache_key: str, payload: dict) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            "INSERT OR REPLACE INTO llm_json_cache (cache_key, response_json, created_at) VALUES (?, ?, ?)",
            (cache_key, json.dumps(payload), now),
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # FDA-timeline adjudication cache (parallel to adjudication_cache so
    # the candidate summary can compare outcomes from both methods when
    # both have been run against the same KnowledgeCache).
    # ------------------------------------------------------------------

    def get_fda_outcome(self, cache_key: str, candidate_id: str) -> CandidateOutcomeRecord | None:
        row = self._conn.execute(
            "SELECT * FROM fda_adjudication_cache WHERE cache_key = ?", (cache_key,)
        ).fetchone()
        if row is None:
            return None
        return CandidateOutcomeRecord(
            candidate_id=candidate_id,
            outcome=CandidateOutcome(row["outcome"]),
            confidence=row["confidence"],
            reasoning=row["reasoning"],
            evidence_sources=json.loads(row["evidence_sources"]),
            approval_date=_parse_iso_date(row["approval_date"]),
            commercialization_date=_parse_iso_date(row["commercialization_date"]),
        )

    def put_fda_outcome(self, cache_key: str, record: CandidateOutcomeRecord) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            """
            INSERT OR REPLACE INTO fda_adjudication_cache
                (cache_key, outcome, confidence, reasoning, evidence_sources,
                 approval_date, commercialization_date, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                cache_key,
                record.outcome.value,
                record.confidence,
                record.reasoning,
                json.dumps(record.evidence_sources),
                record.approval_date.isoformat() if record.approval_date else None,
                record.commercialization_date.isoformat() if record.commercialization_date else None,
                now,
            ),
        )
        self._conn.commit()
