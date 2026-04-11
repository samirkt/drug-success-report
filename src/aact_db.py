"""
aact_db.py

Minimal utilities for connecting to the AACT PostgreSQL database.

Environment variables supported:
    AACT_DB_HOST        default: aact-db.ctti-clinicaltrials.org
    AACT_DB_PORT        default: 5432
    AACT_DB_NAME        default: aact
    AACT_DB_USER        required unless passed explicitly
    AACT_DB_PASSWORD    required unless passed explicitly
    AACT_DB_SCHEMA      default: ctgov
    AACT_DB_SSLMODE     default: prefer

Install:
    pip install "psycopg[binary]"

Example:
    from aact_db import test_connection, fetch_all

    print(test_connection())

    rows = fetch_all("SELECT nct_id, brief_title FROM studies LIMIT 5;")
    for row in rows:
        print(row["nct_id"], row["brief_title"])
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator, Mapping, Sequence

import psycopg
from psycopg.rows import dict_row


@dataclass(frozen=True)
class AACTConfig:
    host: str = os.getenv("AACT_DB_HOST", "aact-db.ctti-clinicaltrials.org")
    port: int = int(os.getenv("AACT_DB_PORT", "5432"))
    dbname: str = os.getenv("AACT_DB_NAME", "aact")
    user: str | None = os.getenv("AACT_DB_USER")
    password: str | None = os.getenv("AACT_DB_PASSWORD")
    schema: str = os.getenv("AACT_DB_SCHEMA", "ctgov")
    sslmode: str = os.getenv("AACT_DB_SSLMODE", "prefer")

    def validate(self) -> None:
        if not self.user:
            raise ValueError(
                "Missing AACT username. Set AACT_DB_USER or pass user=..."
            )
        if not self.password:
            raise ValueError(
                "Missing AACT password. Set AACT_DB_PASSWORD or pass password=..."
            )


def load_config(
    *,
    host: str | None = None,
    port: int | None = None,
    dbname: str | None = None,
    user: str | None = None,
    password: str | None = None,
    schema: str | None = None,
    sslmode: str | None = None,
) -> AACTConfig:
    base = AACTConfig()
    return AACTConfig(
        host=host or base.host,
        port=port or base.port,
        dbname=dbname or base.dbname,
        user=user or base.user,
        password=password or base.password,
        schema=schema or base.schema,
        sslmode=sslmode or base.sslmode,
    )


def get_connection(config: AACTConfig | None = None) -> psycopg.Connection:
    """
    Open a psycopg connection and set the session search_path so
    unqualified table names like 'studies' work for local AACT restores.
    """
    config = config or load_config()
    config.validate()

    conn = psycopg.connect(
        host=config.host,
        port=config.port,
        dbname=config.dbname,
        user=config.user,
        password=config.password,
        sslmode=config.sslmode,
        row_factory=dict_row,
    )

    # AACT snapshot tables are typically under the ctgov schema.
    # Setting this search_path is convenient for unqualified table names.
    with conn.cursor() as cur:
        cur.execute(f"SET search_path TO {config.schema}, public;")

    return conn


@contextmanager
def connection(config: AACTConfig | None = None) -> Iterator[psycopg.Connection]:
    conn = get_connection(config)
    try:
        yield conn
    finally:
        conn.close()


def fetch_all(
    sql: str,
    params: Sequence[Any] | Mapping[str, Any] | None = None,
    *,
    config: AACTConfig | None = None,
) -> list[dict[str, Any]]:
    with connection(config) as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        return list(cur.fetchall())



def fetch_one(
    sql: str,
    params: Sequence[Any] | Mapping[str, Any] | None = None,
    *,
    config: AACTConfig | None = None,
) -> dict[str, Any] | None:
    with connection(config) as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        row = cur.fetchone()
        return dict(row) if row is not None else None



def execute(
    sql: str,
    params: Sequence[Any] | Mapping[str, Any] | None = None,
    *,
    config: AACTConfig | None = None,
) -> None:
    with connection(config) as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        conn.commit()



def test_connection(*, config: AACTConfig | None = None) -> dict[str, Any]:
    """
    Runs a tiny sanity check against AACT.
    """
    row = fetch_one(
        """
        SELECT
            current_database() AS database_name,
            current_user AS username,
            current_schema() AS schema_name,
            COUNT(*) AS study_count
        FROM studies;
        """,
        config=config,
    )
    if row is None:
        raise RuntimeError("AACT test query returned no rows.")
    return row


if __name__ == "__main__":
    result = test_connection()
    print("Connected successfully:")
    for k, v in result.items():
        print(f"{k}: {v}")
