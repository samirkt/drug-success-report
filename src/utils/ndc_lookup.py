"""
Local FDA lookup backed by SQLite.

One-time setup:
    from fda_local import build_db
    build_db()                  # reads dumps in FDA_DATA_DIR, writes FDA_DB

Runtime:
    from fda_local import get_drug, get_drugs_with_indications_bulk
    get_drug("pembrolizumab")
    get_drugs_with_indications_bulk([["pembrolizumab", "Keytruda"], "aspirin"])

RAM during queries is just SQLite's page cache (small). The build phase peaks
at the size of the largest JSON partition (~1 GB) since json.load is not
streaming -- run build_db once with the LLM not loaded.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
from pathlib import Path

DEFAULT_DB = Path(os.environ.get("FDA_DB", "./fda.db"))
DEFAULT_DATA_DIR = Path(os.environ.get("FDA_DATA_DIR", "./fda_data"))

_DECORATIVE = re.compile(r"[®™©°\u2018\u2019\u201C\u201D]")
_WHITESPACE = re.compile(r"\s+")


def _normalize(name):
    if not name:
        return ""
    return _WHITESPACE.sub(" ", _DECORATIVE.sub("", name)).strip().lower()


def _safe_list(x):
    if x is None:
        return []
    return x if isinstance(x, list) else [x]


def _names(rec, fields=("generic_name", "brand_name")):
    """Yield normalized names from a record. Handles top-level scalars (NDC)
    and openfda-subfield lists (labels, drugsfda) in one pass."""
    ofda = rec.get("openfda") or {}
    for k in fields:
        v = rec.get(k)
        if isinstance(v, str):
            n = _normalize(v)
            if n:
                yield n
        for v in _safe_list(ofda.get(k)):
            n = _normalize(v)
            if n:
                yield n


# ---------------------------------------------------------------------------
# Build phase (run once)
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE label      (set_id TEXT PRIMARY KEY, effective_time TEXT, indications TEXT);
CREATE TABLE label_app  (set_id TEXT, app_no TEXT);
CREATE TABLE label_name (set_id TEXT, name TEXT);
CREATE TABLE name_app   (name TEXT, app_no TEXT);
CREATE TABLE marketing  (name TEXT, start_date TEXT, end_date TEXT);
"""

_INDEXES = """
CREATE INDEX idx_label_app  ON label_app(app_no);
CREATE INDEX idx_label_name ON label_name(name);
CREATE INDEX idx_name_app   ON name_app(name);
CREATE INDEX idx_marketing  ON marketing(name);
"""


def _read_results(path):
    with open(path) as f:
        return json.load(f).get("results", [])


def build_db(data_dir=None, db_path=None):
    """Build the SQLite index from openFDA bulk dumps. Idempotent: rebuilds from scratch."""
    data_dir = Path(data_dir or DEFAULT_DATA_DIR)
    db_path = Path(db_path or DEFAULT_DB)
    if db_path.exists():
        db_path.unlink()

    conn = sqlite3.connect(db_path)
    conn.executescript("PRAGMA journal_mode = OFF; PRAGMA synchronous = OFF;")
    conn.executescript(_SCHEMA)

    label_files = sorted(data_dir.glob("drug-label-*.json"))
    if not label_files:
        raise FileNotFoundError(f"No drug-label-*.json files in {data_dir}")

    with conn:  # one transaction for the whole load
        for p in label_files:
            labels, apps, names = [], [], []
            for rec in _read_results(p):
                sid = rec.get("set_id")
                if not sid:
                    continue
                inds = rec.get("indications_and_usage")
                labels.append((sid, rec.get("effective_time"),
                               json.dumps(inds) if inds else None))
                ofda = rec.get("openfda") or {}
                for an in _safe_list(ofda.get("application_number")):
                    if an:
                        apps.append((sid, an.strip()))
                for n in _names(rec):
                    names.append((sid, n))
            conn.executemany("INSERT OR IGNORE INTO label VALUES (?,?,?)", labels)
            conn.executemany("INSERT INTO label_app VALUES (?,?)", apps)
            conn.executemany("INSERT INTO label_name VALUES (?,?)", names)

        ndc_path = data_dir / "drug-ndc-0001-of-0001.json"
        if ndc_path.exists():
            apps, mkts = [], []
            for rec in _read_results(ndc_path):
                an = rec.get("application_number")
                start = rec.get("marketing_start_date")
                end = rec.get("marketing_end_date")
                for n in _names(rec):
                    if an:
                        apps.append((n, an.strip()))
                    mkts.append((n, start, end))
            conn.executemany("INSERT INTO name_app VALUES (?,?)", apps)
            conn.executemany("INSERT INTO marketing VALUES (?,?,?)", mkts)

        drugsfda_path = data_dir / "drug-drugsfda-0001-of-0001.json"
        if drugsfda_path.exists():
            apps = []
            for rec in _read_results(drugsfda_path):
                an = rec.get("application_number")
                if not an:
                    continue
                for n in _names(rec, fields=("generic_name", "brand_name", "substance_name")):
                    apps.append((n, an.strip()))
                for prod in _safe_list(rec.get("products")):
                    bn = _normalize(prod.get("brand_name"))
                    if bn:
                        apps.append((bn, an.strip()))
            conn.executemany("INSERT INTO name_app VALUES (?,?)", apps)

    conn.executescript(_INDEXES)
    conn.close()


# ---------------------------------------------------------------------------
# Query phase (lazy read-only connection)
# ---------------------------------------------------------------------------

_conn = None


def _connect():
    global _conn
    if _conn is None:
        _conn = sqlite3.connect(f"file:{DEFAULT_DB}?mode=ro", uri=True)
    return _conn


def get_drug(drug):
    """Returns (is_commercialized, earliest_active_marketing_start_date)."""
    name = _normalize(drug)
    if not name:
        return False, None
    rows = _connect().execute(
        "SELECT start_date, end_date FROM marketing WHERE name = ?", (name,)
    ).fetchall()
    if not rows:
        return False, None
    has_active = any(end is None for _, end in rows)
    starts = [s for s, _ in rows if s]
    if not has_active or not starts:
        return False, None
    return True, min(starts)


# Two-stage resolution as a single SQL: name -> app_no -> label.
# Fallback (direct name -> label) runs only when the first returns nothing.
_Q_VIA_APP = """
SELECT DISTINCT l.set_id, l.effective_time, l.indications
FROM name_app n
JOIN label_app la ON la.app_no = n.app_no
JOIN label l      ON l.set_id  = la.set_id
WHERE n.name = ?
"""

_Q_FALLBACK = """
SELECT DISTINCT l.set_id, l.effective_time, l.indications
FROM label_name ln
JOIN label l ON l.set_id = ln.set_id
WHERE ln.name = ?
"""


def _resolve(name):
    conn = _connect()
    rows = conn.execute(_Q_VIA_APP, (name,)).fetchall()
    if not rows:
        rows = conn.execute(_Q_FALLBACK, (name,)).fetchall()
    return rows


def get_drugs_with_indications_bulk(drug_groups):
    """
    Each input item is a string or a list of candidate terms for one drug.
    Tries terms in order until one yields indications. Returns dict keyed by
    the matching term (or the first term if none matched).
    """
    if not isinstance(drug_groups, list):
        raise TypeError("drug_groups must be a list")

    output, cache = {}, {}
    for item in drug_groups:
        if isinstance(item, str):
            terms = [item]
        else:
            terms = [t for t in _safe_list(item) if isinstance(t, str) and t.strip()]
        if not terms:
            continue

        selected = None
        for term in terms:
            if term not in cache:
                rows = _resolve(_normalize(term))
                seen, indications, sources = set(), [], []
                for sid, eff, inds_json in rows:
                    sources.append({"set_id": sid, "effective_time": eff})
                    for txt in (json.loads(inds_json) if inds_json else []):
                        if txt and txt not in seen:
                            seen.add(txt)
                            indications.append(txt)
                cache[term] = {
                    "query": term,
                    "indications": indications,
                    "indication_count": len(indications),
                    "indication_sources": sources,
                }
            if cache[term]["indication_count"] > 0:
                selected = cache[term]
                break
        if selected is None:
            selected = cache[terms[0]]
        output[selected["query"]] = selected
    return output