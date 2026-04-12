"""
Stage 1: Trial Ingestion

Fetches trial records from ClinicalTrials.gov (via AACT PostgreSQL mirror or
REST API) and normalizes them into a TrialTable.

Each row in the result represents a unique (study, drug, condition) triple.
A single NCT study that lists multiple drugs or multiple conditions will
produce multiple RawTrial rows — one per intervention–condition pairing —
so the clustering stage can treat each pairing as an independent candidate.

Data source notes:
  "aact"  — queries the AACT PostgreSQL mirror (primary, implemented)
  "api"   — ClinicalTrials.gov REST v2 (not yet implemented)

Cap safeguard:
  max_trials limits the number of rows (intervention–condition pairs) fetched.
  The cap is applied in SQL via LIMIT so no excess data crosses the network.
  Default is 500. Pass max_trials=None to disable for a full pipeline run.
"""

from __future__ import annotations

import logging
from typing import Any

import psycopg

from aact_db import AACTConfig, connection, load_config
from ..aact_cache import AACTCache
from ..models import RawTrial, TrialPhase, TrialPValue, TrialStatus, TrialTable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lookup tables — module-level constants, built once
# ---------------------------------------------------------------------------

# Combined phases map to the HIGHER phase (e.g. Phase 1/2 → Phase 2).
_PHASE_MAP: dict[str, TrialPhase] = {
    "PHASE 1":         TrialPhase.PHASE_1,
    "PHASE1":          TrialPhase.PHASE_1,
    "EARLY PHASE 1":   TrialPhase.PHASE_1,
    "PHASE 1/PHASE 2": TrialPhase.PHASE_2,   # combined → higher
    "PHASE 1/2":       TrialPhase.PHASE_2,   # combined → higher
    "PHASE 2":         TrialPhase.PHASE_2,
    "PHASE2":          TrialPhase.PHASE_2,
    "PHASE 2/PHASE 3": TrialPhase.PHASE_3,   # combined → higher
    "PHASE 2/3":       TrialPhase.PHASE_3,   # combined → higher
    "PHASE 3":         TrialPhase.PHASE_3,
    "PHASE3":          TrialPhase.PHASE_3,
    "PHASE 4":         TrialPhase.PHASE_4,
    "PHASE4":          TrialPhase.PHASE_4,
    "N/A":             TrialPhase.NOT_APPLICABLE,
    "NA":              TrialPhase.NOT_APPLICABLE,
    "NOT APPLICABLE":  TrialPhase.NOT_APPLICABLE,
}

_STATUS_MAP: dict[str, TrialStatus] = {
    "RECRUITING":              TrialStatus.RECRUITING,
    # Pre-open trials are treated as recruiting-class for funnel purposes.
    "NOT YET RECRUITING":      TrialStatus.RECRUITING,
    "ENROLLING BY INVITATION": TrialStatus.RECRUITING,
    "COMPLETED":               TrialStatus.COMPLETED,
    "TERMINATED":              TrialStatus.TERMINATED,
    "WITHDRAWN":               TrialStatus.WITHDRAWN,
    "ACTIVE, NOT RECRUITING":  TrialStatus.ACTIVE_NOT_RECRUITING,
    "ACTIVE_NOT_RECRUITING":   TrialStatus.ACTIVE_NOT_RECRUITING,  # REST API style
    "SUSPENDED":               TrialStatus.SUSPENDED,
}

# Hardcoded row-level exclusion rules applied during ingestion.
_HARDCODED_ROW_FILTER_RULES: list[dict[str, Any]] = [
    {
        "field": "indication",
        "op": "contains",
        "value": "healthy",
        "case_sensitive": False,
    }
]

# ---------------------------------------------------------------------------
# Base SQL — shared across any future keyword/filter variants
# ---------------------------------------------------------------------------

_BASE_SQL = """
SELECT
    s.nct_id,
    s.brief_title,
    s.phase,
    s.overall_status,
    s.start_date,
    s.completion_date,
    i.name        AS intervention,
    c.name        AS indication,
    (
        SELECT sp.name
        FROM   sponsors sp
        WHERE  sp.nct_id = s.nct_id
          AND  sp.lead_or_collaborator = 'lead'
        ORDER  BY sp.id
        LIMIT  1
    ) AS sponsor,
    COALESCE(
        ARRAY(
            SELECT DISTINCT bi.mesh_term
            FROM   browse_interventions bi
            WHERE  bi.nct_id = s.nct_id
              AND  bi.mesh_term IS NOT NULL
        ),
        ARRAY[]::TEXT[]
    ) AS mesh_intervention_terms,
    COALESCE(
        ARRAY(
            SELECT DISTINCT bc.mesh_term
            FROM   browse_conditions bc
            WHERE  bc.nct_id = s.nct_id
              AND  bc.mesh_term IS NOT NULL
        ),
        ARRAY[]::TEXT[]
    ) AS mesh_condition_terms,
    COALESCE(
        ARRAY(
            SELECT DISTINCT mt.tree_number
            FROM   browse_conditions bc
            JOIN   mesh_terms mt ON mt.downcase_mesh_term = bc.downcase_mesh_term
            WHERE  bc.nct_id = s.nct_id
              AND  mt.tree_number IS NOT NULL
              AND  (mt.tree_number LIKE 'C%%' OR mt.tree_number LIKE 'F%%')
        ),
        ARRAY[]::TEXT[]
    ) AS mesh_condition_tree_numbers
FROM       studies       s
JOIN       interventions i ON i.nct_id = s.nct_id
                           AND upper(i.intervention_type) = 'DRUG'
JOIN       conditions    c ON c.nct_id = s.nct_id
WHERE upper(s.study_type) = 'INTERVENTIONAL'
"""


class TrialIngestionStage:
    """
    Pulls trials from ClinicalTrials.gov and extracts structured metadata.

    Each output RawTrial represents one (study, drug, condition) pair.
    A study with N drugs and M conditions produces up to N×M RawTrial rows.

    Inputs:  data source config (AACT credentials or API params)
    Outputs: TrialTable
    """

    def __init__(
        self,
        source: str = "aact",
        filters: dict | None = None,
        max_trials: int | None = 500,
        aact_config: AACTConfig | None = None,
        filter_single_arm: bool = False,
        ct_cache: AACTCache | None = None,
    ):
        """
        Args:
            source:             "aact" (PostgreSQL mirror) or "api" (REST, not yet implemented)
            filters:            query filters — supported keys:
                                  "mesh_term"  exact MeSH term, e.g. "Peptides"
                                  "keyword"    ILIKE substring match on MeSH terms
            max_trials:         maximum rows (intervention–condition pairs) to fetch.
                                Enforced via SQL LIMIT. None = no cap (full run).
            aact_config:        explicit AACT connection config; None = read from env vars
            filter_single_arm:  when True, drop basket and umbrella trials, keeping only
                                NCT IDs with exactly one unique intervention and one unique
                                condition.
            ct_cache:           optional AACTCache to reuse raw fetch results across runs.
        """
        self.source = source
        self.filters = filters or {}
        self.max_trials = max_trials
        self.aact_config = aact_config
        self.filter_single_arm = filter_single_arm
        self.ct_cache = ct_cache

    def run(self) -> TrialTable:
        """Fetch and normalize trials. Returns a populated TrialTable."""
        raw_records = self._fetch(self.source, self.filters)

        # Always classify single-arm status (needed for p-value enrichment)
        single_arm_map = self._classify_single_arm(raw_records)

        if self.filter_single_arm:
            raw_records, stats = self._apply_single_arm_filter(raw_records, single_arm_map)
            logger.info(
                "Single-arm filter: kept %d / %d NCT IDs  "
                "(%d basket, %d umbrella, %d multi-arm dropped, %d rows dropped)",
                stats["kept"], stats["total"],
                stats["basket_dropped"], stats["umbrella_dropped"],
                stats["multi_dropped"], stats["rows_dropped"],
            )
            print(
                f"  Single-arm filter:  {stats['rows_dropped']} rows dropped "
                f"({stats['basket_dropped']} basket, {stats['umbrella_dropped']} umbrella, "
                f"{stats['multi_dropped']} multi-arm trials)"
            )
            print(
                f"  {100*stats['kept'] / stats['total']:0.1f}% of candidates kept"
            )
        raw_records, rows_dropped = self._apply_row_filter_rules(raw_records, _HARDCODED_ROW_FILTER_RULES)
        logger.info(
            "Hardcoded row filters: dropped %d rows, kept %d rows",
            rows_dropped, len(raw_records),
        )
        print(f"  Row filters:        {rows_dropped} rows dropped")

        # Batch-fetch p-values for single-arm trials (AACT only)
        single_arm_ids = [nct_id for nct_id, is_sa in single_arm_map.items() if is_sa]
        p_value_map: dict[str, list[TrialPValue]] = {}
        if self.source == "aact" and single_arm_ids:
            p_value_map = self._fetch_p_values(single_arm_ids)

        trials = []
        for r in raw_records:
            trial = self._normalize(r)
            nct_id = r.get("nct_id", "")
            trial.is_single_arm = single_arm_map.get(nct_id, False)
            trial.primary_p_values = p_value_map.get(nct_id, [])
            trials.append(trial)
        return TrialTable(trials=trials)

    def _apply_row_filter_rules(
        self,
        rows: list[dict],
        rules: list[dict[str, Any]],
    ) -> tuple[list[dict], int]:
        """
        Exclude rows that match any configured rule.

        A row is dropped when at least one rule matches.
        """
        kept_rows = [row for row in rows if not any(self._row_matches_rule(row, rule) for rule in rules)]
        return kept_rows, len(rows) - len(kept_rows)

    def _row_matches_rule(self, row: dict, rule: dict[str, Any]) -> bool:
        """Return True when a row matches a single rule."""
        field = (rule.get("field") or "").strip()
        op = (rule.get("op") or "contains").strip().lower()
        value = rule.get("value")
        case_sensitive = bool(rule.get("case_sensitive", False))

        if not field or value is None:
            return False

        candidate = row.get(field)
        if candidate is None:
            return False

        candidate_str = str(candidate)
        value_str = str(value)
        if not case_sensitive:
            candidate_str = candidate_str.lower()
            value_str = value_str.lower()

        if op == "contains":
            return value_str in candidate_str

        logger.warning("Unsupported row filter op=%r for field=%r; skipping rule.", op, field)
        return False

    # ------------------------------------------------------------------
    # Single-arm filter
    # ------------------------------------------------------------------

    def _classify_single_arm(self, rows: list[dict]) -> dict[str, bool]:
        """
        Return {nct_id: is_single_arm} for every NCT ID in rows.
        Single-arm = exactly 1 unique intervention AND 1 unique condition.
        Always runs (regardless of filter_single_arm flag).
        """
        from collections import defaultdict

        nct_rows: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            nct_id = row.get("nct_id")
            if nct_id:
                nct_rows[nct_id].append(row)
        return {
            nct_id: (
                len({r.get("intervention") for r in group}) == 1
                and len({r.get("indication") for r in group}) == 1
            )
            for nct_id, group in nct_rows.items()
        }

    def _apply_single_arm_filter(
        self, rows: list[dict], single_arm_map: dict[str, bool]
    ) -> tuple[list[dict], dict]:
        """
        Filter rows to only single-arm NCT IDs. Returns (kept_rows, stats).
        Called only when filter_single_arm=True.
        """
        from collections import defaultdict

        kept_rows = [r for r in rows if single_arm_map.get(r["nct_id"], False)]
        total = len(single_arm_map)
        kept = sum(single_arm_map.values())

        nct_rows: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            nct_rows[row["nct_id"]].append(row)

        basket_dropped = umbrella_dropped = multi_dropped = 0
        for nct_id, group in nct_rows.items():
            if single_arm_map[nct_id]:
                continue
            n_i = len({r["intervention"] for r in group})
            n_c = len({r["indication"] for r in group})
            if n_i == 1:
                basket_dropped += 1
            elif n_c == 1:
                umbrella_dropped += 1
            else:
                multi_dropped += 1

        stats = {
            "total": total,
            "kept": kept,
            "basket_dropped": basket_dropped,
            "umbrella_dropped": umbrella_dropped,
            "multi_dropped": multi_dropped,
            "rows_dropped": len(rows) - len(kept_rows),
        }
        return kept_rows, stats

    def _fetch_p_values(
        self, single_arm_nct_ids: list[str]
    ) -> dict[str, list[TrialPValue]]:
        """
        Batch-fetch primary-outcome p-values from AACT for single-arm trials.
        Returns {nct_id: [TrialPValue, ...]}. Excludes rows where p_value IS NULL.
        """
        if not single_arm_nct_ids:
            return {}

        cache_key: str | None = None
        if self.ct_cache is not None:
            cache_key = AACTCache.make_pvalues_key(single_arm_nct_ids)
            cached = self.ct_cache.get_pvalues(cache_key)
            if cached is not None:
                logger.info(
                    "AACT cache hit: p-values for %d NCT IDs served from %s",
                    len(cached), cache_key[:12],
                )
                return cached

        sql = """
            SELECT
                o.nct_id,
                o.title             AS outcome_title,
                oa.p_value,
                oa.p_value_description,
                oa.method           AS statistical_method,
                oa.param_type,
                oa.param_value,
                oa.ci_lower_limit,
                oa.ci_upper_limit
            FROM outcomes           o
            JOIN outcome_analyses  oa ON oa.outcome_id = o.id
                                     AND oa.nct_id     = o.nct_id
            WHERE o.nct_id          = ANY(%(nct_ids)s)
              AND o.outcome_type    = 'PRIMARY'
              AND oa.p_value       IS NOT NULL
        """
        result: dict[str, list[TrialPValue]] = {}
        try:
            with connection(self.aact_config) as conn:
                with conn.cursor() as cur:
                    cur.execute(sql, {"nct_ids": single_arm_nct_ids})
                    for row in (dict(r) for r in cur.fetchall()):
                        pv = TrialPValue(
                            nct_id=row["nct_id"],
                            outcome_title=row.get("outcome_title") or "",
                            p_value=float(row["p_value"]),
                            p_value_description=row.get("p_value_description"),
                            statistical_method=row.get("statistical_method"),
                            param_type=row.get("param_type"),
                            param_value=float(row["param_value"]) if row.get("param_value") is not None else None,
                            ci_lower_limit=float(row["ci_lower_limit"]) if row.get("ci_lower_limit") is not None else None,
                            ci_upper_limit=float(row["ci_upper_limit"]) if row.get("ci_upper_limit") is not None else None,
                        )
                        result.setdefault(pv.nct_id, []).append(pv)
        except psycopg.OperationalError as exc:
            logger.warning("Could not fetch p-values from AACT: %s", exc)
        logger.info(
            "P-value fetch: found data for %d / %d single-arm NCT IDs",
            len(result), len(single_arm_nct_ids),
        )

        if self.ct_cache is not None and cache_key is not None:
            self.ct_cache.put_pvalues(cache_key, result)

        return result

    # ------------------------------------------------------------------
    # Source dispatch
    # ------------------------------------------------------------------

    def _fetch(self, source: str, filters: dict) -> list[dict]:
        """Dispatch to the appropriate data source."""
        if source == "aact":
            return self._fetch_aact(filters)
        elif source == "api":
            return self._fetch_api(filters)
        else:
            raise ValueError(f"Unknown data source: {source!r}. Use 'aact' or 'api'.")

    # ------------------------------------------------------------------
    # AACT source
    # ------------------------------------------------------------------

    def _fetch_aact(self, filters: dict) -> list[dict[str, Any]]:
        """
        Query the AACT PostgreSQL mirror.

        Returns one dict per (study, drug, condition) combination so that
        multi-drug or multi-condition studies are fully expanded.
        """
        cache_key: str | None = None
        if self.ct_cache is not None:
            cache_key = AACTCache.make_rows_key(self.source, filters, self.max_trials)
            cached = self.ct_cache.get_rows(cache_key)
            if cached is not None:
                logger.info("AACT cache hit: %d rows served from %s", len(cached), cache_key[:12])
                return cached

        sql, params = self._build_aact_sql(filters)

        if self.max_trials is not None:
            logger.warning(
                "Trial cap active: fetching at most %d rows (intervention–condition pairs). "
                "Set max_trials=None to disable for a full pipeline run.",
                self.max_trials,
            )

        try:
            with connection(self.aact_config) as conn:
                with conn.cursor() as cur:
                    cur.execute(sql, params)
                    rows = [dict(r) for r in cur.fetchall()]
        except psycopg.OperationalError as exc:
            raise RuntimeError(
                "Could not connect to the AACT database. "
                "Verify that AACT_DB_USER and AACT_DB_PASSWORD are set, "
                "or pass an explicit aact_config."
            ) from exc

        if self.max_trials is not None and len(rows) >= self.max_trials:
            logger.warning(
                "Fetch returned %d rows and hit the max_trials cap (%d). "
                "Results are truncated — increase max_trials or set it to None "
                "when ready for a full run.",
                len(rows), self.max_trials,
            )

        logger.info("Fetched %d intervention–condition rows from AACT.", len(rows))

        if self.ct_cache is not None and cache_key is not None:
            self.ct_cache.put_rows(cache_key, rows)

        return rows

    def _build_aact_sql(self, filters: dict) -> tuple[str, dict]:
        """
        Construct the parameterized AACT query based on active filters.
        Returns (sql_string, params_dict).
        """
        clauses: list[str] = [_BASE_SQL]
        params: dict[str, Any] = {}

        mesh_term = filters.get("mesh_term")
        keyword = filters.get("keyword")

        if mesh_term:
            clauses.append(
                """
                AND EXISTS (
                    SELECT 1 FROM browse_interventions bi
                    WHERE  bi.nct_id = s.nct_id
                      AND  bi.mesh_term = %(mesh_term)s
                )
                """
            )
            params["mesh_term"] = mesh_term
        if keyword:
            clauses.append(
                """
                AND (
                    s.brief_title ILIKE ANY(%(title_patterns)s)
                    OR s.official_title ILIKE ANY(%(title_patterns)s)
                    OR EXISTS (
                        SELECT 1
                        FROM brief_summaries bs
                        WHERE bs.nct_id = s.nct_id
                        AND bs.description ILIKE ANY(%(summary_patterns)s)
                    )
                )
                """
            )
            params["title_patterns"] = ["%peptide%", "%peptides%"]
            params["summary_patterns"] = ["%peptide%", "%peptides%"]

        clauses.append("ORDER BY s.nct_id, i.id, c.id")

        if self.max_trials is not None:
            clauses.append("LIMIT %(max_trials)s")
            params["max_trials"] = self.max_trials

        return "\n".join(clauses), params

    # ------------------------------------------------------------------
    # REST API source (not yet implemented)
    # ------------------------------------------------------------------

    def _fetch_api(self, filters: dict) -> list[dict]:
        """ClinicalTrials.gov REST API v2 — not yet implemented."""
        raise NotImplementedError(
            "REST API source is not yet implemented. Use source='aact'."
        )

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------

    def _normalize(self, raw: dict) -> RawTrial:
        """
        Map an AACT result row to a RawTrial dataclass.

        Expected keys (from _fetch_aact SQL aliases):
            nct_id, brief_title, phase, overall_status,
            intervention, indication, sponsor
        """
        return RawTrial(
            nct_id=raw.get("nct_id") or "",
            title=raw.get("brief_title") or "",
            intervention=raw.get("intervention") or "",
            indication=raw.get("indication") or "",
            sponsor=raw.get("sponsor") or "",
            phase=self._parse_phase(raw.get("phase") or ""),
            status=self._parse_status(raw.get("overall_status") or ""),
            raw_data=dict(raw),
            start_date=raw.get("start_date"),
            completion_date=raw.get("completion_date"),
            mesh_intervention_terms=list(raw.get("mesh_intervention_terms") or []),
            mesh_condition_terms=list(raw.get("mesh_condition_terms") or []),
            mesh_condition_tree_numbers=list(raw.get("mesh_condition_tree_numbers") or []),
        )

    def _parse_phase(self, phase_str: str) -> TrialPhase:
        """
        Convert a free-text phase string to a TrialPhase enum.

        Combined phases (e.g. "Phase 1/Phase 2") map to the HIGHER phase.
        Unrecognized values map to TrialPhase.UNKNOWN.
        """
        return _PHASE_MAP.get(phase_str.upper().strip(), TrialPhase.UNKNOWN)

    def _parse_status(self, status_str: str) -> TrialStatus:
        """
        Convert a free-text status string to a TrialStatus enum.

        Unrecognized values map to TrialStatus.UNKNOWN.
        """
        return _STATUS_MAP.get(status_str.upper().strip(), TrialStatus.UNKNOWN)
