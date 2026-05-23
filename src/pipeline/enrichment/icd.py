"""ICD-10 enrichment — maps each candidate's indication to ICD-10-CM codes.

Calls the NLM Clinical Tables API via :mod:`pipeline.icd_lookup`. For each
candidate the lookup tries ``mesh_indication`` first (a canonical MeSH
term is far closer to NLM's ICD-10-CM index than free-text trial copy)
and falls back to the raw ``indication`` string when MeSH yields no
match. Both keys are written to the same on-disk cache so subsequent
runs over the same vocabulary are free. The stage is gated by
``config.enable_icd_enrichment`` (default False) because it makes a
network call per unique unresolved key.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from ..icd_lookup import IcdCache, get_icd_from_nih
from ..models import CandidateTable

if TYPE_CHECKING:
    from ..pipeline import PipelineConfig
    from utils.tiered_router import CostLedger

logger = logging.getLogger(__name__)


class IcdEnrichment:
    """Populate ``Candidate.icd10_codes`` via NLM."""

    name = "icd10"

    def __init__(
        self,
        cache_path: Optional[str | Path] = None,
        max_workers: int = 4,
        timeout: float = 10.0,
    ) -> None:
        self._cache_path = Path(cache_path) if cache_path else None
        self._max_workers = max_workers
        self._timeout = timeout

    def is_available(self, config: "PipelineConfig") -> bool:
        return bool(getattr(config, "enable_icd10", False))

    def run(
        self,
        candidates: CandidateTable,
        *,
        ledger: "CostLedger",
    ) -> CandidateTable:
        cache: Optional[IcdCache] = (
            IcdCache(self._cache_path) if self._cache_path is not None else None
        )
        total = len(candidates.candidates)

        # Two-phase resolution: try MeSH for every candidate that has one,
        # then fall back to the raw indication only for candidates the
        # MeSH phase didn't resolve. Skipping the second phase for
        # already-resolved candidates avoids ~one NLM call per candidate
        # in the common case where MeSH succeeds.
        results: dict[str, Optional[list[str]]] = {}

        mesh_candidates = [
            c for c in candidates.candidates if (c.mesh_indication or "").strip()
        ]
        mesh_keys = {(c.mesh_indication or "").strip() for c in mesh_candidates}
        if mesh_keys:
            self._resolve_keys(mesh_keys, cache, results, phase="mesh")

        resolved_via_mesh: set[int] = set()
        for cand in mesh_candidates:
            codes = results.get((cand.mesh_indication or "").strip())
            if codes:
                cand.icd10_codes = list(codes)
                resolved_via_mesh.add(id(cand))

        # Phase 2: raw indication for everyone the MeSH phase didn't resolve.
        fallback_candidates = [
            c for c in candidates.candidates
            if id(c) not in resolved_via_mesh and (c.indication or "").strip()
        ]
        indication_keys = {
            (c.indication or "").strip() for c in fallback_candidates
        }
        if indication_keys:
            self._resolve_keys(indication_keys, cache, results, phase="indication")

        n_via_indication = 0
        for cand in fallback_candidates:
            codes = results.get((cand.indication or "").strip())
            if codes:
                cand.icd10_codes = list(codes)
                n_via_indication += 1

        n_via_mesh = len(resolved_via_mesh)
        n_ok = n_via_mesh + n_via_indication
        logger.info(
            "IcdEnrichment: %d/%d candidates received ICD-10 codes "
            "(via mesh: %d, via indication fallback: %d)",
            n_ok, total, n_via_mesh, n_via_indication,
        )
        ledger.record_coverage(self.name, n_ok, total)
        return candidates

    def _resolve_keys(
        self,
        keys: set[str],
        cache: Optional[IcdCache],
        results: dict[str, Optional[list[str]]],
        *,
        phase: str,
    ) -> None:
        """Cache-check + fetch a batch of lookup keys, mutating ``results`` in place."""
        to_fetch: list[str] = []
        cached_count = 0
        for key in keys:
            if key in results:
                continue
            if cache is not None:
                hit, value = cache.get(key)
                if hit:
                    results[key] = value
                    cached_count += 1
                    continue
            to_fetch.append(key)

        unique_count = len(keys)
        fetch_count = len(to_fetch)
        cache_pct = (100.0 * cached_count / unique_count) if unique_count else 0.0
        logger.info(
            "IcdEnrichment[%s]: %d unique keys — %d cached (%.1f%%), "
            "%d to fetch from NLM (workers=%d)",
            phase, unique_count, cached_count, cache_pct,
            fetch_count, self._max_workers,
        )
        if fetch_count > 0:
            self._fetch_with_progress(to_fetch, cache, results)

    def _fetch_with_progress(
        self,
        names: list[str],
        cache: Optional[IcdCache],
        results: dict[str, Optional[list[str]]],
        *,
        log_every: int = 50,
    ) -> None:
        """Fetch missing indications via NLM and log throughput."""

        def _lookup(name: str) -> tuple[str, Optional[list[str]]]:
            # Skip the cache check (we already classified these as misses)
            # but write back into the cache on success so subsequent runs
            # benefit. Failures don't write — they get a clean retry.
            try:
                codes = get_icd_from_nih(name, timeout=self._timeout)
                if cache is not None:
                    cache.put(name, codes)
                return name, codes
            except Exception as exc:
                logger.warning("IcdEnrichment: lookup failed for %r: %s", name, exc)
                return name, None

        total = len(names)
        completed = 0
        ok = 0
        failed = 0
        start = time.monotonic()
        last_log = start

        with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            futures = [pool.submit(_lookup, n) for n in names]
            for fut in as_completed(futures):
                name, codes = fut.result()
                results[name] = codes
                completed += 1
                if codes is None:
                    failed += 1
                else:
                    ok += 1
                if completed % log_every == 0 or completed == total:
                    now = time.monotonic()
                    elapsed = now - start
                    rate = completed / elapsed if elapsed > 0 else 0.0
                    interval = now - last_log
                    interval_rate = (
                        log_every / interval if interval > 0 and completed != total else rate
                    )
                    remaining = (total - completed) / rate if rate > 0 else 0.0
                    logger.info(
                        "IcdEnrichment: fetched %d/%d (%.1f%% — %.1f req/s "
                        "instant, %.1f avg, ok=%d fail=%d, ETA %ds)",
                        completed, total, 100.0 * completed / total,
                        interval_rate, rate, ok, failed, int(remaining),
                    )
                    last_log = now


__all__ = ["IcdEnrichment"]
