"""ICD-10 enrichment — maps each candidate's indication to ICD-10-CM codes.

Calls the NLM Clinical Tables API via :mod:`pipeline.icd_lookup`. Lookups
are cached on disk so re-runs over the same indication strings are free.
The stage is gated by ``config.enable_icd_enrichment`` (default False)
because it makes a network call per unique indication.
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

        # Group by unique indication so we don't pay 1 RPC per duplicate row.
        unique_indications: dict[str, list] = {}
        for cand in candidates.candidates:
            ind = (cand.indication or "").strip()
            if not ind:
                continue
            unique_indications.setdefault(ind, []).append(cand)

        if not unique_indications:
            ledger.record_coverage(self.name, 0, len(candidates.candidates))
            return candidates

        # Split into cache hits vs net-new fetches up front so the user
        # can see the hit-rate before any network calls go out.
        results: dict[str, Optional[list[str]]] = {}
        to_fetch: list[str] = []
        for name in unique_indications.keys():
            if cache is not None:
                hit, value = cache.get(name)
                if hit:
                    results[name] = value
                    continue
            to_fetch.append(name)

        cached_count = len(results)
        fetch_count = len(to_fetch)
        unique_count = len(unique_indications)
        cache_pct = (100.0 * cached_count / unique_count) if unique_count else 0.0
        logger.info(
            "IcdEnrichment: %d unique indications — %d cached (%.1f%%), "
            "%d to fetch from NLM (workers=%d)",
            unique_count, cached_count, cache_pct, fetch_count, self._max_workers,
        )

        if fetch_count > 0:
            self._fetch_with_progress(to_fetch, cache, results)

        n_ok = 0
        for name, cands in unique_indications.items():
            codes = results.get(name)
            if not codes:
                continue
            for cand in cands:
                cand.icd10_codes = list(codes)
                n_ok += 1

        total = len(candidates.candidates)
        logger.info(
            "IcdEnrichment: %d/%d candidates received ICD-10 codes "
            "(unique indications: %d, cache hits: %d, fetched: %d)",
            n_ok, total, unique_count, cached_count, fetch_count,
        )
        ledger.record_coverage(self.name, n_ok, total)
        return candidates

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
