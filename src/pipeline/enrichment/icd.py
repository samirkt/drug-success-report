"""ICD-10 enrichment — maps each candidate's indication to ICD-10-CM codes.

Calls the NLM Clinical Tables API via :mod:`pipeline.icd_lookup`. Lookups
are cached on disk so re-runs over the same indication strings are free.
The stage is gated by ``config.enable_icd_enrichment`` (default False)
because it makes a network call per unique indication.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from ..icd_lookup import IcdCache, get_icd_cached
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

        def _lookup(name: str) -> tuple[str, Optional[list[str]]]:
            try:
                return name, get_icd_cached(name, cache, timeout=self._timeout)
            except Exception as exc:  # network / parse error
                logger.warning("IcdEnrichment: lookup failed for %r: %s", name, exc)
                return name, None

        results: dict[str, Optional[list[str]]] = {}
        with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            for name, codes in pool.map(_lookup, unique_indications.keys()):
                results[name] = codes

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
            "(unique indications looked up: %d)",
            n_ok, total, len(unique_indications),
        )
        ledger.record_coverage(self.name, n_ok, total)
        return candidates


__all__ = ["IcdEnrichment"]
