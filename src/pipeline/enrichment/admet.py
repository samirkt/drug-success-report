"""ADMET enrichment adapter.

Bridges the standalone ``pipeline.admet`` package (zero pipeline imports)
to the ``EnrichmentStage`` Protocol used by the orchestrator. This file
is the only place where ``pipeline.admet`` and ``pipeline.models`` meet.

Modeled directly on
:class:`pipeline.enrichment.smiles_standardization.SmilesStandardizationEnrichment`:
lazy import in ``is_available``, fields-only mutation in ``run``, and a
single ``ledger.record_coverage`` call at the end. The ``admet_ai``
extra is a heavy install (PyTorch + Lightning), so missing it must
degrade to a one-line skip rather than abort the run.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from ..models import CandidateTable

if TYPE_CHECKING:
    from ..pipeline import PipelineConfig
    from utils.tiered_router import CostLedger

logger = logging.getLogger(__name__)


class AdmetEnrichment:
    """Predict ~104 ADMET properties per candidate from canonical SMILES."""

    name = "admet"

    def __init__(self, cache_path: Optional[Path] = None) -> None:
        self._cache_path = cache_path
        self._predictor = None
        self._field_name = None
        self._columns: tuple[str, ...] = ()

    def is_available(self, config: "PipelineConfig") -> bool:
        if not getattr(config, "enable_admet", True):
            return False
        if self._predictor is not None:
            return True
        try:
            from ..admet import (
                ADMET_COLUMNS,
                AdmetCache,
                AdmetPredictor,
                field_name,
            )
            import admet_ai  # noqa: F401  presence check
        except ImportError as exc:
            logger.warning(
                "AdmetEnrichment: skipped — missing %s. "
                "Install with: pip install admet-ai",
                exc.name or "admet_ai",
            )
            return False
        try:
            cache = AdmetCache(self._cache_path) if self._cache_path else None
            self._predictor = AdmetPredictor(cache=cache)
        except Exception as exc:
            logger.warning("AdmetEnrichment: predictor init failed: %s", exc)
            return False
        self._field_name = field_name
        self._columns = ADMET_COLUMNS
        return True

    def run(
        self,
        candidates: CandidateTable,
        *,
        ledger: "CostLedger",
    ) -> CandidateTable:
        total = len(candidates.candidates)

        smiles_for: dict[str, str] = {}
        for c in candidates.candidates:
            smi = (c.smiles_canonical or c.smiles or "").strip()
            if smi:
                smiles_for[c.candidate_id] = smi

        unique_smiles = list(dict.fromkeys(smiles_for.values()))
        try:
            preds = self._predictor.predict(unique_smiles)
        except Exception as exc:
            logger.warning("AdmetEnrichment: batch predict failed: %s", exc)
            ledger.record_coverage(self.name, 0, total)
            return candidates

        n_populated = 0
        n_failed = 0
        for c in candidates.candidates:
            smi = smiles_for.get(c.candidate_id)
            if not smi:
                continue
            pred = preds.get(smi)
            if pred is None:
                n_failed += 1
                continue
            for col in self._columns:
                setattr(c, self._field_name(col), pred.get(col))
            n_populated += 1

        n_skipped = total - len(smiles_for)
        pct = (100.0 * n_populated / total) if total else 0.0
        logger.info(
            "Enrichment admet: populated=%d failed=%d skipped_no_smiles=%d "
            "(coverage: %d/%d, %.0f%%)",
            n_populated, n_failed, n_skipped, n_populated, total, pct,
        )
        ledger.record_coverage(self.name, n_populated, total)
        return candidates


__all__ = ["AdmetEnrichment"]
