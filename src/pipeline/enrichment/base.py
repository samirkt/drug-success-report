"""Shared Protocol and orchestrator for candidate enrichment stages.

An enrichment stage adds fields to `Candidate` objects in-place (or returns
a new `CandidateTable` carrying the enriched rows). Enrichments are
strictly field-only: they must not change `drug_name`, `indication`,
`highest_phase`, or `candidate_id`, so `KnowledgeCache` keys remain stable
whether the enrichment runs or not.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Iterable, Protocol, runtime_checkable

from ..models import CandidateTable

if TYPE_CHECKING:
    from ..pipeline import PipelineConfig
    from utils.tiered_router import CostLedger

logger = logging.getLogger(__name__)


@runtime_checkable
class EnrichmentStage(Protocol):
    """Interface every candidate enrichment must implement."""

    name: str

    def is_available(self, config: "PipelineConfig") -> bool:
        """Return True if this stage can run with the current config/data.

        When False, the orchestrator logs a single skip message and moves
        on — enrichments are additive, so a missing data source should
        never fail the pipeline.
        """
        ...

    def run(
        self,
        candidates: CandidateTable,
        *,
        ledger: "CostLedger",
    ) -> CandidateTable:
        """Attach enrichment fields and return the (possibly same) table."""
        ...


def run_enrichments(
    candidates: CandidateTable,
    stages: Iterable[EnrichmentStage],
    config: "PipelineConfig",
    ledger: "CostLedger",
) -> CandidateTable:
    """Run each stage in order, skipping unavailable ones.

    A stage's own exceptions do not propagate — they are logged and the
    remaining stages still run. Enrichments are supplementary; a bad
    ChEMBL snapshot or flaky HTTP call must never abort the pipeline.
    """
    table = candidates
    for stage in stages:
        try:
            if not stage.is_available(config):
                logger.info("Enrichment %s: skipped (unavailable)", stage.name)
                continue
            table = stage.run(table, ledger=ledger)
        except Exception as exc:
            logger.warning(
                "Enrichment %s raised %s: %s — continuing without its output.",
                stage.name,
                type(exc).__name__,
                exc,
            )
    return table


__all__ = ["EnrichmentStage", "run_enrichments"]
