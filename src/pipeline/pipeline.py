"""
Pipeline orchestrator.

Builds and executes the full research pipeline:

  TrialIngestion
      → CandidateClustering
          → [AttributeClassification || OutcomeAdjudication]  (parallel)
              → FunnelAggregation
                  → AutomatedReport
"""

import concurrent.futures
import logging
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Optional

from .models import (
    AttributeTable,
    CandidateTable,
    FunnelResults,
    OutcomeTable,
    ReportOutput,
    TrialTable,
)
from .stages import (
    AttributeClassificationStage,
    CandidateClusteringStage,
    FunnelAggregationStage,
    OutcomeAdjudicationStage,
    ReportingStage,
    TrialIngestionStage,
)
from utils.tiered_router import CostLedger

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Top-level configuration passed through to each stage."""

    # Ingestion
    data_source: str = "aact"         # "aact" | "api"
    ingestion_filters: dict = field(default_factory=dict)
    max_trials: int | None = 50      # cap on rows fetched; None = full run
    filter_single_arm: bool = False   # drop basket and umbrella trials

    # Clustering
    clustering_method: str = "hybrid" # "embeddings" | "fuzzy" | "hybrid"
    llm_adjudicate_clusters: bool = True

    # Classification
    use_literature_lookup: bool = True

    # Adjudication
    use_regulatory_data: bool = True

    # Knowledge cache
    cache_path: str | None = "knowledge_cache.db"

    # AACT (clinical trials DB) fetch cache — dev convenience to skip re-hitting AACT
    use_ct_cache: bool = False
    ct_cache_path: str = "aact_cache.pkl"

    # DrugBank normalization / deduplication
    drugbank_csv_path: Optional[Path] = None
    drop_unmatched_drugbank: bool = True

    # Reporting
    report_output_path: str | None = None
    report_formats: list[str] = field(default_factory=lambda: ["html"])
    peptide_only_report: bool = True
    # Time-period cohorts for the multi-period LOA comparison.
    # None = auto-split (historical vs. last decade, or median if all recent).
    # Override with e.g. [(2005, 2014), (2015, 2024)].
    time_periods: list[tuple[int, int]] | None = field(
        default_factory=lambda: [(1962, 2000), (2000, 2006), (2007, 2024)]
    )

    # Aggregation: cohort-promotion rule for stale-status trials.
    # A trial with a non-terminal status (Unknown / Active not recruiting /
    # Recruiting) is promoted into the cohort set when its latest activity
    # date is at least `stale_trial_cutoff_years` before
    # `aggregation_reference_date`. This compensates for pre-FDAAA-2007
    # registry records whose status field was never updated.
    aggregation_reference_date: Optional[date] = None  # None → date.today() at run time
    stale_trial_cutoff_years: float = 3.0


@dataclass
class PipelineResult:
    """Collected outputs from every stage, available after pipeline.run()."""
    trial_table: TrialTable | None = None
    candidate_table: CandidateTable | None = None
    attribute_table: AttributeTable | None = None
    outcome_table: OutcomeTable | None = None
    funnel_results: FunnelResults | None = None
    report: ReportOutput | None = None
    cost_ledger: CostLedger | None = None


class Pipeline:
    """
    Builds and executes the drug-development research pipeline.

    Usage:
        config = PipelineConfig(data_source="api", ingestion_filters={"keyword": "peptide"})
        pipeline = Pipeline(config)
        result = pipeline.run()
    """

    def __init__(self, config: PipelineConfig | None = None):
        self.config = config or PipelineConfig()
        self._stages = self._build_stages()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> PipelineResult:
        """Execute all pipeline stages in order. Returns a PipelineResult."""
        result = PipelineResult()

        logger.info("Stage 1/5 — Trial Ingestion")
        result.trial_table = self._run_ingestion()

        logger.info("Stage 2/5 — Candidate Clustering")
        result.candidate_table = self._run_clustering(result.trial_table)

        logger.info("Stage 3/5 — Attribute Classification + Outcome Adjudication")
        result.attribute_table, result.outcome_table = self._run_parallel_stages(result.candidate_table)

        logger.info("Stage 4/5 — Funnel Aggregation")
        result.funnel_results = self._run_aggregation(
            result.candidate_table,
            result.attribute_table,
            result.outcome_table,
            result.trial_table,
        )

        logger.info("Stage 5/5 — Automated Report")
        result.report = self._run_reporting(
            result.candidate_table,
            result.attribute_table,
            result.outcome_table,
            result.funnel_results,
            result.trial_table,
        )

        logger.info("Pipeline complete.")
        self._ledger.log()
        result.cost_ledger = self._ledger
        return result

    # ------------------------------------------------------------------
    # Stage execution methods
    # ------------------------------------------------------------------

    def _run_ingestion(self) -> TrialTable:
        return self._stages["ingestion"].run()

    def _run_clustering(self, trial_table: TrialTable) -> CandidateTable:
        return self._stages["clustering"].run(trial_table)

    def _run_parallel_stages(
        self, candidate_table: CandidateTable
    ) -> tuple[AttributeTable, OutcomeTable]:
        """Run classification and adjudication.

        In all-modalities mode: run concurrently against the full candidate set.
        In peptide-only mode: classify first, filter to peptides, then adjudicate
        only the filtered set — avoiding wasted LLM calls on non-peptide candidates.
        """
        if not self.config.peptide_only_report:
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                future_attrs = executor.submit(
                    self._stages["classification"].run, candidate_table
                )
                future_outcomes = executor.submit(
                    self._stages["adjudication"].run, candidate_table
                )
                attribute_table = future_attrs.result()
                outcome_table = future_outcomes.result()
            return attribute_table, outcome_table

        attribute_table = self._stages["classification"].run(candidate_table)
        peptide_table = self._filter_to_peptides(candidate_table, attribute_table)
        outcome_table = self._stages["adjudication"].run(peptide_table)
        return attribute_table, outcome_table

    def _filter_to_peptides(
        self, candidate_table: CandidateTable, attribute_table: AttributeTable
    ) -> CandidateTable:
        """Return a CandidateTable containing only peptide-classified candidates."""
        peptide_ids = {
            cid
            for cid, attrs in attribute_table.attributes.items()
            if "peptide" in attrs.drug_modality.lower()
        }
        filtered = [c for c in candidate_table.candidates if c.candidate_id in peptide_ids]
        logger.info(
            "Peptide filter: %d / %d candidates queued for adjudication",
            len(filtered),
            len(candidate_table.candidates),
        )
        return CandidateTable(candidates=filtered)

    def _run_aggregation(
        self,
        candidate_table: CandidateTable,
        attribute_table: AttributeTable,
        outcome_table: OutcomeTable,
        trial_table: TrialTable | None = None,
    ) -> FunnelResults:
        return self._stages["aggregation"].run(candidate_table, attribute_table, outcome_table, trial_table=trial_table)

    def _run_reporting(
        self,
        candidate_table: CandidateTable,
        attribute_table: AttributeTable,
        outcome_table: OutcomeTable,
        funnel_results: FunnelResults,
        trial_table: TrialTable | None = None,
    ) -> ReportOutput:
        return self._stages["reporting"].run(
            candidate_table, attribute_table, outcome_table, funnel_results,
            trial_table=trial_table,
        )

    # ------------------------------------------------------------------
    # Stage construction
    # ------------------------------------------------------------------

    def _build_stages(self) -> dict:
        cfg = self.config
        self._ledger = CostLedger()

        cache = None
        if cfg.cache_path is not None:
            from .knowledge_cache import KnowledgeCache
            cache = KnowledgeCache(cfg.cache_path)

        ct_cache = None
        if cfg.use_ct_cache:
            from .aact_cache import AACTCache
            ct_cache = AACTCache(cfg.ct_cache_path)

        return {
            "ingestion": TrialIngestionStage(
                source=cfg.data_source,
                filters=cfg.ingestion_filters,
                max_trials=cfg.max_trials,
                filter_single_arm=cfg.filter_single_arm,
                ct_cache=ct_cache,
            ),
            "clustering": CandidateClusteringStage(
                method=cfg.clustering_method,
                llm_adjudicate=cfg.llm_adjudicate_clusters,
                drugbank_csv_path=cfg.drugbank_csv_path,
                drop_unmatched_drugbank=cfg.drop_unmatched_drugbank,
            ),
            "classification": AttributeClassificationStage(
                use_literature=cfg.use_literature_lookup,
                cache=cache,
                ledger=self._ledger,
            ),
            "adjudication": OutcomeAdjudicationStage(
                use_regulatory_data=cfg.use_regulatory_data,
                cache=cache,
                ledger=self._ledger,
            ),
            "aggregation": FunnelAggregationStage(
                reference_date=cfg.aggregation_reference_date,
                stale_cutoff_years=cfg.stale_trial_cutoff_years,
            ),
            "reporting": ReportingStage(
                output_path=cfg.report_output_path,
                formats=cfg.report_formats,
                peptide_only=cfg.peptide_only_report,
                time_periods=cfg.time_periods,
                reference_date=cfg.aggregation_reference_date,
                stale_cutoff_years=cfg.stale_trial_cutoff_years,
            ),
        }
