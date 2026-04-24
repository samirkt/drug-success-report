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
import random
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
    FDAAdjudicationConfig,
    FDAAdjudicationStage,
    FunnelAggregationStage,
    OutcomeAdjudicationStage,
    ReportingStage,
    TrialIngestionStage,
)
from utils.tiered_router import CostLedger

logger = logging.getLogger(__name__)

# Defaults for the OpenAI-compatible LLM backend on the FDA-timeline
# adjudication stage. Points at local Ollama serving Qwen 2.5 14B Instruct
# — sized for a 36GB-unified-memory Mac with headroom for parallel KV-cache
# slots (Q4_K_M weights ~9GB). The pre-LLM section extractor in
# `pipeline/fda/section_extract.py` keeps inputs ~1.5-3k chars, well within
# the smaller model's strong-extraction range. Users can override via
# `PipelineConfig.fda_llm_base_url` / `fda_llm_model`, or CLI flags. Pass
# `--fda-llm-model qwen2.5:32b-instruct` to opt back into the larger model.
DEFAULT_OPENAI_COMPAT_BASE_URL = "http://localhost:11434/v1"
DEFAULT_OPENAI_COMPAT_MODEL = "qwen2.5:14b-instruct"

# Timeouts for the FDA-timeline adjudication stage. The LLM ceiling is
# generous to absorb cold-prompt latency on local models without masking a
# truly hung server; the HTTP ceiling matches FDAClient's prior hardcoded
# default. Both surfaces are tunable via PipelineConfig and CLI flags.
DEFAULT_FDA_LLM_TIMEOUT = 180.0
DEFAULT_FDA_HTTP_TIMEOUT = 30.0


@dataclass
class PipelineConfig:
    """Top-level configuration passed through to each stage."""

    # Ingestion
    data_source: str = "aact"         # "aact" | "api"
    ingestion_filters: dict = field(default_factory=dict)
    max_trials: int | None = 50      # cap on rows fetched; None = full run
    filter_single_arm: bool = False   # drop basket and umbrella trials

    # Post-clustering candidate sampling. When `max_candidates` is set, after
    # clustering (and after the --year-range filter) the pipeline draws a
    # deterministic random sample of that many candidates using
    # `random.Random(sample_seed)` and prunes `TrialTable` to the union of
    # those candidates' `trial_ids`. This preserves each sampled candidate's
    # complete trial set — unlike `max_trials`, which caps raw ingestion rows
    # and can leave candidates with only a partial trial set after clustering.
    max_candidates: int | None = None
    sample_seed: int = 42

    # Classification
    use_literature_lookup: bool = True

    # Adjudication
    # Which adjudication method to use. "fda_timeline" reconstructs each
    # drug's FDA approval timeline from openFDA and matches indications
    # against it. "llm_direct" asks the LLM directly whether each
    # drug-indication pair is approved/failed (backed by KnowledgeCache).
    adjudication_method: str = "fda_timeline"
    # Path to the DrugBank-derived synonyms CSV used for codename↔INN
    # recovery during clustering.
    drugbank_synonyms_csv: Path = Path("data/drugbank_synonyms.csv")

    # FDA-timeline adjudication (only used when adjudication_method="fda_timeline")
    fda_cache_dir: Path = Path(".fda_cache")
    openfda_api_key: Optional[str] = None  # falls back to OPENFDA_API_KEY env var
    fda_adjudication_as_of: Optional[date] = None  # None → date.today() at run time
    fda_failure_window_days: int = 730  # ClinSR's 2-year PTnT threshold

    # LLM backend for FDA-timeline adjudication. Affects ONLY that stage —
    # `llm_direct` adjudication and classification continue to use Anthropic
    # through their existing plumbing regardless of this setting.
    #   "openai_compat" (default): any OpenAI-compatible chat-completions
    #       endpoint. Unset base_url/model fall back to local Ollama at
    #       DEFAULT_OPENAI_COMPAT_BASE_URL serving DEFAULT_OPENAI_COMPAT_MODEL,
    #       tuned for a 36GB M3 Max.
    #   "anthropic": Claude Sonnet via the Anthropic SDK.
    fda_llm_backend: str = "openai_compat"
    fda_llm_base_url: Optional[str] = None  # defaults when openai_compat and unset
    fda_llm_model: Optional[str] = None  # defaults when openai_compat and unset
    fda_llm_api_key: Optional[str] = None  # optional; local Ollama needs none

    # Performance & timeout knobs for the FDA-timeline stage.
    # `fda_adjudication_workers > 1` enables candidate-level parallelism via
    # ThreadPoolExecutor. Set <= your Ollama `OLLAMA_NUM_PARALLEL` setting;
    # raising past that just queues at the server. Default 1 preserves the
    # previous strictly-sequential behavior for users who haven't configured
    # Ollama for parallelism. Timeouts are surfaced so a hung HTTP fetch or
    # slow LLM call cannot wedge the whole run.
    fda_adjudication_workers: int = 1
    fda_llm_timeout: float = DEFAULT_FDA_LLM_TIMEOUT
    fda_http_timeout: float = DEFAULT_FDA_HTTP_TIMEOUT

    # Knowledge cache
    cache_path: str | None = "knowledge_cache.db"
    # Temporary dev hack: when True, candidates whose adjudication key isn't
    # already in the knowledge cache are dropped from CandidateTable right
    # after clustering (and year-range filtering). They disappear from the
    # funnel and all CSV outputs — "what do I already know about" mode for
    # fast clustering iteration without burning LLM credits. Remove once
    # clustering stabilizes.
    drop_uncached_candidates: bool = True

    # AACT (clinical trials DB) fetch cache — dev convenience to skip re-hitting AACT
    use_ct_cache: bool = False
    ct_cache_path: str = "aact_cache.pkl"

    # DrugBank normalization / deduplication
    drugbank_csv_path: Optional[Path] = None
    require_drugbank_match: bool = False

    # Candidate enrichments. Each feature is enabled by default; toggling
    # any off simply skips its stage — no other pipeline behavior changes.
    # Enrichments run after clustering, before the year/cache/sample
    # filters. They add fields only; they do not modify `drug_name`,
    # `indication`, `highest_phase`, or `candidate_id`, so every
    # KnowledgeCache key derived from those fields remains stable.
    enable_smiles: bool = True
    enable_targets: bool = True
    enable_icd10: bool = True
    # Pre-filtered ChEMBL targets snapshot built by
    # `scripts/build_chembl_targets_snapshot.py`. When None or missing on
    # disk the targets enrichment logs a warning and skips.
    chembl_snapshot_path: Optional[Path] = None
    # ICD-10 code granularity: "full" (e.g. C34.90), "category" (3-char
    # prefix, e.g. C34), or "chapter" (e.g. C00-D49).
    icd10_granularity: str = "category"

    # Reporting
    report_output_path: str | None = None
    report_formats: list[str] = field(default_factory=lambda: ["html"])
    peptide_only_report: bool = True
    # Time-period cohorts for the multi-period LOA comparison.
    # None = auto-split (historical vs. last decade, or median if all recent).
    # Override with e.g. [(2005, 2014), (2015, 2024)].
    # The trailing (2015, 2023) cohort matches ClinSR's headline 9-year
    # rolling window (Zhou et al., Nat Commun 16:9537, 2025) and enables a
    # direct apples-to-apples comparison against their ~5% LOA figure.
    time_periods: list[tuple[int, int]] | None = field(
        default_factory=lambda: [
            (1962, 2000),
            (2000, 2006),
            (2007, 2026),
        ]
    )

    # Aggregation: cohort-promotion rule for stale-status trials.
    # A trial with a non-terminal status (Unknown / Active not recruiting /
    # Recruiting) is promoted into the cohort set when its latest activity
    # date is at least `stale_trial_cutoff_years` before
    # `aggregation_reference_date`. This compensates for pre-FDAAA-2007
    # registry records whose status field was never updated. Aligned with
    # ClinSR's 2-year Trial Failure Threshold (Zhou et al., Nat Commun
    # 16:9537, 2025).
    aggregation_reference_date: Optional[date] = field(default_factory=lambda: date(2006, 12, 31))  # anchor for 2000-2006 cohort; set to None for date.today()
    stale_trial_cutoff_years: float = 2.0
    # When True (ClinSR-aligned default), approved / commercialized
    # candidates are credited as having reached Phase 1, Phase 2, and
    # Phase 3 in `phases_observed` even if no trial record survives for
    # those phases. Set to False to preserve strict forward-looking
    # semantics (no back-propagation).
    back_propagate_approval: bool = True

    # Global cohort filter. When set, candidates whose earliest trial start
    # year falls outside [start, end] (inclusive) are dropped after
    # clustering, before classification / adjudication / aggregation. This
    # restricts the *entire* report — including all funnel slices, LOA
    # charts, and per-period breakdowns — to the selected window, unlike
    # `time_periods` which only drives the multi-period comparison chart.
    candidate_year_range: Optional[tuple[int, int]] = None


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
        result.candidate_table = self._run_enrichments(result.candidate_table)
        result.candidate_table = self._filter_by_year_range(result.candidate_table)
        result.candidate_table, result.trial_table = self._filter_to_cached_candidates(
            result.candidate_table, result.trial_table
        )
        result.candidate_table, result.trial_table = self._sample_candidates(
            result.candidate_table, result.trial_table
        )

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

    def _run_enrichments(self, candidate_table: CandidateTable) -> CandidateTable:
        """Apply each enabled enrichment stage in order.

        Enrichments only add fields to candidates (SMILES, drug targets,
        ICD-10 codes); they never change keys that feed KnowledgeCache.
        Missing data sources log a warning and skip cleanly.
        """
        from .enrichment import run_enrichments

        stages = self._enrichment_stages
        if not stages:
            return candidate_table
        logger.info(
            "Enrichments enabled: smiles=%s, targets=%s, icd10=%s (granularity=%s)",
            self.config.enable_smiles,
            self.config.enable_targets,
            self.config.enable_icd10,
            self.config.icd10_granularity,
        )
        return run_enrichments(candidate_table, stages, self.config, self._ledger)

    def _filter_by_year_range(self, candidate_table: CandidateTable) -> CandidateTable:
        """Drop candidates whose earliest trial start year is outside the configured window.

        Candidates with no `earliest_start_date` are dropped — they carry no
        temporal anchor to place them in a cohort.
        """
        window = self.config.candidate_year_range
        if window is None:
            return candidate_table
        start_yr, end_yr = window
        filtered = [
            c for c in candidate_table.candidates
            if c.earliest_start_date is not None
            and start_yr <= c.earliest_start_date.year <= end_yr
        ]
        logger.info(
            "Year-range filter [%d-%d]: %d / %d candidates retained",
            start_yr, end_yr, len(filtered), len(candidate_table.candidates),
        )
        return CandidateTable(candidates=filtered)

    def _filter_to_cached_candidates(
        self,
        candidate_table: CandidateTable,
        trial_table: TrialTable,
    ) -> tuple[CandidateTable, TrialTable]:
        """Drop candidates whose adjudication key isn't already cached.

        Temporary dev hack (see ``PipelineConfig.drop_uncached_candidates``):
        when enabled, inspects the adjudication cache for each candidate and
        keeps only the ones that already have a stored outcome, pruning
        ``trial_table`` to the matching trials. Lets clustering iteration run
        end-to-end against previously-adjudicated drugs only, with zero LLM
        spend.
        """
        if not self.config.drop_uncached_candidates:
            return candidate_table, trial_table
        if self._cache is None:
            logger.warning(
                "drop_uncached_candidates set but no knowledge cache configured — "
                "skipping filter."
            )
            return candidate_table, trial_table

        from .knowledge_cache import KnowledgeCache

        method = self.config.adjudication_method
        cache = self._cache

        def _is_cached(cand) -> bool:
            key = KnowledgeCache.make_adjudication_key(
                cand.drug_name, cand.indication, cand.highest_phase.value,
            )
            if method == "fda_timeline":
                return cache.get_fda_outcome(key, cand.candidate_id) is not None
            return cache.get_outcome(key, cand.candidate_id) is not None

        kept = [c for c in candidate_table.candidates if _is_cached(c)]
        allowed_nct_ids = {nct for c in kept for nct in c.trial_ids}
        pruned_trials = [t for t in trial_table.trials if t.nct_id in allowed_nct_ids]

        logger.info(
            "drop_uncached_candidates (%s): %d / %d candidates kept, %d / %d trials retained",
            method,
            len(kept), len(candidate_table.candidates),
            len(pruned_trials), len(trial_table.trials),
        )
        return CandidateTable(candidates=kept), TrialTable(trials=pruned_trials)

    def _sample_candidates(
        self,
        candidate_table: CandidateTable,
        trial_table: TrialTable,
    ) -> tuple[CandidateTable, TrialTable]:
        """Draw a deterministic random sample of candidates and prune trials to match.

        Applied after clustering and year-range filtering so each sampled
        candidate retains its complete trial set. Uses an isolated
        `random.Random(seed)` so sampling is reproducible and does not
        perturb any other RNG in the process.
        """
        n = self.config.max_candidates
        if n is None or len(candidate_table.candidates) <= n:
            return candidate_table, trial_table

        rng = random.Random(self.config.sample_seed)
        sampled = rng.sample(candidate_table.candidates, k=n)

        allowed_nct_ids: set[str] = set()
        for cand in sampled:
            allowed_nct_ids.update(cand.trial_ids)
        pruned_trials = [t for t in trial_table.trials if t.nct_id in allowed_nct_ids]

        logger.info(
            "Candidate sample (seed=%d): %d / %d candidates, %d / %d trials retained",
            self.config.sample_seed,
            len(sampled),
            len(candidate_table.candidates),
            len(pruned_trials),
            len(trial_table.trials),
        )
        return CandidateTable(candidates=sampled), TrialTable(trials=pruned_trials)

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
        self._cache = cache

        ct_cache = None
        if cfg.use_ct_cache:
            from .aact_cache import AACTCache
            ct_cache = AACTCache(cfg.ct_cache_path)

        adjudication_stage = self._build_adjudication_stage(cfg, cache)

        self._enrichment_stages = self._build_enrichment_stages()

        return {
            "ingestion": TrialIngestionStage(
                source=cfg.data_source,
                filters=cfg.ingestion_filters,
                max_trials=cfg.max_trials,
                filter_single_arm=cfg.filter_single_arm,
                ct_cache=ct_cache,
            ),
            "clustering": CandidateClusteringStage(
                drugbank_csv_path=cfg.drugbank_csv_path,
                drugbank_synonyms_csv_path=cfg.drugbank_synonyms_csv,
                require_drugbank_match=cfg.require_drugbank_match,
            ),
            "classification": AttributeClassificationStage(
                use_literature=cfg.use_literature_lookup,
                cache=cache,
                ledger=self._ledger,
            ),
            "adjudication": adjudication_stage,
            "aggregation": FunnelAggregationStage(
                reference_date=cfg.aggregation_reference_date,
                stale_cutoff_years=cfg.stale_trial_cutoff_years,
                back_propagate_approval=cfg.back_propagate_approval,
            ),
            "reporting": ReportingStage(
                output_path=cfg.report_output_path,
                formats=cfg.report_formats,
                peptide_only=cfg.peptide_only_report,
                time_periods=cfg.time_periods,
                reference_date=cfg.aggregation_reference_date,
                stale_cutoff_years=cfg.stale_trial_cutoff_years,
                back_propagate_approval=cfg.back_propagate_approval,
                cache=cache,
                adjudication_method=cfg.adjudication_method,
            ),
        }

    def _build_adjudication_stage(self, cfg: "PipelineConfig", cache):
        """Construct the configured adjudicator.

        Both branches produce stages conforming to `run(CandidateTable) -> OutcomeTable`
        so downstream aggregation/reporting stay untouched.
        """
        method = cfg.adjudication_method
        if method == "llm_direct":
            return OutcomeAdjudicationStage(
                cache=cache,
                ledger=self._ledger,
            )
        if method == "fda_timeline":
            import os
            from .fda import AnthropicJSONClient, FDAClient, OpenAICompatJSONClient

            fda = FDAClient(
                cache_dir=cfg.fda_cache_dir,
                openfda_api_key=cfg.openfda_api_key or os.getenv("OPENFDA_API_KEY"),
                timeout=cfg.fda_http_timeout,
            )

            backend = cfg.fda_llm_backend
            if backend == "anthropic":
                anthropic_kwargs = {"cache": cache, "ledger": self._ledger}
                if cfg.fda_llm_model:
                    anthropic_kwargs["model"] = cfg.fda_llm_model
                llm = AnthropicJSONClient(**anthropic_kwargs)
                logger.info(
                    "FDA adjudication using anthropic backend: %s",
                    cfg.fda_llm_model or "default sonnet",
                )
            elif backend == "openai_compat":
                base_url = cfg.fda_llm_base_url or DEFAULT_OPENAI_COMPAT_BASE_URL
                model = cfg.fda_llm_model or DEFAULT_OPENAI_COMPAT_MODEL
                llm = OpenAICompatJSONClient(
                    base_url=base_url,
                    model=model,
                    api_key=cfg.fda_llm_api_key,
                    timeout=cfg.fda_llm_timeout,
                    cache=cache,
                    ledger=self._ledger,
                )
                logger.info(
                    "FDA adjudication using openai_compat backend: %s @ %s "
                    "(workers=%d, llm_timeout=%.0fs, http_timeout=%.0fs)",
                    model,
                    base_url,
                    cfg.fda_adjudication_workers,
                    cfg.fda_llm_timeout,
                    cfg.fda_http_timeout,
                )
            else:
                raise ValueError(
                    f"Unknown fda_llm_backend: {backend!r}. "
                    "Expected 'openai_compat' or 'anthropic'."
                )

            return FDAAdjudicationStage(
                fda_client=fda,
                llm_client=llm,
                config=FDAAdjudicationConfig(
                    as_of=cfg.fda_adjudication_as_of,
                    failure_window_days=cfg.fda_failure_window_days,
                ),
                cache=cache,
                workers=cfg.fda_adjudication_workers,
            )
        raise ValueError(
            f"Unknown adjudication_method: {method!r}. "
            "Expected 'llm_direct' or 'fda_timeline'."
        )

    def _build_enrichment_stages(self) -> list:
        """Instantiate enrichment stages for the features enabled in config.

        Stages are returned in a fixed order (SMILES → targets → ICD-10).
        Each stage's own `is_available` check decides whether it runs;
        toggled-off or missing-data stages are skipped silently here.
        """
        from .enrichment import SmilesEnrichment

        stages: list = []
        if self.config.enable_smiles:
            stages.append(SmilesEnrichment())
        # Targets (step 2) and ICD-10 (step 3) get registered here as they
        # land in subsequent steps of the plan.
        return stages
