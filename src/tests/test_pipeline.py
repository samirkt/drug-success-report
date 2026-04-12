"""
Tests for the Pipeline orchestrator (pipeline/pipeline.py)

Test categories:
  PASS NOW   — PipelineConfig defaults, Pipeline construction, stage wiring,
               full run() with all stages mocked
  FAIL NOW   — full integration run() without mocks (stubs not yet implemented)
"""

import os
from unittest.mock import MagicMock, patch, call
import concurrent.futures

import pytest

from pipeline.models import (
    AttributeTable,
    CandidateTable,
    FunnelResults,
    OutcomeTable,
    ReportOutput,
    TrialTable,
)
from pipeline.pipeline import Pipeline, PipelineConfig, PipelineResult
from pipeline.stages import (
    AttributeClassificationStage,
    CandidateClusteringStage,
    FunnelAggregationStage,
    OutcomeAdjudicationStage,
    ReportingStage,
    TrialIngestionStage,
)


# ---------------------------------------------------------------------------
# PipelineConfig  (PASS NOW)
# ---------------------------------------------------------------------------

class TestPipelineConfig:
    def test_default_data_source(self):
        config = PipelineConfig()
        assert config.data_source == "aact"

    def test_max_trials_can_be_none(self):
        config = PipelineConfig(max_trials=None)
        assert config.max_trials is None

    def test_default_ingestion_filters(self):
        config = PipelineConfig()
        assert config.ingestion_filters == {}

    def test_default_clustering_method(self):
        config = PipelineConfig()
        assert config.clustering_method == "hybrid"

    def test_default_llm_adjudicate_clusters(self):
        config = PipelineConfig()
        assert config.llm_adjudicate_clusters is True

    def test_default_use_literature_lookup(self):
        config = PipelineConfig()
        assert config.use_literature_lookup is True

    def test_default_use_regulatory_data(self):
        config = PipelineConfig()
        assert config.use_regulatory_data is True

    def test_default_report_output_path(self):
        config = PipelineConfig()
        assert config.report_output_path is None

    def test_default_report_formats(self):
        config = PipelineConfig()
        assert config.report_formats == ["html"]

    def test_default_drop_unmatched_drugbank(self):
        config = PipelineConfig()
        assert config.drop_unmatched_drugbank is True

    def test_custom_values(self):
        config = PipelineConfig(
            data_source="aact",
            ingestion_filters={"keyword": "GLP-1"},
            clustering_method="fuzzy",
            llm_adjudicate_clusters=False,
            use_literature_lookup=False,
            use_regulatory_data=False,
            drop_unmatched_drugbank=False,
            report_output_path="/tmp/reports",
            report_formats=["html", "excel"],
        )
        assert config.data_source == "aact"
        assert config.ingestion_filters == {"keyword": "GLP-1"}
        assert config.clustering_method == "fuzzy"
        assert config.llm_adjudicate_clusters is False
        assert config.use_literature_lookup is False
        assert config.use_regulatory_data is False
        assert config.drop_unmatched_drugbank is False
        assert config.report_output_path == "/tmp/reports"
        assert config.report_formats == ["html", "excel"]

    def test_ingestion_filters_default_is_independent_per_instance(self):
        c1 = PipelineConfig()
        c2 = PipelineConfig()
        c1.ingestion_filters["key"] = "value"
        assert "key" not in c2.ingestion_filters

    def test_default_peptide_only_report(self):
        config = PipelineConfig()
        assert config.peptide_only_report is True


# ---------------------------------------------------------------------------
# PipelineResult  (PASS NOW)
# ---------------------------------------------------------------------------

class TestPipelineResult:
    def test_all_fields_default_to_none(self):
        result = PipelineResult()
        assert result.trial_table is None
        assert result.candidate_table is None
        assert result.attribute_table is None
        assert result.outcome_table is None
        assert result.funnel_results is None
        assert result.report is None


# ---------------------------------------------------------------------------
# Pipeline construction  (PASS NOW)
# ---------------------------------------------------------------------------

class TestPipelineConstruction:
    def test_default_config_used_when_none_provided(self):
        pipeline = Pipeline()
        assert pipeline.config is not None
        assert isinstance(pipeline.config, PipelineConfig)

    def test_custom_config_stored(self):
        config = PipelineConfig(data_source="aact")
        pipeline = Pipeline(config)
        assert pipeline.config is config

    def test_build_stages_returns_six_stages(self):
        pipeline = Pipeline()
        assert len(pipeline._stages) == 6

    def test_ingestion_stage_type(self):
        pipeline = Pipeline()
        assert isinstance(pipeline._stages["ingestion"], TrialIngestionStage)

    def test_clustering_stage_type(self):
        pipeline = Pipeline()
        assert isinstance(pipeline._stages["clustering"], CandidateClusteringStage)

    def test_classification_stage_type(self):
        pipeline = Pipeline()
        assert isinstance(pipeline._stages["classification"], AttributeClassificationStage)

    def test_adjudication_stage_type(self):
        pipeline = Pipeline()
        assert isinstance(pipeline._stages["adjudication"], OutcomeAdjudicationStage)

    def test_aggregation_stage_type(self):
        pipeline = Pipeline()
        assert isinstance(pipeline._stages["aggregation"], FunnelAggregationStage)

    def test_reporting_stage_type(self):
        pipeline = Pipeline()
        assert isinstance(pipeline._stages["reporting"], ReportingStage)

    def test_ingestion_stage_receives_data_source(self):
        config = PipelineConfig(data_source="aact")
        pipeline = Pipeline(config)
        assert pipeline._stages["ingestion"].source == "aact"

    def test_ingestion_stage_receives_filters(self):
        config = PipelineConfig(ingestion_filters={"keyword": "GLP-1"})
        pipeline = Pipeline(config)
        assert pipeline._stages["ingestion"].filters == {"keyword": "GLP-1"}

    def test_clustering_stage_receives_method(self):
        config = PipelineConfig(clustering_method="fuzzy")
        pipeline = Pipeline(config)
        assert pipeline._stages["clustering"].method == "fuzzy"

    def test_clustering_stage_receives_llm_adjudicate(self):
        config = PipelineConfig(llm_adjudicate_clusters=False)
        pipeline = Pipeline(config)
        assert pipeline._stages["clustering"].llm_adjudicate is False

    def test_clustering_stage_receives_drop_unmatched_drugbank(self):
        config = PipelineConfig(drop_unmatched_drugbank=False)
        pipeline = Pipeline(config)
        assert pipeline._stages["clustering"].drop_unmatched_drugbank is False

    def test_classification_stage_receives_use_literature(self):
        config = PipelineConfig(use_literature_lookup=False)
        pipeline = Pipeline(config)
        assert pipeline._stages["classification"].use_literature is False

    def test_adjudication_stage_receives_use_regulatory_data(self):
        config = PipelineConfig(use_regulatory_data=False)
        pipeline = Pipeline(config)
        assert pipeline._stages["adjudication"].use_regulatory_data is False

    def test_reporting_stage_receives_output_path(self):
        config = PipelineConfig(report_output_path="/tmp/reports")
        pipeline = Pipeline(config)
        assert pipeline._stages["reporting"].output_path == "/tmp/reports"

    def test_reporting_stage_receives_formats(self):
        config = PipelineConfig(report_formats=["html", "excel"])
        pipeline = Pipeline(config)
        assert pipeline._stages["reporting"].formats == ["html", "excel"]

    def test_reporting_stage_receives_peptide_only_default(self):
        pipeline = Pipeline()
        assert pipeline._stages["reporting"].peptide_only is True

    def test_reporting_stage_receives_peptide_only_false(self):
        config = PipelineConfig(peptide_only_report=False)
        pipeline = Pipeline(config)
        assert pipeline._stages["reporting"].peptide_only is False


# ---------------------------------------------------------------------------
# run() with all stages mocked  (PASS NOW)
# ---------------------------------------------------------------------------

class TestPipelineRunOrchestration:
    """Mock every stage's run() to verify the orchestrator correctly
    chains data through all stages and returns a complete PipelineResult."""

    def _build_mocked_pipeline(
        self,
        sample_trial_table,
        sample_candidate_table,
        sample_attribute_table,
        sample_outcome_table,
        sample_funnel_results,
    ):
        pipeline = Pipeline()
        pipeline._stages["ingestion"].run = MagicMock(return_value=sample_trial_table)
        pipeline._stages["clustering"].run = MagicMock(return_value=sample_candidate_table)
        pipeline._stages["classification"].run = MagicMock(return_value=sample_attribute_table)
        pipeline._stages["adjudication"].run = MagicMock(return_value=sample_outcome_table)
        pipeline._stages["aggregation"].run = MagicMock(return_value=sample_funnel_results)
        pipeline._stages["reporting"].run = MagicMock(return_value=ReportOutput())
        return pipeline

    def test_run_returns_pipeline_result(
        self, sample_trial_table, sample_candidate_table, sample_attribute_table,
        sample_outcome_table, sample_funnel_results
    ):
        pipeline = self._build_mocked_pipeline(
            sample_trial_table, sample_candidate_table, sample_attribute_table,
            sample_outcome_table, sample_funnel_results
        )
        result = pipeline.run()
        assert isinstance(result, PipelineResult)

    def test_run_populates_trial_table(
        self, sample_trial_table, sample_candidate_table, sample_attribute_table,
        sample_outcome_table, sample_funnel_results
    ):
        pipeline = self._build_mocked_pipeline(
            sample_trial_table, sample_candidate_table, sample_attribute_table,
            sample_outcome_table, sample_funnel_results
        )
        result = pipeline.run()
        assert result.trial_table is sample_trial_table

    def test_run_populates_candidate_table(
        self, sample_trial_table, sample_candidate_table, sample_attribute_table,
        sample_outcome_table, sample_funnel_results
    ):
        pipeline = self._build_mocked_pipeline(
            sample_trial_table, sample_candidate_table, sample_attribute_table,
            sample_outcome_table, sample_funnel_results
        )
        result = pipeline.run()
        assert result.candidate_table is sample_candidate_table

    def test_run_populates_attribute_table(
        self, sample_trial_table, sample_candidate_table, sample_attribute_table,
        sample_outcome_table, sample_funnel_results
    ):
        pipeline = self._build_mocked_pipeline(
            sample_trial_table, sample_candidate_table, sample_attribute_table,
            sample_outcome_table, sample_funnel_results
        )
        result = pipeline.run()
        assert result.attribute_table is sample_attribute_table

    def test_run_populates_outcome_table(
        self, sample_trial_table, sample_candidate_table, sample_attribute_table,
        sample_outcome_table, sample_funnel_results
    ):
        pipeline = self._build_mocked_pipeline(
            sample_trial_table, sample_candidate_table, sample_attribute_table,
            sample_outcome_table, sample_funnel_results
        )
        result = pipeline.run()
        assert result.outcome_table is sample_outcome_table

    def test_run_populates_funnel_results(
        self, sample_trial_table, sample_candidate_table, sample_attribute_table,
        sample_outcome_table, sample_funnel_results
    ):
        pipeline = self._build_mocked_pipeline(
            sample_trial_table, sample_candidate_table, sample_attribute_table,
            sample_outcome_table, sample_funnel_results
        )
        result = pipeline.run()
        assert result.funnel_results is sample_funnel_results

    def test_run_populates_report(
        self, sample_trial_table, sample_candidate_table, sample_attribute_table,
        sample_outcome_table, sample_funnel_results
    ):
        pipeline = self._build_mocked_pipeline(
            sample_trial_table, sample_candidate_table, sample_attribute_table,
            sample_outcome_table, sample_funnel_results
        )
        result = pipeline.run()
        assert isinstance(result.report, ReportOutput)

    def test_clustering_receives_trial_table(
        self, sample_trial_table, sample_candidate_table, sample_attribute_table,
        sample_outcome_table, sample_funnel_results
    ):
        pipeline = self._build_mocked_pipeline(
            sample_trial_table, sample_candidate_table, sample_attribute_table,
            sample_outcome_table, sample_funnel_results
        )
        pipeline.run()
        pipeline._stages["clustering"].run.assert_called_once_with(sample_trial_table)

    def test_classification_receives_candidate_table(
        self, sample_trial_table, sample_candidate_table, sample_attribute_table,
        sample_outcome_table, sample_funnel_results
    ):
        pipeline = self._build_mocked_pipeline(
            sample_trial_table, sample_candidate_table, sample_attribute_table,
            sample_outcome_table, sample_funnel_results
        )
        pipeline.run()
        pipeline._stages["classification"].run.assert_called_once_with(sample_candidate_table)

    def test_adjudication_receives_candidate_table(
        self, sample_trial_table, sample_candidate_table, sample_attribute_table,
        sample_outcome_table, sample_funnel_results
    ):
        # Use all-modalities mode so adjudication receives the full candidate table
        # (in peptide-only mode adjudication receives only the peptide-filtered subset)
        pipeline = self._build_mocked_pipeline(
            sample_trial_table, sample_candidate_table, sample_attribute_table,
            sample_outcome_table, sample_funnel_results
        )
        pipeline.config.peptide_only_report = False
        pipeline.run()
        pipeline._stages["adjudication"].run.assert_called_once_with(sample_candidate_table)

    def test_aggregation_receives_all_three_tables(
        self, sample_trial_table, sample_candidate_table, sample_attribute_table,
        sample_outcome_table, sample_funnel_results
    ):
        pipeline = self._build_mocked_pipeline(
            sample_trial_table, sample_candidate_table, sample_attribute_table,
            sample_outcome_table, sample_funnel_results
        )
        pipeline.run()
        pipeline._stages["aggregation"].run.assert_called_once_with(
            sample_candidate_table, sample_attribute_table, sample_outcome_table,
            trial_table=sample_trial_table,
        )

    def test_reporting_receives_all_four_inputs(
        self, sample_trial_table, sample_candidate_table, sample_attribute_table,
        sample_outcome_table, sample_funnel_results
    ):
        pipeline = self._build_mocked_pipeline(
            sample_trial_table, sample_candidate_table, sample_attribute_table,
            sample_outcome_table, sample_funnel_results
        )
        pipeline.run()
        pipeline._stages["reporting"].run.assert_called_once_with(
            sample_candidate_table, sample_attribute_table,
            sample_outcome_table, sample_funnel_results
        )

    def test_classification_and_adjudication_both_called(
        self, sample_trial_table, sample_candidate_table, sample_attribute_table,
        sample_outcome_table, sample_funnel_results
    ):
        """Both parallel stages must be called exactly once."""
        pipeline = self._build_mocked_pipeline(
            sample_trial_table, sample_candidate_table, sample_attribute_table,
            sample_outcome_table, sample_funnel_results
        )
        pipeline.run()
        pipeline._stages["classification"].run.assert_called_once()
        pipeline._stages["adjudication"].run.assert_called_once()


# ---------------------------------------------------------------------------
# run() raises when stubs not implemented  (PASS NOW)
# ---------------------------------------------------------------------------

class TestPipelineRunNotImplemented:
    def test_run_raises_when_api_source_not_implemented(self):
        """The api source is still a stub — pipeline should fail at ingestion."""
        config = PipelineConfig(data_source="api")
        pipeline = Pipeline(config)
        with pytest.raises(NotImplementedError):
            pipeline.run()


# ---------------------------------------------------------------------------
# Full integration  (all stubs need to be implemented)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not os.environ.get("RUN_INTEGRATION"),
    reason="Integration test requires live AACT DB and Anthropic API key. Set RUN_INTEGRATION=1 to run.",
)
class TestPipelineIntegration:
    def test_full_pipeline_with_aact_source(self):
        """End-to-end pipeline run against the live AACT should succeed.

        This test will fail until all stage implementations are complete.
        It does not use mocks — it exercises the real stage logic.
        """
        config = PipelineConfig(
            data_source="aact",
            ingestion_filters={"mesh_term": "Peptides"},
            clustering_method="fuzzy",
            llm_adjudicate_clusters=False,
            use_literature_lookup=False,
            use_regulatory_data=False,
            report_output_path=None,
            report_formats=["html"],
        )
        pipeline = Pipeline(config)
        result = pipeline.run()

        assert isinstance(result, PipelineResult)
        assert isinstance(result.trial_table, TrialTable)
        assert isinstance(result.candidate_table, CandidateTable)
        assert isinstance(result.attribute_table, AttributeTable)
        assert isinstance(result.outcome_table, OutcomeTable)
        assert isinstance(result.funnel_results, FunnelResults)
        assert isinstance(result.report, ReportOutput)
        assert len(result.trial_table) > 0
        assert len(result.candidate_table) > 0
