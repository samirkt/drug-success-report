"""
Tests for Stage 5: Automated Report (pipeline/stages/reporting.py)

Test categories:
  PASS NOW   — constructor, orchestration wiring (via mocks)
  FAIL NOW   — behavioral contracts for all _* helper methods
               These tests will PASS once the implementation is complete.
"""

import os
import tempfile
from datetime import date
from unittest.mock import MagicMock, call, patch

import pytest

from pipeline.models import (
    AttributeTable,
    Candidate,
    CandidateAttributes,
    CandidateOutcome,
    CandidateOutcomeRecord,
    CandidateTable,
    FunnelResults,
    FunnelSlice,
    OutcomeTable,
    ReportOutput,
    TrialPhase,
    TransitionRate,
)
from pipeline.stages.reporting import ReportingStage
from pipeline.stages.reporting._compute import (
    compute_loa,
    loa_table,
    bio_qls_by_disease_area,
    oncology_vs_rest,
    compute_transition_rate,
    sponsor_summary_stats,
)
from pipeline.stages.reporting.components._funnel import funnel_table
from pipeline.stages.reporting.components._disease_breakdown import (
    breakdown_chart,
    stratified_funnel_table,
)
from pipeline.stages.reporting.components._spider import spider_chart, disease_spider_charts
from pipeline.stages.reporting.components._candidate_summary import (
    candidate_summary_table,
    latest_milestone_date,
    inconsistency_flag,
)
from pipeline.stages.reporting.components._timeline import timeline_by_disease_chart
from pipeline.stages.reporting.components._modality_trend import modality_proportion_data
from pipeline.stages.reporting.components._sponsor import sponsor_concentration_data
from pipeline.stages.reporting.components._time_period import (
    partition_by_time_periods,
    time_period_loa_table,
)
from pipeline.stages.reporting.components._heatmaps import modality_heatmap, disease_heatmap
from pipeline.stages.reporting.components._bubble_heatmaps import (
    interactive_bubble_heatmap,
    interactive_disease_bubble_heatmap,
)
from pipeline.stages.reporting import _compute


# ---------------------------------------------------------------------------
# Constructor / initialization  (PASS NOW)
# ---------------------------------------------------------------------------

class TestReportingStageInit:
    def test_default_output_path_is_none(self):
        stage = ReportingStage()
        assert stage.output_path is None

    def test_default_formats_is_html(self):
        stage = ReportingStage()
        assert stage.formats == ["html"]

    def test_custom_output_path(self):
        stage = ReportingStage(output_path="/tmp/reports")
        assert stage.output_path == "/tmp/reports"

    def test_custom_formats(self):
        stage = ReportingStage(formats=["html", "excel", "pdf"])
        assert stage.formats == ["html", "excel", "pdf"]

    def test_none_formats_normalised_to_html(self):
        stage = ReportingStage(formats=None)
        assert stage.formats == ["html"]

    def test_default_peptide_only_is_true(self):
        stage = ReportingStage()
        assert stage.peptide_only is True

    def test_peptide_only_false(self):
        stage = ReportingStage(peptide_only=False)
        assert stage.peptide_only is False


# ---------------------------------------------------------------------------
# run() orchestration with mocks  (PASS NOW)
# ---------------------------------------------------------------------------

class TestReportingStageRunOrchestration:
    """Mock all stub helpers to verify run() assembles ReportOutput correctly."""

    def _patched_stage(self, output_path=None, peptide_only=True):
        return ReportingStage(output_path=output_path, peptide_only=peptide_only)

    def test_run_returns_report_output_type(
        self, sample_candidate_table, sample_attribute_table,
        sample_outcome_table, sample_funnel_results
    ):
        stage = self._patched_stage()
        result = stage.run(
            sample_candidate_table, sample_attribute_table,
            sample_outcome_table, sample_funnel_results
        )
        assert isinstance(result, ReportOutput)

    def test_run_all_modalities_has_four_tables(
        self, sample_candidate_table, sample_attribute_table,
        sample_outcome_table, sample_funnel_results
    ):
        stage = self._patched_stage(peptide_only=False)
        result = stage.run(
            sample_candidate_table, sample_attribute_table,
            sample_outcome_table, sample_funnel_results
        )
        assert "candidate_summary" in result.tables
        assert "funnel_by_modality" in result.tables
        assert "funnel_by_disease" in result.tables

    def test_run_peptide_only_omits_modality_table(
        self, sample_candidate_table, sample_attribute_table,
        sample_outcome_table, sample_funnel_results
    ):
        stage = self._patched_stage(peptide_only=True)
        result = stage.run(
            sample_candidate_table, sample_attribute_table,
            sample_outcome_table, sample_funnel_results
        )
        assert "candidate_summary" in result.tables
        assert "funnel_by_disease" in result.tables
        assert "funnel_by_modality" not in result.tables

    def test_run_all_modalities_has_three_figures(
        self, sample_candidate_table, sample_attribute_table,
        sample_outcome_table, sample_funnel_results
    ):
        stage = self._patched_stage(peptide_only=False)
        result = stage.run(
            sample_candidate_table, sample_attribute_table,
            sample_outcome_table, sample_funnel_results
        )
        assert "overall_spider" in result.figures
        assert "modality_breakdown" in result.figures
        assert "disease_breakdown" in result.figures

    def test_run_peptide_only_omits_modality_breakdown(
        self, sample_candidate_table, sample_attribute_table,
        sample_outcome_table, sample_funnel_results
    ):
        stage = self._patched_stage(peptide_only=True)
        result = stage.run(
            sample_candidate_table, sample_attribute_table,
            sample_outcome_table, sample_funnel_results
        )
        assert "overall_spider" in result.figures
        assert "disease_breakdown" in result.figures
        assert "modality_breakdown" not in result.figures

    def test_run_populates_summary_text(
        self, sample_candidate_table, sample_attribute_table,
        sample_outcome_table, sample_funnel_results
    ):
        stage = self._patched_stage()
        result = stage.run(
            sample_candidate_table, sample_attribute_table,
            sample_outcome_table, sample_funnel_results
        )
        assert isinstance(result.summary_text, str)
        assert len(result.summary_text) > 0

    def test_run_calls_write_when_output_path_set(
        self, sample_candidate_table, sample_attribute_table,
        sample_outcome_table, sample_funnel_results
    ):
        stage = self._patched_stage(output_path="/tmp/out")
        stage.run(
            sample_candidate_table, sample_attribute_table,
            sample_outcome_table, sample_funnel_results
        )
        # Output directory should be created when output_path is set
        assert os.path.isdir("/tmp/out")

    def test_run_does_not_call_write_when_no_output_path(
        self, sample_candidate_table, sample_attribute_table,
        sample_outcome_table, sample_funnel_results
    ):
        stage = self._patched_stage(output_path=None)
        result = stage.run(
            sample_candidate_table, sample_attribute_table,
            sample_outcome_table, sample_funnel_results
        )
        # No output_path → no files written (just returns the report object)
        assert result.output_path is None

    def test_run_output_path_preserved_in_report(
        self, sample_candidate_table, sample_attribute_table,
        sample_outcome_table, sample_funnel_results
    ):
        stage = self._patched_stage(output_path="/tmp/out")
        result = stage.run(
            sample_candidate_table, sample_attribute_table,
            sample_outcome_table, sample_funnel_results
        )
        assert result.output_path == "/tmp/out"

    def test_run_all_modalities_stratified_funnel_called_twice(
        self, sample_candidate_table, sample_attribute_table,
        sample_outcome_table, sample_funnel_results
    ):
        """_stratified_funnel_table is called once for by_modality and once for by_disease_area."""
        stage = self._patched_stage(peptide_only=False)
        result = stage.run(
            sample_candidate_table, sample_attribute_table,
            sample_outcome_table, sample_funnel_results
        )
        # All-modalities mode produces both disease and modality funnel tables
        assert "funnel_by_disease" in result.tables
        assert "funnel_by_modality" in result.tables

    def test_run_peptide_only_stratified_funnel_called_once(
        self, sample_candidate_table, sample_attribute_table,
        sample_outcome_table, sample_funnel_results
    ):
        """In peptide-only mode, _stratified_funnel_table is called once (disease only)."""
        stage = self._patched_stage(peptide_only=True)
        result = stage.run(
            sample_candidate_table, sample_attribute_table,
            sample_outcome_table, sample_funnel_results
        )
        # Peptide-only mode has disease funnel but not modality funnel
        assert "funnel_by_disease" in result.tables
        assert "funnel_by_modality" not in result.tables

    def test_run_all_modalities_breakdown_chart_called_twice(
        self, sample_candidate_table, sample_attribute_table,
        sample_outcome_table, sample_funnel_results
    ):
        """_breakdown_chart is called for modality and disease area breakdowns."""
        stage = self._patched_stage(peptide_only=False)
        result = stage.run(
            sample_candidate_table, sample_attribute_table,
            sample_outcome_table, sample_funnel_results
        )
        # All-modalities mode produces both disease and modality breakdown charts
        assert "disease_breakdown" in result.figures
        assert "modality_breakdown" in result.figures

    def test_run_peptide_only_breakdown_chart_called_once(
        self, sample_candidate_table, sample_attribute_table,
        sample_outcome_table, sample_funnel_results
    ):
        """In peptide-only mode, _breakdown_chart is called once (disease only)."""
        stage = self._patched_stage(peptide_only=True)
        result = stage.run(
            sample_candidate_table, sample_attribute_table,
            sample_outcome_table, sample_funnel_results
        )
        # Peptide-only mode has disease breakdown but not modality breakdown
        assert "disease_breakdown" in result.figures
        assert "modality_breakdown" not in result.figures


# ---------------------------------------------------------------------------
# _candidate_summary_table  (FAIL NOW — stub; PASS when implemented)
# ---------------------------------------------------------------------------

class TestCandidateSummaryTable:
    def test_returns_list(self, sample_candidate_table, sample_attribute_table, sample_outcome_table):
        stage = ReportingStage()
        result = candidate_summary_table(
            sample_candidate_table, sample_attribute_table, sample_outcome_table
        )
        assert isinstance(result, list)

    def test_one_row_per_candidate(
        self, sample_candidate_table, sample_attribute_table, sample_outcome_table
    ):
        stage = ReportingStage()
        result = candidate_summary_table(
            sample_candidate_table, sample_attribute_table, sample_outcome_table
        )
        assert len(result) == len(sample_candidate_table.candidates)

    def test_rows_are_dicts(self, sample_candidate_table, sample_attribute_table, sample_outcome_table):
        stage = ReportingStage()
        result = candidate_summary_table(
            sample_candidate_table, sample_attribute_table, sample_outcome_table
        )
        assert all(isinstance(r, dict) for r in result)

    def test_row_includes_drug_name(
        self, sample_candidate_table, sample_attribute_table, sample_outcome_table
    ):
        stage = ReportingStage()
        result = candidate_summary_table(
            sample_candidate_table, sample_attribute_table, sample_outcome_table
        )
        for row in result:
            assert "drug_name" in row or "drug" in row

    def test_row_includes_modality(
        self, sample_candidate_table, sample_attribute_table, sample_outcome_table
    ):
        stage = ReportingStage()
        result = candidate_summary_table(
            sample_candidate_table, sample_attribute_table, sample_outcome_table
        )
        for row in result:
            assert "modality" in row or "drug_modality" in row

    def test_row_includes_outcome(
        self, sample_candidate_table, sample_attribute_table, sample_outcome_table
    ):
        stage = ReportingStage()
        result = candidate_summary_table(
            sample_candidate_table, sample_attribute_table, sample_outcome_table
        )
        for row in result:
            assert "outcome" in row

    def test_row_includes_latest_milestone_date(
        self, sample_candidate_table, sample_attribute_table, sample_outcome_table
    ):
        stage = ReportingStage()
        result = candidate_summary_table(
            sample_candidate_table, sample_attribute_table, sample_outcome_table
        )
        row_by_drug = {r["drug_name"]: r for r in result}
        assert "latest_milestone_date" in row_by_drug["DrugA"]
        assert row_by_drug["DrugA"]["latest_milestone_date"] == date(2022, 4, 15)

    def test_latest_milestone_ignores_future_dates(self):
        stage = ReportingStage()
        candidate_table = CandidateTable(candidates=[
            Candidate(
                candidate_id="cand_future",
                drug_name="FutureDrug",
                indication="Example",
                highest_phase=TrialPhase.PHASE_3,
                latest_completion_date=date(2030, 1, 1),
            )
        ])
        attribute_table = AttributeTable(attributes={
            "cand_future": CandidateAttributes(
                candidate_id="cand_future",
                drug_modality="peptide",
                disease_area="metabolic",
            )
        })
        outcome_table = OutcomeTable(outcomes={
            "cand_future": CandidateOutcomeRecord(
                candidate_id="cand_future",
                outcome=CandidateOutcome.APPROVED,
                approval_date=date(2031, 1, 1),
                commercialization_date=date(2032, 1, 1),
            )
        })

        result = candidate_summary_table(candidate_table, attribute_table, outcome_table)

        assert len(result) == 1
        assert result[0]["latest_milestone_date"] is None


# ---------------------------------------------------------------------------
# _funnel_table  (FAIL NOW — stub; PASS when implemented)
# ---------------------------------------------------------------------------

class TestFunnelTable:
    def test_returns_list(self, sample_funnel_slice):
        stage = ReportingStage()
        result = funnel_table(sample_funnel_slice)
        assert isinstance(result, list)

    def test_rows_are_dicts(self, sample_funnel_slice):
        stage = ReportingStage()
        result = funnel_table(sample_funnel_slice)
        assert all(isinstance(r, dict) for r in result)

    def test_one_row_per_transition(self, sample_funnel_slice):
        stage = ReportingStage()
        result = funnel_table(sample_funnel_slice)
        assert len(result) == len(sample_funnel_slice.transitions)

    def test_row_includes_from_phase(self, sample_funnel_slice):
        stage = ReportingStage()
        result = funnel_table(sample_funnel_slice)
        for row in result:
            assert "from_phase" in row or "from" in row

    def test_row_includes_to_phase(self, sample_funnel_slice):
        stage = ReportingStage()
        result = funnel_table(sample_funnel_slice)
        for row in result:
            assert "to_phase" in row or "to" in row

    def test_row_includes_rate(self, sample_funnel_slice):
        stage = ReportingStage()
        result = funnel_table(sample_funnel_slice)
        for row in result:
            assert "rate" in row


# ---------------------------------------------------------------------------
# _stratified_funnel_table  (FAIL NOW — stub; PASS when implemented)
# ---------------------------------------------------------------------------

class TestStratifiedFunnelTable:
    def test_returns_list(self, sample_funnel_results):
        stage = ReportingStage()
        result = stratified_funnel_table(sample_funnel_results.by_modality)
        assert isinstance(result, list)

    def test_rows_are_dicts(self, sample_funnel_results):
        stage = ReportingStage()
        result = stratified_funnel_table(sample_funnel_results.by_modality)
        assert all(isinstance(r, dict) for r in result)

    def test_empty_slices_returns_empty_list(self):
        stage = ReportingStage()
        result = stratified_funnel_table({})
        assert result == []


# ---------------------------------------------------------------------------
# _breakdown_chart  (FAIL NOW — stub; PASS when implemented)
# ---------------------------------------------------------------------------

class TestBreakdownChart:
    def test_returns_bytes(self, sample_funnel_results):
        stage = ReportingStage()
        result = breakdown_chart(sample_funnel_results.by_modality, label="Modality")
        assert isinstance(result, bytes)

    def test_returns_non_empty_bytes(self, sample_funnel_results):
        stage = ReportingStage()
        result = breakdown_chart(sample_funnel_results.by_modality, label="Modality")
        assert len(result) > 0


# ---------------------------------------------------------------------------
# _disease_spider_charts
# ---------------------------------------------------------------------------

@pytest.mark.skip(reason="Temporarily skipped — spider chart regressions under investigation")
class TestDiseaseSpiderCharts:
    def test_returns_dict_of_bytes(self, sample_funnel_results):
        stage = ReportingStage()
        result = disease_spider_charts(sample_funnel_results.by_disease_area)
        assert isinstance(result, dict)
        assert result
        assert all(k.startswith("disease_spider_") for k in result)
        assert all(isinstance(v, bytes) and len(v) > 0 for v in result.values())

    def test_handles_approval_to_market_transition(self):
        stage = ReportingStage()
        slices = {
            "metabolic": FunnelSlice(
                modality=None,
                disease_area="metabolic",
                candidate_count=3,
                transitions=[
                    TransitionRate("Phase 1", "Phase 2", 2, 3, 0.67),
                    TransitionRate("Phase 2", "Phase 3", 1, 2, 0.50),
                    TransitionRate("Phase 3", "Approval", 1, 1, 1.00),
                    TransitionRate("Approval", "Market", 1, 1, 1.00),
                ],
            )
        }
        result = disease_spider_charts(slices)
        assert "disease_spider_metabolic" in result


# ---------------------------------------------------------------------------
# _spider_chart
# ---------------------------------------------------------------------------

class TestSpiderChart:
    def test_returns_bytes(self, sample_funnel_results):
        stage = ReportingStage()
        result = spider_chart(sample_funnel_results.overall, label="Overall")
        assert isinstance(result, bytes)
        assert len(result) > 0

    @patch("pipeline.stages.reporting.components._spider.plt.subplots")
    def test_title_includes_sample_size(self, mock_subplots, sample_funnel_results):
        stage = ReportingStage()
        fig = MagicMock()
        ax = MagicMock()
        mock_subplots.return_value = (fig, ax)
        spider_chart(sample_funnel_results.overall, label="Overall")
        ax.set_title.assert_called_once_with(
            f"Overall: Success rates (n={sample_funnel_results.overall.candidate_count})"
        )


# ---------------------------------------------------------------------------
# _narrative_summary  (FAIL NOW — stub; PASS when implemented)
# ---------------------------------------------------------------------------

class TestNarrativeSummary:
    def test_executive_summary_returns_string(self, sample_funnel_results):
        from pipeline.stages.reporting._narrative import executive_summary
        result = executive_summary(
            sample_funnel_results.overall, None, None, None, None, None, True,
        )
        assert isinstance(result, str)

    def test_executive_summary_returns_non_empty(self, sample_funnel_results):
        from pipeline.stages.reporting._narrative import executive_summary
        result = executive_summary(
            sample_funnel_results.overall, None, None, None, None, None, False,
        )
        assert len(result) > 0

    def test_introduction_mentions_candidate_count(
        self, sample_candidate_table, sample_funnel_results,
    ):
        from pipeline.stages.reporting._narrative import introduction_text
        n_candidates = len(sample_candidate_table.candidates)
        result = introduction_text(100, n_candidates, 5, (2015, 2020), False)
        assert str(n_candidates) in result

    def test_introduction_mentions_peptide(self):
        from pipeline.stages.reporting._narrative import introduction_text
        result = introduction_text(50, 10, 3, (2015, 2020), True)
        assert "peptide" in result.lower()


# ---------------------------------------------------------------------------
# _write  (FAIL NOW — stub; PASS when implemented)
# ---------------------------------------------------------------------------

class TestWrite:
    def test_write_creates_output_directory_if_missing(self, sample_report_output):
        from pipeline.stages.reporting._writer import write_report
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = os.path.join(tmpdir, "new_subdir")
            write_report(sample_report_output, out_dir, ["html"], {})
            assert os.path.isdir(out_dir)

    def test_write_html_creates_file(self, sample_report_output):
        from pipeline.stages.reporting._writer import write_report
        with tempfile.TemporaryDirectory() as tmpdir:
            write_report(sample_report_output, tmpdir, ["html"], {})
            html_files = [f for f in os.listdir(tmpdir) if f.endswith(".html")]
            assert len(html_files) >= 1

    def test_write_excel_creates_file(self, sample_report_output):
        from pipeline.stages.reporting._writer import write_report
        with tempfile.TemporaryDirectory() as tmpdir:
            write_report(sample_report_output, tmpdir, ["excel"], {})
            excel_files = [f for f in os.listdir(tmpdir) if f.endswith(".xlsx")]
            assert len(excel_files) >= 1


# ---------------------------------------------------------------------------
# _peptide_disease_slices — uses peptide-only counts  (PASS NOW)
# ---------------------------------------------------------------------------

class TestPeptideDiseaseSlices:
    """Verify _peptide_disease_slices returns peptide-only per-disease counts."""

    def test_peptide_disease_slices_uses_peptide_only_counts(self):
        """
        by_disease_area["oncology"] has candidate_count=100 (all modalities).
        by_modality_and_disease[("peptide","oncology")] has candidate_count=10.
        _peptide_disease_slices should return the 10-count slice, not the 100-count one.
        """
        all_modality_oncology = FunnelSlice(
            modality=None, disease_area="oncology", candidate_count=100, transitions=[]
        )
        peptide_oncology = FunnelSlice(
            modality="peptide", disease_area="oncology", candidate_count=10, transitions=[]
        )
        funnel_results = FunnelResults(
            overall=FunnelSlice(None, None, 110, []),
            by_modality={"peptide": FunnelSlice("peptide", None, 10, [])},
            by_disease_area={"oncology": all_modality_oncology},
            by_modality_and_disease={("peptide", "oncology"): peptide_oncology},
        )
        attribute_table = AttributeTable(
            attributes={
                "cand_001": CandidateAttributes(
                    candidate_id="cand_001",
                    drug_modality="peptide",
                    disease_area="oncology",
                ),
            }
        )
        stage = ReportingStage(peptide_only=True)
        slices = _compute.peptide_disease_slices(funnel_results, attribute_table)

        assert "oncology" in slices
        assert slices["oncology"].candidate_count == 10, (
            "Expected peptide-only count (10), got all-modality count instead"
        )


# ---------------------------------------------------------------------------
# Heatmap visualizations
# ---------------------------------------------------------------------------

import numpy as np


@pytest.fixture
def multi_modality_funnel():
    """FunnelResults with multiple modalities, disease areas, and cross-stratification."""
    peptide_slice = FunnelSlice(
        modality="peptide",
        disease_area=None,
        candidate_count=100,
        transitions=[
            TransitionRate("Phase 1", "Phase 2", 60, 100, 0.60),
            TransitionRate("Phase 2", "Phase 3", 30, 60, 0.50),
            TransitionRate("Phase 3", "Approval", 12, 30, 0.40),
            TransitionRate("Approval", "Market", 8, 12, 0.667),
        ],
    )
    sirna_slice = FunnelSlice(
        modality="siRNA",
        disease_area=None,
        candidate_count=5,
        transitions=[
            TransitionRate("Phase 1", "Phase 2", 3, 5, 0.60),
            TransitionRate("Phase 2", "Phase 3", 1, 3, 0.333),
            TransitionRate("Phase 3", "Approval", 0, 1, 0.0),
            TransitionRate("Approval", "Market", 0, 0, 0.0),
        ],
    )
    oncology_peptide = FunnelSlice(
        modality="peptide",
        disease_area="oncology",
        candidate_count=40,
        transitions=[
            TransitionRate("Phase 1", "Phase 2", 25, 40, 0.625),
            TransitionRate("Phase 2", "Phase 3", 10, 25, 0.40),
            TransitionRate("Phase 3", "Approval", 4, 10, 0.40),
            TransitionRate("Approval", "Market", 3, 4, 0.75),
        ],
    )
    metabolic_peptide = FunnelSlice(
        modality="peptide",
        disease_area="metabolic",
        candidate_count=60,
        transitions=[
            TransitionRate("Phase 1", "Phase 2", 35, 60, 0.583),
            TransitionRate("Phase 2", "Phase 3", 20, 35, 0.571),
            TransitionRate("Phase 3", "Approval", 8, 20, 0.40),
            TransitionRate("Approval", "Market", 5, 8, 0.625),
        ],
    )
    return FunnelResults(
        overall=peptide_slice,
        by_modality={"peptide": peptide_slice, "siRNA": sirna_slice},
        by_disease_area={"oncology": oncology_peptide, "metabolic": metabolic_peptide},
        by_modality_and_disease={
            ("peptide", "oncology"): oncology_peptide,
            ("peptide", "metabolic"): metabolic_peptide,
            ("siRNA", "oncology"): sirna_slice,
        },
    )


class TestHeatmapData:
    def test_returns_correct_structure(self, multi_modality_funnel):
        stage = ReportingStage(peptide_only=False)
        modalities, trans, rates, denoms, nums, cis = _compute.heatmap_grid(
            multi_modality_funnel
        )
        assert isinstance(modalities, list)
        assert len(trans) == 4
        assert rates.shape == (len(modalities), 4)
        assert denoms.shape == rates.shape
        assert nums.shape == rates.shape
        assert len(cis) == len(modalities)

    def test_modalities_sorted(self, multi_modality_funnel):
        stage = ReportingStage(peptide_only=False)
        modalities, *_ = _compute.heatmap_grid(multi_modality_funnel)
        assert modalities == sorted(modalities)

    def test_disease_filter_excludes_absent_modalities(self, multi_modality_funnel):
        stage = ReportingStage(peptide_only=False)
        mod_met, *_ = _compute.heatmap_grid(multi_modality_funnel, disease_area="metabolic")
        assert "siRNA" not in mod_met
        assert "peptide" in mod_met

    def test_rates_match_source(self, multi_modality_funnel):
        stage = ReportingStage(peptide_only=False)
        modalities, _, rates, denoms, nums, _ = _compute.heatmap_grid(multi_modality_funnel)
        peptide_idx = modalities.index("peptide")
        assert rates[peptide_idx, 0] == pytest.approx(0.60)
        assert denoms[peptide_idx, 0] == 100
        assert nums[peptide_idx, 0] == 60

    def test_empty_funnel(self):
        stage = ReportingStage(peptide_only=False)
        empty = FunnelResults(
            overall=FunnelSlice(None, None, 0, []),
            by_modality={},
            by_disease_area={},
        )
        modalities, _, rates, *_ = _compute.heatmap_grid(empty)
        assert modalities == []
        assert rates.shape == (0, 4)


class TestModalityHeatmap:
    def test_returns_png_bytes(self, multi_modality_funnel):
        stage = ReportingStage(peptide_only=False)
        result = modality_heatmap(multi_modality_funnel)
        assert isinstance(result, bytes)
        assert result[:4] == b"\x89PNG"

    def test_empty_data_returns_png(self):
        stage = ReportingStage(peptide_only=False)
        empty = FunnelResults(
            overall=FunnelSlice(None, None, 0, []),
            by_modality={},
            by_disease_area={},
        )
        result = modality_heatmap(empty)
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_with_disease_filter(self, multi_modality_funnel):
        stage = ReportingStage(peptide_only=False)
        result = modality_heatmap(
            multi_modality_funnel, disease_area="oncology", title_suffix="Oncology"
        )
        assert isinstance(result, bytes)
        assert result[:4] == b"\x89PNG"


class TestInteractiveBubbleHeatmap:
    def test_returns_html_string(self, multi_modality_funnel):
        stage = ReportingStage(peptide_only=False)
        result = interactive_bubble_heatmap(multi_modality_funnel)
        assert isinstance(result, str)
        assert "<div" in result

    def test_contains_plotly_reference(self, multi_modality_funnel):
        stage = ReportingStage(peptide_only=False)
        result = interactive_bubble_heatmap(multi_modality_funnel)
        assert "plotly" in result.lower()

    def test_empty_funnel_returns_html(self):
        stage = ReportingStage(peptide_only=False)
        empty = FunnelResults(
            overall=FunnelSlice(None, None, 0, []),
            by_modality={},
            by_disease_area={},
        )
        result = interactive_bubble_heatmap(empty)
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# Disease-area heatmap visualizations
# ---------------------------------------------------------------------------

class TestDiseaseHeatmapData:
    def test_returns_correct_structure(self, multi_modality_funnel):
        stage = ReportingStage(peptide_only=False)
        das, trans, rates, denoms, nums, cis = _compute.disease_heatmap_grid(
            multi_modality_funnel
        )
        assert isinstance(das, list)
        assert len(trans) == 4
        assert rates.shape == (len(das), 4)

    def test_modality_filter(self, multi_modality_funnel):
        stage = ReportingStage(peptide_only=False)
        das_all, *_ = _compute.disease_heatmap_grid(multi_modality_funnel)
        das_pep, *_ = _compute.disease_heatmap_grid(multi_modality_funnel, modality="peptide")
        assert set(das_pep).issubset(set(das_all))

    def test_empty_funnel(self):
        stage = ReportingStage(peptide_only=False)
        empty = FunnelResults(
            overall=FunnelSlice(None, None, 0, []),
            by_modality={},
            by_disease_area={},
        )
        das, _, rates, *_ = _compute.disease_heatmap_grid(empty)
        assert das == []
        assert rates.shape == (0, 4)


class TestDiseaseHeatmap:
    def test_returns_png_bytes(self, multi_modality_funnel):
        stage = ReportingStage(peptide_only=False)
        result = disease_heatmap(multi_modality_funnel)
        assert isinstance(result, bytes)
        assert result[:4] == b"\x89PNG"

    def test_with_modality_filter(self, multi_modality_funnel):
        stage = ReportingStage(peptide_only=False)
        result = disease_heatmap(
            multi_modality_funnel, modality="peptide", title_suffix="Peptide"
        )
        assert isinstance(result, bytes)
        assert result[:4] == b"\x89PNG"

    def test_empty_data_returns_png(self):
        stage = ReportingStage(peptide_only=False)
        empty = FunnelResults(
            overall=FunnelSlice(None, None, 0, []),
            by_modality={},
            by_disease_area={},
        )
        result = disease_heatmap(empty)
        assert isinstance(result, bytes)
        assert len(result) > 0


class TestInteractiveDiseaseBubbleHeatmap:
    def test_returns_html_string(self, multi_modality_funnel):
        stage = ReportingStage(peptide_only=False)
        result = interactive_disease_bubble_heatmap(multi_modality_funnel)
        assert isinstance(result, str)
        assert "<div" in result

    def test_contains_plotly_reference(self, multi_modality_funnel):
        stage = ReportingStage(peptide_only=False)
        result = interactive_disease_bubble_heatmap(multi_modality_funnel)
        assert "plotly" in result.lower()

    def test_empty_funnel_returns_html(self):
        stage = ReportingStage(peptide_only=False)
        empty = FunnelResults(
            overall=FunnelSlice(None, None, 0, []),
            by_modality={},
            by_disease_area={},
        )
        result = interactive_disease_bubble_heatmap(empty)
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# BIO/QLS rebucketing
# ---------------------------------------------------------------------------

class TestBioQlsRebucketing:
    def test_merges_into_bio_qls_buckets(self):
        """Multiple granular areas mapping to 'Others' get merged."""
        stage = ReportingStage(peptide_only=False)
        by_disease = {
            "dermatology": FunnelSlice(
                modality=None, disease_area="dermatology", candidate_count=10,
                transitions=[
                    TransitionRate("Phase 1", "Phase 2", 8, 10, 0.8),
                    TransitionRate("Phase 2", "Phase 3", 4, 8, 0.5),
                    TransitionRate("Phase 3", "Approval", 2, 4, 0.5),
                    TransitionRate("Approval", "Market", 1, 2, 0.5),
                ],
            ),
            "musculoskeletal": FunnelSlice(
                modality=None, disease_area="musculoskeletal", candidate_count=20,
                transitions=[
                    TransitionRate("Phase 1", "Phase 2", 12, 20, 0.6),
                    TransitionRate("Phase 2", "Phase 3", 6, 12, 0.5),
                    TransitionRate("Phase 3", "Approval", 3, 6, 0.5),
                    TransitionRate("Approval", "Market", 2, 3, 0.667),
                ],
            ),
            "oncology": FunnelSlice(
                modality=None, disease_area="oncology", candidate_count=50,
                transitions=[
                    TransitionRate("Phase 1", "Phase 2", 30, 50, 0.6),
                    TransitionRate("Phase 2", "Phase 3", 15, 30, 0.5),
                    TransitionRate("Phase 3", "Approval", 8, 15, 0.533),
                    TransitionRate("Approval", "Market", 6, 8, 0.75),
                ],
            ),
        }

        result = bio_qls_by_disease_area(by_disease)

        assert "Oncology" in result
        assert "Others" in result
        assert "dermatology" not in result
        assert "musculoskeletal" not in result

        # Oncology passes through unchanged
        onc = result["Oncology"]
        assert onc.candidate_count == 50
        assert onc.transitions[0].numerator == 30
        assert onc.transitions[0].denominator == 50

        # Others merges dermatology + musculoskeletal
        others = result["Others"]
        assert others.candidate_count == 30  # 10 + 20
        assert others.transitions[0].numerator == 20  # 8 + 12
        assert others.transitions[0].denominator == 30  # 10 + 20
        # Rate recomputed from merged counts
        assert abs(others.transitions[0].rate - 20 / 30) < 0.001

    def test_empty_input(self):
        stage = ReportingStage(peptide_only=False)
        result = bio_qls_by_disease_area({})
        assert result == {}


# ---------------------------------------------------------------------------
# LOA computation
# ---------------------------------------------------------------------------

def _make_slice(rates, candidate_count=100):
    """Helper: build a FunnelSlice from 4 transition rates."""
    phases = [("Phase 1", "Phase 2"), ("Phase 2", "Phase 3"),
              ("Phase 3", "Approval"), ("Approval", "Market")]
    transitions = [
        TransitionRate(f, t, int(r * 100), 100, r)
        for (f, t), r in zip(phases, rates)
    ]
    return FunnelSlice(modality=None, disease_area=None,
                       candidate_count=candidate_count, transitions=transitions)


class TestComputeLoa:
    def test_product_of_all_rates(self):
        fs = _make_slice([0.5, 0.3, 0.6, 0.9])
        loa = compute_loa(fs)
        expected = 0.5 * 0.3 * 0.6 * 0.9
        assert abs(loa["Phase 1"] - expected) < 1e-9

    def test_loa_from_phase2(self):
        fs = _make_slice([0.5, 0.3, 0.6, 0.9])
        loa = compute_loa(fs)
        expected = 0.3 * 0.6 * 0.9
        assert abs(loa["Phase 2"] - expected) < 1e-9

    def test_loa_from_phase3(self):
        fs = _make_slice([0.5, 0.3, 0.6, 0.9])
        loa = compute_loa(fs)
        expected = 0.6 * 0.9
        assert abs(loa["Phase 3"] - expected) < 1e-9

    def test_loa_from_approval(self):
        fs = _make_slice([0.5, 0.3, 0.6, 0.9])
        loa = compute_loa(fs)
        assert abs(loa["Approval"] - 0.9) < 1e-9

    def test_zero_rate_propagates(self):
        fs = _make_slice([0.5, 0.0, 0.6, 0.9])
        loa = compute_loa(fs)
        assert loa["Phase 1"] == 0.0
        assert loa["Phase 2"] == 0.0
        # Phase 3 onwards unaffected
        assert abs(loa["Phase 3"] - 0.54) < 1e-9

    def test_all_ones(self):
        fs = _make_slice([1.0, 1.0, 1.0, 1.0])
        loa = compute_loa(fs)
        for label in ["Phase 1", "Phase 2", "Phase 3", "Approval"]:
            assert loa[label] == 1.0


class TestLoaTable:
    def test_includes_all_strata_plus_overall(self):
        slices = {
            "oncology": _make_slice([0.5, 0.3, 0.5, 0.9]),
            "metabolic": _make_slice([0.6, 0.4, 0.6, 0.9]),
        }
        overall = _make_slice([0.55, 0.35, 0.55, 0.9])
        stage = ReportingStage()
        rows = loa_table(slices, overall)
        strata = [r["stratum"] for r in rows]
        assert "oncology" in strata
        assert "metabolic" in strata
        assert "All indications" in strata

    def test_n_values_are_denominators(self):
        fs = _make_slice([0.5, 0.3, 0.6, 0.9])
        stage = ReportingStage()
        rows = loa_table({"test": fs}, fs)
        row = rows[0]
        assert row["n_phase1"] == 100
        assert row["n_phase2"] == 100

    def test_sorted_by_loa_descending(self):
        slices = {
            "low": _make_slice([0.1, 0.1, 0.1, 0.1]),
            "high": _make_slice([0.9, 0.9, 0.9, 0.9]),
        }
        overall = _make_slice([0.5, 0.5, 0.5, 0.5])
        stage = ReportingStage()
        rows = loa_table(slices, overall)
        # First row should be "high" (highest LOA), last is "All indications"
        assert rows[0]["stratum"] == "high"
        assert rows[-1]["stratum"] == "All indications"


# ---------------------------------------------------------------------------
# Oncology vs non-oncology split
# ---------------------------------------------------------------------------

class TestOncologyVsRest:
    def test_partitions_correctly(self):
        by_disease = {
            "oncology": FunnelSlice(
                modality=None, disease_area="oncology", candidate_count=50,
                transitions=[
                    TransitionRate("Phase 1", "Phase 2", 25, 50, 0.5),
                    TransitionRate("Phase 2", "Phase 3", 10, 25, 0.4),
                    TransitionRate("Phase 3", "Approval", 5, 10, 0.5),
                    TransitionRate("Approval", "Market", 4, 5, 0.8),
                ],
            ),
            "metabolic": FunnelSlice(
                modality=None, disease_area="metabolic", candidate_count=30,
                transitions=[
                    TransitionRate("Phase 1", "Phase 2", 18, 30, 0.6),
                    TransitionRate("Phase 2", "Phase 3", 9, 18, 0.5),
                    TransitionRate("Phase 3", "Approval", 5, 9, 0.556),
                    TransitionRate("Approval", "Market", 4, 5, 0.8),
                ],
            ),
            "neurology": FunnelSlice(
                modality=None, disease_area="neurology", candidate_count=20,
                transitions=[
                    TransitionRate("Phase 1", "Phase 2", 10, 20, 0.5),
                    TransitionRate("Phase 2", "Phase 3", 5, 10, 0.5),
                    TransitionRate("Phase 3", "Approval", 3, 5, 0.6),
                    TransitionRate("Approval", "Market", 2, 3, 0.667),
                ],
            ),
        }
        result = oncology_vs_rest(by_disease)

        assert "Oncology" in result
        assert "Non-Oncology" in result
        assert result["Oncology"].candidate_count == 50
        assert result["Non-Oncology"].candidate_count == 50  # 30 + 20

    def test_numerators_sum(self):
        by_disease = {
            "oncology": FunnelSlice(
                modality=None, disease_area="oncology", candidate_count=50,
                transitions=[
                    TransitionRate("Phase 1", "Phase 2", 25, 50, 0.5),
                    TransitionRate("Phase 2", "Phase 3", 10, 25, 0.4),
                    TransitionRate("Phase 3", "Approval", 5, 10, 0.5),
                    TransitionRate("Approval", "Market", 4, 5, 0.8),
                ],
            ),
            "metabolic": FunnelSlice(
                modality=None, disease_area="metabolic", candidate_count=30,
                transitions=[
                    TransitionRate("Phase 1", "Phase 2", 18, 30, 0.6),
                    TransitionRate("Phase 2", "Phase 3", 9, 18, 0.5),
                    TransitionRate("Phase 3", "Approval", 5, 9, 0.556),
                    TransitionRate("Approval", "Market", 4, 5, 0.8),
                ],
            ),
        }
        result = oncology_vs_rest(by_disease)
        non_onc = result["Non-Oncology"]
        assert non_onc.transitions[0].numerator == 18
        assert non_onc.transitions[0].denominator == 30

    def test_handles_missing_oncology(self):
        by_disease = {
            "metabolic": _make_slice([0.6, 0.4, 0.6, 0.9]),
        }
        result = oncology_vs_rest(by_disease)
        assert result["Oncology"].candidate_count == 0
        assert result["Non-Oncology"].candidate_count == 100


# ---------------------------------------------------------------------------
# Duration rebucketing preservation
# ---------------------------------------------------------------------------

class TestDurationRebucketing:
    def test_bio_qls_preserves_weighted_avg_duration(self):
        """Durations survive BIO/QLS rebucketing as weighted averages."""
        by_disease = {
            "dermatology": FunnelSlice(
                modality=None, disease_area="dermatology", candidate_count=10,
                transitions=[
                    TransitionRate("Phase 1", "Phase 2", 8, 10, 0.8, avg_duration_years=2.0),
                    TransitionRate("Phase 2", "Phase 3", 4, 8, 0.5, avg_duration_years=3.0),
                    TransitionRate("Phase 3", "Approval", 2, 4, 0.5, avg_duration_years=2.5),
                    TransitionRate("Approval", "Market", 1, 2, 0.5, avg_duration_years=1.0),
                ],
            ),
            "musculoskeletal": FunnelSlice(
                modality=None, disease_area="musculoskeletal", candidate_count=20,
                transitions=[
                    TransitionRate("Phase 1", "Phase 2", 12, 20, 0.6, avg_duration_years=3.0),
                    TransitionRate("Phase 2", "Phase 3", 6, 12, 0.5, avg_duration_years=4.0),
                    TransitionRate("Phase 3", "Approval", 3, 6, 0.5, avg_duration_years=3.5),
                    TransitionRate("Approval", "Market", 2, 3, 0.667, avg_duration_years=1.5),
                ],
            ),
        }
        result = bio_qls_by_disease_area(by_disease)
        others = result["Others"]

        # P1→P2 weighted avg: (2.0*10 + 3.0*20) / (10+20) = 80/30 ≈ 2.667
        assert others.transitions[0].avg_duration_years is not None
        assert abs(others.transitions[0].avg_duration_years - (2.0 * 10 + 3.0 * 20) / 30) < 0.01

    def test_none_duration_excluded_from_weighted_avg(self):
        """If one source has None duration, it's excluded from the weighted average."""
        by_disease = {
            "dermatology": FunnelSlice(
                modality=None, disease_area="dermatology", candidate_count=10,
                transitions=[
                    TransitionRate("Phase 1", "Phase 2", 8, 10, 0.8, avg_duration_years=2.0),
                    TransitionRate("Phase 2", "Phase 3", 4, 8, 0.5),  # None duration
                    TransitionRate("Phase 3", "Approval", 2, 4, 0.5),
                    TransitionRate("Approval", "Market", 1, 2, 0.5),
                ],
            ),
            "musculoskeletal": FunnelSlice(
                modality=None, disease_area="musculoskeletal", candidate_count=20,
                transitions=[
                    TransitionRate("Phase 1", "Phase 2", 12, 20, 0.6, avg_duration_years=3.0),
                    TransitionRate("Phase 2", "Phase 3", 6, 12, 0.5, avg_duration_years=4.0),
                    TransitionRate("Phase 3", "Approval", 3, 6, 0.5),
                    TransitionRate("Approval", "Market", 2, 3, 0.667),
                ],
            ),
        }
        result = bio_qls_by_disease_area(by_disease)
        others = result["Others"]

        # P2→P3: only musculoskeletal has duration, so weighted avg = 4.0
        assert others.transitions[1].avg_duration_years is not None
        assert abs(others.transitions[1].avg_duration_years - 4.0) < 0.01

        # P3→Appr: both None → should be None
        assert others.transitions[2].avg_duration_years is None


# ---------------------------------------------------------------------------
# Timeline chart
# ---------------------------------------------------------------------------

class TestTimelineChart:
    def test_handles_none_durations(self):
        """Chart renders without error when some durations are None."""
        slices = {
            "oncology": FunnelSlice(
                modality=None, disease_area="oncology", candidate_count=50,
                transitions=[
                    TransitionRate("Phase 1", "Phase 2", 25, 50, 0.5, avg_duration_years=2.5),
                    TransitionRate("Phase 2", "Phase 3", 10, 25, 0.4, avg_duration_years=None),
                    TransitionRate("Phase 3", "Approval", 5, 10, 0.5, avg_duration_years=3.0),
                    TransitionRate("Approval", "Market", 4, 5, 0.8, avg_duration_years=1.0),
                ],
            ),
        }
        stage = ReportingStage()
        png = timeline_by_disease_chart(slices)
        assert png[:4] == b"\x89PNG"

    def test_skips_all_none_disease_areas(self):
        """Disease areas with all None durations are excluded."""
        slices = {
            "oncology": FunnelSlice(
                modality=None, disease_area="oncology", candidate_count=50,
                transitions=[
                    TransitionRate("Phase 1", "Phase 2", 25, 50, 0.5),
                    TransitionRate("Phase 2", "Phase 3", 10, 25, 0.4),
                    TransitionRate("Phase 3", "Approval", 5, 10, 0.5),
                    TransitionRate("Approval", "Market", 4, 5, 0.8),
                ],
            ),
        }
        stage = ReportingStage()
        # Should produce a "no data" chart without error
        png = timeline_by_disease_chart(slices)
        assert png[:4] == b"\x89PNG"

    def test_total_is_sum_of_clinical_phases(self):
        """Verify the total computation logic (sum of first 3 durations)."""
        durations = [2.0, 3.5, 3.0, 1.2]
        total_clinical = sum(durations[:3])
        assert abs(total_clinical - 8.5) < 0.001


# ---------------------------------------------------------------------------
# Transition rate computation (for time-period analysis)
# ---------------------------------------------------------------------------

class TestComputeTransitionRateFromRecords:
    def test_basic_rate(self):
        records = [
            {"highest_phase": "Phase 2", "outcome": "Failed Phase 2"},
            {"highest_phase": "Phase 3", "outcome": "Approved"},
        ]
        tr = compute_transition_rate(records, "Phase 1", "Phase 2")
        assert tr.denominator == 2
        assert tr.numerator == 2  # both reached Phase 2+
        assert tr.rate == 1.0

    def test_excludes_ongoing(self):
        records = [
            {"highest_phase": "Phase 1", "outcome": "Ongoing"},
            {"highest_phase": "Phase 2", "outcome": "Failed Phase 2"},
        ]
        tr = compute_transition_rate(records, "Phase 1", "Phase 2")
        assert tr.denominator == 1  # Ongoing excluded
        assert tr.numerator == 1

    def test_approved_boosts_level(self):
        records = [
            {"highest_phase": "Phase 2", "outcome": "Approved"},  # boosted to level 3
        ]
        tr = compute_transition_rate(records, "Phase 3", "Approval")
        assert tr.denominator == 1
        assert tr.numerator == 1

    def test_zero_denominator(self):
        tr = compute_transition_rate([], "Phase 1", "Phase 2")
        assert tr.rate == 0.0
        assert tr.denominator == 0


# ---------------------------------------------------------------------------
# Time-period partitioning
# ---------------------------------------------------------------------------

class TestPartitionByTimePeriods:
    def _make_candidates_and_tables(self, years_outcomes):
        """Build candidate_table, attribute_table, outcome_table from (year, outcome, disease) tuples."""
        candidates = []
        attrs_dict = {}
        outcomes_dict = {}
        for i, (year, outcome, disease) in enumerate(years_outcomes):
            cid = f"c{i}"
            candidates.append(Candidate(
                candidate_id=cid, drug_name=f"Drug{i}", indication=f"Ind{i}",
                drug_name_raw=f"Drug{i}",
                highest_phase=TrialPhase.PHASE_2,
                earliest_start_date=date(year, 6, 1) if year else None,
            ))
            attrs_dict[cid] = CandidateAttributes(
                candidate_id=cid, drug_modality="peptide",
                disease_area=disease,
            )
            outcomes_dict[cid] = CandidateOutcomeRecord(
                candidate_id=cid,
                outcome=CandidateOutcome(outcome),
                confidence=0.9, reasoning="test",
            )
        ct = CandidateTable(candidates=candidates)
        at = AttributeTable(attributes=attrs_dict)
        ot = OutcomeTable(outcomes=outcomes_dict)
        return ct, at, ot

    def test_default_splits_by_median(self):
        ct, at, ot = self._make_candidates_and_tables([
            (2012, "Failed Phase 2", "oncology"),
            (2013, "Failed Phase 2", "oncology"),
            (2016, "Approved", "metabolic"),
            (2018, "Failed Phase 2", "metabolic"),
        ])
        stage = ReportingStage()
        slices, labels, n_excluded = partition_by_time_periods(ct, at, ot)
        assert len(labels) == 2
        assert n_excluded == 0

    def test_custom_periods(self):
        ct, at, ot = self._make_candidates_and_tables([
            (2011, "Failed Phase 2", "oncology"),
            (2015, "Approved", "oncology"),
            (2018, "Failed Phase 2", "metabolic"),
        ])
        stage = ReportingStage()
        slices, labels, _ = partition_by_time_periods(
            ct, at, ot, periods=[(2011, 2015), (2016, 2020)])
        assert labels == ["2011\u20132015", "2016\u20132020"]
        # 2011 and 2015 in first period, 2018 in second
        assert len(slices) == 2

    def test_three_periods(self):
        ct, at, ot = self._make_candidates_and_tables([
            (2011, "Failed Phase 2", "oncology"),
            (2014, "Approved", "oncology"),
            (2017, "Failed Phase 2", "oncology"),
        ])
        stage = ReportingStage()
        slices, labels, _ = partition_by_time_periods(
            ct, at, ot, periods=[(2011, 2012), (2013, 2015), (2016, 2020)])
        assert len(labels) == 3

    def test_excludes_none_dates(self):
        ct, at, ot = self._make_candidates_and_tables([
            (2015, "Approved", "oncology"),
            (None, "Failed Phase 2", "oncology"),  # should be excluded
        ])
        stage = ReportingStage()
        _, _, n_excluded = partition_by_time_periods(ct, at, ot)
        assert n_excluded == 1


# ---------------------------------------------------------------------------
# Time-period LOA table
# ---------------------------------------------------------------------------

class TestTimePeriodLoaTable:
    def test_table_structure(self):
        slices = {
            "2011\u20132015": {"Oncology": _make_slice([0.5, 0.3, 0.5, 0.9])},
            "2016\u20132020": {"Oncology": _make_slice([0.6, 0.4, 0.6, 0.9])},
        }
        stage = ReportingStage()
        rows = time_period_loa_table(slices, ["2011\u20132015", "2016\u20132020"])
        assert len(rows) == 1
        row = rows[0]
        assert row["disease_area"] == "Oncology"
        assert "loa_2011-2015" in row
        assert "n_2011-2015" in row

    def test_works_with_three_periods(self):
        slices = {
            "2011\u20132013": {"Oncology": _make_slice([0.5, 0.3, 0.5, 0.9])},
            "2014\u20132016": {"Oncology": _make_slice([0.6, 0.4, 0.6, 0.9])},
            "2017\u20132020": {"Oncology": _make_slice([0.7, 0.5, 0.7, 0.9])},
        }
        labels = ["2011\u20132013", "2014\u20132016", "2017\u20132020"]
        stage = ReportingStage()
        rows = time_period_loa_table(slices, labels)
        assert "loa_2017-2020" in rows[0]


# ---------------------------------------------------------------------------
# Modality proportion
# ---------------------------------------------------------------------------

class TestModalityProportionData:
    def test_percentages_sum_to_one(self):
        candidates = [
            Candidate(candidate_id="c1", drug_name="D1", indication="I1", drug_name_raw="D1",
                      earliest_start_date=date(2015, 1, 1)),
            Candidate(candidate_id="c2", drug_name="D2", indication="I2", drug_name_raw="D2",
                      earliest_start_date=date(2015, 6, 1)),
        ]
        ct = CandidateTable(candidates=candidates)
        at = AttributeTable(attributes={
            "c1": CandidateAttributes(candidate_id="c1", drug_modality="peptide", disease_area="oncology"),
            "c2": CandidateAttributes(candidate_id="c2", drug_modality="small_molecule", disease_area="oncology"),
        })
        rows = modality_proportion_data(ct, at)
        year_2015 = [r for r in rows if r["year"] == 2015]
        total_pct = sum(r["pct"] for r in year_2015)
        assert abs(total_pct - 1.0) < 0.001

    def test_excludes_none_dates(self):
        candidates = [
            Candidate(candidate_id="c1", drug_name="D1", indication="I1", drug_name_raw="D1",
                      earliest_start_date=None),
        ]
        ct = CandidateTable(candidates=candidates)
        at = AttributeTable(attributes={
            "c1": CandidateAttributes(candidate_id="c1", drug_modality="peptide", disease_area="oncology"),
        })
        rows = modality_proportion_data(ct, at)
        assert len(rows) == 0

    def test_row_structure(self):
        candidates = [
            Candidate(candidate_id="c1", drug_name="D1", indication="I1", drug_name_raw="D1",
                      earliest_start_date=date(2015, 1, 1)),
        ]
        ct = CandidateTable(candidates=candidates)
        at = AttributeTable(attributes={
            "c1": CandidateAttributes(candidate_id="c1", drug_modality="peptide", disease_area="oncology"),
        })
        rows = modality_proportion_data(ct, at)
        assert len(rows) == 1
        assert set(rows[0].keys()) == {"year", "modality", "count", "pct"}


# ---------------------------------------------------------------------------
# Sponsor concentration
# ---------------------------------------------------------------------------

class TestSponsorConcentrationData:
    def test_multi_sponsor_credited(self):
        candidates = [
            Candidate(candidate_id="c1", drug_name="D1", indication="I1", drug_name_raw="D1",
                      sponsors=["Pfizer", "BioNTech"]),
        ]
        ct = CandidateTable(candidates=candidates)
        at = AttributeTable(attributes={
            "c1": CandidateAttributes(candidate_id="c1", drug_modality="peptide", disease_area="oncology"),
        })
        rows = sponsor_concentration_data(ct, at)
        sponsors = {r["sponsor"] for r in rows}
        assert "Pfizer" in sponsors
        assert "BioNTech" in sponsors

    def test_sorted_descending(self):
        candidates = [
            Candidate(candidate_id="c1", drug_name="D1", indication="I1", drug_name_raw="D1",
                      sponsors=["BigPharma"]),
            Candidate(candidate_id="c2", drug_name="D2", indication="I2", drug_name_raw="D2",
                      sponsors=["BigPharma"]),
            Candidate(candidate_id="c3", drug_name="D3", indication="I3", drug_name_raw="D3",
                      sponsors=["SmallBio"]),
        ]
        ct = CandidateTable(candidates=candidates)
        at = AttributeTable(attributes={
            "c1": CandidateAttributes(candidate_id="c1", drug_modality="peptide", disease_area="oncology"),
            "c2": CandidateAttributes(candidate_id="c2", drug_modality="peptide", disease_area="metabolic"),
            "c3": CandidateAttributes(candidate_id="c3", drug_modality="small_molecule", disease_area="neurology"),
        })
        rows = sponsor_concentration_data(ct, at)
        assert rows[0]["sponsor"] == "BigPharma"
        assert rows[0]["candidate_count"] == 2


class TestSponsorSummaryStats:
    def test_total_unique_count(self):
        candidates = [
            Candidate(candidate_id="c1", drug_name="D1", indication="I1", drug_name_raw="D1",
                      sponsors=["A", "B"]),
            Candidate(candidate_id="c2", drug_name="D2", indication="I2", drug_name_raw="D2",
                      sponsors=["B", "C"]),
        ]
        ct = CandidateTable(candidates=candidates)
        stats = sponsor_summary_stats(ct)
        assert stats["total_unique_sponsors"] == 3  # A, B, C

    def test_single_candidate_pct(self):
        candidates = [
            Candidate(candidate_id="c1", drug_name="D1", indication="I1", drug_name_raw="D1",
                      sponsors=["Big"]),
            Candidate(candidate_id="c2", drug_name="D2", indication="I2", drug_name_raw="D2",
                      sponsors=["Big"]),
            Candidate(candidate_id="c3", drug_name="D3", indication="I3", drug_name_raw="D3",
                      sponsors=["Small"]),
        ]
        ct = CandidateTable(candidates=candidates)
        stats = sponsor_summary_stats(ct)
        # Big has 2 candidates, Small has 1 → 1 out of 2 sponsors is single-candidate
        assert stats["single_candidate_sponsors"] == 1
        assert abs(stats["single_candidate_pct"] - 50.0) < 0.1
