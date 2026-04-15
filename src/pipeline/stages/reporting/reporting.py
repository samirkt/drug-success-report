"""
ReportingStage — orchestrates report composition and writing.

Kept in its own module (separate from the package `__init__.py`) so the
package surface stays minimal and the stage class is easy to locate.
"""

from __future__ import annotations

from datetime import date

from ...models import (
    AttributeTable,
    CandidateTable,
    FunnelResults,
    OutcomeTable,
    ReportOutput,
    TrialTable,
)
from . import _compute, _narrative
from ._composer import ReportComposer
from ._types import ReportContext
from ._writer import write_report


class ReportingStage:
    """
    Renders pipeline results into a structured report artifact.

    Inputs:  CandidateTable, AttributeTable, OutcomeTable, FunnelResults, TrialTable
    Outputs: ReportOutput
    """

    def __init__(
        self,
        output_path: str | None = None,
        formats: list[str] | None = None,
        peptide_only: bool = True,
        time_periods: list[tuple[int, int]] | None = None,
        reference_date: date | None = None,
        stale_cutoff_years: float = 2.0,
        back_propagate_approval: bool = True,
    ):
        self.output_path = output_path
        self.formats = formats or ["html"]
        self.peptide_only = peptide_only
        self.time_periods = time_periods
        self.reference_date = reference_date
        self.stale_cutoff_years = stale_cutoff_years
        self.back_propagate_approval = back_propagate_approval

    def run(
        self,
        candidate_table: CandidateTable,
        attribute_table: AttributeTable,
        outcome_table: OutcomeTable,
        funnel_results: FunnelResults,
        trial_table: TrialTable | None = None,
    ) -> ReportOutput:
        """Assemble and write the report. Returns a ReportOutput."""
        ctx = ReportContext(
            candidate_table=candidate_table,
            attribute_table=attribute_table,
            outcome_table=outcome_table,
            funnel_results=funnel_results,
            peptide_only=self.peptide_only,
            time_periods=self.time_periods,
            trial_table=trial_table,
            reference_date=self.reference_date,
            stale_cutoff_years=self.stale_cutoff_years,
            back_propagate_approval=self.back_propagate_approval,
        )

        composer = ReportComposer()
        tables, figures, html_figures, narrative, export_data, sections = composer.compose(ctx)

        # Compute metadata for executive summary and introduction
        overall = funnel_results.overall
        n_transitions = sum(t.denominator for t in overall.transitions)
        n_candidates = len(candidate_table.candidates)
        sponsors = set()
        for c in candidate_table.candidates:
            sponsors.update(s for s in c.sponsors if s)
        n_sponsors = len(sponsors)

        # Date range
        dates = [c.earliest_start_date for c in candidate_table.candidates
                 if c.earliest_start_date is not None]
        date_range = (min(d.year for d in dates), max(d.year for d in dates)) if dates else None

        # Compute slices needed for executive summary
        bio_qls_slices = None
        onc_split = None
        by_modality = None
        timelines_slices = None
        stats = None

        if not self.peptide_only:
            bio_qls_slices = _compute.bio_qls_by_disease_area(funnel_results.by_disease_area)
            onc_split = _compute.oncology_vs_rest(funnel_results.by_disease_area)
            by_modality = funnel_results.by_modality
            timelines_slices = bio_qls_slices
            stats = _compute.sponsor_summary_stats(candidate_table)

        exec_summary = _narrative.executive_summary(
            overall, bio_qls_slices, by_modality, onc_split,
            timelines_slices, stats, self.peptide_only,
        )
        introduction = _narrative.introduction_text(
            n_transitions, n_candidates, n_sponsors, date_range, self.peptide_only,
        )

        summary_text = f"Executive Summary:\n{exec_summary}\n\n{introduction}"

        report = ReportOutput(
            summary_text=summary_text,
            tables=tables,
            figures=figures,
            html_figures=html_figures,
            output_path=self.output_path,
        )

        if self.output_path:
            write_report(
                report, self.output_path, self.formats, export_data,
                sections=sections,
                exec_summary=exec_summary,
                introduction=introduction,
            )

        return report
