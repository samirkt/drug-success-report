"""Shared types for the reporting package."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from ...models import (
        AttributeTable,
        CandidateTable,
        FunnelResults,
        OutcomeTable,
    )


@dataclass
class ReportContext:
    """Immutable bag of inputs passed to every component."""

    candidate_table: CandidateTable
    attribute_table: AttributeTable
    outcome_table: OutcomeTable
    funnel_results: FunnelResults
    peptide_only: bool
    time_periods: list[tuple[int, int]] | None


@dataclass
class ComponentResult:
    """Output of a single report component."""

    tables: dict[str, list[dict]] = field(default_factory=dict)
    figures: dict[str, bytes] = field(default_factory=dict)
    html_figures: dict[str, str] = field(default_factory=dict)
    narrative: str = ""
    export_data: dict = field(default_factory=dict)


@runtime_checkable
class ReportComponent(Protocol):
    """Interface every report section must satisfy."""

    key: str
    order: int
    title: str

    def should_include(self, ctx: ReportContext) -> bool: ...
    def render(self, ctx: ReportContext) -> ComponentResult: ...
