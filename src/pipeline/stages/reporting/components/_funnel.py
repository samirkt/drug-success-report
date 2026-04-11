"""Funnel table component (chart removed)."""

from .._types import ComponentResult, ReportContext
from .. import _narrative
from ....models import FunnelSlice


# ---------------------------------------------------------------------------
# Standalone functions
# ---------------------------------------------------------------------------

def funnel_table(slice_) -> list[dict]:
    """Convert a FunnelSlice's transition rates into table rows."""
    return [
        {
            "from_phase": t.from_phase,
            "to_phase": t.to_phase,
            "numerator": t.numerator,
            "denominator": t.denominator,
            "rate": t.rate,
        }
        for t in slice_.transitions
    ]


# ---------------------------------------------------------------------------
# Component class
# ---------------------------------------------------------------------------

class FunnelComponent:
    key = "funnel"
    order = 10
    title = "Phase Transition Success Rates"

    def should_include(self, ctx: ReportContext) -> bool:
        return False

    def render(self, ctx: ReportContext) -> ComponentResult:
        if ctx.peptide_only:
            funnel = ctx.funnel_results.by_modality.get("peptide", ctx.funnel_results.overall)
        else:
            funnel = ctx.funnel_results.overall

        return ComponentResult(
            tables={
                "funnel_overall": funnel_table(funnel),
            },
            figures={},
            narrative=_narrative.overall_success_narrative(funnel),
        )
