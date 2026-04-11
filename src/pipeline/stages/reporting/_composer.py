"""Report composer: discovers, orders, and runs report components."""

from __future__ import annotations

from ._types import ComponentResult, ReportComponent, ReportContext


class ReportComposer:
    """Run all applicable components in order, merge results."""

    def __init__(self, components: list | None = None):
        if components is None:
            from .components import ALL_COMPONENTS
            components = [cls() for cls in ALL_COMPONENTS]
        self.components: list[ReportComponent] = components

    def compose(
        self, ctx: ReportContext,
    ) -> tuple[dict[str, list[dict]], dict[str, bytes], dict[str, str], str, dict, list[dict]]:
        """Returns (tables, figures, html_figures, narrative, export_data, sections).

        ``sections`` is a list of per-component dicts used by the HTML writer
        to interleave narrative text with figures and tables.
        """
        tables: dict[str, list[dict]] = {}
        figures: dict[str, bytes] = {}
        html_figures: dict[str, str] = {}
        export_data: dict = {}
        narratives: list[str] = []
        sections: list[dict] = []

        for comp in sorted(self.components, key=lambda c: c.order):
            if not comp.should_include(ctx):
                continue
            result = comp.render(ctx)
            tables.update(result.tables)
            figures.update(result.figures)
            html_figures.update(result.html_figures)
            export_data.update(result.export_data)
            if result.narrative:
                narratives.append(result.narrative)
            sections.append({
                "key": comp.key,
                "title": comp.title,
                "narrative": result.narrative,
                "figures": dict(result.figures),
                "tables": dict(result.tables),
                "html_figures": dict(result.html_figures),
            })

        return tables, figures, html_figures, " ".join(narratives), export_data, sections
