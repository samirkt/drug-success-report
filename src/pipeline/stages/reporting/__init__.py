"""
Stage 5: Automated Report

Component-based report generation. Each report section (funnel, heatmap, LOA,
etc.) is a self-contained component in the ``components/`` sub-package.
The composer discovers, orders, and runs them; the writer serializes output.

Public API:
    ReportingStage  — pipeline stage (defined in ``reporting.py``)
"""

from .reporting import ReportingStage

__all__ = ["ReportingStage"]
