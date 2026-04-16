"""FDA integration primitives: openFDA/DailyMed access, approval-timeline
reconstruction, indication extraction/matching via LLM, and NDC commercial
status.

Used by `stages/adjudication_fda.py` to compute outcomes for drug-indication
candidates, but the primitives are reusable outside adjudication.
"""

from .commercial import CommercialStatus, CommercialStatusChecker
from .fda_client import Application, FDAClient, NDCRecord, Submission
from .llm_adjudicator import (
    ExtractedIndication,
    IndicationAdjudicator,
    LLMClient,
    MatchResult,
    MatchVerdict,
    extract_pdf_text,
)
from .llm_client_anthropic import AnthropicJSONClient
from .timeline import (
    DrugApprovalTimeline,
    IndicationApprovalEvent,
    TimelineBuilder,
)

__all__ = [
    "AnthropicJSONClient",
    "Application",
    "CommercialStatus",
    "CommercialStatusChecker",
    "DrugApprovalTimeline",
    "ExtractedIndication",
    "FDAClient",
    "IndicationAdjudicator",
    "IndicationApprovalEvent",
    "LLMClient",
    "MatchResult",
    "MatchVerdict",
    "NDCRecord",
    "Submission",
    "TimelineBuilder",
    "extract_pdf_text",
]
