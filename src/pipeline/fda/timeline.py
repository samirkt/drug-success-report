"""Reconstructs an indication-level approval timeline for a drug.

For each Application, we walk its Submissions in chronological order
and, for each approved submission, determine which indications became
approved on that date. The output is a list of IndicationApprovalEvents
that together form the temporal ground-truth for the drug.

Strategy by submission type:

  ORIG (original approval)
    -> Fetch the earliest available label (DailyMed current label as a
       proxy when historical SPLs are unavailable) and LLM-extract the
       indications. Attribute them all to the ORIG approval date.
       NOTE: the current label may contain indications added later; this
       is corrected below by diffing against supplemental letters.

  SUPPL (supplemental approval) with new-indication class
    -> Fetch the approval letter PDF, LLM-extract the specific indication
       added. Attribute to the submission's approval date.

  SUPPL (supplemental) without new-indication class
    -> Skip (labeling, manufacturing, etc. don't add indications).

After the pass, indications that appear in the current label but never
surfaced from any submission extraction are attributed to the ORIG
approval with a confidence penalty.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

from .fda_client import Application, FDAClient, Submission
from .llm_adjudicator import ExtractedIndication, IndicationAdjudicator, extract_pdf_text

logger = logging.getLogger(__name__)


@dataclass
class IndicationApprovalEvent:
    """A single indication approved on a specific date for an application."""

    application_number: str
    approval_date: date
    submission_type: str  # "ORIG" or "SUPPL"
    submission_number: int
    indication: ExtractedIndication
    source: str  # "approval_letter" | "current_label" | "inferred"
    confidence: float  # 0.0..1.0


@dataclass
class DrugApprovalTimeline:
    """Full indication-level timeline for a drug across all its applications."""

    drug_name: str
    applications: list[Application]
    events: list[IndicationApprovalEvent] = field(default_factory=list)

    @property
    def earliest_approval_date(self) -> Optional[date]:
        dates = [e.approval_date for e in self.events]
        return min(dates) if dates else None


class TimelineBuilder:
    """Builds a DrugApprovalTimeline by combining submissions + LLM extraction."""

    def __init__(
        self,
        fda: FDAClient,
        adjudicator: IndicationAdjudicator,
        pdf_extractor=extract_pdf_text,
    ):
        self.fda = fda
        self.adjudicator = adjudicator
        self.pdf_extractor = pdf_extractor

    def build(self, drug_name: str) -> DrugApprovalTimeline:
        applications = self.fda.find_applications_by_drug(drug_name)
        timeline = DrugApprovalTimeline(drug_name=drug_name, applications=applications)

        for app in applications:
            self._process_application(drug_name, app, timeline)

        return timeline

    def _process_application(
        self,
        drug_name: str,
        app: Application,
        timeline: DrugApprovalTimeline,
    ) -> None:
        # Sort submissions chronologically so we attribute indications
        # to the earliest submission that introduced them.
        ordered = sorted(
            [s for s in app.submissions if s.is_approved and s.submission_status_date],
            key=lambda s: s.submission_status_date,  # type: ignore[return-value]
        )

        original: Optional[Submission] = next(
            (s for s in ordered if s.submission_type == "ORIG"), None
        )
        supplementals = [s for s in ordered if s.submission_type == "SUPPL"]

        # --- Original approval ---
        orig_indications: list[ExtractedIndication] = []
        if original:
            orig_indications = self._extract_original_indications(drug_name, app)
            for ind in orig_indications:
                timeline.events.append(
                    IndicationApprovalEvent(
                        application_number=app.application_number,
                        approval_date=original.submission_status_date,  # type: ignore[arg-type]
                        submission_type="ORIG",
                        submission_number=original.submission_number,
                        indication=ind,
                        source="current_label",
                        # ORIG attribution from current label is lower-
                        # confidence: a new indication added via SUPPL
                        # will also appear in the current label.
                        confidence=0.75,
                    )
                )

        # --- Supplemental new-indication approvals ---
        previously_seen = {_normalize(ind.indication_text) for ind in orig_indications}
        for sub in supplementals:
            if not sub.is_new_indication:
                continue

            letter_inds = self._extract_supplemental_indications(drug_name, sub)
            for ind in letter_inds:
                key = _normalize(ind.indication_text)
                if key in previously_seen:
                    # Already attributed to an earlier submission.
                    continue
                previously_seen.add(key)

                timeline.events.append(
                    IndicationApprovalEvent(
                        application_number=app.application_number,
                        approval_date=sub.submission_status_date,  # type: ignore[arg-type]
                        submission_type="SUPPL",
                        submission_number=sub.submission_number,
                        indication=ind,
                        source="approval_letter",
                        confidence=0.90,
                    )
                )

        # --- Correction pass ---
        # Any indication in ORIG list that also shows up in a SUPPL
        # letter should be re-attributed to the SUPPL date. This catches
        # the common case where the current label contains indications
        # added years after original approval.
        self._reattribute_later_indications(timeline, app)

    def _extract_original_indications(
        self, drug_name: str, app: Application
    ) -> list[ExtractedIndication]:
        label_text = self.fda.get_current_label_text(app.application_number)
        if not label_text:
            return []
        return self.adjudicator.extract_from_text(
            label_text, drug_name, source_kind="label"
        )

    def _extract_supplemental_indications(
        self, drug_name: str, sub: Submission
    ) -> list[ExtractedIndication]:
        pdf_bytes = self.fda.get_approval_letter_pdf(sub)
        if not pdf_bytes:
            return []
        try:
            text = self.pdf_extractor(pdf_bytes)
        except Exception as e:
            logger.warning(
                "PDF extraction failed for %s sub %s: %s",
                sub.application_number,
                sub.submission_number,
                e,
            )
            return []
        if not text:
            return []
        return self.adjudicator.extract_from_text(
            text, drug_name, source_kind="approval_letter"
        )

    def _reattribute_later_indications(
        self, timeline: DrugApprovalTimeline, app: Application
    ) -> None:
        """If the same indication appears under both ORIG (from current
        label) and a SUPPL approval letter, drop the ORIG attribution.
        """
        events_for_app = [
            e for e in timeline.events if e.application_number == app.application_number
        ]
        suppl_keys = {
            _normalize(e.indication.indication_text)
            for e in events_for_app
            if e.submission_type == "SUPPL"
        }

        timeline.events = [
            e
            for e in timeline.events
            if not (
                e.application_number == app.application_number
                and e.submission_type == "ORIG"
                and _normalize(e.indication.indication_text) in suppl_keys
            )
        ]


def _normalize(s: str) -> str:
    return " ".join(s.lower().split())
