"""Commercial-availability check via NDC directory."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Optional

from .fda_client import FDAClient, NDCRecord


@dataclass
class CommercialStatus:
    is_commercialized: bool
    commercialization_date: Optional[date]  # earliest marketing_start_date
    discontinuation_date: Optional[date]  # latest marketing_end_date if all ended
    active_ndc_count: int
    total_ndc_count: int
    reasoning: str


class CommercialStatusChecker:
    def __init__(self, fda: FDAClient):
        self.fda = fda

    def check(
        self, application_numbers: list[str], as_of: Optional[date] = None
    ) -> CommercialStatus:
        as_of = as_of or date.today()
        all_records: list[NDCRecord] = []
        for app_no in application_numbers:
            all_records.extend(self.fda.get_ndc_records(app_no))

        if not all_records:
            return CommercialStatus(
                is_commercialized=False,
                commercialization_date=None,
                discontinuation_date=None,
                active_ndc_count=0,
                total_ndc_count=0,
                reasoning="No NDC records found for any application.",
            )

        active = [r for r in all_records if r.is_currently_marketed(as_of)]
        start_dates = [
            r.marketing_start_date for r in all_records if r.marketing_start_date
        ]
        end_dates = [r.marketing_end_date for r in all_records if r.marketing_end_date]

        earliest_start = min(start_dates) if start_dates else None

        if active:
            return CommercialStatus(
                is_commercialized=True,
                commercialization_date=earliest_start,
                discontinuation_date=None,
                active_ndc_count=len(active),
                total_ndc_count=len(all_records),
                reasoning=(
                    f"{len(active)} of {len(all_records)} NDC records are "
                    f"currently marketed."
                ),
            )

        # No active records but some history -- withdrawn/discontinued.
        latest_end = max(end_dates) if end_dates else None
        return CommercialStatus(
            is_commercialized=False,
            commercialization_date=earliest_start,
            discontinuation_date=latest_end,
            active_ndc_count=0,
            total_ndc_count=len(all_records),
            reasoning=(
                f"All {len(all_records)} NDC records have marketing end dates "
                f"in the past (latest: {latest_end}). Drug appears discontinued."
            ),
        )
