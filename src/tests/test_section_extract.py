"""Tests for the pre-LLM narrowing helpers in pipeline.fda.section_extract.

Both helpers must be conservative: any uncertainty falls back to the
original input so the LLM still has a chance to find the indication.
"""

from __future__ import annotations

import pytest

from pipeline.fda.section_extract import (
    extract_letter_indication_clause,
    narrow_label_indications,
)


# ---------------------------------------------------------------------------
# extract_letter_indication_clause
# ---------------------------------------------------------------------------


def _approval_letter(indication_clause: str, *, pad: int = 5000) -> str:
    """Build a synthetic approval-letter PDF dump with realistic structure.

    The clause goes in the middle so anchor matching has to actually work
    rather than tripping on letterhead at offset 0.
    """
    header = "Department of Health and Human Services\n" * 50
    boilerplate_tail = (
        "REMS: A Risk Evaluation and Mitigation Strategy is required.\n"
        "Pediatric Research Equity Act assessments are deferred. "
        "Postmarketing Requirements include Study 1234. Sincerely, "
        "FDA Center for Drug Evaluation and Research."
    )
    middle_padding = "We have reviewed your application. " * (pad // 35)
    return f"{header}\n{indication_clause}\n{middle_padding}\n{boilerplate_tail}"


class TestLetterExtraction:
    def test_short_input_passes_through_untouched(self):
        text = "Short letter that is well below the min_input_chars threshold."
        assert extract_letter_indication_clause(text) == text

    def test_extracts_window_around_single_indication_clause(self):
        clause = (
            "This supplemental new drug application provides for the use of "
            "DRUGX for the treatment of metastatic non-small cell lung cancer "
            "in adult patients whose tumors express PD-L1."
        )
        text = _approval_letter(clause)
        out = extract_letter_indication_clause(text)
        assert "metastatic non-small cell lung cancer" in out
        assert len(out) < len(text)

    def test_drops_post_indication_boilerplate(self):
        clause = "DRUGX is indicated for the treatment of plaque psoriasis."
        text = _approval_letter(clause)
        out = extract_letter_indication_clause(text)
        # Anything past the REMS / Pediatric / Sincerely tail must be excluded
        assert "Sincerely" not in out
        assert "Pediatric Research Equity Act" not in out

    def test_no_anchors_falls_back_to_full_text(self):
        # No anchor phrases AT ALL — fall back so the LLM can try.
        text = (
            "Greetings from the agency. " * 200
            + "We acknowledge receipt of your submission. " * 200
        )
        assert extract_letter_indication_clause(text) == text

    def test_overlapping_anchors_are_merged(self):
        # Two anchors close together should produce one merged window,
        # not two duplicated chunks.
        clause = (
            "This application provides for the treatment of hypertension. "
            "It is indicated for adult patients with stage 2 hypertension."
        )
        text = _approval_letter(clause)
        out = extract_letter_indication_clause(text)
        # Single merged window — separator only appears between distinct windows
        assert out.count("[...]") <= 1
        assert "hypertension" in out

    def test_extraction_smaller_than_input(self):
        clause = "This sNDA provides for a new indication for treatment of asthma."
        text = _approval_letter(clause, pad=20_000)
        out = extract_letter_indication_clause(text)
        assert len(out) < len(text) * 0.9


# ---------------------------------------------------------------------------
# narrow_label_indications
# ---------------------------------------------------------------------------


class TestLabelNarrowing:
    def test_strips_limitations_of_use_subsection(self):
        text = (
            "1 INDICATIONS AND USAGE\n"
            "DRUGX is indicated for the treatment of moderate to severe rheumatoid arthritis.\n"
            "Limitations of Use\n"
            "DRUGX is not indicated for use in combination with biologic DMARDs.\n"
        )
        out = narrow_label_indications(text)
        assert "moderate to severe rheumatoid arthritis" in out
        assert "Limitations of Use" not in out
        assert "biologic DMARDs" not in out

    def test_no_limitations_section_passes_through(self):
        text = "DRUGX is indicated for the treatment of type 2 diabetes mellitus."
        assert narrow_label_indications(text) == text

    def test_empty_input_returns_empty(self):
        assert narrow_label_indications("") == ""

    def test_limitations_at_start_falls_back(self):
        # If stripping the tail would leave essentially nothing, return
        # the original so the LLM still sees something.
        text = "Limitations of Use\nDRUGX is not for combination therapy."
        assert narrow_label_indications(text) == text

    def test_handles_singular_limitation_spelling(self):
        text = (
            "DRUGX is indicated for migraine prevention in adults.\n"
            "Limitation of Use\n"
            "Not for acute migraine treatment.\n"
        )
        out = narrow_label_indications(text)
        assert "migraine prevention" in out
        assert "acute migraine treatment" not in out
