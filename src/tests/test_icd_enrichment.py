"""Tests for the IcdEnrichment stage."""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import patch

import pytest

from pipeline.enrichment.icd import IcdEnrichment
from pipeline.models import Candidate, CandidateTable


@dataclass
class _FakeConfig:
    enable_icd10: bool = True


class _FakeLedger:
    def __init__(self):
        self.coverage_calls: list[tuple] = []

    def record_coverage(self, name, enriched, total):
        self.coverage_calls.append((name, enriched, total))


def _candidates() -> CandidateTable:
    return CandidateTable(candidates=[
        Candidate(candidate_id="c1", drug_name="DrugA", indication="Type 2 Diabetes"),
        Candidate(candidate_id="c2", drug_name="DrugB", indication="Type 2 Diabetes"),
        Candidate(candidate_id="c3", drug_name="DrugC", indication="Influenza"),
        Candidate(candidate_id="c4", drug_name="DrugD", indication=""),
    ])


class TestIsAvailable:

    def test_disabled_when_flag_off(self):
        assert IcdEnrichment().is_available(_FakeConfig(enable_icd10=False)) is False

    def test_enabled_when_flag_on(self):
        assert IcdEnrichment().is_available(_FakeConfig(enable_icd10=True)) is True


class TestRun:

    @patch("pipeline.enrichment.icd.get_icd_cached")
    def test_populates_codes_per_unique_indication(self, mock_lookup):
        mock_lookup.side_effect = lambda name, cache, timeout: {
            "Type 2 Diabetes": ["E11", "E11.9"],
            "Influenza": ["J10", "J11"],
        }.get(name)

        cands = _candidates()
        ledger = _FakeLedger()
        IcdEnrichment().run(cands, ledger=ledger)

        # 3 candidates with non-empty indications got codes; c4 (empty) did not.
        assert cands.candidates[0].icd10_codes == ["E11", "E11.9"]
        assert cands.candidates[1].icd10_codes == ["E11", "E11.9"]
        assert cands.candidates[2].icd10_codes == ["J10", "J11"]
        assert cands.candidates[3].icd10_codes == []

        # Two unique indications -> two lookups (Type 2 Diabetes is reused for c1+c2).
        assert mock_lookup.call_count == 2
        assert ("icd10", 3, 4) in ledger.coverage_calls

    @patch("pipeline.enrichment.icd.get_icd_cached")
    def test_lookup_failure_is_logged_and_skipped(self, mock_lookup):
        mock_lookup.side_effect = RuntimeError("network down")
        cands = _candidates()
        ledger = _FakeLedger()
        IcdEnrichment().run(cands, ledger=ledger)
        assert all(c.icd10_codes == [] for c in cands.candidates)
        assert ("icd10", 0, 4) in ledger.coverage_calls
