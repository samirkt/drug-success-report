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

    @patch("pipeline.enrichment.icd.get_icd_from_nih")
    def test_populates_codes_per_unique_indication(self, mock_lookup):
        mock_lookup.side_effect = lambda name, timeout: {
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

    @patch("pipeline.enrichment.icd.get_icd_from_nih")
    def test_lookup_failure_is_logged_and_skipped(self, mock_lookup):
        mock_lookup.side_effect = RuntimeError("network down")
        cands = _candidates()
        ledger = _FakeLedger()
        IcdEnrichment().run(cands, ledger=ledger)
        assert all(c.icd10_codes == [] for c in cands.candidates)
        assert ("icd10", 0, 4) in ledger.coverage_calls

    @patch("pipeline.enrichment.icd.get_icd_from_nih")
    def test_cache_hits_skip_network(self, mock_lookup, tmp_path):
        # Pre-populate cache with one positive and one negative entry.
        from pipeline.icd_lookup import IcdCache
        cache_path = tmp_path / "icd.sqlite"
        cache = IcdCache(cache_path)
        cache.put("Type 2 Diabetes", ["E11"])
        cache.put("Influenza", None)  # cached negative

        mock_lookup.side_effect = AssertionError(
            "should not be called when both indications are cached"
        )

        cands = _candidates()
        ledger = _FakeLedger()
        IcdEnrichment(cache_path=cache_path).run(cands, ledger=ledger)

        # Positive cache hit populated; negative cache hit stayed empty.
        assert cands.candidates[0].icd10_codes == ["E11"]
        assert cands.candidates[2].icd10_codes == []
        assert mock_lookup.call_count == 0
        assert ("icd10", 2, 4) in ledger.coverage_calls

    @patch("pipeline.enrichment.icd.get_icd_from_nih")
    def test_successful_fetch_writes_to_cache(self, mock_lookup, tmp_path):
        mock_lookup.side_effect = lambda name, timeout: {
            "Type 2 Diabetes": ["E11"],
        }.get(name)

        from pipeline.icd_lookup import IcdCache
        cache_path = tmp_path / "icd.sqlite"
        cands = _candidates()
        IcdEnrichment(cache_path=cache_path).run(cands, ledger=_FakeLedger())

        # Re-open the cache and check both positive (Type 2 Diabetes) and
        # negative (Influenza, returned None) results were written back.
        cache = IcdCache(cache_path)
        assert cache.get("Type 2 Diabetes") == (True, ["E11"])
        assert cache.get("Influenza") == (True, None)

    @patch("pipeline.enrichment.icd.get_icd_from_nih")
    def test_mesh_indication_preferred_over_raw_indication(self, mock_lookup):
        # MeSH succeeds — raw indication should never be queried.
        mock_lookup.side_effect = lambda name, timeout: {
            "Diabetes Mellitus, Type 2": ["E11", "E11.9"],
        }.get(name)

        cands = CandidateTable(candidates=[
            Candidate(
                candidate_id="c1",
                drug_name="DrugA",
                indication="T2DM",  # ill-formed; would miss NLM
                mesh_indication="Diabetes Mellitus, Type 2",
            ),
        ])
        IcdEnrichment().run(cands, ledger=_FakeLedger())

        assert cands.candidates[0].icd10_codes == ["E11", "E11.9"]
        # MeSH key resolved first, so only one lookup was attempted.
        called_with = [c.args[0] for c in mock_lookup.call_args_list]
        assert called_with == ["Diabetes Mellitus, Type 2"]

    @patch("pipeline.enrichment.icd.get_icd_from_nih")
    def test_falls_back_to_indication_when_mesh_misses(self, mock_lookup):
        # MeSH returns None → fall back to the raw indication.
        mock_lookup.side_effect = lambda name, timeout: {
            "Diabetes Mellitus, Type 2": None,
            "Type 2 Diabetes": ["E11"],
        }.get(name)

        cands = CandidateTable(candidates=[
            Candidate(
                candidate_id="c1",
                drug_name="DrugA",
                indication="Type 2 Diabetes",
                mesh_indication="Diabetes Mellitus, Type 2",
            ),
        ])
        IcdEnrichment().run(cands, ledger=_FakeLedger())

        assert cands.candidates[0].icd10_codes == ["E11"]
        called_with = [c.args[0] for c in mock_lookup.call_args_list]
        assert set(called_with) == {"Diabetes Mellitus, Type 2", "Type 2 Diabetes"}

    @patch("pipeline.enrichment.icd.get_icd_from_nih")
    def test_no_mesh_uses_indication_only(self, mock_lookup):
        # Backwards-compatible path: when mesh_indication is missing, the
        # raw indication is the only key tried — same as before.
        mock_lookup.side_effect = lambda name, timeout: {
            "Influenza": ["J10"],
        }.get(name)

        cands = CandidateTable(candidates=[
            Candidate(candidate_id="c1", drug_name="DrugA", indication="Influenza"),
        ])
        IcdEnrichment().run(cands, ledger=_FakeLedger())

        assert cands.candidates[0].icd10_codes == ["J10"]
        assert mock_lookup.call_count == 1
