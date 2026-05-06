"""Tests for the AdmetEnrichment adapter.

These tests stub out ``AdmetPredictor`` so they run without invoking the
real ADMET model. The standalone predictor itself is covered in
``test_admet_predictor.py``.
"""
from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from typing import Optional

import pytest

from pipeline.admet import ADMET_COLUMNS, field_name
from pipeline.enrichment.admet import AdmetEnrichment
from pipeline.models import Candidate, CandidateTable, TrialPhase


@dataclass
class _DummyConfig:
    enable_admet: bool = True


class _StubLedger:
    def __init__(self) -> None:
        self.coverage_calls: list[tuple[str, int, int]] = []

    def record_coverage(self, name: str, enriched: int, total: int) -> None:
        self.coverage_calls.append((name, enriched, total))


class _StubPredictor:
    """Returns a deterministic full-row dict for whitelisted SMILES."""

    def __init__(
        self,
        responses: dict[str, Optional[dict[str, float]]],
        *,
        raise_on_call: bool = False,
    ):
        self._responses = responses
        self._raise = raise_on_call
        self.calls: list[list[str]] = []

    def predict(self, smiles_list: list[str]):
        self.calls.append(list(smiles_list))
        if self._raise:
            raise RuntimeError("stub predictor failure")
        return {s: self._responses.get(s) for s in smiles_list}


def _full_row(value: float) -> dict[str, float]:
    return {col: value + i * 0.001 for i, col in enumerate(ADMET_COLUMNS)}


def _bare(candidate_id: str, *, smiles: Optional[str] = None, canonical: Optional[str] = None) -> Candidate:
    return Candidate(
        candidate_id=candidate_id,
        drug_name=f"drug-{candidate_id}",
        indication="Test Indication",
        highest_phase=TrialPhase.PHASE_2,
        smiles=smiles,
        smiles_canonical=canonical,
    )


def _make_table(*candidates: Candidate) -> CandidateTable:
    return CandidateTable(candidates=list(candidates))


def _wire_stub(stage: AdmetEnrichment, predictor: _StubPredictor) -> None:
    """Bypass ``is_available`` so tests don't require admet_ai."""
    stage._predictor = predictor
    stage._field_name = field_name
    stage._columns = ADMET_COLUMNS


def test_disabled_via_config_returns_false():
    stage = AdmetEnrichment()
    assert stage.is_available(_DummyConfig(enable_admet=False)) is False


def test_missing_admet_ai_returns_false(monkeypatch: pytest.MonkeyPatch):
    """Simulate a deploy without admet_ai installed."""
    monkeypatch.setitem(sys.modules, "admet_ai", None)
    stage = AdmetEnrichment()
    # Force a fresh is_available so the cached predictor doesn't short-circuit.
    assert stage._predictor is None
    assert stage.is_available(_DummyConfig()) is False


def test_full_population_from_canonical_smiles():
    cand = _bare("c1", smiles="CCO_raw", canonical="CCO")
    stage = AdmetEnrichment()
    _wire_stub(stage, _StubPredictor({"CCO": _full_row(1.0)}))
    ledger = _StubLedger()

    stage.run(_make_table(cand), ledger=ledger)

    expected = _full_row(1.0)
    for col in ADMET_COLUMNS:
        assert getattr(cand, field_name(col)) == expected[col]
    assert ledger.coverage_calls == [("admet", 1, 1)]


def test_falls_back_to_raw_smiles_when_canonical_missing():
    cand = _bare("c1", smiles="CC", canonical=None)
    stage = AdmetEnrichment()
    predictor = _StubPredictor({"CC": _full_row(0.5)})
    _wire_stub(stage, predictor)

    stage.run(_make_table(cand), ledger=_StubLedger())

    assert predictor.calls == [["CC"]]
    assert getattr(cand, field_name(ADMET_COLUMNS[0])) == _full_row(0.5)[ADMET_COLUMNS[0]]


def test_canonical_preferred_over_raw_when_both_present():
    cand = _bare("c1", smiles="CC", canonical="CCO")
    stage = AdmetEnrichment()
    predictor = _StubPredictor({"CCO": _full_row(1.0)})
    _wire_stub(stage, predictor)

    stage.run(_make_table(cand), ledger=_StubLedger())

    assert predictor.calls == [["CCO"]]


def test_skips_candidates_with_no_smiles():
    has = _bare("c1", canonical="CCO")
    none = _bare("c2", smiles=None, canonical=None)
    blank = _bare("c3", smiles="   ", canonical="")
    stage = AdmetEnrichment()
    predictor = _StubPredictor({"CCO": _full_row(2.0)})
    _wire_stub(stage, predictor)
    ledger = _StubLedger()

    stage.run(_make_table(has, none, blank), ledger=ledger)

    assert getattr(has, field_name(ADMET_COLUMNS[0])) is not None
    assert getattr(none, field_name(ADMET_COLUMNS[0])) is None
    assert getattr(blank, field_name(ADMET_COLUMNS[0])) is None
    assert ledger.coverage_calls == [("admet", 1, 3)]


def test_predictor_failure_recorded_as_zero_coverage():
    cand = _bare("c1", canonical="CCO")
    stage = AdmetEnrichment()
    _wire_stub(stage, _StubPredictor({}, raise_on_call=True))
    ledger = _StubLedger()

    stage.run(_make_table(cand), ledger=ledger)

    assert getattr(cand, field_name(ADMET_COLUMNS[0])) is None
    assert ledger.coverage_calls == [("admet", 0, 1)]


def test_per_smiles_failure_does_not_taint_other_candidates():
    bad = _bare("c1", canonical="BAD")
    good = _bare("c2", canonical="CCO")
    stage = AdmetEnrichment()
    _wire_stub(stage, _StubPredictor({"BAD": None, "CCO": _full_row(3.0)}))
    ledger = _StubLedger()

    stage.run(_make_table(bad, good), ledger=ledger)

    assert getattr(bad, field_name(ADMET_COLUMNS[0])) is None
    assert getattr(good, field_name(ADMET_COLUMNS[0])) == _full_row(3.0)[ADMET_COLUMNS[0]]
    assert ledger.coverage_calls == [("admet", 1, 2)]


def test_nan_within_row_propagates_as_none():
    cand = _bare("c1", canonical="CCO")
    row = _full_row(1.0)
    row[ADMET_COLUMNS[0]] = None  # simulating predictor's NaN→None translation
    stage = AdmetEnrichment()
    _wire_stub(stage, _StubPredictor({"CCO": row}))

    stage.run(_make_table(cand), ledger=_StubLedger())

    assert getattr(cand, field_name(ADMET_COLUMNS[0])) is None
    # Other columns must still be populated.
    assert getattr(cand, field_name(ADMET_COLUMNS[1])) is not None


def test_dedup_smiles_collapses_to_single_predictor_call():
    c1 = _bare("c1", canonical="CCO")
    c2 = _bare("c2", canonical="CCO")
    stage = AdmetEnrichment()
    predictor = _StubPredictor({"CCO": _full_row(1.0)})
    _wire_stub(stage, predictor)

    stage.run(_make_table(c1, c2), ledger=_StubLedger())

    assert predictor.calls == [["CCO"]]
    assert getattr(c1, field_name(ADMET_COLUMNS[0])) == _full_row(1.0)[ADMET_COLUMNS[0]]
    assert getattr(c2, field_name(ADMET_COLUMNS[0])) == _full_row(1.0)[ADMET_COLUMNS[0]]


def test_empty_candidate_table_records_zero_zero():
    stage = AdmetEnrichment()
    _wire_stub(stage, _StubPredictor({}))
    ledger = _StubLedger()

    stage.run(_make_table(), ledger=ledger)

    assert ledger.coverage_calls == [("admet", 0, 0)]
