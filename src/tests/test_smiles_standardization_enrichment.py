"""Tests for SmilesStandardizationEnrichment.

The enrichment writes canonical SMILES into ``Candidate.smiles_canonical``
and tags every candidate with a status string. Failure modes (missing
SMILES, unparseable SMILES, standardizer exception) must surface as
the documented status values without dropping the candidate.

Tests that need RDKit + chembl_structure_pipeline skip cleanly when
those packages aren't installed in the active environment.
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass, field
from typing import Optional
from unittest.mock import patch

import pytest

from pipeline.enrichment.smiles_standardization import SmilesStandardizationEnrichment
from pipeline.models import CandidateTable

# RDKit / chembl_structure_pipeline are required deps for the live tests.
# CI environments without them just skip — the enrichment's is_available
# guard already handles that path; the missing-deps test below exercises it.
rdkit = pytest.importorskip("rdkit")
csp = pytest.importorskip("chembl_structure_pipeline")


@dataclass
class _DummyConfig:
    enable_smiles_standardization: bool = True


class _StubLedger:
    def __init__(self) -> None:
        self.coverage_calls: list[tuple[str, int, int]] = []

    def record_coverage(self, name: str, enriched: int, total: int) -> None:
        self.coverage_calls.append((name, enriched, total))


@pytest.fixture
def stub_ledger() -> _StubLedger:
    return _StubLedger()


def _make_table(*candidates):
    return CandidateTable(candidates=list(candidates))


def _bare_candidate(candidate_id: str, smiles: Optional[str]):
    """Build a Candidate without depending on the conftest fixtures' fields."""
    from pipeline.models import Candidate, TrialPhase

    return Candidate(
        candidate_id=candidate_id,
        drug_name=f"drug_{candidate_id}",
        indication="Test Indication",
        highest_phase=TrialPhase.PHASE_2,
        smiles=smiles,
    )


def test_status_ok_canonical_set(stub_ledger):
    cand = _bare_candidate("c1", "CC(=O)O")
    stage = SmilesStandardizationEnrichment()
    assert stage.is_available(_DummyConfig()) is True
    stage.run(_make_table(cand), ledger=stub_ledger)
    assert cand.smiles_standardization_status == "ok"
    assert cand.smiles_canonical is not None
    assert cand.smiles_canonical != ""
    # Raw input untouched
    assert cand.smiles == "CC(=O)O"


def test_status_empty_when_smiles_none(stub_ledger):
    cand = _bare_candidate("c1", None)
    stage = SmilesStandardizationEnrichment()
    stage.is_available(_DummyConfig())
    stage.run(_make_table(cand), ledger=stub_ledger)
    assert cand.smiles_standardization_status == "empty"
    assert cand.smiles_canonical is None
    assert cand.smiles is None


def test_status_empty_when_smiles_blank(stub_ledger):
    cand = _bare_candidate("c1", "   ")
    stage = SmilesStandardizationEnrichment()
    stage.is_available(_DummyConfig())
    stage.run(_make_table(cand), ledger=stub_ledger)
    assert cand.smiles_standardization_status == "empty"
    assert cand.smiles_canonical is None


def test_status_failed_parse_for_garbage(stub_ledger):
    cand = _bare_candidate("c1", "not a valid SMILES &&&")
    stage = SmilesStandardizationEnrichment()
    stage.is_available(_DummyConfig())
    stage.run(_make_table(cand), ledger=stub_ledger)
    assert cand.smiles_standardization_status == "failed_parse"
    assert cand.smiles_canonical is None
    assert cand.smiles == "not a valid SMILES &&&"


def test_status_failed_standardize_when_pipeline_raises(stub_ledger):
    cand = _bare_candidate("c1", "CC(=O)O")
    stage = SmilesStandardizationEnrichment()
    stage.is_available(_DummyConfig())

    def _boom(_mol):
        raise RuntimeError("synthetic standardize failure")

    # Patch the bound reference on the stage's lazy-imported module handle.
    with patch.object(stage._deps[1], "standardize_mol", side_effect=_boom):
        stage.run(_make_table(cand), ledger=stub_ledger)

    assert cand.smiles_standardization_status == "failed_standardize"
    assert cand.smiles_canonical is None


def test_strips_salt_via_get_parent_mol(stub_ledger):
    # Hydrochloride salt — chembl_structure_pipeline's salt list strips Cl.
    cand = _bare_candidate("c1", "Cl.CCN")
    stage = SmilesStandardizationEnrichment()
    stage.is_available(_DummyConfig())
    stage.run(_make_table(cand), ledger=stub_ledger)
    assert cand.smiles_standardization_status == "ok"
    assert cand.smiles_canonical == "CCN"
    # Raw stays as-is
    assert cand.smiles == "Cl.CCN"


def test_ledger_records_coverage(stub_ledger):
    cands = [
        _bare_candidate("c1", "CC(=O)O"),     # ok
        _bare_candidate("c2", None),          # empty (not counted as enriched)
        _bare_candidate("c3", "junk!@#"),     # failed_parse
        _bare_candidate("c4", "CCO"),         # ok
    ]
    stage = SmilesStandardizationEnrichment()
    stage.is_available(_DummyConfig())
    stage.run(_make_table(*cands), ledger=stub_ledger)
    assert stub_ledger.coverage_calls == [("smiles_standardization", 2, 4)]


def test_disabled_via_config_skips():
    cand = _bare_candidate("c1", "CC(=O)O")
    stage = SmilesStandardizationEnrichment()
    cfg = _DummyConfig(enable_smiles_standardization=False)
    assert stage.is_available(cfg) is False
    # Status should remain None — enrichment never ran.
    assert cand.smiles_standardization_status is None
    assert cand.smiles_canonical is None


def test_missing_deps_skips_gracefully():
    """is_available returns False when the underlying packages can't import."""
    stage = SmilesStandardizationEnrichment()

    real_import = __import__

    def _fake_import(name, *args, **kwargs):
        if name.startswith("chembl_structure_pipeline"):
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    # Force a fresh import attempt by clearing the cached deps handle.
    stage._deps = None
    with patch("builtins.__import__", side_effect=_fake_import):
        assert stage.is_available(_DummyConfig()) is False
