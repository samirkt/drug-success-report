"""Tests for pipeline.enrichment.smiles.SmilesEnrichment."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from pipeline.enrichment.smiles import SmilesEnrichment
from pipeline.models import CandidateTable
from utils.tiered_router import CostLedger


_SCHEMA_V2_CSV = (
    "drug_id,query_name,query_norm,query_kind,modality,aa_sequence,aa_length,"
    "approval_groups,is_approved,indications,"
    "cf_superclass,cf_class,cf_subclass,cf_direct_parent,cf_alt_parents,"
    "cf_substituents,peptide_like_cf,peptide_drug_cat,"
    "smiles,inchi,logp,molecular_weight\n"
    "DB00001,Lepirudin,lepirudin,canonical,peptide,,,approved,1,,"
    ",,,,,,0,0,"
    ",,,6979.5\n"
    "DB00030,Insulin Human,insulin human,canonical,peptide,,,approved,1,,"
    ",,,,,,0,0,"
    ",,,5807.6\n"
    "DB00050,Cetuximab,cetuximab,canonical,biologic,,,approved,1,,"
    ",,,,,,0,0,"
    ",,,145781.6\n"
    "DB00945,Aspirin,aspirin,canonical,small molecule,,,approved,1,,"
    ",,,,,,0,0,"
    "CC(=O)OC1=CC=CC=C1C(=O)O,InChI=1S/C9H8O4,1.2,180.16\n"
)

_LEGACY_CSV = (
    "drug_id,query_name,query_norm,query_kind,modality\n"
    "DB00001,Lepirudin,lepirudin,canonical,peptide\n"
    "DB00945,Aspirin,aspirin,canonical,small molecule\n"
)


@pytest.fixture
def drugbank_csv(tmp_path: Path) -> Path:
    p = tmp_path / "drugbank_approvals.csv"
    p.write_text(_SCHEMA_V2_CSV)
    return p


@pytest.fixture
def legacy_drugbank_csv(tmp_path: Path) -> Path:
    p = tmp_path / "drugbank_legacy.csv"
    p.write_text(_LEGACY_CSV)
    return p


def _make_config(**overrides):
    class _Cfg:
        enable_smiles = True
        drugbank_csv_path: Path | None = None

    cfg = _Cfg()
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


class TestIsAvailable:
    def test_disabled_flag_returns_false(self, drugbank_csv):
        cfg = _make_config(enable_smiles=False, drugbank_csv_path=drugbank_csv)
        assert SmilesEnrichment().is_available(cfg) is False

    def test_missing_csv_returns_false(self, tmp_path):
        cfg = _make_config(drugbank_csv_path=tmp_path / "does_not_exist.csv")
        assert SmilesEnrichment().is_available(cfg) is False

    def test_no_csv_path_returns_false(self):
        cfg = _make_config(drugbank_csv_path=None)
        assert SmilesEnrichment().is_available(cfg) is False

    def test_valid_csv_returns_true(self, drugbank_csv):
        cfg = _make_config(drugbank_csv_path=drugbank_csv)
        assert SmilesEnrichment().is_available(cfg) is True


class TestRun:
    def test_attaches_smiles_when_drugbank_id_matches(
        self, drugbank_csv, sample_candidate, another_candidate
    ):
        # sample_candidate has drugbank_id DB00001 (no SMILES in fixture),
        # replace with aspirin (DB00945, has SMILES) to exercise the match.
        aspirin = replace(sample_candidate, drugbank_id="DB00945")
        table = CandidateTable(candidates=[aspirin, another_candidate])

        cfg = _make_config(drugbank_csv_path=drugbank_csv)
        stage = SmilesEnrichment()
        assert stage.is_available(cfg) is True
        ledger = CostLedger()

        stage.run(table, ledger=ledger)

        assert aspirin.smiles == "CC(=O)OC1=CC=CC=C1C(=O)O"
        # DB00030 (insulin) has no SMILES cell in the fixture
        assert another_candidate.smiles is None

    def test_records_coverage_in_ledger(self, drugbank_csv, sample_candidate):
        aspirin = replace(sample_candidate, drugbank_id="DB00945")
        insulin_nothing = replace(
            sample_candidate, candidate_id="c2", drugbank_id="DB00030"
        )
        no_drugbank = replace(
            sample_candidate, candidate_id="c3", drugbank_id=None
        )
        table = CandidateTable(candidates=[aspirin, insulin_nothing, no_drugbank])

        ledger = CostLedger()
        stage = SmilesEnrichment()
        cfg = _make_config(drugbank_csv_path=drugbank_csv)
        stage.is_available(cfg)
        stage.run(table, ledger=ledger)

        assert ledger.coverage["smiles"] == (1, 3)

    def test_candidates_without_drugbank_id_are_passed_through(
        self, drugbank_csv, sample_candidate
    ):
        unmatched = replace(sample_candidate, drugbank_id=None, smiles=None)
        table = CandidateTable(candidates=[unmatched])

        stage = SmilesEnrichment()
        cfg = _make_config(drugbank_csv_path=drugbank_csv)
        stage.is_available(cfg)
        stage.run(table, ledger=CostLedger())

        assert unmatched.smiles is None

    def test_legacy_csv_without_smiles_column_is_non_fatal(
        self, legacy_drugbank_csv, sample_candidate
    ):
        aspirin = replace(sample_candidate, drugbank_id="DB00945")
        table = CandidateTable(candidates=[aspirin])

        stage = SmilesEnrichment()
        cfg = _make_config(drugbank_csv_path=legacy_drugbank_csv)
        assert stage.is_available(cfg) is True

        ledger = CostLedger()
        stage.run(table, ledger=ledger)

        assert aspirin.smiles is None
        assert ledger.coverage["smiles"] == (0, 1)

    def test_does_not_mutate_cache_key_inputs(
        self, drugbank_csv, sample_candidate
    ):
        """Invariant: enrichment must not touch drug_name, indication, or
        highest_phase — the fields KnowledgeCache keys derive from."""
        aspirin = replace(sample_candidate, drugbank_id="DB00945")
        before = (
            aspirin.drug_name,
            aspirin.indication,
            aspirin.highest_phase,
            aspirin.candidate_id,
        )

        stage = SmilesEnrichment()
        cfg = _make_config(drugbank_csv_path=drugbank_csv)
        stage.is_available(cfg)
        stage.run(CandidateTable(candidates=[aspirin]), ledger=CostLedger())

        after = (
            aspirin.drug_name,
            aspirin.indication,
            aspirin.highest_phase,
            aspirin.candidate_id,
        )
        assert before == after
