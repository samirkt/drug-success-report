"""Unit tests for the RegulatoryIndex and synonym loader."""
from __future__ import annotations

from datetime import date
from pathlib import Path

from pipeline.drugbank_norm import load_drugbank_synonyms, match_drug_name
from pipeline.regulatory import RegulatoryIndex
import pandas as pd


def _write_products(tmp_path: Path) -> Path:
    path = tmp_path / "drugbank_products.csv"
    path.write_text(
        "drugbank_id,drug_name_norm,primary_name,appl_no,approved,approval_date,first_marketed_date,us_country_hit,groups\n"
        "DB00001,lepirudin,Lepirudin,NDA020807,true,1998-03-06,1998-03-06,true,approved|withdrawn\n"
        "DB00002,cetuximab,Cetuximab,BLA125084,true,2004-02-12,2004-02-12,true,approved|investigational\n"
        # Approved outside the US — should be filtered out of the index.
        "DB00500,somedrug,SomeDrug,,true,,2010-01-01,false,approved\n"
        # Approved but no date signal at all — also filtered.
        "DB00600,nodrug,NoDrug,,true,,,true,approved\n"
        "DB00700,investigational,Investigational,,false,,,false,investigational\n",
        encoding="utf-8",
    )
    return path


def _write_synonyms(tmp_path: Path) -> Path:
    path = tmp_path / "drugbank_synonyms.csv"
    path.write_text(
        "drugbank_id,synonym_norm,kind\n"
        "DB00001,lepirudin,primary_name\n"
        "DB00001,refludan,international_brand\n"
        "DB00001,r hirudin,synonym\n"
        "DB00002,cetuximab,primary_name\n"
        "DB00002,chimeric moab c225,synonym\n"
        "DB00002,erbitux,international_brand\n",
        encoding="utf-8",
    )
    return path


class TestRegulatoryIndex:
    def test_lookup_by_drugbank_id(self, tmp_path: Path):
        idx = RegulatoryIndex.from_csv(_write_products(tmp_path))
        rec = idx.lookup(drugbank_id="DB00001")
        assert rec is not None
        assert rec.approval_date == date(1998, 3, 6)
        assert rec.appl_no == "NDA020807"
        assert rec.evidence_source == "drugs_at_fda:NDA020807"

    def test_lookup_by_name_canonicalization(self, tmp_path: Path):
        idx = RegulatoryIndex.from_csv(_write_products(tmp_path))
        # "Lepirudin (rDNA)" → canonicalizes to "lepirudin"
        rec = idx.lookup(drug_name="Lepirudin (rDNA)")
        assert rec is not None
        assert rec.drugbank_id == "DB00001"

    def test_non_us_approved_drugs_are_excluded(self, tmp_path: Path):
        idx = RegulatoryIndex.from_csv(_write_products(tmp_path))
        assert idx.lookup(drugbank_id="DB00500") is None

    def test_approved_but_no_date_excluded(self, tmp_path: Path):
        idx = RegulatoryIndex.from_csv(_write_products(tmp_path))
        assert idx.lookup(drugbank_id="DB00600") is None

    def test_unapproved_excluded(self, tmp_path: Path):
        idx = RegulatoryIndex.from_csv(_write_products(tmp_path))
        assert idx.lookup(drugbank_id="DB00700") is None

    def test_missing_csv_returns_empty_index(self, tmp_path: Path):
        idx = RegulatoryIndex.from_csv(tmp_path / "nonexistent.csv")
        assert len(idx) == 0
        assert idx.lookup(drugbank_id="DB00001") is None

    def test_drugbank_id_takes_precedence_over_name(self, tmp_path: Path):
        idx = RegulatoryIndex.from_csv(_write_products(tmp_path))
        rec = idx.lookup(drug_name="cetuximab", drugbank_id="DB00001")
        assert rec.drugbank_id == "DB00001"


class TestSynonymLoader:
    def test_load_drugbank_synonyms_shape(self, tmp_path: Path):
        syn_path = _write_synonyms(tmp_path)
        forward, reverse = load_drugbank_synonyms(syn_path)
        assert "DB00001" in forward
        assert "lepirudin" in forward["DB00001"]
        assert "refludan" in forward["DB00001"]
        assert reverse["erbitux"] == "DB00002"
        assert reverse["chimeric moab c225"] == "DB00002"

    def test_load_missing_file_returns_empty_maps(self, tmp_path: Path):
        forward, reverse = load_drugbank_synonyms(tmp_path / "no.csv")
        assert forward == {}
        assert reverse == {}


class TestMatchDrugNameWithSynonyms:
    def test_synonym_recovery_when_canonical_lookup_fails(self):
        # Build a minimal drugbank_approvals-like frame with no "erbitux" row.
        df = pd.DataFrame({
            "drug_id": ["DB00001"],
            "query_name": ["lepirudin"],
            "query_norm": ["lepirudin"],
        })
        best_rows = df
        best_rows_norm = df
        synonym_reverse = {"erbitux": "DB00002"}
        # Lookup "Erbitux" should not match canonical/first-word, but should
        # recover via the synonym map.
        result = match_drug_name(
            "Erbitux", best_rows, best_rows_norm, synonym_reverse=synonym_reverse,
        )
        assert result == "DB00002"

    def test_synonym_map_not_used_when_canonical_matches(self):
        df = pd.DataFrame({
            "drug_id": ["DB00001"],
            "query_name": ["lepirudin"],
            "query_norm": ["lepirudin"],
        })
        # A bogus synonym map that would return a different ID if consulted.
        synonym_reverse = {"lepirudin": "DB99999"}
        result = match_drug_name(
            "Lepirudin", df, df, synonym_reverse=synonym_reverse,
        )
        assert result == "DB00001"

    def test_synonym_map_none_preserves_legacy_behavior(self):
        df = pd.DataFrame({
            "drug_id": ["DB00001"],
            "query_name": ["lepirudin"],
            "query_norm": ["lepirudin"],
        })
        assert match_drug_name("Erbitux", df, df) is None
