"""
Tests for pipeline/drugbank_norm.py

Test categories:
  PASS NOW — all normalization and lookup functions are fully implemented
"""

import io
from pathlib import Path

import pandas as pd
import pytest

from pipeline.drugbank_norm import canonicalize_drug_name, load_drugbank_lookup, match_drug_name


# ---------------------------------------------------------------------------
# TestCanonicalizeDrugName
# ---------------------------------------------------------------------------

class TestCanonicalizeDrugName:
    def test_lowercases_input(self):
        assert canonicalize_drug_name("Lepirudin") == "lepirudin"

    def test_collapses_whitespace(self):
        assert canonicalize_drug_name("insulin  human") == "insulin human"

    def test_strips_leading_trailing_whitespace(self):
        assert canonicalize_drug_name("  lepirudin  ") == "lepirudin"

    def test_strips_parenthetical(self):
        assert canonicalize_drug_name("Lepirudin (rDNA origin)") == "lepirudin"

    def test_strips_bracketed(self):
        assert canonicalize_drug_name("Insulin [recombinant]") == "insulin"

    def test_hyphen_to_space(self):
        assert canonicalize_drug_name("GLP-1") == "glp 1"

    def test_removes_formulation_words(self):
        assert canonicalize_drug_name("insulin glargine injection") == "insulin glargine"

    def test_roman_numeral_normalization(self):
        assert canonicalize_drug_name("Angiotensin II") == "angiotensin 2"

    def test_unicode_diacritics(self):
        assert canonicalize_drug_name("café drug") == "cafe drug"

    def test_empty_string(self):
        assert canonicalize_drug_name("") == ""

    def test_only_drop_words(self):
        assert canonicalize_drug_name("injection solution oral") == ""


# ---------------------------------------------------------------------------
# TestLoadDrugbankLookup
# ---------------------------------------------------------------------------

class TestLoadDrugbankLookup:
    @pytest.fixture
    def csv_file(self, tmp_path, drugbank_csv_content):
        p = tmp_path / "drugbank_approvals.csv"
        p.write_text(drugbank_csv_content)
        return p

    def test_returns_two_dataframes(self, csv_file):
        result = load_drugbank_lookup(csv_file)
        assert isinstance(result, tuple)
        assert len(result) == 2
        best_rows, best_rows_norm = result
        assert isinstance(best_rows, pd.DataFrame)
        assert isinstance(best_rows_norm, pd.DataFrame)

    def test_best_rows_keyed_by_query_name(self, csv_file):
        best_rows, _ = load_drugbank_lookup(csv_file)
        assert best_rows["query_name"].is_unique

    def test_best_rows_norm_keyed_by_query_norm(self, csv_file):
        _, best_rows_norm = load_drugbank_lookup(csv_file)
        assert best_rows_norm["query_norm"].is_unique

    def test_most_complete_row_wins_on_duplicate_name(self, csv_file):
        """For duplicate query_name, the row with more non-null columns is kept."""
        best_rows, _ = load_drugbank_lookup(csv_file)
        # "lepirudin" appears twice — first row has modality (4 non-null), second has NaN modality (3 non-null)
        lepirudin_row = best_rows[best_rows["query_name"] == "lepirudin"]
        assert len(lepirudin_row) == 1
        assert lepirudin_row.iloc[0]["modality"] == "peptide"


# ---------------------------------------------------------------------------
# TestMatchDrugName
# ---------------------------------------------------------------------------

class TestMatchDrugName:
    @pytest.fixture
    def lookup_tables(self, tmp_path, drugbank_csv_content):
        p = tmp_path / "drugbank_approvals.csv"
        p.write_text(drugbank_csv_content)
        return load_drugbank_lookup(p)

    def test_exact_normalized_match(self, lookup_tables):
        best_rows, best_rows_norm = lookup_tables
        result = match_drug_name("lepirudin", best_rows, best_rows_norm)
        assert result == "DB00001"

    def test_first_word_fallback(self, lookup_tables):
        """'lepirudin hcl' has no exact match but first word 'lepirudin' does."""
        best_rows, best_rows_norm = lookup_tables
        result = match_drug_name("lepirudin hcl", best_rows, best_rows_norm)
        assert result == "DB00001"

    def test_parenthetical_stripped_before_match(self, lookup_tables):
        """'Lepirudin (rDNA origin) HCl' should match DB00001 after canonicalization."""
        best_rows, best_rows_norm = lookup_tables
        result = match_drug_name("Lepirudin (rDNA origin) HCl", best_rows, best_rows_norm)
        assert result == "DB00001"

    def test_no_match_returns_none(self, lookup_tables):
        best_rows, best_rows_norm = lookup_tables
        result = match_drug_name("unknowndrug xyz", best_rows, best_rows_norm)
        assert result is None

    def test_exact_match_preferred_over_first_word(self, lookup_tables):
        """'insulin human' matches exactly (DB00030); first word 'insulin' → DB00050.
        Exact match must win."""
        best_rows, best_rows_norm = lookup_tables
        result = match_drug_name("insulin human", best_rows, best_rows_norm)
        assert result == "DB00030"
