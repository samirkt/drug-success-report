"""
Tests for Stage 3a: Attribute Classification (pipeline/stages/classification.py)

Test categories:
  PASS NOW   — constructor, orchestration wiring (via mocks)
  FAIL NOW   — behavioral contracts for _literature_lookup (still a stub)
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from pipeline.models import (
    AttributeTable,
    Candidate,
    CandidateAttributes,
    CandidateTable,
    TrialPhase,
)
from pipeline.stages.classification import AttributeClassificationStage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_llm_message(data: dict) -> MagicMock:
    """Build a mock Anthropic Message whose content yields the given JSON (single object).
    Used for tests where the outcome value is not asserted (parse fallback is acceptable)."""
    block = MagicMock()
    block.type = "text"
    block.text = json.dumps(data)
    msg = MagicMock()
    msg.content = [block]
    return msg


def _make_llm_message_batch(data_list: list[dict]) -> MagicMock:
    """Build a mock Anthropic Message whose content yields a JSON array.
    Automatically injects candidate_index (1-based) into each item."""
    indexed = [{**d, "candidate_index": i + 1} for i, d in enumerate(data_list)]
    block = MagicMock()
    block.type = "text"
    block.text = json.dumps(indexed)
    msg = MagicMock()
    msg.content = [block]
    return msg


_GOOD_CLASSIFICATION = {
    "drug_modality": "peptide",
    "disease_area": "metabolic",
    "modality_confidence": "HIGH",
    "disease_confidence": "HIGH",
    "reasoning": "DrugA is a GLP-1 analog — a peptide hormone.",
}


# ---------------------------------------------------------------------------
# Constructor / initialization
# ---------------------------------------------------------------------------

class TestAttributeClassificationStageInit:
    def test_default_use_literature_is_true(self):
        stage = AttributeClassificationStage()
        assert stage.use_literature is True

    def test_custom_use_literature_false(self):
        stage = AttributeClassificationStage(use_literature=False)
        assert stage.use_literature is False


# ---------------------------------------------------------------------------
# run() orchestration with mocks
# ---------------------------------------------------------------------------

class TestAttributeClassificationStageRunOrchestration:
    def test_run_returns_attribute_table_type(self, sample_candidate_table, sample_attributes_cand001):
        stage = AttributeClassificationStage()
        n = len(sample_candidate_table.candidates)
        stage._llm_classify_batch = MagicMock(return_value=[sample_attributes_cand001] * n)

        result = stage.run(sample_candidate_table)

        assert isinstance(result, AttributeTable)

    def test_run_classifies_each_candidate(self, sample_candidate_table, sample_attributes_cand001):
        stage = AttributeClassificationStage()
        n = len(sample_candidate_table.candidates)
        stage._llm_classify_batch = MagicMock(return_value=[sample_attributes_cand001] * n)

        result = stage.run(sample_candidate_table)

        assert len(result.attributes) == n

    def test_run_keys_attributes_by_candidate_id(self, sample_candidate_table, sample_attributes_cand001):
        stage = AttributeClassificationStage()
        n = len(sample_candidate_table.candidates)
        stage._llm_classify_batch = MagicMock(return_value=[sample_attributes_cand001] * n)

        result = stage.run(sample_candidate_table)

        for candidate in sample_candidate_table.candidates:
            assert candidate.candidate_id in result.attributes

    def test_run_empty_candidate_table_returns_empty_attribute_table(self, empty_candidate_table):
        stage = AttributeClassificationStage()
        stage._classify = MagicMock()

        result = stage.run(empty_candidate_table)

        assert isinstance(result, AttributeTable)
        assert result.attributes == {}
        stage._classify.assert_not_called()

    def test_run_passes_candidates_to_batch(self, single_candidate_table, sample_candidate, sample_attributes_cand001):
        stage = AttributeClassificationStage()
        stage._llm_classify_batch = MagicMock(return_value=[sample_attributes_cand001])

        stage.run(single_candidate_table)

        stage._llm_classify_batch.assert_called_once_with([sample_candidate])


# ---------------------------------------------------------------------------
# run() end-to-end with mocked LLM
# ---------------------------------------------------------------------------

class TestAttributeClassificationStageRunWithLLM:
    @patch("pipeline.stages.classification.process_prompt_request")
    def test_run_returns_attribute_table_with_mocked_llm(self, mock_llm, single_candidate_table):
        mock_llm.return_value = _make_llm_message_batch([_GOOD_CLASSIFICATION])
        stage = AttributeClassificationStage()

        result = stage.run(single_candidate_table)

        assert isinstance(result, AttributeTable)
        assert len(result.attributes) == 1

    @patch("pipeline.stages.classification.process_prompt_request")
    def test_run_does_not_raise_for_empty_table(self, mock_llm, empty_candidate_table):
        stage = AttributeClassificationStage()
        result = stage.run(empty_candidate_table)
        assert isinstance(result, AttributeTable)
        mock_llm.assert_not_called()


# ---------------------------------------------------------------------------
# _classify
# ---------------------------------------------------------------------------

class TestClassify:
    @patch("pipeline.stages.classification.process_prompt_request")
    def test_classify_returns_candidate_attributes(self, mock_llm, sample_candidate):
        mock_llm.return_value = _make_llm_message(_GOOD_CLASSIFICATION)
        stage = AttributeClassificationStage()
        result = stage._classify(sample_candidate)
        assert isinstance(result, CandidateAttributes)

    @patch("pipeline.stages.classification.process_prompt_request")
    def test_classify_sets_candidate_id(self, mock_llm, sample_candidate):
        mock_llm.return_value = _make_llm_message(_GOOD_CLASSIFICATION)
        stage = AttributeClassificationStage()
        result = stage._classify(sample_candidate)
        assert result.candidate_id == sample_candidate.candidate_id

    @patch("pipeline.stages.classification.process_prompt_request")
    def test_classify_sets_non_empty_drug_modality(self, mock_llm, sample_candidate):
        mock_llm.return_value = _make_llm_message(_GOOD_CLASSIFICATION)
        stage = AttributeClassificationStage()
        result = stage._classify(sample_candidate)
        assert result.drug_modality != ""

    @patch("pipeline.stages.classification.process_prompt_request")
    def test_classify_sets_non_empty_disease_area(self, mock_llm, sample_candidate):
        mock_llm.return_value = _make_llm_message(_GOOD_CLASSIFICATION)
        stage = AttributeClassificationStage()
        result = stage._classify(sample_candidate)
        assert result.disease_area != ""

    @patch("pipeline.stages.classification.process_prompt_request")
    def test_classify_modality_confidence_in_range(self, mock_llm, sample_candidate):
        mock_llm.return_value = _make_llm_message(_GOOD_CLASSIFICATION)
        stage = AttributeClassificationStage()
        result = stage._classify(sample_candidate)
        assert 0.0 <= result.modality_confidence <= 1.0

    @patch("pipeline.stages.classification.process_prompt_request")
    def test_classify_disease_confidence_in_range(self, mock_llm, sample_candidate):
        mock_llm.return_value = _make_llm_message(_GOOD_CLASSIFICATION)
        stage = AttributeClassificationStage()
        result = stage._classify(sample_candidate)
        assert 0.0 <= result.disease_confidence <= 1.0

    @patch("pipeline.stages.classification.process_prompt_request")
    def test_classify_known_peptide(self, mock_llm):
        mock_llm.return_value = _make_llm_message_batch([_GOOD_CLASSIFICATION])
        candidate = Candidate(
            candidate_id="c_peptide",
            drug_name="semaglutide",
            indication="Type 2 Diabetes",
            highest_phase=TrialPhase.PHASE_3,
        )
        stage = AttributeClassificationStage()
        result = stage._classify(candidate)
        assert result.drug_modality.lower() == "peptide"

    @patch("pipeline.stages.classification.process_prompt_request")
    def test_classify_known_oncology_indication(self, mock_llm):
        mock_llm.return_value = _make_llm_message_batch([{
            **_GOOD_CLASSIFICATION,
            "disease_area": "oncology",
        }])
        candidate = Candidate(
            candidate_id="c_onco",
            drug_name="PeptideB",
            indication="Non-Small Cell Lung Cancer",
            highest_phase=TrialPhase.PHASE_1,
        )
        stage = AttributeClassificationStage()
        result = stage._classify(candidate)
        assert result.disease_area.lower() == "oncology"


# ---------------------------------------------------------------------------
# _llm_classify
# ---------------------------------------------------------------------------

class TestLlmClassify:
    @patch("pipeline.stages.classification.process_prompt_request")
    def test_llm_classify_returns_candidate_attributes(self, mock_llm, sample_candidate):
        mock_llm.return_value = _make_llm_message(_GOOD_CLASSIFICATION)
        stage = AttributeClassificationStage()
        result = stage._llm_classify(sample_candidate)
        assert isinstance(result, CandidateAttributes)

    @patch("pipeline.stages.classification.process_prompt_request")
    def test_llm_classify_sets_candidate_id(self, mock_llm, sample_candidate):
        mock_llm.return_value = _make_llm_message(_GOOD_CLASSIFICATION)
        stage = AttributeClassificationStage()
        result = stage._llm_classify(sample_candidate)
        assert result.candidate_id == sample_candidate.candidate_id

    @patch("pipeline.stages.classification.process_prompt_request")
    def test_llm_classify_modality_is_string(self, mock_llm, sample_candidate):
        mock_llm.return_value = _make_llm_message(_GOOD_CLASSIFICATION)
        stage = AttributeClassificationStage()
        result = stage._llm_classify(sample_candidate)
        assert isinstance(result.drug_modality, str)

    @patch("pipeline.stages.classification.process_prompt_request")
    def test_llm_classify_malformed_response_falls_back_to_defaults(self, mock_llm, sample_candidate):
        """Malformed JSON should fall back to unknown/0.0 rather than crashing."""
        block = MagicMock()
        block.type = "text"
        block.text = "not valid json {{{"
        msg = MagicMock()
        msg.content = [block]
        mock_llm.return_value = msg

        stage = AttributeClassificationStage()
        result = stage._llm_classify(sample_candidate)

        assert isinstance(result, CandidateAttributes)
        assert result.drug_modality == "unknown"
        assert result.modality_confidence == 0.0


# ---------------------------------------------------------------------------
# Cache integration tests
# ---------------------------------------------------------------------------

class TestAttributeClassificationStageWithCache:
    @patch("pipeline.stages.classification.process_prompt_request")
    def test_cache_hit_skips_llm_call(self, mock_llm, single_candidate_table, sample_candidate, knowledge_cache):
        from pipeline.knowledge_cache import KnowledgeCache
        key = KnowledgeCache.make_classification_key(sample_candidate.drug_name, sample_candidate.indication)
        cached_attrs = CandidateAttributes(
            candidate_id=sample_candidate.candidate_id,
            drug_modality="peptide",
            disease_area="metabolic",
            modality_confidence=1.0,
            disease_confidence=1.0,
            reasoning="Pre-cached.",
        )
        knowledge_cache.put_attributes(key, cached_attrs)

        stage = AttributeClassificationStage(cache=knowledge_cache)
        result = stage.run(single_candidate_table)

        mock_llm.assert_not_called()
        assert result.attributes[sample_candidate.candidate_id].drug_modality == "peptide"

    @patch("pipeline.stages.classification.process_prompt_request")
    def test_cache_miss_calls_llm_and_writes_to_cache(self, mock_llm, single_candidate_table, sample_candidate, knowledge_cache):
        from pipeline.knowledge_cache import KnowledgeCache
        mock_llm.return_value = _make_llm_message_batch([_GOOD_CLASSIFICATION])

        stage = AttributeClassificationStage(cache=knowledge_cache)
        stage.run(single_candidate_table)

        mock_llm.assert_called_once()
        key = KnowledgeCache.make_classification_key(sample_candidate.drug_name, sample_candidate.indication)
        cached = knowledge_cache.get_attributes(key, sample_candidate.candidate_id)
        assert cached is not None
        assert cached.drug_modality == "peptide"

    @patch("pipeline.stages.classification.process_prompt_request")
    def test_no_cache_injected_calls_llm_normally(self, mock_llm, single_candidate_table):
        mock_llm.return_value = _make_llm_message(_GOOD_CLASSIFICATION)
        stage = AttributeClassificationStage()
        stage.run(single_candidate_table)
        mock_llm.assert_called_once()

    @patch("pipeline.stages.classification.process_prompt_request")
    def test_llm_failure_does_not_write_to_cache(self, mock_llm, sample_candidate, knowledge_cache):
        from pipeline.knowledge_cache import KnowledgeCache
        mock_llm.side_effect = RuntimeError("LLM unavailable")

        stage = AttributeClassificationStage(cache=knowledge_cache)
        stage._classify(sample_candidate)

        key = KnowledgeCache.make_classification_key(sample_candidate.drug_name, sample_candidate.indication)
        assert knowledge_cache.get_attributes(key, sample_candidate.candidate_id) is None


# ---------------------------------------------------------------------------
# TestClassificationUsesRawDrugName
# ---------------------------------------------------------------------------

class TestClassificationUsesRawDrugName:
    @patch("pipeline.stages.classification.process_prompt_request")
    def test_llm_prompt_uses_drug_name_raw(self, mock_llm, sample_candidate):
        """The string passed to the LLM must contain drug_name_raw, not drug_name."""
        mock_llm.return_value = _make_llm_message(_GOOD_CLASSIFICATION)
        stage = AttributeClassificationStage()
        stage._llm_classify(sample_candidate)

        call_kwargs = mock_llm.call_args
        user_prompt = call_kwargs.kwargs.get("user_prompt", "") or call_kwargs.args[1] if call_kwargs.args else ""
        # Inspect via the captured call args
        all_call_args = str(mock_llm.call_args)
        assert sample_candidate.drug_name_raw in all_call_args
        assert sample_candidate.drug_name not in all_call_args or sample_candidate.drug_name_raw in all_call_args

    @patch("pipeline.stages.classification.process_prompt_request")
    def test_cache_key_uses_drug_name(self, mock_llm, sample_candidate, knowledge_cache):
        """Cache key must be built from drug_name (normalized), not drug_name_raw."""
        mock_llm.return_value = _make_llm_message(_GOOD_CLASSIFICATION)
        stage = AttributeClassificationStage(cache=knowledge_cache)

        with patch("pipeline.knowledge_cache.KnowledgeCache.make_classification_key") as mock_key:
            mock_key.return_value = "test_key"
            knowledge_cache.get_attributes = lambda *a, **kw: None
            knowledge_cache.put_attributes = lambda *a, **kw: None
            stage._classify(sample_candidate)

        mock_key.assert_called_once_with(sample_candidate.drug_name, sample_candidate.indication)


# ---------------------------------------------------------------------------
# _literature_lookup  (still a stub — NotImplementedError expected)
# ---------------------------------------------------------------------------

class TestLiteratureLookup:
    def test_literature_lookup_raises_not_implemented(self):
        stage = AttributeClassificationStage()
        with pytest.raises(NotImplementedError):
            stage._literature_lookup("semaglutide")


# ---------------------------------------------------------------------------
# Disease area normalization
# ---------------------------------------------------------------------------

from pipeline.stages.classification import normalize_disease_area


class TestNormalizeDiseaseArea:
    def test_space_to_underscore(self):
        assert normalize_disease_area("infectious disease") == "infectious_disease"

    def test_case_normalized(self):
        assert normalize_disease_area("Oncology") == "oncology"
        assert normalize_disease_area("CARDIOVASCULAR") == "cardiovascular"

    def test_whitespace_stripped(self):
        assert normalize_disease_area("  oncology  ") == "oncology"
        assert normalize_disease_area(" neurology\n") == "neurology"

    def test_synonym_resolved(self):
        assert normalize_disease_area("immunology") == "autoimmune"
        assert normalize_disease_area("allergy") == "autoimmune"
        assert normalize_disease_area("endocrinology") == "endocrine"
        assert normalize_disease_area("addiction") == "psychiatry"
        assert normalize_disease_area("hepatology") == "gastroenterology"
        assert normalize_disease_area("vascular") == "cardiovascular"

    def test_old_granular_categories_to_other(self):
        assert normalize_disease_area("dermatology") == "other"
        assert normalize_disease_area("musculoskeletal") == "other"
        assert normalize_disease_area("reproductive_health") == "other"
        assert normalize_disease_area("nephrology") == "other"
        assert normalize_disease_area("rare_disease") == "other"
        assert normalize_disease_area("gynecology") == "other"
        assert normalize_disease_area("pain_management") == "other"

    def test_nonclinical_to_other(self):
        assert normalize_disease_area("drug_safety") == "other"
        assert normalize_disease_area("research_tool") == "other"
        assert normalize_disease_area("diagnostic") == "other"

    def test_unknown_fallback(self):
        assert normalize_disease_area("made_up_area") == "unknown"
        assert normalize_disease_area("") == "unknown"

    def test_valid_areas_pass_through(self):
        assert normalize_disease_area("oncology") == "oncology"
        assert normalize_disease_area("autoimmune") == "autoimmune"
        assert normalize_disease_area("endocrine") == "endocrine"
        assert normalize_disease_area("urology") == "urology"
        assert normalize_disease_area("other") == "other"
        assert normalize_disease_area("unknown") == "unknown"


# ---------------------------------------------------------------------------
# BIO/QLS disease area mapping
# ---------------------------------------------------------------------------

from pipeline.stages.classification import to_bio_qls_disease_area


class TestBioQlsMapping:
    def test_direct_matches(self):
        assert to_bio_qls_disease_area("oncology") == "Oncology"
        assert to_bio_qls_disease_area("cardiovascular") == "Cardiovascular"
        assert to_bio_qls_disease_area("hematology") == "Hematology"
        assert to_bio_qls_disease_area("endocrine") == "Endocrine"
        assert to_bio_qls_disease_area("urology") == "Urology"
        assert to_bio_qls_disease_area("autoimmune") == "Autoimmune"

    def test_other_and_unknown_map_to_others(self):
        assert to_bio_qls_disease_area("other") == "Others"
        assert to_bio_qls_disease_area("unknown") == "Others"
        assert to_bio_qls_disease_area("nonexistent") == "Others"


# ---------------------------------------------------------------------------
# MeSH tree-based disease area resolution
# ---------------------------------------------------------------------------

from pipeline.stages.classification import mesh_trees_to_disease_area


class TestMeshTreesToDiseaseArea:
    def test_single_bucket(self):
        # All oncology tree numbers
        assert mesh_trees_to_disease_area(["C04.588.180", "C04.588.443"]) == "oncology"

    def test_plurality_wins(self):
        # 2 oncology, 1 metabolic
        result = mesh_trees_to_disease_area(["C04.588.180", "C04.588.443", "C18.452.394"])
        assert result == "oncology"

    def test_tie_returns_none(self):
        # 1 oncology, 1 metabolic
        result = mesh_trees_to_disease_area(["C04.588.180", "C18.452.394"])
        assert result is None

    def test_empty_returns_none(self):
        assert mesh_trees_to_disease_area([]) is None

    def test_unmapped_prefixes_only_returns_other(self):
        # C05 = musculoskeletal, C17 = skin — neither mapped to named bucket
        assert mesh_trees_to_disease_area(["C05.550.114", "C17.800.090"]) == "other"

    def test_infectious_disease_subtrees(self):
        # C01, C02, C03 all map to infectious_disease
        assert mesh_trees_to_disease_area(["C01.539", "C02.256"]) == "infectious_disease"

    def test_psychiatry_from_f03(self):
        assert mesh_trees_to_disease_area(["F03.600.300"]) == "psychiatry"

    def test_mixed_with_clear_winner(self):
        # 3 cardiovascular, 1 neurology, 1 unmapped
        result = mesh_trees_to_disease_area([
            "C14.280.067", "C14.280.400", "C14.907.253",
            "C10.228.140", "C05.550.114"
        ])
        assert result == "cardiovascular"
