"""
Tests for Stage 3b: Outcome Adjudication (pipeline/stages/adjudication.py)

Test categories:
  PASS NOW   — constructor, orchestration wiring (via mocks)
  FAIL NOW   — behavioral contracts for _infer_from_trials,
               _check_regulatory_status (still stubs)
"""

import json
from datetime import date
from unittest.mock import MagicMock, patch

import pytest

from pipeline.models import (
    Candidate,
    CandidateOutcome,
    CandidateOutcomeRecord,
    CandidateTable,
    OutcomeTable,
    TrialPhase,
    TrialStatus,
)
from pipeline.stages.adjudication import OutcomeAdjudicationStage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_llm_message_batch(data_list: list[dict]) -> MagicMock:
    """Build a mock Anthropic Message whose content yields a JSON array of adjudication results.
    Automatically injects candidate_index (1-based) into each item."""
    indexed = [{**d, "candidate_index": i + 1} for i, d in enumerate(data_list)]
    block = MagicMock()
    block.type = "text"
    block.text = json.dumps(indexed)
    msg = MagicMock()
    msg.content = [block]
    return msg


_GOOD_ADJUDICATION = {
    "outcome": "ONGOING",
    "confidence": "MEDIUM",
    "reasoning": "The drug is still in Phase 2 trials with no termination signal.",
    "evidence_sources": ["trial_metadata"],
    "approval_date": None,
    "commercialization_date": None,
}

_APPROVED_ADJUDICATION = {
    "outcome": "APPROVED",
    "confidence": "HIGH",
    "reasoning": "Drug approved by FDA in 2022.",
    "evidence_sources": ["FDA Orange Book"],
    "approval_date": "2022-04-15",
    "commercialization_date": None,
}


# ---------------------------------------------------------------------------
# Constructor / initialization
# ---------------------------------------------------------------------------

class TestOutcomeAdjudicationStageInit:
    def test_default_use_regulatory_data_is_true(self):
        stage = OutcomeAdjudicationStage()
        assert stage.use_regulatory_data is True

    def test_custom_use_regulatory_data_false(self):
        stage = OutcomeAdjudicationStage(use_regulatory_data=False)
        assert stage.use_regulatory_data is False


# ---------------------------------------------------------------------------
# run() orchestration with mocks
# ---------------------------------------------------------------------------

class TestOutcomeAdjudicationStageRunOrchestration:
    def test_run_returns_outcome_table_type(self, sample_candidate_table, sample_outcome_cand001):
        stage = OutcomeAdjudicationStage()
        n = len(sample_candidate_table.candidates)
        stage._llm_adjudicate_batch = MagicMock(return_value=[sample_outcome_cand001] * n)

        result = stage.run(sample_candidate_table)

        assert isinstance(result, OutcomeTable)

    def test_run_adjudicates_each_candidate(self, sample_candidate_table, sample_outcome_cand001):
        stage = OutcomeAdjudicationStage()
        n = len(sample_candidate_table.candidates)
        stage._llm_adjudicate_batch = MagicMock(return_value=[sample_outcome_cand001] * n)

        result = stage.run(sample_candidate_table)

        assert len(result.outcomes) == n

    def test_run_keys_outcomes_by_candidate_id(self, sample_candidate_table, sample_outcome_cand001):
        stage = OutcomeAdjudicationStage()
        n = len(sample_candidate_table.candidates)
        stage._llm_adjudicate_batch = MagicMock(return_value=[sample_outcome_cand001] * n)

        result = stage.run(sample_candidate_table)

        for candidate in sample_candidate_table.candidates:
            assert candidate.candidate_id in result.outcomes

    def test_run_empty_candidate_table_returns_empty_outcome_table(self, empty_candidate_table):
        stage = OutcomeAdjudicationStage()
        stage._adjudicate = MagicMock()

        result = stage.run(empty_candidate_table)

        assert isinstance(result, OutcomeTable)
        assert result.outcomes == {}
        stage._adjudicate.assert_not_called()

    def test_run_passes_candidates_to_batch(self, single_candidate_table, sample_candidate, sample_outcome_cand001):
        stage = OutcomeAdjudicationStage()
        stage._llm_adjudicate_batch = MagicMock(return_value=[sample_outcome_cand001])

        stage.run(single_candidate_table)

        stage._llm_adjudicate_batch.assert_called_once_with([sample_candidate])


# ---------------------------------------------------------------------------
# run() end-to-end with mocked LLM
# ---------------------------------------------------------------------------

class TestOutcomeAdjudicationStageRunWithLLM:
    @patch("pipeline.stages.adjudication.process_prompt_request")
    def test_run_returns_outcome_table_with_mocked_llm(self, mock_llm, single_candidate_table):
        mock_llm.return_value = _make_llm_message_batch([_GOOD_ADJUDICATION])
        stage = OutcomeAdjudicationStage()

        result = stage.run(single_candidate_table)

        assert isinstance(result, OutcomeTable)
        assert len(result.outcomes) == 1

    @patch("pipeline.stages.adjudication.process_prompt_request")
    def test_run_does_not_raise_for_empty_table(self, mock_llm, empty_candidate_table):
        stage = OutcomeAdjudicationStage()
        result = stage.run(empty_candidate_table)
        assert isinstance(result, OutcomeTable)
        mock_llm.assert_not_called()


# ---------------------------------------------------------------------------
# _adjudicate
# ---------------------------------------------------------------------------

class TestAdjudicate:
    @patch("pipeline.stages.adjudication.process_prompt_request")
    def test_adjudicate_returns_candidate_outcome_record(self, mock_llm, sample_candidate):
        mock_llm.return_value = _make_llm_message_batch([_GOOD_ADJUDICATION])
        stage = OutcomeAdjudicationStage()
        result = stage._adjudicate(sample_candidate)
        assert isinstance(result, CandidateOutcomeRecord)

    @patch("pipeline.stages.adjudication.process_prompt_request")
    def test_adjudicate_sets_candidate_id(self, mock_llm, sample_candidate):
        mock_llm.return_value = _make_llm_message_batch([_GOOD_ADJUDICATION])
        stage = OutcomeAdjudicationStage()
        result = stage._adjudicate(sample_candidate)
        assert result.candidate_id == sample_candidate.candidate_id

    @patch("pipeline.stages.adjudication.process_prompt_request")
    def test_adjudicate_sets_valid_outcome(self, mock_llm, sample_candidate):
        mock_llm.return_value = _make_llm_message_batch([_GOOD_ADJUDICATION])
        stage = OutcomeAdjudicationStage()
        result = stage._adjudicate(sample_candidate)
        assert isinstance(result.outcome, CandidateOutcome)

    @patch("pipeline.stages.adjudication.process_prompt_request")
    def test_adjudicate_confidence_in_range(self, mock_llm, sample_candidate):
        mock_llm.return_value = _make_llm_message_batch([_GOOD_ADJUDICATION])
        stage = OutcomeAdjudicationStage()
        result = stage._adjudicate(sample_candidate)
        assert 0.0 <= result.confidence <= 1.0

    @patch("pipeline.stages.adjudication.process_prompt_request")
    def test_adjudicate_ongoing_phase2_infers_ongoing(self, mock_llm):
        mock_llm.return_value = _make_llm_message_batch([{**_GOOD_ADJUDICATION, "outcome": "ONGOING"}])
        candidate = Candidate(
            candidate_id="c_ongoing",
            drug_name="OngoingDrug",
            indication="Hypertension",
            trial_ids=["NCT100"],
            highest_phase=TrialPhase.PHASE_2,
            sponsors=["CardioInc"],
        )
        stage = OutcomeAdjudicationStage()
        result = stage._adjudicate(candidate)
        assert result.outcome == CandidateOutcome.ONGOING

    @patch("pipeline.stages.adjudication.process_prompt_request")
    def test_adjudicate_failed_phase1(self, mock_llm):
        mock_llm.return_value = _make_llm_message_batch([{**_GOOD_ADJUDICATION, "outcome": "FAILED_PHASE_1"}])
        candidate = Candidate(
            candidate_id="c_fail",
            drug_name="FailedDrug",
            indication="Diabetes",
            trial_ids=["NCT999"],
            highest_phase=TrialPhase.PHASE_1,
            sponsors=["PharmaCo"],
        )
        stage = OutcomeAdjudicationStage()
        result = stage._adjudicate(candidate)
        assert result.outcome == CandidateOutcome.FAILED_PHASE_1


# ---------------------------------------------------------------------------
# _llm_adjudicate
# ---------------------------------------------------------------------------

class TestLlmAdjudicate:
    @patch("pipeline.stages.adjudication.process_prompt_request")
    def test_llm_adjudicate_returns_candidate_outcome_record(self, mock_llm, sample_candidate):
        mock_llm.return_value = _make_llm_message_batch([_GOOD_ADJUDICATION])
        stage = OutcomeAdjudicationStage()
        result = stage._llm_adjudicate(sample_candidate, context=None)
        assert isinstance(result, CandidateOutcomeRecord)

    @patch("pipeline.stages.adjudication.process_prompt_request")
    def test_llm_adjudicate_called_with_context_dict(self, mock_llm, sample_candidate):
        mock_llm.return_value = _make_llm_message_batch([_GOOD_ADJUDICATION])
        stage = OutcomeAdjudicationStage()
        result = stage._llm_adjudicate(sample_candidate, context={"some_key": "some_value"})
        assert isinstance(result, CandidateOutcomeRecord)

    @patch("pipeline.stages.adjudication.process_prompt_request")
    def test_llm_adjudicate_outcome_is_valid_enum(self, mock_llm, sample_candidate):
        mock_llm.return_value = _make_llm_message_batch([_GOOD_ADJUDICATION])
        stage = OutcomeAdjudicationStage()
        result = stage._llm_adjudicate(sample_candidate, context=None)
        assert isinstance(result.outcome, CandidateOutcome)

    @patch("pipeline.stages.adjudication.process_prompt_request")
    def test_llm_adjudicate_malformed_response_falls_back_to_unknown(self, mock_llm, sample_candidate):
        """Malformed JSON should fall back to UNKNOWN/0.0 rather than crashing."""
        block = MagicMock()
        block.type = "text"
        block.text = "not valid json {{{"
        msg = MagicMock()
        msg.content = [block]
        mock_llm.return_value = msg

        stage = OutcomeAdjudicationStage()
        result = stage._llm_adjudicate(sample_candidate, context=None)

        assert isinstance(result, CandidateOutcomeRecord)
        assert result.outcome == CandidateOutcome.UNKNOWN
        assert result.confidence == 0.0

    @patch("pipeline.stages.adjudication.process_prompt_request")
    def test_llm_adjudicate_unknown_outcome_string_falls_back(self, mock_llm, sample_candidate):
        """Unrecognized outcome string should fall back to UNKNOWN."""
        mock_llm.return_value = _make_llm_message_batch([{
            **_GOOD_ADJUDICATION,
            "outcome": "NOT_A_REAL_OUTCOME",
        }])
        stage = OutcomeAdjudicationStage()
        result = stage._llm_adjudicate(sample_candidate, context=None)
        assert result.outcome == CandidateOutcome.UNKNOWN


# ---------------------------------------------------------------------------
# Cache integration tests
# ---------------------------------------------------------------------------

class TestOutcomeAdjudicationStageWithCache:
    @patch("pipeline.stages.adjudication.process_prompt_request")
    def test_cache_hit_skips_llm_call(self, mock_llm, single_candidate_table, sample_candidate, knowledge_cache):
        from pipeline.knowledge_cache import KnowledgeCache
        key = KnowledgeCache.make_adjudication_key(
            sample_candidate.drug_name, sample_candidate.indication, sample_candidate.highest_phase.value
        )
        cached_record = CandidateOutcomeRecord(
            candidate_id=sample_candidate.candidate_id,
            outcome=CandidateOutcome.APPROVED,
            confidence=0.9,
            reasoning="Pre-cached.",
            evidence_sources=["cache"],
        )
        knowledge_cache.put_outcome(key, cached_record)

        stage = OutcomeAdjudicationStage(cache=knowledge_cache)
        result = stage.run(single_candidate_table)

        mock_llm.assert_not_called()
        assert result.outcomes[sample_candidate.candidate_id].outcome == CandidateOutcome.APPROVED

    @patch("pipeline.stages.adjudication.process_prompt_request")
    def test_cache_miss_calls_llm_and_writes_to_cache(self, mock_llm, single_candidate_table, sample_candidate, knowledge_cache):
        from pipeline.knowledge_cache import KnowledgeCache
        mock_llm.return_value = _make_llm_message_batch([_GOOD_ADJUDICATION])

        stage = OutcomeAdjudicationStage(cache=knowledge_cache)
        stage.run(single_candidate_table)

        mock_llm.assert_called_once()
        key = KnowledgeCache.make_adjudication_key(
            sample_candidate.drug_name, sample_candidate.indication, sample_candidate.highest_phase.value
        )
        cached = knowledge_cache.get_outcome(key, sample_candidate.candidate_id)
        assert cached is not None
        assert cached.outcome == CandidateOutcome.ONGOING

    @patch("pipeline.stages.adjudication.process_prompt_request")
    def test_no_cache_injected_calls_llm_normally(self, mock_llm, single_candidate_table):
        mock_llm.return_value = _make_llm_message_batch([_GOOD_ADJUDICATION])
        stage = OutcomeAdjudicationStage()
        stage.run(single_candidate_table)
        mock_llm.assert_called_once()

    @patch("pipeline.stages.adjudication.process_prompt_request")
    def test_llm_failure_does_not_write_to_cache(self, mock_llm, sample_candidate, knowledge_cache):
        from pipeline.knowledge_cache import KnowledgeCache
        mock_llm.side_effect = RuntimeError("LLM unavailable")

        stage = OutcomeAdjudicationStage(cache=knowledge_cache)
        stage._adjudicate(sample_candidate)

        key = KnowledgeCache.make_adjudication_key(
            sample_candidate.drug_name, sample_candidate.indication, sample_candidate.highest_phase.value
        )
        assert knowledge_cache.get_outcome(key, sample_candidate.candidate_id) is None


# ---------------------------------------------------------------------------
# TestAdjudicationUsesRawDrugName
# ---------------------------------------------------------------------------

class TestAdjudicationUsesRawDrugName:
    @patch("pipeline.stages.adjudication.process_prompt_request")
    def test_llm_prompt_uses_drug_name_raw(self, mock_llm, sample_candidate):
        """The prompt sent to the LLM must contain drug_name_raw, not drug_name."""
        mock_llm.return_value = _make_llm_message_batch([_GOOD_ADJUDICATION])
        stage = OutcomeAdjudicationStage()
        stage._llm_adjudicate(sample_candidate, context=None)

        all_call_args = str(mock_llm.call_args)
        assert sample_candidate.drug_name_raw in all_call_args

    @patch("pipeline.stages.adjudication.process_prompt_request")
    def test_cache_key_uses_drug_name(self, mock_llm, sample_candidate, knowledge_cache):
        """Cache key must be built from drug_name (normalized), not drug_name_raw."""
        mock_llm.return_value = _make_llm_message_batch([_GOOD_ADJUDICATION])
        stage = OutcomeAdjudicationStage(cache=knowledge_cache)

        with patch("pipeline.knowledge_cache.KnowledgeCache.make_adjudication_key") as mock_key:
            mock_key.return_value = "test_key"
            knowledge_cache.get_outcome = lambda *a, **kw: None
            knowledge_cache.put_outcome = lambda *a, **kw: None
            stage._adjudicate(sample_candidate)

        mock_key.assert_called_once_with(
            sample_candidate.drug_name,
            sample_candidate.indication,
            sample_candidate.highest_phase.value,
        )


# ---------------------------------------------------------------------------
# _infer_from_trials  (still a stub — NotImplementedError expected)
# ---------------------------------------------------------------------------

class TestInferFromTrials:
    def test_infer_from_trials_raises_not_implemented(self, sample_candidate):
        stage = OutcomeAdjudicationStage()
        with pytest.raises(NotImplementedError):
            stage._infer_from_trials(sample_candidate)


# ---------------------------------------------------------------------------
# _check_regulatory_status  (still a stub — NotImplementedError expected)
# ---------------------------------------------------------------------------

class TestCheckRegulatoryStatus:
    def test_check_regulatory_status_raises_not_implemented(self):
        stage = OutcomeAdjudicationStage()
        with pytest.raises(NotImplementedError):
            stage._check_regulatory_status("semaglutide", "Type 2 Diabetes")


# ---------------------------------------------------------------------------
# _llm_adjudicate — date field extraction  (PASS NOW)
# ---------------------------------------------------------------------------

class TestLlmAdjudicateDateExtraction:
    @patch("pipeline.stages.adjudication.process_prompt_request")
    def test_llm_adjudicate_approved_outcome_has_approval_date(self, mock_llm, sample_candidate):
        mock_llm.return_value = _make_llm_message_batch([_APPROVED_ADJUDICATION])
        stage = OutcomeAdjudicationStage()
        result = stage._llm_adjudicate(sample_candidate, context=None)
        assert result.approval_date == date(2022, 4, 15)

    @patch("pipeline.stages.adjudication.process_prompt_request")
    def test_llm_adjudicate_commercialized_outcome_has_commercialization_date(self, mock_llm, sample_candidate):
        data = {
            **_APPROVED_ADJUDICATION,
            "outcome": "COMMERCIALIZED",
            "approval_date": "2021-07-01",
            "commercialization_date": "2022-01-15",
        }
        mock_llm.return_value = _make_llm_message_batch([data])
        stage = OutcomeAdjudicationStage()
        result = stage._llm_adjudicate(sample_candidate, context=None)
        assert result.approval_date == date(2021, 7, 1)
        assert result.commercialization_date == date(2022, 1, 15)

    @patch("pipeline.stages.adjudication.process_prompt_request")
    def test_llm_adjudicate_ongoing_outcome_has_null_approval_date(self, mock_llm, sample_candidate):
        mock_llm.return_value = _make_llm_message_batch([_GOOD_ADJUDICATION])
        stage = OutcomeAdjudicationStage()
        result = stage._llm_adjudicate(sample_candidate, context=None)
        assert result.approval_date is None

    @patch("pipeline.stages.adjudication.process_prompt_request")
    def test_llm_adjudicate_invalid_date_string_falls_back_to_none(self, mock_llm, sample_candidate):
        data = {**_APPROVED_ADJUDICATION, "approval_date": "not-a-date"}
        mock_llm.return_value = _make_llm_message_batch([data])
        stage = OutcomeAdjudicationStage()
        result = stage._llm_adjudicate(sample_candidate, context=None)
        assert result.approval_date is None

    @patch("pipeline.stages.adjudication.process_prompt_request")
    def test_llm_adjudicate_missing_date_keys_default_to_none(self, mock_llm, sample_candidate):
        data = {
            "outcome": "APPROVED",
            "confidence": "HIGH",
            "reasoning": "Approved.",
            "evidence_sources": [],
        }
        mock_llm.return_value = _make_llm_message_batch([data])
        stage = OutcomeAdjudicationStage()
        result = stage._llm_adjudicate(sample_candidate, context=None)
        assert result.approval_date is None
        assert result.commercialization_date is None
