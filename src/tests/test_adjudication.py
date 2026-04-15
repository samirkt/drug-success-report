"""
Tests for Stage 3b: Outcome Adjudication (pipeline/stages/adjudication.py)

Test categories:
  PASS NOW   — constructor, orchestration wiring, run() with mocked LLM,
               cache integration, date extraction, deterministic failure
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
# run() orchestration with mocked LLM
# ---------------------------------------------------------------------------

class TestOutcomeAdjudicationStageRunOrchestration:
    @patch("utils.tiered_router.process_prompt_request")
    def test_run_returns_outcome_table_type(self, mock_llm, sample_candidate_table):
        n = len(sample_candidate_table.candidates)
        mock_llm.return_value = _make_llm_message_batch([_GOOD_ADJUDICATION] * n)
        stage = OutcomeAdjudicationStage()

        result = stage.run(sample_candidate_table)

        assert isinstance(result, OutcomeTable)

    @patch("utils.tiered_router.process_prompt_request")
    def test_run_adjudicates_each_candidate(self, mock_llm, sample_candidate_table):
        n = len(sample_candidate_table.candidates)
        mock_llm.return_value = _make_llm_message_batch([_GOOD_ADJUDICATION] * n)
        stage = OutcomeAdjudicationStage()

        result = stage.run(sample_candidate_table)

        assert len(result.outcomes) == n

    @patch("utils.tiered_router.process_prompt_request")
    def test_run_keys_outcomes_by_candidate_id(self, mock_llm, sample_candidate_table):
        n = len(sample_candidate_table.candidates)
        mock_llm.return_value = _make_llm_message_batch([_GOOD_ADJUDICATION] * n)
        stage = OutcomeAdjudicationStage()

        result = stage.run(sample_candidate_table)

        for candidate in sample_candidate_table.candidates:
            assert candidate.candidate_id in result.outcomes

    def test_run_empty_candidate_table_returns_empty_outcome_table(self, empty_candidate_table):
        stage = OutcomeAdjudicationStage()

        result = stage.run(empty_candidate_table)

        assert isinstance(result, OutcomeTable)
        assert result.outcomes == {}

    @patch("utils.tiered_router.process_prompt_request")
    def test_run_calls_llm(self, mock_llm, single_candidate_table):
        mock_llm.return_value = _make_llm_message_batch([_GOOD_ADJUDICATION])
        stage = OutcomeAdjudicationStage()

        stage.run(single_candidate_table)

        mock_llm.assert_called()


# ---------------------------------------------------------------------------
# run() end-to-end with mocked LLM
# ---------------------------------------------------------------------------

class TestOutcomeAdjudicationStageRunWithLLM:
    @patch("utils.tiered_router.process_prompt_request")
    def test_run_returns_outcome_table_with_mocked_llm(self, mock_llm, single_candidate_table):
        mock_llm.return_value = _make_llm_message_batch([_GOOD_ADJUDICATION])
        stage = OutcomeAdjudicationStage()

        result = stage.run(single_candidate_table)

        assert isinstance(result, OutcomeTable)
        assert len(result.outcomes) == 1

    @patch("utils.tiered_router.process_prompt_request")
    def test_run_does_not_raise_for_empty_table(self, mock_llm, empty_candidate_table):
        stage = OutcomeAdjudicationStage()
        result = stage.run(empty_candidate_table)
        assert isinstance(result, OutcomeTable)
        mock_llm.assert_not_called()


# ---------------------------------------------------------------------------
# Single-candidate adjudication via run()
# ---------------------------------------------------------------------------

class TestAdjudicate:
    @patch("utils.tiered_router.process_prompt_request")
    def test_adjudicate_returns_candidate_outcome_record(self, mock_llm, sample_candidate):
        mock_llm.return_value = _make_llm_message_batch([_GOOD_ADJUDICATION])
        stage = OutcomeAdjudicationStage()
        result = stage.run(CandidateTable(candidates=[sample_candidate]))
        outcome = result.outcomes[sample_candidate.candidate_id]
        assert isinstance(outcome, CandidateOutcomeRecord)

    @patch("utils.tiered_router.process_prompt_request")
    def test_adjudicate_sets_candidate_id(self, mock_llm, sample_candidate):
        mock_llm.return_value = _make_llm_message_batch([_GOOD_ADJUDICATION])
        stage = OutcomeAdjudicationStage()
        result = stage.run(CandidateTable(candidates=[sample_candidate]))
        outcome = result.outcomes[sample_candidate.candidate_id]
        assert outcome.candidate_id == sample_candidate.candidate_id

    @patch("utils.tiered_router.process_prompt_request")
    def test_adjudicate_sets_valid_outcome(self, mock_llm, sample_candidate):
        mock_llm.return_value = _make_llm_message_batch([_GOOD_ADJUDICATION])
        stage = OutcomeAdjudicationStage()
        result = stage.run(CandidateTable(candidates=[sample_candidate]))
        outcome = result.outcomes[sample_candidate.candidate_id]
        assert isinstance(outcome.outcome, CandidateOutcome)

    @patch("utils.tiered_router.process_prompt_request")
    def test_adjudicate_confidence_in_range(self, mock_llm, sample_candidate):
        mock_llm.return_value = _make_llm_message_batch([_GOOD_ADJUDICATION])
        stage = OutcomeAdjudicationStage()
        result = stage.run(CandidateTable(candidates=[sample_candidate]))
        outcome = result.outcomes[sample_candidate.candidate_id]
        assert 0.0 <= outcome.confidence <= 1.0

    @patch("utils.tiered_router.process_prompt_request")
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
        result = stage.run(CandidateTable(candidates=[candidate]))
        outcome = result.outcomes["c_ongoing"]
        assert outcome.outcome == CandidateOutcome.ONGOING

    @patch("utils.tiered_router.process_prompt_request")
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
        result = stage.run(CandidateTable(candidates=[candidate]))
        outcome = result.outcomes["c_fail"]
        assert outcome.outcome == CandidateOutcome.FAILED_PHASE_1


# ---------------------------------------------------------------------------
# LLM response handling via run()
# ---------------------------------------------------------------------------

class TestLlmAdjudicate:
    @patch("utils.tiered_router.process_prompt_request")
    def test_llm_adjudicate_returns_candidate_outcome_record(self, mock_llm, sample_candidate):
        mock_llm.return_value = _make_llm_message_batch([_GOOD_ADJUDICATION])
        stage = OutcomeAdjudicationStage()
        result = stage.run(CandidateTable(candidates=[sample_candidate]))
        outcome = result.outcomes[sample_candidate.candidate_id]
        assert isinstance(outcome, CandidateOutcomeRecord)

    @patch("utils.tiered_router.process_prompt_request")
    def test_llm_adjudicate_outcome_is_valid_enum(self, mock_llm, sample_candidate):
        mock_llm.return_value = _make_llm_message_batch([_GOOD_ADJUDICATION])
        stage = OutcomeAdjudicationStage()
        result = stage.run(CandidateTable(candidates=[sample_candidate]))
        outcome = result.outcomes[sample_candidate.candidate_id]
        assert isinstance(outcome.outcome, CandidateOutcome)

    @patch("utils.tiered_router.process_prompt_request")
    def test_llm_adjudicate_malformed_response_falls_back_to_unknown(self, mock_llm, sample_candidate):
        """Malformed JSON should fall back to UNKNOWN/0.0 rather than crashing."""
        block = MagicMock()
        block.type = "text"
        block.text = "not valid json {{{"
        msg = MagicMock()
        msg.content = [block]
        mock_llm.return_value = msg

        stage = OutcomeAdjudicationStage()
        result = stage.run(CandidateTable(candidates=[sample_candidate]))

        outcome = result.outcomes[sample_candidate.candidate_id]
        assert isinstance(outcome, CandidateOutcomeRecord)
        assert outcome.outcome == CandidateOutcome.UNKNOWN
        assert outcome.confidence == 0.0

    @patch("utils.tiered_router.process_prompt_request")
    def test_llm_adjudicate_unknown_outcome_string_falls_back(self, mock_llm, sample_candidate):
        """Unrecognized outcome string should fall back to UNKNOWN."""
        mock_llm.return_value = _make_llm_message_batch([{
            **_GOOD_ADJUDICATION,
            "outcome": "NOT_A_REAL_OUTCOME",
        }])
        stage = OutcomeAdjudicationStage()
        result = stage.run(CandidateTable(candidates=[sample_candidate]))
        outcome = result.outcomes[sample_candidate.candidate_id]
        assert outcome.outcome == CandidateOutcome.UNKNOWN


# ---------------------------------------------------------------------------
# Cache integration tests
# ---------------------------------------------------------------------------

class TestOutcomeAdjudicationStageWithCache:
    @patch("utils.tiered_router.process_prompt_request")
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

    @patch("utils.tiered_router.process_prompt_request")
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

    @patch("utils.tiered_router.process_prompt_request")
    def test_no_cache_injected_calls_llm_normally(self, mock_llm, single_candidate_table):
        mock_llm.return_value = _make_llm_message_batch([_GOOD_ADJUDICATION])
        stage = OutcomeAdjudicationStage()
        stage.run(single_candidate_table)
        mock_llm.assert_called_once()

    @patch("utils.tiered_router.process_prompt_request")
    def test_llm_failure_writes_defaults_to_cache(self, mock_llm, single_candidate_table, sample_candidate, knowledge_cache):
        from pipeline.knowledge_cache import KnowledgeCache
        mock_llm.side_effect = RuntimeError("LLM unavailable")

        stage = OutcomeAdjudicationStage(cache=knowledge_cache)
        stage.run(single_candidate_table)

        key = KnowledgeCache.make_adjudication_key(
            sample_candidate.drug_name, sample_candidate.indication, sample_candidate.highest_phase.value
        )
        cached = knowledge_cache.get_outcome(key, sample_candidate.candidate_id)
        assert cached is not None
        assert cached.outcome == CandidateOutcome.UNKNOWN
        assert cached.confidence == 0.0


# ---------------------------------------------------------------------------
# TestAdjudicationUsesRawDrugName
# ---------------------------------------------------------------------------

class TestAdjudicationUsesRawDrugName:
    @patch("utils.tiered_router.process_prompt_request")
    def test_llm_prompt_uses_drug_name_raw(self, mock_llm, sample_candidate):
        """The prompt sent to the LLM must contain drug_name_raw, not drug_name."""
        mock_llm.return_value = _make_llm_message_batch([_GOOD_ADJUDICATION])
        stage = OutcomeAdjudicationStage()
        stage.run(CandidateTable(candidates=[sample_candidate]))

        all_call_args = str(mock_llm.call_args_list)
        assert sample_candidate.drug_name_raw in all_call_args

    @patch("utils.tiered_router.process_prompt_request")
    def test_cache_key_uses_drug_name(self, mock_llm, sample_candidate, knowledge_cache):
        """Cache key must be built from drug_name (normalized), not drug_name_raw."""
        mock_llm.return_value = _make_llm_message_batch([_GOOD_ADJUDICATION])
        stage = OutcomeAdjudicationStage(cache=knowledge_cache)

        with patch("pipeline.knowledge_cache.KnowledgeCache.make_adjudication_key", wraps=knowledge_cache.make_adjudication_key) as mock_key:
            stage.run(CandidateTable(candidates=[sample_candidate]))

        mock_key.assert_any_call(
            sample_candidate.drug_name,
            sample_candidate.indication,
            sample_candidate.highest_phase.value,
        )


# ---------------------------------------------------------------------------
# Date field extraction via run()
# ---------------------------------------------------------------------------

class TestLlmAdjudicateDateExtraction:
    @patch("utils.tiered_router.process_prompt_request")
    def test_llm_adjudicate_approved_outcome_has_approval_date(self, mock_llm, sample_candidate):
        mock_llm.return_value = _make_llm_message_batch([_APPROVED_ADJUDICATION])
        stage = OutcomeAdjudicationStage()
        result = stage.run(CandidateTable(candidates=[sample_candidate]))
        outcome = result.outcomes[sample_candidate.candidate_id]
        assert outcome.approval_date == date(2022, 4, 15)

    @patch("utils.tiered_router.process_prompt_request")
    def test_llm_adjudicate_commercialized_outcome_has_commercialization_date(self, mock_llm, sample_candidate):
        data = {
            **_APPROVED_ADJUDICATION,
            "outcome": "COMMERCIALIZED",
            "approval_date": "2021-07-01",
            "commercialization_date": "2022-01-15",
        }
        mock_llm.return_value = _make_llm_message_batch([data])
        stage = OutcomeAdjudicationStage()
        result = stage.run(CandidateTable(candidates=[sample_candidate]))
        outcome = result.outcomes[sample_candidate.candidate_id]
        assert outcome.approval_date == date(2021, 7, 1)
        assert outcome.commercialization_date == date(2022, 1, 15)

    @patch("utils.tiered_router.process_prompt_request")
    def test_llm_adjudicate_ongoing_outcome_has_null_approval_date(self, mock_llm, sample_candidate):
        mock_llm.return_value = _make_llm_message_batch([_GOOD_ADJUDICATION])
        stage = OutcomeAdjudicationStage()
        result = stage.run(CandidateTable(candidates=[sample_candidate]))
        outcome = result.outcomes[sample_candidate.candidate_id]
        assert outcome.approval_date is None

    @patch("utils.tiered_router.process_prompt_request")
    def test_llm_adjudicate_invalid_date_string_falls_back_to_none(self, mock_llm, sample_candidate):
        data = {**_APPROVED_ADJUDICATION, "approval_date": "not-a-date"}
        mock_llm.return_value = _make_llm_message_batch([data])
        stage = OutcomeAdjudicationStage()
        result = stage.run(CandidateTable(candidates=[sample_candidate]))
        outcome = result.outcomes[sample_candidate.candidate_id]
        assert outcome.approval_date is None

    @patch("utils.tiered_router.process_prompt_request")
    def test_llm_adjudicate_missing_date_keys_default_to_none(self, mock_llm, sample_candidate):
        data = {
            "outcome": "APPROVED",
            "confidence": "HIGH",
            "reasoning": "Approved.",
            "evidence_sources": [],
        }
        mock_llm.return_value = _make_llm_message_batch([data])
        stage = OutcomeAdjudicationStage()
        result = stage.run(CandidateTable(candidates=[sample_candidate]))
        outcome = result.outcomes[sample_candidate.candidate_id]
        assert outcome.approval_date is None
        assert outcome.commercialization_date is None


# ---------------------------------------------------------------------------
# Regulatory approval shortcut (Fix #1 + Fix #5)
# ---------------------------------------------------------------------------


class _FakeRegulatoryIndex:
    """Minimal stand-in for pipeline.regulatory.RegulatoryIndex."""
    def __init__(self, hits):
        from pipeline.regulatory import ApprovalRecord
        self._by_db_id = {}
        self._by_name = {}
        for h in hits:
            rec = ApprovalRecord(**h)
            self._by_db_id[rec.drugbank_id] = rec
            self._by_name[rec.drug_name_norm] = rec

    def lookup(self, drug_name=None, drugbank_id=None):
        if drugbank_id and drugbank_id in self._by_db_id:
            return self._by_db_id[drugbank_id]
        if drug_name:
            from pipeline.drugbank_norm import canonicalize_drug_name
            n = canonicalize_drug_name(drug_name)
            return self._by_name.get(n)
        return None


class TestRegulatoryApprovalShortcut:
    def _candidate(self, cid="c1", drug="CompoundX", drugbank_id="DB99", phase=TrialPhase.PHASE_3):
        from pipeline.models import RawTrial
        trials = [
            RawTrial(
                nct_id="NCT99", title="", intervention=drug, indication="Type 2 Diabetes",
                sponsor="Acme", phase=phase, status=TrialStatus.TERMINATED,
                start_date=date(2015, 1, 1), completion_date=date(2018, 1, 1),
            ),
        ]
        c = Candidate(
            candidate_id=cid, drug_name=drug, drug_name_raw=drug,
            indication="Type 2 Diabetes", trial_ids=["NCT99"],
            highest_phase=phase, sponsors=["Acme"],
            earliest_start_date=date(2015, 1, 1),
            latest_completion_date=date(2018, 1, 1),
            drugbank_id=drugbank_id,
        )
        c._raw_trials = trials  # type: ignore[attr-defined]
        return c

    def _reg_index(self, appl="NDA12345", db_id="DB99", name="compoundx",
                   approval_date=date(2020, 6, 1), first=date(2020, 6, 1)):
        return _FakeRegulatoryIndex([{
            "drugbank_id": db_id,
            "drug_name_norm": name,
            "appl_no": appl,
            "approval_date": approval_date,
            "first_marketed_date": first,
            "evidence_source": f"drugs_at_fda:{appl}" if appl else "drugs_at_fda:drugbank_products",
        }])

    def test_regulatory_hit_overrides_all_terminated_deterministic_failure(self):
        c = self._candidate(phase=TrialPhase.PHASE_1)
        stage = OutcomeAdjudicationStage(regulatory_index=self._reg_index())
        result = stage.run(CandidateTable(candidates=[c]))
        out = result.outcomes["c1"]
        assert out.outcome == CandidateOutcome.APPROVED
        assert out.approval_date == date(2020, 6, 1)
        assert out.evidence_sources == ["drugs_at_fda:NDA12345"]

    def test_use_regulatory_data_false_disables_shortcut(self):
        c = self._candidate(phase=TrialPhase.PHASE_3)
        stage = OutcomeAdjudicationStage(
            use_regulatory_data=False, regulatory_index=self._reg_index(),
        )
        # With a Phase 3 all-terminated candidate and regulatory disabled,
        # the deterministic-failure path should fire.
        result = stage.run(CandidateTable(candidates=[c]))
        assert result.outcomes["c1"].outcome == CandidateOutcome.FAILED_PHASE_3

    def test_no_regulatory_hit_falls_through_to_deterministic_failure(self):
        c = self._candidate(drugbank_id="DB_UNLISTED", drug="UnlistedDrug", phase=TrialPhase.PHASE_2)
        stage = OutcomeAdjudicationStage(regulatory_index=self._reg_index())
        result = stage.run(CandidateTable(candidates=[c]))
        assert result.outcomes["c1"].outcome == CandidateOutcome.FAILED_PHASE_2

    def test_approval_predating_earliest_trial_by_many_years_is_ignored(self):
        """An approval date far earlier than the candidate's earliest trial
        strongly suggests the approval is for a different indication."""
        c = self._candidate(phase=TrialPhase.PHASE_2)
        # Earliest trial in 2015; "approval" in 1990 → 25 years before trial → skip
        reg = self._reg_index(
            approval_date=date(1990, 1, 1), first=date(1990, 1, 1),
        )
        stage = OutcomeAdjudicationStage(regulatory_index=reg)
        result = stage.run(CandidateTable(candidates=[c]))
        assert result.outcomes["c1"].outcome == CandidateOutcome.FAILED_PHASE_2

    def test_lookup_by_drug_name_when_drugbank_id_missing(self):
        c = self._candidate(drugbank_id=None, drug="CompoundX", phase=TrialPhase.PHASE_1)
        reg = self._reg_index(db_id="DB99", name="compoundx")
        stage = OutcomeAdjudicationStage(regulatory_index=reg)
        result = stage.run(CandidateTable(candidates=[c]))
        assert result.outcomes["c1"].outcome == CandidateOutcome.APPROVED
