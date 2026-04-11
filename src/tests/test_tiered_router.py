"""
Tests for the tiered routing system and classification/adjudication stages.

All tests are fully mocked — no API credentials or network calls required.
Run with:  cd src && python -m pytest tests/test_tiered_router.py -v
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import date
from unittest.mock import MagicMock, patch

import pytest

from pipeline.models import (
    Candidate,
    CandidateOutcome,
    CandidateTable,
    RawTrial,
    TrialPhase,
    TrialStatus,
)
from utils.tiered_router import (
    CONFIDENCE_THRESHOLD,
    CostLedger,
    MODEL_OPUS,
    MODEL_SONNET,
    confidence_below_threshold,
    default_escalation_predicate,
    outcome_is_unknown,
    tiered_batch_call,
)
from pipeline.stages.adjudication import (
    OutcomeAdjudicationStage,
    try_deterministic_failure,
    _parse_adjudication,
    _default_adjudication,
)
from pipeline.stages.classification import (
    AttributeClassificationStage,
    VALID_MODALITIES,
    _classification_escalation_predicate,
    _parse_classification,
    _default_classification,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_candidate(
    cid: str = "test_drug__test_indication",
    drug: str = "test_drug",
    indication: str = "test_indication",
    phase: TrialPhase = TrialPhase.PHASE_2,
    raw_trials: list[RawTrial] | None = None,
) -> Candidate:
    c = Candidate(
        candidate_id=cid,
        drug_name=drug,
        indication=indication,
        trial_ids=["NCT001"],
        highest_phase=phase,
        sponsors=["TestPharma"],
        drug_name_raw=drug,
    )
    if raw_trials is not None:
        c._raw_trials = raw_trials
    return c


def _make_raw_trial(
    status: TrialStatus = TrialStatus.COMPLETED,
    phase: TrialPhase = TrialPhase.PHASE_2,
) -> RawTrial:
    return RawTrial(
        nct_id="NCT001",
        title="Test Trial",
        intervention="test_drug",
        indication="test_indication",
        sponsor="TestPharma",
        phase=phase,
        status=status,
    )


def _mock_message(response_json: list[dict], model: str = MODEL_SONNET):
    """Create a mock Anthropic Message with usage stats."""
    msg = MagicMock()
    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = json.dumps(response_json)
    msg.content = [text_block]
    msg.usage = MagicMock()
    msg.usage.input_tokens = 500
    msg.usage.output_tokens = 200
    msg.usage.cache_creation_input_tokens = 0
    msg.usage.cache_read_input_tokens = 0
    return msg


# ===================================================================
# 1. CostLedger tests
# ===================================================================


class TestCostLedger:
    def test_empty_ledger(self):
        ledger = CostLedger()
        assert ledger.total_input_tokens == 0
        assert ledger.total_output_tokens == 0
        assert ledger.total_cost_usd == 0.0
        assert len(ledger.entries) == 0

    def test_single_record(self):
        ledger = CostLedger()
        ledger.record(
            stage="Classification",
            model=MODEL_SONNET,
            input_tokens=1_000_000,
            output_tokens=1_000_000,
            n_candidates=50,
        )
        assert len(ledger.entries) == 1
        assert ledger.total_input_tokens == 1_000_000
        assert ledger.total_output_tokens == 1_000_000
        # Sonnet: $3/M input + $15/M output = $18
        assert abs(ledger.total_cost_usd - 18.0) < 0.001

    def test_multiple_records_accumulate(self):
        ledger = CostLedger()
        ledger.record(
            stage="Classification",
            model=MODEL_SONNET,
            input_tokens=100,
            output_tokens=50,
            n_candidates=5,
        )
        ledger.record(
            stage="Adjudication",
            model=MODEL_OPUS,
            input_tokens=200,
            output_tokens=100,
            n_candidates=3,
        )
        assert len(ledger.entries) == 2
        assert ledger.total_input_tokens == 300
        assert ledger.total_output_tokens == 150

    def test_log_does_not_raise(self):
        ledger = CostLedger()
        ledger.record(
            stage="test", model=MODEL_SONNET,
            input_tokens=10, output_tokens=5, n_candidates=1,
        )
        ledger.log()  # should not raise

    def test_cache_token_cost_calculation(self):
        """Prompt caching splits input tokens into three buckets with different rates."""
        ledger = CostLedger()
        # Sonnet: $3/M input base rate
        # 100k uncached @ $3/M = $0.30
        # 500k cache_creation @ $3/M * 1.25 = $1.875
        # 400k cache_read @ $3/M * 0.10 = $0.12
        # 200k output @ $15/M = $3.00
        # Total = $5.295
        ledger.record(
            stage="Classification",
            model=MODEL_SONNET,
            input_tokens=100_000,
            output_tokens=200_000,
            cache_creation_input_tokens=500_000,
            cache_read_input_tokens=400_000,
            n_candidates=50,
        )
        assert abs(ledger.total_cost_usd - 5.295) < 0.001
        # total_input_tokens includes all three buckets
        assert ledger.total_input_tokens == 1_000_000
        assert ledger.total_output_tokens == 200_000

    def test_cache_tokens_default_to_zero(self):
        """Existing calls without cache params still work correctly."""
        ledger = CostLedger()
        ledger.record(
            stage="test", model=MODEL_SONNET,
            input_tokens=1_000_000, output_tokens=1_000_000, n_candidates=10,
        )
        # Same as before: $3 + $15 = $18
        assert abs(ledger.total_cost_usd - 18.0) < 0.001


# ===================================================================
# 2. Escalation predicate tests
# ===================================================================


class TestEscalationPredicates:
    def test_confidence_high_no_escalation(self):
        assert not confidence_below_threshold({"confidence": "HIGH"})

    def test_confidence_medium_no_escalation(self):
        assert not confidence_below_threshold({"confidence": "MEDIUM"})

    def test_confidence_low_triggers_escalation(self):
        assert confidence_below_threshold({"confidence": "LOW"})

    def test_missing_confidence_triggers_escalation(self):
        assert confidence_below_threshold({})

    def test_modality_confidence_also_checked(self):
        assert confidence_below_threshold({"modality_confidence": "LOW"})
        assert not confidence_below_threshold({"modality_confidence": "HIGH"})

    def test_outcome_unknown_triggers(self):
        assert outcome_is_unknown({"outcome": "UNKNOWN"})

    def test_outcome_known_does_not_trigger(self):
        assert not outcome_is_unknown({"outcome": "APPROVED"})

    def test_default_predicate_combines(self):
        # LOW confidence → escalate
        assert default_escalation_predicate(
            {"confidence": "LOW", "outcome": "APPROVED"}
        )
        # UNKNOWN outcome → escalate
        assert default_escalation_predicate(
            {"confidence": "HIGH", "outcome": "UNKNOWN"}
        )
        # Both fine → no escalation
        assert not default_escalation_predicate(
            {"confidence": "HIGH", "outcome": "APPROVED"}
        )


# ===================================================================
# 3. Tiered batch call routing tests
# ===================================================================


class TestTieredBatchCall:
    """Tests for tiered_batch_call routing logic."""

    def _fields_fn(self, i, c):
        return [f"Drug: {c.drug_name}", f"Indication: {c.indication}"]

    def _parse_fn(self, data, c):
        return {"candidate_id": c.candidate_id, **data}

    def _default_fn(self, c):
        return {"candidate_id": c.candidate_id, "outcome": "UNKNOWN"}

    @patch("utils.tiered_router.process_prompt_request")
    def test_no_escalation_all_high_confidence(self, mock_request):
        """When all Sonnet results are HIGH confidence, no Opus call is made."""
        candidates = [_make_candidate(cid=f"c{i}") for i in range(3)]
        sonnet_response = [
            {"candidate_index": i, "confidence": "HIGH", "outcome": "APPROVED"}
            for i in range(1, 4)
        ]
        mock_request.return_value = _mock_message(sonnet_response)

        results = tiered_batch_call(
            system_prompt="test",
            user_template="{{CANDIDATES_BLOCK}}",
            candidates=candidates,
            fields_fn=self._fields_fn,
            parse_fn=self._parse_fn,
            default_fn=self._default_fn,
            stage_name="Test",
        )

        assert len(results) == 3
        assert mock_request.call_count == 1  # Sonnet only
        # Verify it was Sonnet
        assert mock_request.call_args_list[0].kwargs.get("model") == MODEL_SONNET

    @patch("utils.tiered_router.process_prompt_request")
    def test_partial_escalation(self, mock_request):
        """When some Sonnet results are LOW, only those are sent to Opus."""
        candidates = [_make_candidate(cid=f"c{i}") for i in range(3)]

        sonnet_response = [
            {"candidate_index": 1, "confidence": "HIGH", "outcome": "APPROVED"},
            {"candidate_index": 2, "confidence": "LOW", "outcome": "UNKNOWN"},
            {"candidate_index": 3, "confidence": "HIGH", "outcome": "COMMERCIALIZED"},
        ]
        opus_response = [
            {"candidate_index": 1, "confidence": "HIGH", "outcome": "FAILED_PHASE_2"},
        ]

        mock_request.side_effect = [
            _mock_message(sonnet_response),
            _mock_message(opus_response, MODEL_OPUS),
        ]

        results = tiered_batch_call(
            system_prompt="test",
            user_template="{{CANDIDATES_BLOCK}}",
            candidates=candidates,
            fields_fn=self._fields_fn,
            parse_fn=self._parse_fn,
            default_fn=self._default_fn,
            stage_name="Test",
        )

        assert len(results) == 3
        assert mock_request.call_count == 2  # Sonnet + Opus
        # Second call should be Opus
        assert mock_request.call_args_list[1].kwargs.get("model") == MODEL_OPUS
        # Candidate 2 should have the Opus result
        assert results[1]["outcome"] == "FAILED_PHASE_2"
        # Candidates 1 and 3 keep Sonnet results
        assert results[0]["outcome"] == "APPROVED"
        assert results[2]["outcome"] == "COMMERCIALIZED"

    @patch("utils.tiered_router.process_prompt_request")
    def test_full_escalation_on_sonnet_failure(self, mock_request):
        """When Sonnet fails entirely, all candidates go to Opus."""
        candidates = [_make_candidate(cid="c1")]
        opus_response = [
            {"candidate_index": 1, "confidence": "HIGH", "outcome": "APPROVED"},
        ]

        mock_request.side_effect = [
            Exception("Sonnet API error"),
            _mock_message(opus_response, MODEL_OPUS),
        ]

        results = tiered_batch_call(
            system_prompt="test",
            user_template="{{CANDIDATES_BLOCK}}",
            candidates=candidates,
            fields_fn=self._fields_fn,
            parse_fn=self._parse_fn,
            default_fn=self._default_fn,
            stage_name="Test",
        )

        assert len(results) == 1
        assert results[0]["outcome"] == "APPROVED"
        assert mock_request.call_count == 2

    @patch("utils.tiered_router.process_prompt_request")
    def test_ledger_records_costs(self, mock_request):
        """CostLedger should record entries for each model call."""
        candidates = [_make_candidate()]
        sonnet_response = [
            {"candidate_index": 1, "confidence": "HIGH", "outcome": "APPROVED"},
        ]
        mock_request.return_value = _mock_message(sonnet_response)

        ledger = CostLedger()
        tiered_batch_call(
            system_prompt="test",
            user_template="{{CANDIDATES_BLOCK}}",
            candidates=candidates,
            fields_fn=self._fields_fn,
            parse_fn=self._parse_fn,
            default_fn=self._default_fn,
            stage_name="Test",
            ledger=ledger,
        )

        assert len(ledger.entries) == 1
        assert ledger.entries[0]["model"] == MODEL_SONNET
        assert ledger.total_input_tokens == 500
        assert ledger.total_output_tokens == 200


# ===================================================================
# 4. Deterministic failure detection tests
# ===================================================================


class TestDeterministicFailure:
    def test_all_terminated_phase2(self):
        """All trials terminated at Phase 2 → FAILED_PHASE_2."""
        trials = [
            _make_raw_trial(TrialStatus.TERMINATED, TrialPhase.PHASE_2),
            _make_raw_trial(TrialStatus.WITHDRAWN, TrialPhase.PHASE_2),
        ]
        candidate = _make_candidate(phase=TrialPhase.PHASE_2, raw_trials=trials)
        result = try_deterministic_failure(candidate)
        assert result is not None
        assert result.outcome == CandidateOutcome.FAILED_PHASE_2
        assert result.confidence == 1.0
        assert "deterministic" in result.reasoning.lower()

    def test_mixed_statuses_no_deterministic(self):
        """Mix of Completed + Terminated → not deterministic."""
        trials = [
            _make_raw_trial(TrialStatus.COMPLETED, TrialPhase.PHASE_2),
            _make_raw_trial(TrialStatus.TERMINATED, TrialPhase.PHASE_2),
        ]
        candidate = _make_candidate(phase=TrialPhase.PHASE_2, raw_trials=trials)
        result = try_deterministic_failure(candidate)
        assert result is None

    def test_no_raw_trials_returns_none(self):
        """Candidate without _raw_trials → not deterministic."""
        candidate = _make_candidate()
        result = try_deterministic_failure(candidate)
        assert result is None

    def test_phase4_not_deterministic(self):
        """Phase 4 terminated trials should not produce deterministic failure
        (Phase 4 is post-approval monitoring, not a standard failure point)."""
        trials = [_make_raw_trial(TrialStatus.TERMINATED, TrialPhase.PHASE_4)]
        candidate = _make_candidate(phase=TrialPhase.PHASE_4, raw_trials=trials)
        result = try_deterministic_failure(candidate)
        assert result is None


# ===================================================================
# 5. Classification stage tests
# ===================================================================


class TestClassificationStage:
    def test_valid_modalities_set(self):
        """All 12 valid modality values (11 + unknown) are in the set."""
        assert len(VALID_MODALITIES) == 13
        assert "peptide" in VALID_MODALITIES
        assert "monoclonal_antibody" in VALID_MODALITIES
        assert "cell_therapy" in VALID_MODALITIES
        assert "unknown" in VALID_MODALITIES

    def test_escalation_predicate_unknown_modality(self):
        assert _classification_escalation_predicate(
            {"drug_modality": "unknown", "modality_confidence": "HIGH"}
        )

    def test_escalation_predicate_valid_high_confidence(self):
        assert not _classification_escalation_predicate(
            {"drug_modality": "peptide", "modality_confidence": "HIGH"}
        )

    def test_parse_classification_normalizes_modality(self):
        candidate = _make_candidate()
        data = {
            "drug_modality": "peptide",
            "disease_area": "metabolic",
            "modality_confidence": "HIGH",
            "disease_confidence": "MEDIUM",
            "reasoning": "GLP-1 analog",
        }
        result = _parse_classification(data, candidate)
        assert result.drug_modality == "peptide"
        assert result.disease_area == "metabolic"
        assert result.modality_confidence == 1.0
        assert result.disease_confidence == 0.5

    def test_parse_classification_rejects_invalid_modality(self):
        candidate = _make_candidate()
        data = {"drug_modality": "nanobody", "disease_area": "oncology"}
        result = _parse_classification(data, candidate)
        assert result.drug_modality == "unknown"

    def test_default_classification(self):
        candidate = _make_candidate()
        result = _default_classification(candidate)
        assert result.drug_modality == "unknown"
        assert result.disease_area == "unknown"


# ===================================================================
# 6. Adjudication stage tests
# ===================================================================


class TestAdjudicationStage:
    def test_adj_fields_includes_trial_statuses(self):
        """When _raw_trials is present, trial statuses appear in the prompt fields."""
        trials = [
            _make_raw_trial(TrialStatus.COMPLETED),
            _make_raw_trial(TrialStatus.TERMINATED),
            _make_raw_trial(TrialStatus.COMPLETED),
        ]
        candidate = _make_candidate(raw_trials=trials)
        fields = OutcomeAdjudicationStage._adj_fields(1, candidate)
        status_field = [f for f in fields if f.startswith("Trial statuses:")]
        assert len(status_field) == 1
        assert "Completed" in status_field[0]
        assert "Terminated" in status_field[0]

    def test_adj_fields_without_raw_trials(self):
        """Without _raw_trials, the Trial statuses line is absent."""
        candidate = _make_candidate()
        fields = OutcomeAdjudicationStage._adj_fields(1, candidate)
        status_field = [f for f in fields if f.startswith("Trial statuses:")]
        assert len(status_field) == 0

    def test_parse_adjudication(self):
        candidate = _make_candidate()
        data = {
            "outcome": "APPROVED",
            "confidence": "HIGH",
            "reasoning": "FDA approved 2023",
            "evidence_sources": ["fda_data"],
            "approval_date": "2023-06-15",
        }
        result = _parse_adjudication(data, candidate)
        assert result.outcome == CandidateOutcome.APPROVED
        assert result.confidence == 1.0
        assert result.approval_date == date(2023, 6, 15)

    def test_default_adjudication(self):
        candidate = _make_candidate()
        result = _default_adjudication(candidate)
        assert result.outcome == CandidateOutcome.UNKNOWN
        assert result.confidence == 0.0
