"""
Tests for Stage 1: Trial Ingestion (pipeline/stages/ingestion.py)

Test categories:
  PASS NOW   — constructor, attribute defaults, orchestration wiring (via mocks),
               _parse_phase, _parse_status, _normalize (all implemented)
  FAIL NOW   — _fetch_aact behavioral tests (require a live AACT DB connection),
               _fetch_api (still a NotImplementedError stub)
"""

from datetime import date
from unittest.mock import MagicMock, call, patch

import pytest

from pipeline.models import RawTrial, TrialPhase, TrialStatus, TrialTable
from pipeline.stages.ingestion import TrialIngestionStage


# ---------------------------------------------------------------------------
# Constructor / initialization  (PASS NOW)
# ---------------------------------------------------------------------------

class TestTrialIngestionStageInit:
    def test_default_source_is_aact(self):
        stage = TrialIngestionStage()
        assert stage.source == "aact"

    def test_default_filters_is_empty_dict(self):
        stage = TrialIngestionStage()
        assert stage.filters == {}

    def test_max_trials_can_be_none(self):
        stage = TrialIngestionStage(max_trials=None)
        assert stage.max_trials is None

    def test_custom_max_trials(self):
        stage = TrialIngestionStage(max_trials=74)
        assert stage.max_trials == 74

    def test_custom_source(self):
        stage = TrialIngestionStage(source="aact")
        assert stage.source == "aact"

    def test_custom_filters(self):
        filters = {"mesh_term": "Peptides"}
        stage = TrialIngestionStage(filters=filters)
        assert stage.filters == filters

    def test_none_filters_normalised_to_empty_dict(self):
        stage = TrialIngestionStage(filters=None)
        assert stage.filters == {}

    def test_aact_config_defaults_to_none(self):
        stage = TrialIngestionStage()
        assert stage.aact_config is None

# ---------------------------------------------------------------------------
# run() orchestration with mocks  (PASS NOW)
# ---------------------------------------------------------------------------

class TestTrialIngestionStageRunOrchestration:
    """Mock _fetch and _normalize to verify that run() correctly
    wires _fetch → _normalize → TrialTable assembly."""

    def test_run_returns_trial_table_type(self, sample_raw_trial):
        stage = TrialIngestionStage()
        stage._fetch = MagicMock(return_value=[{"nct_id": "NCT001"}])
        stage._normalize = MagicMock(return_value=sample_raw_trial)

        result = stage.run()

        assert isinstance(result, TrialTable)

    def test_run_calls_fetch_with_source_and_filters(self):
        stage = TrialIngestionStage(source="aact", filters={"mesh_term": "Peptides"})
        stage._fetch = MagicMock(return_value=[])
        stage._normalize = MagicMock()

        stage.run()

        stage._fetch.assert_called_once_with("aact", {"mesh_term": "Peptides"})

    def test_run_normalizes_each_raw_record(self, sample_raw_trial):
        raw_records = [{"id": 1}, {"id": 2}, {"id": 3}]
        stage = TrialIngestionStage()
        stage._fetch = MagicMock(return_value=raw_records)
        stage._normalize = MagicMock(return_value=sample_raw_trial)

        result = stage.run()

        assert stage._normalize.call_count == 3
        assert len(result) == 3

    def test_run_returns_empty_table_when_fetch_returns_no_records(self):
        stage = TrialIngestionStage()
        stage._fetch = MagicMock(return_value=[])
        stage._normalize = MagicMock()

        result = stage.run()

        assert isinstance(result, TrialTable)
        assert len(result) == 0
        stage._normalize.assert_not_called()

    def test_run_preserves_order_of_normalized_trials(self):
        trial_a = MagicMock(spec=RawTrial)
        trial_b = MagicMock(spec=RawTrial)
        stage = TrialIngestionStage()
        stage._fetch = MagicMock(return_value=[{"order": 1}, {"order": 2}])
        stage._normalize = MagicMock(side_effect=[trial_a, trial_b])

        result = stage.run()

        assert result.trials[0] is trial_a
        assert result.trials[1] is trial_b

    def test_run_applies_hardcoded_row_filter_rules(self, sample_raw_trial):
        stage = TrialIngestionStage()
        stage._fetch = MagicMock(return_value=[
            {"nct_id": "NCT1", "indication": "Healthy Volunteers"},
            {"nct_id": "NCT2", "indication": "Type 2 Diabetes"},
        ])
        stage._normalize = MagicMock(return_value=sample_raw_trial)
        stage._classify_single_arm = MagicMock(return_value={"NCT1": True, "NCT2": True})
        stage._fetch_p_values = MagicMock(return_value={})

        result = stage.run()

        assert len(result) == 1
        stage._normalize.assert_called_once_with({"nct_id": "NCT2", "indication": "Type 2 Diabetes"})


class TestRowFilterRules:
    def test_row_matches_contains_rule_case_insensitive(self):
        stage = TrialIngestionStage()
        row = {"indication": "Healthy Volunteers"}
        rule = {"field": "indication", "op": "contains", "value": "healthy"}
        assert stage._row_matches_rule(row, rule) is True

    def test_apply_row_filter_rules_drops_matching_rows(self):
        stage = TrialIngestionStage()
        rows = [
            {"nct_id": "NCT1", "indication": "Healthy Volunteers"},
            {"nct_id": "NCT2", "indication": "Type 2 Diabetes"},
        ]
        rules = [{"field": "indication", "op": "contains", "value": "healthy"}]
        kept_rows, rows_dropped = stage._apply_row_filter_rules(rows, rules)
        assert rows_dropped == 1
        assert len(kept_rows) == 1
        assert kept_rows[0]["nct_id"] == "NCT2"


# ---------------------------------------------------------------------------
# run() raises when the api source stub is used  (PASS NOW)
# ---------------------------------------------------------------------------

class TestTrialIngestionStageRunApiStub:
    def test_run_raises_not_implemented_for_api_source(self):
        """The REST API path is still a stub — run() should propagate its error."""
        stage = TrialIngestionStage(source="api")
        with pytest.raises(NotImplementedError):
            stage.run()


# ---------------------------------------------------------------------------
# _fetch dispatch  (PASS NOW)
# ---------------------------------------------------------------------------

class TestFetchDispatch:
    def test_fetch_api_raises_not_implemented(self):
        """api source is still a stub."""
        stage = TrialIngestionStage()
        with pytest.raises(NotImplementedError):
            stage._fetch("api", {})

    def test_fetch_unknown_source_raises_value_error(self):
        stage = TrialIngestionStage()
        with pytest.raises(ValueError, match="Unknown data source"):
            stage._fetch("unknown_source", {})

    def test_fetch_aact_calls_fetch_aact(self):
        """_fetch dispatches to _fetch_aact for the aact source."""
        stage = TrialIngestionStage()
        stage._fetch_aact = MagicMock(return_value=[])

        result = stage._fetch("aact", {"mesh_term": "Peptides"})

        stage._fetch_aact.assert_called_once_with({"mesh_term": "Peptides"})
        assert result == []


# ---------------------------------------------------------------------------
# _fetch_aact  (FAIL NOW — requires live AACT DB; PASS when DB is available)
# ---------------------------------------------------------------------------

class TestFetchAact:
    def test_fetch_aact_returns_list(self):
        """_fetch_aact should return a list of dicts when connected to AACT."""
        stage = TrialIngestionStage(filters={"mesh_term": "Peptides"}, max_trials=10)
        result = stage._fetch_aact({"mesh_term": "Peptides"})
        assert isinstance(result, list)

    def test_fetch_aact_returns_dicts(self):
        stage = TrialIngestionStage(max_trials=5)
        result = stage._fetch_aact({})
        assert all(isinstance(r, dict) for r in result)

    def test_fetch_aact_respects_max_trials_cap(self):
        """Result set should not exceed max_trials."""
        stage = TrialIngestionStage(max_trials=10)
        result = stage._fetch_aact({})
        assert len(result) <= 10

    def test_fetch_aact_rows_have_required_keys(self):
        """Each row should have the keys _normalize expects."""
        required = {"nct_id", "brief_title", "phase", "overall_status",
                    "intervention", "indication", "sponsor"}
        stage = TrialIngestionStage(max_trials=5)
        result = stage._fetch_aact({})
        for row in result:
            assert required.issubset(row.keys()), (
                f"Row missing keys: {required - row.keys()}"
            )

    def test_fetch_aact_mesh_term_filter_narrows_results(self):
        """A specific MeSH term filter should return fewer rows than no filter."""
        stage_unfiltered = TrialIngestionStage(max_trials=50)
        stage_filtered = TrialIngestionStage(
            filters={"mesh_term": "Peptides"}, max_trials=50
        )
        unfiltered = stage_unfiltered._fetch_aact({})
        filtered = stage_filtered._fetch_aact({"mesh_term": "Peptides"})
        assert len(filtered) <= len(unfiltered)

    def test_fetch_aact_one_row_per_intervention_condition_pair(self):
        """
        A study with multiple drugs or multiple conditions must produce
        multiple rows — one per (intervention, condition) pairing.
        """
        stage = TrialIngestionStage(max_trials=100)
        rows = stage._fetch_aact({})
        nct_counts: dict[str, int] = {}
        for row in rows:
            nct_counts[row["nct_id"]] = nct_counts.get(row["nct_id"], 0) + 1
        # At least one study should have >1 row if multi-drug/condition trials exist
        multi_row_studies = [nct for nct, count in nct_counts.items() if count > 1]
        assert len(multi_row_studies) > 0, (
            "Expected some studies to produce multiple rows for different "
            "intervention–condition pairings, but all studies had exactly one row."
        )

    def test_fetch_aact_bad_credentials_raises_runtime_error(self):
        """An invalid AACT config should raise RuntimeError, not OperationalError."""
        from aact_db import AACTConfig
        bad_config = AACTConfig(user="bad_user", password="bad_password")
        stage = TrialIngestionStage(aact_config=bad_config)
        with pytest.raises(RuntimeError, match="Could not connect"):
            stage._fetch_aact({})


# ---------------------------------------------------------------------------
# _normalize  (PASS NOW — implemented; uses AACT column names)
# ---------------------------------------------------------------------------

class TestNormalize:
    def _aact_row(self, **overrides) -> dict:
        """Return a minimal AACT-style row dict, with optional field overrides."""
        base = {
            "nct_id": "NCT00000001",
            "brief_title": "Phase 2 Study of DrugA in Diabetes",
            "intervention": "DrugA",
            "indication": "Type 2 Diabetes",
            "sponsor": "PharmaCo",
            "phase": "Phase 2",
            "overall_status": "Completed",
            "start_date": date(2020, 1, 1),
            "completion_date": date(2022, 12, 31),
        }
        base.update(overrides)
        return base

    def test_normalize_returns_raw_trial(self):
        stage = TrialIngestionStage()
        result = stage._normalize(self._aact_row())
        assert isinstance(result, RawTrial)

    def test_normalize_maps_nct_id(self):
        stage = TrialIngestionStage()
        result = stage._normalize(self._aact_row(nct_id="NCT00000042"))
        assert result.nct_id == "NCT00000042"

    def test_normalize_maps_brief_title(self):
        stage = TrialIngestionStage()
        result = stage._normalize(self._aact_row(brief_title="A Phase 3 Study"))
        assert result.title == "A Phase 3 Study"

    def test_normalize_maps_intervention(self):
        stage = TrialIngestionStage()
        result = stage._normalize(self._aact_row(intervention="semaglutide"))
        assert result.intervention == "semaglutide"

    def test_normalize_maps_indication(self):
        stage = TrialIngestionStage()
        result = stage._normalize(self._aact_row(indication="Type 2 Diabetes"))
        assert result.indication == "Type 2 Diabetes"

    def test_normalize_maps_sponsor(self):
        stage = TrialIngestionStage()
        result = stage._normalize(self._aact_row(sponsor="NovoNordisk"))
        assert result.sponsor == "NovoNordisk"

    def test_normalize_maps_phase_enum(self):
        stage = TrialIngestionStage()
        result = stage._normalize(self._aact_row(phase="Phase 2"))
        assert result.phase == TrialPhase.PHASE_2

    def test_normalize_maps_status_enum(self):
        stage = TrialIngestionStage()
        result = stage._normalize(self._aact_row(overall_status="Completed"))
        assert result.status == TrialStatus.COMPLETED

    def test_normalize_stores_full_row_in_raw_data(self):
        stage = TrialIngestionStage()
        row = self._aact_row()
        result = stage._normalize(row)
        assert isinstance(result.raw_data, dict)
        assert result.raw_data["nct_id"] == "NCT00000001"

    def test_normalize_null_intervention_becomes_empty_string(self):
        stage = TrialIngestionStage()
        result = stage._normalize(self._aact_row(intervention=None))
        assert result.intervention == ""

    def test_normalize_null_indication_becomes_empty_string(self):
        stage = TrialIngestionStage()
        result = stage._normalize(self._aact_row(indication=None))
        assert result.indication == ""

    def test_normalize_null_sponsor_becomes_empty_string(self):
        stage = TrialIngestionStage()
        result = stage._normalize(self._aact_row(sponsor=None))
        assert result.sponsor == ""

    def test_normalize_null_phase_maps_to_unknown(self):
        stage = TrialIngestionStage()
        result = stage._normalize(self._aact_row(phase=None))
        assert result.phase == TrialPhase.UNKNOWN

    def test_normalize_null_status_maps_to_unknown(self):
        stage = TrialIngestionStage()
        result = stage._normalize(self._aact_row(overall_status=None))
        assert result.status == TrialStatus.UNKNOWN

    def test_normalize_empty_dict_does_not_raise(self):
        """Normalize should handle a fully empty row gracefully."""
        stage = TrialIngestionStage()
        result = stage._normalize({})
        assert isinstance(result, RawTrial)
        assert result.nct_id == ""
        assert result.phase == TrialPhase.UNKNOWN
        assert result.status == TrialStatus.UNKNOWN

    def test_normalize_maps_mesh_condition_terms(self):
        stage = TrialIngestionStage()
        result = stage._normalize(self._aact_row(
            mesh_condition_terms=["Diabetes Mellitus, Type 2", "Insulin Resistance"]
        ))
        assert result.mesh_condition_terms == ["Diabetes Mellitus, Type 2", "Insulin Resistance"]

    def test_normalize_maps_mesh_intervention_terms(self):
        stage = TrialIngestionStage()
        result = stage._normalize(self._aact_row(
            mesh_intervention_terms=["Insulin", "Hypoglycemic Agents"]
        ))
        assert result.mesh_intervention_terms == ["Insulin", "Hypoglycemic Agents"]

    def test_normalize_mesh_condition_terms_defaults_to_empty_list_when_absent(self):
        stage = TrialIngestionStage()
        result = stage._normalize(self._aact_row())
        assert result.mesh_condition_terms == []

    def test_normalize_mesh_intervention_terms_defaults_to_empty_list_when_absent(self):
        stage = TrialIngestionStage()
        result = stage._normalize(self._aact_row())
        assert result.mesh_intervention_terms == []

    def test_normalize_mesh_condition_terms_handles_none(self):
        stage = TrialIngestionStage()
        result = stage._normalize(self._aact_row(mesh_condition_terms=None))
        assert result.mesh_condition_terms == []


# ---------------------------------------------------------------------------
# _parse_phase  (PASS NOW — implemented)
# ---------------------------------------------------------------------------

class TestParsePhase:
    def test_parse_phase_returns_trial_phase_enum(self):
        stage = TrialIngestionStage()
        result = stage._parse_phase("Phase 1")
        assert isinstance(result, TrialPhase)

    @pytest.mark.parametrize("phase_str,expected", [
        # Standard AACT values
        ("Phase 1",          TrialPhase.PHASE_1),
        ("phase 1",          TrialPhase.PHASE_1),   # case-insensitive
        ("EARLY PHASE 1",    TrialPhase.PHASE_1),
        ("Phase 2",          TrialPhase.PHASE_2),
        ("Phase 3",          TrialPhase.PHASE_3),
        ("Phase 4",          TrialPhase.PHASE_4),
        ("N/A",              TrialPhase.NOT_APPLICABLE),
        ("Not Applicable",   TrialPhase.NOT_APPLICABLE),
        # REST API compact style (no space)
        ("PHASE1",           TrialPhase.PHASE_1),
        ("PHASE2",           TrialPhase.PHASE_2),
        ("PHASE3",           TrialPhase.PHASE_3),
        # Combined phases → HIGHER phase
        ("Phase 1/Phase 2",  TrialPhase.PHASE_2),
        ("Phase 1/2",        TrialPhase.PHASE_2),
        ("Phase 2/Phase 3",  TrialPhase.PHASE_3),
        ("Phase 2/3",        TrialPhase.PHASE_3),
    ])
    def test_parse_phase_known_values(self, phase_str, expected):
        stage = TrialIngestionStage()
        assert stage._parse_phase(phase_str) == expected

    def test_parse_phase_unknown_returns_unknown(self):
        stage = TrialIngestionStage()
        assert stage._parse_phase("Some Unusual Phase String") == TrialPhase.UNKNOWN

    def test_parse_phase_empty_string_returns_unknown(self):
        stage = TrialIngestionStage()
        assert stage._parse_phase("") == TrialPhase.UNKNOWN

    def test_parse_phase_none_string_returns_unknown(self):
        stage = TrialIngestionStage()
        assert stage._parse_phase("None") == TrialPhase.UNKNOWN

    def test_combined_phase_maps_to_higher_not_lower(self):
        """Verify the policy: Phase 1/2 → Phase 2 (higher), not Phase 1 (lower)."""
        stage = TrialIngestionStage()
        assert stage._parse_phase("Phase 1/Phase 2") == TrialPhase.PHASE_2
        assert stage._parse_phase("Phase 2/Phase 3") == TrialPhase.PHASE_3


# ---------------------------------------------------------------------------
# _parse_status  (PASS NOW — implemented)
# ---------------------------------------------------------------------------

class TestParseStatus:
    def test_parse_status_returns_trial_status_enum(self):
        stage = TrialIngestionStage()
        result = stage._parse_status("Completed")
        assert isinstance(result, TrialStatus)

    @pytest.mark.parametrize("status_str,expected", [
        ("Recruiting",              TrialStatus.RECRUITING),
        ("RECRUITING",              TrialStatus.RECRUITING),
        ("Not Yet Recruiting",      TrialStatus.RECRUITING),
        ("Enrolling by invitation", TrialStatus.RECRUITING),
        ("Completed",               TrialStatus.COMPLETED),
        ("COMPLETED",               TrialStatus.COMPLETED),
        ("Terminated",              TrialStatus.TERMINATED),
        ("TERMINATED",              TrialStatus.TERMINATED),
        ("Withdrawn",               TrialStatus.WITHDRAWN),
        ("Active, not recruiting",  TrialStatus.ACTIVE_NOT_RECRUITING),
        ("ACTIVE_NOT_RECRUITING",   TrialStatus.ACTIVE_NOT_RECRUITING),
        ("Suspended",               TrialStatus.SUSPENDED),
    ])
    def test_parse_status_known_values(self, status_str, expected):
        stage = TrialIngestionStage()
        assert stage._parse_status(status_str) == expected

    def test_parse_status_unknown_returns_unknown(self):
        stage = TrialIngestionStage()
        assert stage._parse_status("Some Weird Status") == TrialStatus.UNKNOWN

    def test_parse_status_empty_string_returns_unknown(self):
        stage = TrialIngestionStage()
        assert stage._parse_status("") == TrialStatus.UNKNOWN


# ---------------------------------------------------------------------------
# _normalize — date field mapping  (PASS NOW)
# ---------------------------------------------------------------------------

class TestNormalizeDates:
    def _aact_row(self, **overrides) -> dict:
        base = {
            "nct_id": "NCT00000001",
            "brief_title": "Phase 2 Study of DrugA in Diabetes",
            "intervention": "DrugA",
            "indication": "Type 2 Diabetes",
            "sponsor": "PharmaCo",
            "phase": "Phase 2",
            "overall_status": "Completed",
            "start_date": date(2020, 1, 1),
            "completion_date": date(2022, 12, 31),
        }
        base.update(overrides)
        return base

    def test_normalize_maps_start_date(self):
        stage = TrialIngestionStage()
        result = stage._normalize(self._aact_row(start_date=date(2020, 1, 1)))
        assert result.start_date == date(2020, 1, 1)

    def test_normalize_maps_completion_date(self):
        stage = TrialIngestionStage()
        result = stage._normalize(self._aact_row(completion_date=date(2022, 12, 31)))
        assert result.completion_date == date(2022, 12, 31)

    def test_normalize_null_start_date_is_none(self):
        stage = TrialIngestionStage()
        result = stage._normalize(self._aact_row(start_date=None))
        assert result.start_date is None

    def test_normalize_null_completion_date_is_none(self):
        stage = TrialIngestionStage()
        result = stage._normalize(self._aact_row(completion_date=None))
        assert result.completion_date is None

    def test_normalize_missing_date_keys_defaults_to_none(self):
        stage = TrialIngestionStage()
        row = {
            "nct_id": "NCT001",
            "brief_title": "Title",
            "intervention": "DrugX",
            "indication": "Diabetes",
            "sponsor": "Co",
            "phase": "Phase 1",
            "overall_status": "Completed",
        }
        result = stage._normalize(row)
        assert result.start_date is None
        assert result.completion_date is None


# ---------------------------------------------------------------------------
# _fetch_aact — date keys present  (FAIL NOW — requires live AACT DB)
# ---------------------------------------------------------------------------

class TestFetchAactDateKeys:
    def test_fetch_aact_rows_have_date_keys(self):
        stage = TrialIngestionStage(max_trials=5)
        result = stage._fetch_aact({})
        for row in result:
            assert "start_date" in row, "Row missing 'start_date' key"
            assert "completion_date" in row, "Row missing 'completion_date' key"
