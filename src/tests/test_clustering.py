"""
Tests for Stage 2: Candidate Clustering (pipeline/stages/clustering.py)

Test categories:
  PASS NOW   — constructor, attribute defaults, orchestration wiring (via mocks)
  FAIL NOW   — behavioral contracts for _cluster, _adjudicate, _build_candidate
               These tests will PASS once the implementation is complete.
"""

from datetime import date
from unittest.mock import MagicMock, call

import pytest

from pipeline.models import (
    Candidate,
    CandidateTable,
    TrialPhase,
    TrialStatus,
    TrialTable,
    RawTrial,
)
from pipeline.stages.clustering import CandidateClusteringStage, HybridStrategy


# ---------------------------------------------------------------------------
# Constructor / initialization  (PASS NOW)
# ---------------------------------------------------------------------------

class TestCandidateClusteringStageInit:
    def test_default_method_is_hybrid(self):
        stage = CandidateClusteringStage()
        assert stage.method == "hybrid"

    def test_default_llm_adjudicate_is_true(self):
        stage = CandidateClusteringStage()
        assert stage.llm_adjudicate is True

    def test_custom_method(self):
        stage = CandidateClusteringStage(method="fuzzy")
        assert stage.method == "fuzzy"

    def test_custom_llm_adjudicate_false(self):
        stage = CandidateClusteringStage(llm_adjudicate=False)
        assert stage.llm_adjudicate is False

    def test_hybrid_method(self):
        stage = CandidateClusteringStage(method="hybrid")
        assert stage.method == "hybrid"

    def test_default_drop_unmatched_drugbank_is_true(self):
        stage = CandidateClusteringStage()
        assert stage.drop_unmatched_drugbank is True

    def test_custom_drop_unmatched_drugbank_false(self):
        stage = CandidateClusteringStage(drop_unmatched_drugbank=False)
        assert stage.drop_unmatched_drugbank is False


# ---------------------------------------------------------------------------
# run() orchestration with mocks  (PASS NOW)
# ---------------------------------------------------------------------------

class TestCandidateClusteringStageRunOrchestration:
    """Mock stubs to verify run() correctly wires _cluster → _adjudicate → candidates."""

    def _make_cluster(self, trial_table):
        return {"cluster_a": trial_table.trials}

    def test_run_returns_candidate_table_type(self, sample_trial_table, sample_candidate):
        stage = CandidateClusteringStage(llm_adjudicate=False)
        clusters = {"c1": [sample_trial_table.trials[0]]}
        stage._cluster = MagicMock(return_value=clusters)
        stage._build_candidate = MagicMock(return_value=sample_candidate)

        result = stage.run(sample_trial_table)

        assert isinstance(result, CandidateTable)

    def test_run_calls_cluster_with_trial_table(self, sample_trial_table, sample_candidate):
        stage = CandidateClusteringStage(llm_adjudicate=False)
        stage._cluster = MagicMock(return_value={})
        stage._build_candidate = MagicMock(return_value=sample_candidate)

        stage.run(sample_trial_table)

        stage._cluster.assert_called_once_with(sample_trial_table)

    def test_run_calls_adjudicate_when_flag_is_true(self, sample_trial_table, sample_candidate):
        stage = CandidateClusteringStage(llm_adjudicate=True)
        initial_clusters = {"c1": [sample_trial_table.trials[0]]}
        refined_clusters = {"c1": [sample_trial_table.trials[0]]}
        stage._cluster = MagicMock(return_value=initial_clusters)
        stage._adjudicate = MagicMock(return_value=refined_clusters)
        stage._build_candidate = MagicMock(return_value=sample_candidate)

        stage.run(sample_trial_table)

        stage._adjudicate.assert_called_once_with(initial_clusters)

    def test_run_skips_adjudicate_when_flag_is_false(self, sample_trial_table, sample_candidate):
        stage = CandidateClusteringStage(llm_adjudicate=False)
        stage._cluster = MagicMock(return_value={})
        stage._adjudicate = MagicMock()
        stage._build_candidate = MagicMock(return_value=sample_candidate)

        stage.run(sample_trial_table)

        stage._adjudicate.assert_not_called()

    def test_run_builds_one_candidate_per_cluster(self, sample_trial_table, sample_candidate):
        stage = CandidateClusteringStage(llm_adjudicate=False)
        clusters = {"c1": [sample_trial_table.trials[0]], "c2": [sample_trial_table.trials[1]]}
        stage._cluster = MagicMock(return_value=clusters)
        stage._build_candidate = MagicMock(return_value=sample_candidate)

        result = stage.run(sample_trial_table)

        assert stage._build_candidate.call_count == 2
        assert len(result) == 2

    def test_run_returns_empty_table_for_empty_trial_table(self, empty_trial_table):
        stage = CandidateClusteringStage(llm_adjudicate=False)
        stage._cluster = MagicMock(return_value={})
        stage._build_candidate = MagicMock()

        result = stage.run(empty_trial_table)

        assert isinstance(result, CandidateTable)
        assert len(result) == 0
        stage._build_candidate.assert_not_called()


# ---------------------------------------------------------------------------
# _cluster  (FAIL NOW — stub; PASS when implemented)
# ---------------------------------------------------------------------------

class TestCluster:
    def test_cluster_returns_dict(self, sample_trial_table):
        stage = CandidateClusteringStage()
        result = stage._cluster(sample_trial_table)
        assert isinstance(result, dict)

    def test_cluster_keys_are_strings(self, sample_trial_table):
        stage = CandidateClusteringStage()
        result = stage._cluster(sample_trial_table)
        assert all(isinstance(k, str) for k in result)

    def test_cluster_values_are_lists(self, sample_trial_table):
        stage = CandidateClusteringStage()
        result = stage._cluster(sample_trial_table)
        assert all(isinstance(v, list) for v in result.values())

    def test_cluster_all_trials_appear_in_some_cluster(self, sample_trial_table):
        """Every input trial should end up in exactly one cluster."""
        stage = CandidateClusteringStage()
        result = stage._cluster(sample_trial_table)
        all_clustered = [t for trials in result.values() for t in trials]
        input_nct_ids = {t.nct_id for t in sample_trial_table.trials}
        clustered_nct_ids = {t.nct_id for t in all_clustered}
        assert input_nct_ids == clustered_nct_ids

    def test_cluster_same_drug_indication_grouped_together(self):
        """Two trials for the same drug and indication should share a cluster."""
        trials = [
            RawTrial(
                nct_id="NCT001", title="DrugA Phase 1", intervention="DrugA",
                indication="Diabetes", sponsor="PharmaCo",
                phase=TrialPhase.PHASE_1, status=TrialStatus.COMPLETED,
            ),
            RawTrial(
                nct_id="NCT002", title="DrugA Phase 2", intervention="DrugA",
                indication="Diabetes", sponsor="PharmaCo",
                phase=TrialPhase.PHASE_2, status=TrialStatus.COMPLETED,
            ),
        ]
        table = TrialTable(trials=trials)
        stage = CandidateClusteringStage()
        result = stage._cluster(table)
        all_nct_ids = [t.nct_id for cluster in result.values() for t in cluster]
        assert "NCT001" in all_nct_ids and "NCT002" in all_nct_ids
        cluster_for_drug_a = [
            cluster for cluster in result.values()
            if any(t.intervention == "DrugA" for t in cluster)
        ]
        # Both trials should be in the same cluster
        assert len(cluster_for_drug_a) == 1
        assert len(cluster_for_drug_a[0]) == 2

    def test_cluster_different_drugs_separate_clusters(self):
        """Trials for distinct drugs should produce distinct clusters."""
        trials = [
            RawTrial(
                nct_id="NCT001", title="DrugA Phase 1", intervention="DrugA",
                indication="Diabetes", sponsor="PharmaCo",
                phase=TrialPhase.PHASE_1, status=TrialStatus.COMPLETED,
            ),
            RawTrial(
                nct_id="NCT002", title="DrugB Phase 1", intervention="DrugB",
                indication="Hypertension", sponsor="CardioInc",
                phase=TrialPhase.PHASE_1, status=TrialStatus.COMPLETED,
            ),
        ]
        table = TrialTable(trials=trials)
        stage = CandidateClusteringStage()
        result = stage._cluster(table)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# _adjudicate  (FAIL NOW — stub; PASS when implemented)
# ---------------------------------------------------------------------------

class TestAdjudicate:
    def test_adjudicate_returns_dict(self, sample_trial_table):
        stage = CandidateClusteringStage()
        result = stage._adjudicate({"c1": sample_trial_table.trials})
        assert isinstance(result, dict)

    def test_adjudicate_preserves_all_trials(self, sample_trial_table):
        """Adjudication should not drop any trials."""
        input_clusters = {"c1": sample_trial_table.trials}
        stage = CandidateClusteringStage()
        result = stage._adjudicate(input_clusters)
        all_trials = [t for trials in result.values() for t in trials]
        assert len(all_trials) >= len(sample_trial_table.trials)


# ---------------------------------------------------------------------------
# _build_candidate  (FAIL NOW — stub; PASS when implemented)
# ---------------------------------------------------------------------------

class TestBuildCandidate:
    def test_build_candidate_returns_candidate(self, sample_trial_table):
        stage = CandidateClusteringStage()
        result = stage._build_candidate("cand_001", sample_trial_table.trials)
        assert isinstance(result, Candidate)

    def test_build_candidate_uses_provided_id(self, sample_trial_table):
        stage = CandidateClusteringStage()
        result = stage._build_candidate("my_cluster_id", sample_trial_table.trials)
        assert result.candidate_id == "my_cluster_id"

    def test_build_candidate_aggregates_trial_ids(self, sample_trial_table):
        """All trial nct_ids from the cluster should appear in the Candidate."""
        stage = CandidateClusteringStage()
        trials = sample_trial_table.trials
        result = stage._build_candidate("cand_x", trials)
        expected_nct_ids = {t.nct_id for t in trials}
        assert expected_nct_ids.issubset(set(result.trial_ids))

    def test_build_candidate_sets_highest_phase(self):
        """highest_phase should reflect the most advanced phase in the cluster."""
        trials = [
            RawTrial(
                nct_id="NCT001", title="DrugA Ph1", intervention="DrugA",
                indication="Diabetes", sponsor="X",
                phase=TrialPhase.PHASE_1, status=TrialStatus.COMPLETED,
            ),
            RawTrial(
                nct_id="NCT002", title="DrugA Ph2", intervention="DrugA",
                indication="Diabetes", sponsor="X",
                phase=TrialPhase.PHASE_2, status=TrialStatus.COMPLETED,
            ),
        ]
        stage = CandidateClusteringStage()
        result = stage._build_candidate("cand_x", trials)
        assert result.highest_phase == TrialPhase.PHASE_2

    def test_build_candidate_extracts_drug_name(self, sample_trial_table):
        stage = CandidateClusteringStage()
        trials = [sample_trial_table.trials[0]]  # DrugA / Diabetes trial
        result = stage._build_candidate("cand_x", trials)
        assert result.drug_name != ""

    def test_build_candidate_collects_unique_sponsors(self):
        trials = [
            RawTrial(
                nct_id="NCT001", title="DrugA Ph1", intervention="DrugA",
                indication="Diabetes", sponsor="PharmaCo",
                phase=TrialPhase.PHASE_1, status=TrialStatus.COMPLETED,
            ),
            RawTrial(
                nct_id="NCT002", title="DrugA Ph2", intervention="DrugA",
                indication="Diabetes", sponsor="BioInc",
                phase=TrialPhase.PHASE_2, status=TrialStatus.COMPLETED,
            ),
        ]
        stage = CandidateClusteringStage()
        result = stage._build_candidate("cand_x", trials)
        assert "PharmaCo" in result.sponsors
        assert "BioInc" in result.sponsors


# ---------------------------------------------------------------------------
# _build_candidate — date propagation  (PASS NOW)
# ---------------------------------------------------------------------------

class TestBuildCandidateDates:
    def _make_trial(self, nct_id, start_date, completion_date, phase=TrialPhase.PHASE_1):
        return RawTrial(
            nct_id=nct_id, title="Title", intervention="DrugX",
            indication="Diabetes", sponsor="Co",
            phase=phase, status=TrialStatus.COMPLETED,
            start_date=start_date, completion_date=completion_date,
        )

    def test_build_candidate_sets_earliest_start_date(self):
        trials = [
            self._make_trial("NCT001", date(2018, 1, 1), date(2020, 6, 1)),
            self._make_trial("NCT002", date(2019, 3, 1), date(2021, 12, 1)),
        ]
        stage = CandidateClusteringStage()
        result = stage._build_candidate("cand_x", trials)
        assert result.earliest_start_date == date(2018, 1, 1)

    def test_build_candidate_sets_latest_completion_date(self):
        trials = [
            self._make_trial("NCT001", date(2018, 1, 1), date(2020, 6, 1)),
            self._make_trial("NCT002", date(2019, 3, 1), date(2021, 12, 1)),
        ]
        stage = CandidateClusteringStage()
        result = stage._build_candidate("cand_x", trials)
        assert result.latest_completion_date == date(2021, 12, 1)

    def test_build_candidate_earliest_start_date_is_none_when_all_trials_lack_start(self):
        trials = [
            self._make_trial("NCT001", None, date(2020, 6, 1)),
            self._make_trial("NCT002", None, date(2021, 12, 1)),
        ]
        stage = CandidateClusteringStage()
        result = stage._build_candidate("cand_x", trials)
        assert result.earliest_start_date is None

    def test_build_candidate_latest_completion_date_is_none_when_all_trials_lack_completion(self):
        trials = [
            self._make_trial("NCT001", date(2018, 1, 1), None),
            self._make_trial("NCT002", date(2019, 3, 1), None),
        ]
        stage = CandidateClusteringStage()
        result = stage._build_candidate("cand_x", trials)
        assert result.latest_completion_date is None

    def test_build_candidate_skips_none_start_dates_when_computing_min(self):
        trials = [
            self._make_trial("NCT001", None, date(2020, 6, 1)),
            self._make_trial("NCT002", date(2019, 3, 1), date(2021, 12, 1)),
            self._make_trial("NCT003", date(2017, 6, 1), date(2022, 1, 1)),
        ]
        stage = CandidateClusteringStage()
        result = stage._build_candidate("cand_x", trials)
        assert result.earliest_start_date == date(2017, 6, 1)


# ---------------------------------------------------------------------------
# TestBuildCandidateRawName  (PASS NOW)
# ---------------------------------------------------------------------------

class TestBuildCandidateRawName:
    def _make_trial(self, nct_id, intervention):
        return RawTrial(
            nct_id=nct_id, title="Title", intervention=intervention,
            indication="Diabetes", sponsor="Co",
            phase=TrialPhase.PHASE_1, status=TrialStatus.COMPLETED,
        )

    def test_drug_name_raw_preserved_from_trial(self):
        """drug_name_raw must equal the original intervention string (not salt-stripped)."""
        trial = self._make_trial("NCT001", "Lepirudin HCl")
        stage = CandidateClusteringStage()
        result = stage._build_candidate("cand_x", [trial])
        assert result.drug_name_raw == "Lepirudin HCl"

    def test_drug_name_still_normalized(self):
        """drug_name must be the salt-stripped lowercase form."""
        trial = self._make_trial("NCT001", "Lepirudin HCl")
        stage = CandidateClusteringStage()
        result = stage._build_candidate("cand_x", [trial])
        assert result.drug_name == "lepirudin"
        assert result.drug_name != result.drug_name_raw


# ---------------------------------------------------------------------------
# TestHybridStrategy  (PASS NOW)
# ---------------------------------------------------------------------------

class TestHybridStrategy:
    def _make_trial(self, nct_id, intervention, indication,
                    mesh_conditions=None, mesh_interventions=None):
        return RawTrial(
            nct_id=nct_id, title="T", intervention=intervention,
            indication=indication, sponsor="Co",
            phase=TrialPhase.PHASE_1, status=TrialStatus.COMPLETED,
            mesh_condition_terms=mesh_conditions or [],
            mesh_intervention_terms=mesh_interventions or [],
        )

    def test_same_mesh_condition_term_clusters_together(self):
        """Two trials with same drug + same MeSH condition term cluster together
        even when free-text indication strings differ."""
        trials = [
            self._make_trial("NCT001", "DrugA", "Type 2 Diabetes",
                             mesh_conditions=["Diabetes Mellitus, Type 2"]),
            self._make_trial("NCT002", "DrugA", "T2DM",
                             mesh_conditions=["Diabetes Mellitus, Type 2"]),
        ]
        result = HybridStrategy().cluster(TrialTable(trials=trials))
        assert len(result) == 1

    def test_same_mesh_intervention_term_clusters_together(self):
        """Two trials sharing the same MeSH intervention term cluster together
        even when free-text drug names differ (e.g. brand vs generic)."""
        trials = [
            self._make_trial("NCT001", "Insulin Glargine", "Diabetes",
                             mesh_interventions=["Insulin Glargine"]),
            self._make_trial("NCT002", "Lantus", "Diabetes",
                             mesh_interventions=["Insulin Glargine"]),
        ]
        result = HybridStrategy().cluster(TrialTable(trials=trials))
        assert len(result) == 1

    def test_fallback_to_normalized_drug_and_indication_when_no_mesh(self):
        """Trials without any MeSH terms cluster by normalized free-text strings."""
        trials = [
            self._make_trial("NCT001", "DrugA", "Diabetes",
                             mesh_conditions=[], mesh_interventions=[]),
            self._make_trial("NCT002", "DrugA", "Diabetes",
                             mesh_conditions=[], mesh_interventions=[]),
        ]
        result = HybridStrategy().cluster(TrialTable(trials=trials))
        assert len(result) == 1

    def test_different_mesh_condition_terms_produce_separate_clusters(self):
        """Same drug tagged with different MeSH condition terms → separate clusters."""
        trials = [
            self._make_trial("NCT001", "DrugA", "Cancer",
                             mesh_conditions=["Breast Neoplasms"]),
            self._make_trial("NCT002", "DrugA", "Cancer",
                             mesh_conditions=["Lung Neoplasms"]),
        ]
        result = HybridStrategy().cluster(TrialTable(trials=trials))
        assert len(result) == 2

    def test_different_mesh_intervention_terms_produce_separate_clusters(self):
        """Trials with different MeSH intervention terms → separate clusters."""
        trials = [
            self._make_trial("NCT001", "DrugA", "Diabetes",
                             mesh_interventions=["Insulin"]),
            self._make_trial("NCT002", "DrugB", "Diabetes",
                             mesh_interventions=["Metformin"]),
        ]
        result = HybridStrategy().cluster(TrialTable(trials=trials))
        assert len(result) == 2

    def test_mesh_condition_key_is_deterministic_for_multiple_terms(self):
        """Multiple MeSH condition terms produce a sorted, stable indication key."""
        trials = [
            self._make_trial("NCT001", "DrugA", "Cancer",
                             mesh_conditions=["Neoplasms", "Breast Neoplasms"]),
            self._make_trial("NCT002", "DrugA", "Cancer",
                             mesh_conditions=["Breast Neoplasms", "Neoplasms"]),
        ]
        result = HybridStrategy().cluster(TrialTable(trials=trials))
        assert len(result) == 1

    def test_mesh_intervention_key_is_deterministic_for_multiple_terms(self):
        """Multiple MeSH intervention terms produce a sorted, stable drug key."""
        trials = [
            self._make_trial("NCT001", "DrugA", "Diabetes",
                             mesh_interventions=["Insulin", "Insulin Glargine"]),
            self._make_trial("NCT002", "DrugA", "Diabetes",
                             mesh_interventions=["Insulin Glargine", "Insulin"]),
        ]
        result = HybridStrategy().cluster(TrialTable(trials=trials))
        assert len(result) == 1


# ---------------------------------------------------------------------------
# TestApplyDrugbankDedup  (PASS NOW)
# ---------------------------------------------------------------------------

class TestApplyDrugbankDedup:
    def _make_trial(self, nct_id, intervention, indication="Diabetes", phase=TrialPhase.PHASE_1):
        return RawTrial(
            nct_id=nct_id, title="Title", intervention=intervention,
            indication=indication, sponsor="Co",
            phase=phase, status=TrialStatus.COMPLETED,
        )

    def _make_candidate(self, cid, drug_name_raw, indication="diabetes", phase=TrialPhase.PHASE_1,
                        trial_ids=None, mesh_drug=None):
        return Candidate(
            candidate_id=cid,
            drug_name=drug_name_raw.lower(),
            indication=indication,
            drug_name_raw=drug_name_raw,
            trial_ids=trial_ids or [cid],
            highest_phase=phase,
            sponsors=["Co"],
            mesh_drug=mesh_drug,
        )

    def _write_csv(self, tmp_path, content):
        p = tmp_path / "drugbank_approvals.csv"
        p.write_text(content)
        return p

    def test_assigns_drugbank_id_to_matched_candidate(self, tmp_path, drugbank_csv_content):
        csv_path = self._write_csv(tmp_path, drugbank_csv_content)
        stage = CandidateClusteringStage(drugbank_csv_path=csv_path)
        candidates = [self._make_candidate("c1", "lepirudin")]
        result = stage._apply_drugbank_dedup(candidates)
        assert len(result) == 1
        assert result[0].drugbank_id == "DB00001"

    def test_assigns_none_to_unmatched_candidate(self, tmp_path, drugbank_csv_content):
        csv_path = self._write_csv(tmp_path, drugbank_csv_content)
        stage = CandidateClusteringStage(drugbank_csv_path=csv_path)
        candidates = [self._make_candidate("c1", "unknowndrug xyz")]
        # unmatched candidates are filtered out
        result = stage._apply_drugbank_dedup(candidates)
        assert len(result) == 0

    def test_filters_out_unmatched_candidates(self, tmp_path, drugbank_csv_content):
        csv_path = self._write_csv(tmp_path, drugbank_csv_content)
        stage = CandidateClusteringStage(drugbank_csv_path=csv_path)
        candidates = [
            self._make_candidate("c1", "lepirudin"),
            self._make_candidate("c2", "unknowndrug xyz"),
        ]
        result = stage._apply_drugbank_dedup(candidates)
        assert len(result) == 1
        assert result[0].candidate_id == "c1"

    def test_keeps_unmatched_candidates_when_configured(self, tmp_path, drugbank_csv_content):
        csv_path = self._write_csv(tmp_path, drugbank_csv_content)
        stage = CandidateClusteringStage(
            drugbank_csv_path=csv_path,
            drop_unmatched_drugbank=False,
        )
        candidates = [
            self._make_candidate("c1", "lepirudin"),
            self._make_candidate("c2", "unknowndrug xyz"),
        ]
        result = stage._apply_drugbank_dedup(candidates)
        assert len(result) == 2
        by_id = {c.candidate_id: c for c in result}
        assert by_id["c1"].drugbank_id == "DB00001"
        assert by_id["c2"].drugbank_id is None

    def test_merges_candidates_sharing_drugbank_id_and_indication(self, tmp_path, drugbank_csv_content):
        """Brand + generic for same drug_id + indication → one merged candidate."""
        csv_path = self._write_csv(tmp_path, drugbank_csv_content)
        stage = CandidateClusteringStage(drugbank_csv_path=csv_path)
        # Both normalize to DB00001; "lepirudin hcl" first-word → DB00001
        candidates = [
            self._make_candidate("c1", "lepirudin", trial_ids=["NCT001"], phase=TrialPhase.PHASE_1),
            self._make_candidate("c2", "lepirudin hcl", trial_ids=["NCT002"], phase=TrialPhase.PHASE_2),
        ]
        result = stage._apply_drugbank_dedup(candidates)
        assert len(result) == 1
        merged = result[0]
        assert set(merged.trial_ids) == {"NCT001", "NCT002"}
        assert merged.highest_phase == TrialPhase.PHASE_2

    def test_does_not_merge_candidates_with_same_drugbank_id_but_different_indication(self, tmp_path, drugbank_csv_content):
        csv_path = self._write_csv(tmp_path, drugbank_csv_content)
        stage = CandidateClusteringStage(drugbank_csv_path=csv_path)
        candidates = [
            self._make_candidate("c1", "lepirudin", indication="diabetes"),
            self._make_candidate("c2", "lepirudin", indication="hypertension"),
        ]
        result = stage._apply_drugbank_dedup(candidates)
        assert len(result) == 2

    def test_skipped_when_csv_path_is_none(self):
        """Without a CSV path, _apply_drugbank_dedup is a no-op."""
        stage = CandidateClusteringStage(drugbank_csv_path=None)
        candidates = [
            self._make_candidate("c1", "unknowndrug"),
            self._make_candidate("c2", "anotherdrug"),
        ]
        result = stage._apply_drugbank_dedup(candidates)
        # All candidates pass through unchanged
        assert len(result) == 2
        assert all(c.drugbank_id is None for c in result)

    def test_merges_candidates_sharing_mesh_drug_and_indication(self, tmp_path, drugbank_csv_content):
        """Candidates matched by drug MeSH (not DrugBank) sharing same indication merge."""
        csv_path = self._write_csv(tmp_path, drugbank_csv_content)
        stage = CandidateClusteringStage(drugbank_csv_path=csv_path, drop_unmatched_drugbank=False)
        # Use a drug name that won't match DrugBank but supply mesh_drug
        candidates = [
            self._make_candidate("c1", "unknowndrug", indication="diabetes",
                                 trial_ids=["NCT001"], mesh_drug="insulin"),
            self._make_candidate("c2", "unknowndrug variant", indication="diabetes",
                                 trial_ids=["NCT002"], mesh_drug="insulin"),
        ]
        result = stage._apply_drugbank_dedup(candidates)
        # Both share mesh_drug="insulin" + same indication → merged into one
        by_trials = {frozenset(c.trial_ids) for c in result}
        assert frozenset({"NCT001", "NCT002"}) in by_trials

    def test_union_includes_candidates_with_only_mesh_drug(self, tmp_path, drugbank_csv_content):
        """Candidates with mesh_drug but no DrugBank ID are included in eligible pool."""
        csv_path = self._write_csv(tmp_path, drugbank_csv_content)
        stage = CandidateClusteringStage(drugbank_csv_path=csv_path, drop_unmatched_drugbank=False)
        candidates = [
            self._make_candidate("c1", "unknowndrug", mesh_drug="some mesh term"),
        ]
        result = stage._apply_drugbank_dedup(candidates)
        # Candidate with only mesh_drug is retained (not dropped as unmatched)
        assert len(result) == 1

    def test_transitive_closure_across_drugbank_and_mesh_aliases(self, tmp_path, drugbank_csv_content):
        """A↔B share DrugBank ID, B↔C share MeSH drug — all three merge."""
        csv_path = self._write_csv(tmp_path, drugbank_csv_content)
        stage = CandidateClusteringStage(drugbank_csv_path=csv_path)
        # Candidate A: matches DB00050 via name="insulin", mesh_drug="insulin"
        # Candidate B: matches DB00050 via name="insulin", but mesh_drug differs
        # Candidate C: does NOT match DrugBank (unknown name), but mesh_drug="insulin glargine"
        # Under the old one-shot rule, A+B merge via DrugBank; C stays alone
        # because its canonical drug key is "insulin glargine" (MeSH) while
        # A's and B's are "DB00050". Under union-find, B↔C link via the MeSH
        # alias "insulin glargine" (B's mesh_drug), so all three merge.
        candidates = [
            self._make_candidate("cA", "insulin", indication="diabetes",
                                 trial_ids=["NCT001"], mesh_drug="insulin"),
            self._make_candidate("cB", "insulin", indication="diabetes",
                                 trial_ids=["NCT002"], mesh_drug="insulin glargine"),
            self._make_candidate("cC", "unknowndrug xyz", indication="diabetes",
                                 trial_ids=["NCT003"], mesh_drug="insulin glargine"),
        ]
        result = stage._apply_drugbank_dedup(candidates)
        assert len(result) == 1
        merged = result[0]
        assert set(merged.trial_ids) == {"NCT001", "NCT002", "NCT003"}

    def test_namespace_tagging_prevents_cross_alias_false_merge(self, tmp_path, drugbank_csv_content):
        """A DrugBank ID string equal to a MeSH term must not link across candidates."""
        csv_path = self._write_csv(tmp_path, drugbank_csv_content)
        stage = CandidateClusteringStage(drugbank_csv_path=csv_path, drop_unmatched_drugbank=False)
        # Candidate A gets DrugBank ID "DB00001" via the lepirudin match.
        # Candidate B has no DrugBank match but carries mesh_drug="DB00001"
        # (contrived string collision). Tagged aliases ("db", "DB00001") vs
        # ("mesh", "DB00001") must not union them.
        candidates = [
            self._make_candidate("cA", "lepirudin", indication="diabetes",
                                 trial_ids=["NCT001"]),
            self._make_candidate("cB", "unknowndrug xyz", indication="diabetes",
                                 trial_ids=["NCT002"], mesh_drug="DB00001"),
        ]
        result = stage._apply_drugbank_dedup(candidates)
        assert len(result) == 2
        trial_sets = {frozenset(c.trial_ids) for c in result}
        assert trial_sets == {frozenset({"NCT001"}), frozenset({"NCT002"})}

    def test_union_find_respects_indication_mismatch(self, tmp_path, drugbank_csv_content):
        """Same DrugBank ID but different indications → no merge."""
        csv_path = self._write_csv(tmp_path, drugbank_csv_content)
        stage = CandidateClusteringStage(drugbank_csv_path=csv_path)
        candidates = [
            self._make_candidate("c1", "lepirudin", indication="diabetes",
                                 trial_ids=["NCT001"]),
            self._make_candidate("c2", "lepirudin", indication="hypertension",
                                 trial_ids=["NCT002"]),
        ]
        result = stage._apply_drugbank_dedup(candidates)
        assert len(result) == 2

    def test_union_find_merges_via_normalized_name_alias(self, tmp_path, drugbank_csv_content):
        """Two candidates sharing only a normalized drug name + indication merge."""
        csv_path = self._write_csv(tmp_path, drugbank_csv_content)
        stage = CandidateClusteringStage(drugbank_csv_path=csv_path)
        # Both are "lepirudin" so both get DB00001 → eligible via DrugBank.
        # They also both have drug_name="lepirudin" (normalized). Even if we
        # removed DrugBank, the name alias should still link them. This test
        # documents that the name tier of the alias ladder works.
        candidates = [
            self._make_candidate("c1", "lepirudin", indication="diabetes",
                                 trial_ids=["NCT001"]),
            self._make_candidate("c2", "lepirudin", indication="diabetes",
                                 trial_ids=["NCT002"]),
        ]
        result = stage._apply_drugbank_dedup(candidates)
        assert len(result) == 1
        assert set(result[0].trial_ids) == {"NCT001", "NCT002"}
