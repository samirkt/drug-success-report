"""
Tests for Stage 2: Candidate Clustering (pipeline/stages/clustering.py)

The clustering stage is a single-pass grouping by (drug_key, indication_key):
every RawTrial resolves to exactly one drug_key from a priority ladder
(DrugBank exact → synonym reverse → mesh-list leaf → canonicalized row name)
and one indication_key (mesh-list leaf → normalized indication text). There
is no union-find, no alias-set merging, and no transitive closure.
"""

from datetime import date
from unittest.mock import MagicMock

import pytest

from pipeline.models import (
    Candidate,
    CandidateTable,
    TrialPhase,
    TrialStatus,
    TrialTable,
    RawTrial,
)
from pipeline.stages.clustering import CandidateClusteringStage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _trial(
    nct_id,
    intervention,
    indication="diabetes",
    phase=TrialPhase.PHASE_1,
    status=TrialStatus.COMPLETED,
    mesh_interventions=None,
    mesh_conditions=None,
    sponsor="Co",
    start_date=None,
    completion_date=None,
    last_update_submitted_date=None,
):
    return RawTrial(
        nct_id=nct_id,
        title="T",
        intervention=intervention,
        indication=indication,
        sponsor=sponsor,
        phase=phase,
        status=status,
        start_date=start_date,
        completion_date=completion_date,
        last_update_submitted_date=last_update_submitted_date,
        mesh_intervention_terms=mesh_interventions or [],
        mesh_condition_terms=mesh_conditions or [],
    )


def _write_drugbank_csv(tmp_path, rows=None):
    p = tmp_path / "drugbank_approvals.csv"
    if rows is None:
        rows = [
            "DB00001,lepirudin,lepirudin,peptide",
            "DB00030,insulin human,insulin human,peptide",
            "DB00050,insulin,insulin,small molecule",
        ]
    p.write_text("drug_id,query_name,query_norm,modality\n" + "\n".join(rows) + "\n")
    return p


def _write_synonyms_csv(tmp_path, rows):
    p = tmp_path / "drugbank_synonyms.csv"
    p.write_text("drugbank_id,synonym_norm,kind\n" + "\n".join(rows) + "\n")
    return p


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------


class TestCandidateClusteringStageInit:
    def test_default_no_drugbank_path(self):
        stage = CandidateClusteringStage()
        assert stage.drugbank_csv_path is None
        assert stage.drugbank_synonyms_csv_path is None

    def test_custom_drugbank_path(self, tmp_path):
        csv = _write_drugbank_csv(tmp_path)
        stage = CandidateClusteringStage(drugbank_csv_path=csv)
        assert stage.drugbank_csv_path == csv


# ---------------------------------------------------------------------------
# run() orchestration with mocks
# ---------------------------------------------------------------------------


class TestCandidateClusteringStageRunOrchestration:
    def test_run_returns_candidate_table(self, sample_trial_table, sample_candidate):
        stage = CandidateClusteringStage()
        stage._cluster = MagicMock(return_value={"c1": [sample_trial_table.trials[0]]})
        stage._build_candidate = MagicMock(return_value=sample_candidate)

        result = stage.run(sample_trial_table)

        assert isinstance(result, CandidateTable)

    def test_run_calls_cluster_with_trial_table(self, sample_trial_table, sample_candidate):
        stage = CandidateClusteringStage()
        stage._cluster = MagicMock(return_value={})
        stage._build_candidate = MagicMock(return_value=sample_candidate)

        stage.run(sample_trial_table)

        stage._cluster.assert_called_once_with(sample_trial_table)

    def test_run_builds_one_candidate_per_cluster(self, sample_trial_table, sample_candidate):
        stage = CandidateClusteringStage()
        clusters = {
            "c1": [sample_trial_table.trials[0]],
            "c2": [sample_trial_table.trials[1]],
        }
        stage._cluster = MagicMock(return_value=clusters)
        stage._build_candidate = MagicMock(return_value=sample_candidate)

        result = stage.run(sample_trial_table)

        assert stage._build_candidate.call_count == 2
        assert len(result) == 2

    def test_run_returns_empty_table_for_empty_trial_table(self, empty_trial_table):
        stage = CandidateClusteringStage()
        stage._cluster = MagicMock(return_value={})
        stage._build_candidate = MagicMock()

        result = stage.run(empty_trial_table)

        assert isinstance(result, CandidateTable)
        assert len(result) == 0
        stage._build_candidate.assert_not_called()


# ---------------------------------------------------------------------------
# Single-pass clustering invariants
# ---------------------------------------------------------------------------


class TestClusterSinglePass:
    def test_every_trial_lands_in_exactly_one_cluster(self, sample_trial_table):
        stage = CandidateClusteringStage()
        clusters = stage._cluster(sample_trial_table)
        all_clustered = [t for trials in clusters.values() for t in trials]
        input_ids = {t.nct_id for t in sample_trial_table.trials}
        assert {t.nct_id for t in all_clustered} == input_ids

    def test_same_row_drug_same_indication_merges(self):
        trials = TrialTable(trials=[
            _trial("NCT001", "DrugA", indication="Diabetes"),
            _trial("NCT002", "DrugA", indication="Diabetes"),
        ])
        clusters = CandidateClusteringStage()._cluster(trials)
        assert len(clusters) == 1
        (only_cluster,) = clusters.values()
        assert {t.nct_id for t in only_cluster} == {"NCT001", "NCT002"}

    def test_different_drugs_do_not_merge(self):
        trials = TrialTable(trials=[
            _trial("NCT001", "DrugA", indication="Diabetes"),
            _trial("NCT002", "DrugB", indication="Diabetes"),
        ])
        clusters = CandidateClusteringStage()._cluster(trials)
        assert len(clusters) == 2

    def test_same_drug_different_indications_do_not_merge(self):
        trials = TrialTable(trials=[
            _trial("NCT001", "DrugA", indication="Diabetes"),
            _trial("NCT002", "DrugA", indication="Hypertension"),
        ])
        clusters = CandidateClusteringStage()._cluster(trials)
        assert len(clusters) == 2

    def test_salt_variants_merge_via_canonicalization(self):
        """'Lepirudin HCl' canonicalizes to 'lepirudin' so it clusters with plain Lepirudin."""
        trials = TrialTable(trials=[
            _trial("NCT001", "Lepirudin HCl", indication="diabetes"),
            _trial("NCT002", "Lepirudin", indication="diabetes"),
        ])
        clusters = CandidateClusteringStage()._cluster(trials)
        assert len(clusters) == 1


# ---------------------------------------------------------------------------
# DrugBank + synonym resolution (drug_key)
# ---------------------------------------------------------------------------


class TestClusterDrugBankResolution:
    def test_drugbank_match_yields_db_prefixed_candidate_id(self, tmp_path):
        csv = _write_drugbank_csv(tmp_path)
        stage = CandidateClusteringStage(drugbank_csv_path=csv)
        trials = TrialTable(trials=[_trial("NCT001", "Lepirudin")])

        result = stage.run(trials)

        assert len(result) == 1
        only = result.candidates[0]
        assert only.drugbank_id == "DB00001"
        assert only.candidate_id.startswith("db:DB00001")

    def test_codename_and_inn_merge_via_synonyms(self, tmp_path):
        db_csv = _write_drugbank_csv(
            tmp_path,
            rows=["DB00002,vopratelimab,vopratelimab,monoclonal antibody"],
        )
        syn_csv = _write_synonyms_csv(
            tmp_path,
            rows=[
                "DB00002,vopratelimab,primary_name",
                "DB00002,bms 986156,synonym",
            ],
        )
        stage = CandidateClusteringStage(
            drugbank_csv_path=db_csv,
            drugbank_synonyms_csv_path=syn_csv,
        )
        trials = TrialTable(trials=[
            _trial("NCT001", "BMS-986156", indication="oncology", phase=TrialPhase.PHASE_1),
            _trial("NCT002", "Vopratelimab", indication="oncology", phase=TrialPhase.PHASE_2),
        ])

        result = stage.run(trials)

        assert len(result) == 1
        merged = result.candidates[0]
        assert merged.drugbank_id == "DB00002"
        assert set(merged.trial_ids) == {"NCT001", "NCT002"}
        assert merged.highest_phase == TrialPhase.PHASE_2

    def test_first_word_groups_unresolved_variants_into_name_tier(self, tmp_path):
        """Plain "insulin" resolves to DrugBank; "insulin lispro" falls through
        to the first-word fallback and lands in the ``("name", "insulin")``
        tier — two separate candidates, NOT one merged DB cluster."""
        csv = _write_drugbank_csv(tmp_path)  # has 'insulin' = DB00050
        stage = CandidateClusteringStage(drugbank_csv_path=csv)
        trials = TrialTable(trials=[
            _trial("NCT001", "insulin", indication="diabetes"),
            _trial("NCT002", "insulin lispro", indication="diabetes"),
        ])

        result = stage.run(trials)

        assert len(result) == 2
        by_prefix = {
            c.candidate_id.split("__")[0]: c for c in result.candidates
        }
        assert "db:DB00050" in by_prefix
        assert by_prefix["db:DB00050"].drugbank_id == "DB00050"
        assert by_prefix["db:DB00050"].trial_ids == ["NCT001"]
        assert "name:insulin lispro" in by_prefix
        assert by_prefix["name:insulin lispro"].drugbank_id is None
        assert by_prefix["name:insulin lispro"].trial_ids == ["NCT002"]

    def test_first_word_groups_same_prefix_unresolved_together(self, tmp_path):
        """Two different insulin variants ("lispro", "aspart"), neither an
        exact DrugBank match, share the same first-word fallback key."""
        csv = _write_drugbank_csv(tmp_path)  # has 'insulin' = DB00050
        stage = CandidateClusteringStage(drugbank_csv_path=csv)
        trials = TrialTable(trials=[
            _trial("NCT001", "insulin lispro", indication="diabetes"),
            _trial("NCT002", "insulin aspart", indication="diabetes"),
        ])

        result = stage.run(trials)

        assert len(result) == 2
        assert {c.candidate_id.split("__")[0] for c in result.candidates} == {
            "name:insulin lispro",
            "name:insulin aspart",
        }

    def test_first_word_fallback_requires_known_drugbank_head(self, tmp_path):
        """Stopword-collapse guard: a first word that is NOT in DrugBank
        must not trigger the fallback. "small molecule alpha" and "small
        molecule beta" stay separate."""
        csv = _write_drugbank_csv(tmp_path)  # does not list 'small'
        stage = CandidateClusteringStage(drugbank_csv_path=csv)
        trials = TrialTable(trials=[
            _trial("NCT001", "small molecule alpha", indication="diabetes"),
            _trial("NCT002", "small molecule beta", indication="diabetes"),
        ])

        result = stage.run(trials)

        assert len(result) == 2
        for cand in result.candidates:
            assert cand.drugbank_id is None
            assert not cand.candidate_id.startswith("name:small__")

    def test_first_word_fallback_skipped_for_single_token(self, tmp_path):
        """A one-token unresolved row_norm is left alone — no point re-
        looking-up the already-failed row_norm as its own first word."""
        csv = _write_drugbank_csv(tmp_path)  # does not list 'lispro'
        stage = CandidateClusteringStage(drugbank_csv_path=csv)
        trials = TrialTable(trials=[_trial("NCT001", "lispro", indication="diabetes")])

        result = stage.run(trials)

        assert len(result) == 1
        assert result.candidates[0].candidate_id.startswith("name:lispro__")

    def test_first_word_fallback_does_not_hijack_db_cluster(self, tmp_path):
        """The variant rows must NOT be pulled into the ``("db", DB00050)``
        cluster for plain "insulin" — they live in a parallel name-tier
        cluster keyed on the shared first word."""
        csv = _write_drugbank_csv(tmp_path)  # has 'insulin' = DB00050
        stage = CandidateClusteringStage(drugbank_csv_path=csv)
        trials = TrialTable(trials=[
            _trial("NCT001", "insulin", indication="diabetes"),
            _trial("NCT002", "insulin lispro", indication="diabetes"),
            _trial("NCT003", "insulin aspart", indication="diabetes"),
        ])

        result = stage.run(trials)

        assert len(result) == 3
        by_prefix = {
            c.candidate_id.split("__")[0]: c for c in result.candidates
        }
        assert by_prefix["db:DB00050"].trial_ids == ["NCT001"]
        assert by_prefix["name:insulin lispro"].trial_ids == ["NCT002"]
        assert by_prefix["name:insulin aspart"].trial_ids == ["NCT003"]


# ---------------------------------------------------------------------------
# MeSH-leaf fallback (drug_key step 3)
# ---------------------------------------------------------------------------


class TestMeshLeafFallback:
    def test_mesh_leaf_chosen_when_no_drugbank_match(self, tmp_path):
        csv = _write_drugbank_csv(tmp_path)
        stage = CandidateClusteringStage(drugbank_csv_path=csv)
        trials = TrialTable(trials=[
            _trial("NCT001", "NovelPeptide-001", indication="diabetes",
                   mesh_interventions=["NovelPeptide-001"]),
            _trial("NCT002", "NovelPeptide-001", indication="diabetes",
                   mesh_interventions=["NovelPeptide-001"]),
        ])

        result = stage.run(trials)

        assert len(result) == 1

    def test_combo_therapy_row_picks_its_own_drug_leaf(self, tmp_path):
        """A Paclitaxel row in a study with both 'Paclitaxel' and 'Carboplatin'
        listed at study level picks Paclitaxel (the one matching the row's
        drug name), not the first-listed leaf. The carboplatin row does the
        symmetric thing — neither contaminates the other."""
        csv = _write_drugbank_csv(
            tmp_path,
            rows=[
                "DB00072,paclitaxel,paclitaxel,small molecule",
                "DB00073,carboplatin,carboplatin,small molecule",
            ],
        )
        stage = CandidateClusteringStage(drugbank_csv_path=csv)
        # Combo study NCT001 has both drugs as separate rows, each with the
        # study-level mesh-list list ["Paclitaxel", "Carboplatin"]. Row-level
        # `intervention` is the distinguishing field.
        combo_mesh = ["Paclitaxel", "Carboplatin"]
        trials = TrialTable(trials=[
            _trial("NCT001-A", "Paclitaxel", indication="ovarian cancer",
                   mesh_interventions=combo_mesh),
            _trial("NCT001-B", "Carboplatin", indication="ovarian cancer",
                   mesh_interventions=combo_mesh),
            _trial("NCT002",   "Paclitaxel", indication="ovarian cancer",
                   mesh_interventions=["Paclitaxel"]),
        ])

        result = stage.run(trials)

        # Two clusters: paclitaxel (2 trials) and carboplatin (1 trial).
        by_id = {c.drugbank_id: c for c in result.candidates}
        assert set(by_id.keys()) == {"DB00072", "DB00073"}
        assert set(by_id["DB00072"].trial_ids) == {"NCT001-A", "NCT002"}
        assert by_id["DB00073"].trial_ids == ["NCT001-B"]

    def test_ancestor_string_never_merges_unrelated_drugs(self, tmp_path):
        """Even if an ancestor-style term ('Organic Chemicals') ends up in the
        row's mesh_intervention_terms (it shouldn't post-SQL-filter; this is
        a regression guard), it must not cause two unrelated drugs to merge."""
        csv = _write_drugbank_csv(
            tmp_path,
            rows=[
                "DB00072,paclitaxel,paclitaxel,small molecule",
                "DB00100,atorvastatin,atorvastatin,small molecule",
            ],
        )
        stage = CandidateClusteringStage(drugbank_csv_path=csv)
        # Both rows carry an unrelated ancestor term as a MeSH leaf (simulating
        # a stale cache or a misconfigured SQL filter). With row-drug matching,
        # the ancestor is ignored because it does not match either row's drug.
        trials = TrialTable(trials=[
            _trial("NCT001", "Paclitaxel", indication="breast cancer",
                   mesh_interventions=["Paclitaxel", "Organic Chemicals"]),
            _trial("NCT002", "Atorvastatin", indication="hyperlipidemia",
                   mesh_interventions=["Atorvastatin", "Organic Chemicals"]),
        ])

        result = stage.run(trials)

        # Different drugs AND different indications — two clusters.
        assert len(result) == 2


# ---------------------------------------------------------------------------
# indication_key resolution
# ---------------------------------------------------------------------------


class TestIndicationKey:
    def test_mesh_leaf_preferred_over_free_text(self):
        trials = TrialTable(trials=[
            _trial("NCT001", "DrugA", indication="Type 2 Diabetes",
                   mesh_conditions=["Diabetes Mellitus, Type 2"]),
            _trial("NCT002", "DrugA", indication="T2DM",
                   mesh_conditions=["Diabetes Mellitus, Type 2"]),
        ])

        clusters = CandidateClusteringStage()._cluster(trials)

        assert len(clusters) == 1

    def test_different_mesh_condition_leaves_produce_separate_clusters(self):
        trials = TrialTable(trials=[
            _trial("NCT001", "DrugA", mesh_conditions=["Breast Neoplasms"]),
            _trial("NCT002", "DrugA", mesh_conditions=["Lung Neoplasms"]),
        ])

        clusters = CandidateClusteringStage()._cluster(trials)

        assert len(clusters) == 2

    def test_free_text_fallback_when_no_mesh_leaf(self):
        trials = TrialTable(trials=[
            _trial("NCT001", "DrugA", indication="Diabetes",
                   mesh_conditions=[]),
            _trial("NCT002", "DrugA", indication="Diabetes",
                   mesh_conditions=[]),
        ])

        clusters = CandidateClusteringStage()._cluster(trials)

        assert len(clusters) == 1


# ---------------------------------------------------------------------------
# _build_candidate
# ---------------------------------------------------------------------------


class TestBuildCandidate:
    def test_returns_candidate_instance(self, sample_trial_table):
        stage = CandidateClusteringStage()
        result = stage._build_candidate("my_id", sample_trial_table.trials)
        assert isinstance(result, Candidate)
        assert result.candidate_id == "my_id"

    def test_aggregates_trial_ids(self, sample_trial_table):
        stage = CandidateClusteringStage()
        result = stage._build_candidate("x", sample_trial_table.trials)
        assert set(result.trial_ids) == {t.nct_id for t in sample_trial_table.trials}

    def test_highest_phase_is_most_advanced_in_cluster(self):
        trials = [
            _trial("NCT001", "DrugA", phase=TrialPhase.PHASE_1),
            _trial("NCT002", "DrugA", phase=TrialPhase.PHASE_2),
        ]
        stage = CandidateClusteringStage()
        result = stage._build_candidate("x", trials)
        assert result.highest_phase == TrialPhase.PHASE_2

    def test_drug_name_raw_preserved_from_first_trial(self):
        trial = _trial("NCT001", "Lepirudin HCl")
        stage = CandidateClusteringStage()
        result = stage._build_candidate("x", [trial])
        assert result.drug_name_raw == "Lepirudin HCl"

    def test_drug_name_normalized(self):
        trial = _trial("NCT001", "Lepirudin HCl")
        stage = CandidateClusteringStage()
        result = stage._build_candidate("x", [trial])
        assert result.drug_name == "lepirudin"

    def test_collects_unique_sponsors(self):
        trials = [
            _trial("NCT001", "DrugA", sponsor="PharmaCo"),
            _trial("NCT002", "DrugA", sponsor="BioInc"),
            _trial("NCT003", "DrugA", sponsor="PharmaCo"),
        ]
        stage = CandidateClusteringStage()
        result = stage._build_candidate("x", trials)
        assert result.sponsors == ["PharmaCo", "BioInc"]

    def test_earliest_and_latest_dates(self):
        trials = [
            _trial("NCT001", "DrugA", start_date=date(2018, 1, 1),
                   completion_date=date(2020, 6, 1)),
            _trial("NCT002", "DrugA", start_date=date(2019, 3, 1),
                   completion_date=date(2021, 12, 1)),
        ]
        stage = CandidateClusteringStage()
        result = stage._build_candidate("x", trials)
        assert result.earliest_start_date == date(2018, 1, 1)
        assert result.latest_completion_date == date(2021, 12, 1)

    def test_dates_none_when_unavailable(self):
        trials = [
            _trial("NCT001", "DrugA", start_date=None, completion_date=None),
        ]
        stage = CandidateClusteringStage()
        result = stage._build_candidate("x", trials)
        assert result.earliest_start_date is None
        assert result.latest_completion_date is None
        assert result.latest_update_submitted_date is None

    def test_latest_update_submitted_date_takes_max_across_cluster(self):
        trials = [
            _trial(
                "NCT001",
                "DrugA",
                last_update_submitted_date=date(2022, 5, 1),
            ),
            _trial(
                "NCT002",
                "DrugA",
                last_update_submitted_date=date(2024, 1, 15),
            ),
            _trial(
                "NCT003",
                "DrugA",
                last_update_submitted_date=None,
            ),
        ]
        stage = CandidateClusteringStage()
        result = stage._build_candidate("x", trials)
        assert result.latest_update_submitted_date == date(2024, 1, 15)

    def test_mesh_drug_picks_most_common_leaf(self):
        trials = [
            _trial("NCT001", "DrugA", mesh_interventions=["Insulin"]),
            _trial("NCT002", "DrugA", mesh_interventions=["Insulin"]),
            _trial("NCT003", "DrugA", mesh_interventions=["Insulin Glargine"]),
        ]
        stage = CandidateClusteringStage()
        result = stage._build_candidate("x", trials)
        assert result.mesh_drug == "insulin"

    def test_drugbank_id_extracted_from_tuple_candidate_id(self):
        """When _build_candidate is called via run(), the cluster id is a
        (drug_key, indication_key) tuple; _build_candidate parses the DrugBank
        ID from that tuple so the Candidate record exposes it."""
        trials = [_trial("NCT001", "Lepirudin")]
        stage = CandidateClusteringStage()
        result = stage._build_candidate((("db", "DB00001"), "diabetes"), trials)
        assert result.drugbank_id == "DB00001"
