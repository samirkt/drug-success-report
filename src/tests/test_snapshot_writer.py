"""Tests for the analytical-snapshot writers (Parquet + run manifest).

Covers `write_candidate_parquet`, `write_trial_parquet`, `write_run_manifest`
in `pipeline/stages/reporting/_writer.py`, plus the manifest-payload helpers
in `pipeline/pipeline.py` (`_read_user_version`, `_file_mtime_iso`,
`_serialize_config`).

These outputs feed downstream modeling/analysis without re-running the
pipeline, so the contracts under test are: list-typed columns stay as
lists (not pipe-joined strings), dates stay as dates, both adjudicator
outcomes surface side-by-side, and the run manifest captures snapshot
versions for reproducibility.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import replace
from datetime import date

import pandas as pd
import pytest

from pipeline.knowledge_cache import KnowledgeCache
from pipeline.models import (
    AttributeTable,
    Candidate,
    CandidateOutcome,
    CandidateOutcomeRecord,
    CandidateTable,
    OutcomeTable,
    RawTrial,
    TrialPhase,
    TrialStatus,
    TrialTable,
)
from pipeline.pipeline import (
    PipelineConfig,
    _file_mtime_iso,
    _read_user_version,
    _serialize_config,
)
from pipeline.stages.reporting._writer import (
    write_candidate_parquet,
    write_run_manifest,
    write_smiles_standardization_log,
    write_trial_parquet,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _enriched_candidate(**overrides) -> Candidate:
    """Build a Candidate with every list-typed enrichment populated.

    Defaults to the same shape as the conftest `sample_candidate`, but
    with all enrichment lists non-empty so list-column assertions are
    meaningful. Any keyword overrides win.
    """
    base = dict(
        candidate_id="cand_001",
        drug_name="DrugA",
        drug_name_raw="DrugA",
        indication="Type 2 Diabetes",
        trial_ids=["NCT00000001", "NCT00000003"],
        highest_phase=TrialPhase.PHASE_2,
        sponsors=["PharmaCo", "BigPharma"],
        earliest_start_date=date(2020, 1, 1),
        latest_completion_date=date(2022, 12, 31),
        drugbank_id="DB00001",
        mesh_drug="Aspirin",
        mesh_indication="Diabetes Mellitus, Type 2",
        mesh_condition_tree_numbers=["C18.452.394.750", "C19.246.300"],
        smiles="CC(=O)O",
        drug_targets=["P12345", "Q67890"],
        target_names=["Target A", "Target B"],
        icd10_description="Type 2 diabetes mellitus",
        opentargets_moa="Inhibits X | Activates Y",
        opentargets_action_type="INHIBITOR | AGONIST",
        opentargets_targets=["GENE1", "GENE2", "GENE3"],
        opentargets_pathways=["Pathway A", "Pathway B"],
        opentargets_indication_max_phase=4,
        opentargets_tractability_modalities=["SM", "AB"],
        opentargets_tractability_labels=[
            "Clinical_Precedence_sm",
            "Predicted_Tractable_ab_High_Confidence",
        ],
        opentargets_loeuf_min=0.42,
        opentargets_genetic_score=0.71,
        opentargets_genetic_score_max_any_indication=0.85,
    )
    base.update(overrides)
    return Candidate(**base)


def _put_llm_direct(cache: KnowledgeCache, cand: Candidate, outcome: CandidateOutcome) -> None:
    key = KnowledgeCache.make_adjudication_key(
        cand.drug_name, cand.indication, cand.highest_phase.value,
    )
    cache.put_outcome(key, CandidateOutcomeRecord(
        candidate_id=cand.candidate_id,
        outcome=outcome,
        confidence=0.81,
        reasoning="LLM said so.",
        evidence_sources=["llm:direct"],
    ))


def _put_fda_timeline(cache: KnowledgeCache, cand: Candidate, outcome: CandidateOutcome) -> None:
    key = KnowledgeCache.make_adjudication_key(
        cand.drug_name, cand.indication, cand.highest_phase.value,
    )
    cache.put_fda_outcome(key, CandidateOutcomeRecord(
        candidate_id=cand.candidate_id,
        outcome=outcome,
        confidence=0.93,
        reasoning="FDA timeline says so.",
        evidence_sources=["openfda:NDA123456"],
        approval_date=date(2023, 1, 15),
        commercialization_date=None,
    ))


# ---------------------------------------------------------------------------
# write_candidate_parquet — list / date / bool typing
# ---------------------------------------------------------------------------

class TestCandidateParquetTypes:

    def test_list_columns_native(self, tmp_path):
        cand = _enriched_candidate()
        write_candidate_parquet(
            CandidateTable(candidates=[cand]),
            AttributeTable(),
            OutcomeTable(),
            str(tmp_path),
        )
        df = pd.read_parquet(tmp_path / "candidate_detail.parquet")
        assert len(df) == 1
        # Every list-typed column should round-trip as a Python list /
        # numpy array of strings — not as a pipe-joined string.
        for col in (
            "drug_targets", "target_names", "opentargets_targets",
            "opentargets_pathways", "mesh_condition_tree_numbers",
            "trial_ids", "sponsors", "outcome_evidence_sources",
            "opentargets_moa", "opentargets_action_type",
        ):
            value = df[col].iloc[0]
            assert not isinstance(value, str), f"{col} should be a list, got str"
            # Coerce numpy arrays / lists to a real list for content check.
            as_list = list(value)
            for elem in as_list:
                assert isinstance(elem, str)

        assert list(df["drug_targets"].iloc[0]) == ["P12345", "Q67890"]
        assert list(df["opentargets_targets"].iloc[0]) == ["GENE1", "GENE2", "GENE3"]
        assert list(df["sponsors"].iloc[0]) == ["PharmaCo", "BigPharma"]

    def test_dates_native(self, tmp_path):
        cand = _enriched_candidate(
            earliest_start_date=date(2018, 6, 1),
            latest_completion_date=date(2024, 3, 15),
            latest_update_submitted_date=date(2025, 9, 20),
        )
        write_candidate_parquet(
            CandidateTable(candidates=[cand]),
            AttributeTable(),
            OutcomeTable(),
            str(tmp_path),
        )
        df = pd.read_parquet(tmp_path / "candidate_detail.parquet")
        # pandas reads parquet date columns as datetime64[ns] / Timestamp;
        # the key invariant is that the value is not a string.
        start = df["earliest_start_date"].iloc[0]
        assert not isinstance(start, str)
        # Coerce to date for value check (handles both Timestamp and date).
        if hasattr(start, "date"):
            start_d = start.date()
        else:
            start_d = start
        assert start_d == date(2018, 6, 1)

        update = df["latest_update_submitted_date"].iloc[0]
        assert not isinstance(update, str)
        if hasattr(update, "date"):
            update = update.date()
        assert update == date(2025, 9, 20)

    def test_opentargets_moa_split(self, tmp_path):
        cands = [
            _enriched_candidate(candidate_id="c1", opentargets_moa="A | B"),
            _enriched_candidate(candidate_id="c2", opentargets_moa=None),
            _enriched_candidate(candidate_id="c3", opentargets_moa="Solo"),
        ]
        write_candidate_parquet(
            CandidateTable(candidates=cands),
            AttributeTable(),
            OutcomeTable(),
            str(tmp_path),
        )
        df = pd.read_parquet(tmp_path / "candidate_detail.parquet")
        df = df.set_index("candidate_id")
        assert list(df.loc["c1", "opentargets_moa"]) == ["A", "B"]
        assert list(df.loc["c2", "opentargets_moa"]) == []
        assert list(df.loc["c3", "opentargets_moa"]) == ["Solo"]

    def test_opentargets_action_type_split(self, tmp_path):
        cand = _enriched_candidate(
            opentargets_action_type="INHIBITOR | AGONIST | MODULATOR",
        )
        write_candidate_parquet(
            CandidateTable(candidates=[cand]),
            AttributeTable(),
            OutcomeTable(),
            str(tmp_path),
        )
        df = pd.read_parquet(tmp_path / "candidate_detail.parquet")
        assert list(df["opentargets_action_type"].iloc[0]) == [
            "INHIBITOR", "AGONIST", "MODULATOR",
        ]

    def test_opentargets_tractability_genetics_roundtrip(self, tmp_path):
        """Tractability list columns plus LOEUF + genetic-score scalars
        land as native types on candidate_detail.parquet."""
        cand = _enriched_candidate()
        write_candidate_parquet(
            CandidateTable(candidates=[cand]),
            AttributeTable(),
            OutcomeTable(),
            str(tmp_path),
        )
        df = pd.read_parquet(tmp_path / "candidate_detail.parquet")
        # Tractability stays list-typed (not pipe-joined strings).
        for col in (
            "opentargets_tractability_modalities",
            "opentargets_tractability_labels",
        ):
            value = df[col].iloc[0]
            assert not isinstance(value, str), f"{col} should be a list"
            for elem in list(value):
                assert isinstance(elem, str)
        assert list(df["opentargets_tractability_modalities"].iloc[0]) == [
            "SM", "AB",
        ]
        assert list(df["opentargets_tractability_labels"].iloc[0]) == [
            "Clinical_Precedence_sm",
            "Predicted_Tractable_ab_High_Confidence",
        ]
        # Scalars round-trip as floats.
        assert df["opentargets_loeuf_min"].iloc[0] == pytest.approx(0.42)
        assert df["opentargets_genetic_score"].iloc[0] == pytest.approx(0.71)
        assert (
            df["opentargets_genetic_score_max_any_indication"].iloc[0]
            == pytest.approx(0.85)
        )

    def test_opentargets_genetics_null_when_unset(self, tmp_path):
        """Candidate with default OT genetics/tractability fields lands
        as empty lists + null scalars — keeps the column shape stable
        across runs when the enrichment is off."""
        cand = _enriched_candidate(
            opentargets_tractability_modalities=[],
            opentargets_tractability_labels=[],
            opentargets_loeuf_min=None,
            opentargets_genetic_score=None,
            opentargets_genetic_score_max_any_indication=None,
        )
        write_candidate_parquet(
            CandidateTable(candidates=[cand]),
            AttributeTable(),
            OutcomeTable(),
            str(tmp_path),
        )
        df = pd.read_parquet(tmp_path / "candidate_detail.parquet")
        assert list(df["opentargets_tractability_modalities"].iloc[0]) == []
        assert list(df["opentargets_tractability_labels"].iloc[0]) == []
        # Numeric Optional[float] columns of all-None values round-trip
        # as NaN under pyarrow's Float64 inference — use pd.isna so the
        # assertion works for both None and NaN.
        assert pd.isna(df["opentargets_loeuf_min"].iloc[0])
        assert pd.isna(df["opentargets_genetic_score"].iloc[0])
        assert pd.isna(
            df["opentargets_genetic_score_max_any_indication"].iloc[0]
        )


# ---------------------------------------------------------------------------
# write_candidate_parquet — both-adjudicators surfacing
# ---------------------------------------------------------------------------

class TestCandidateParquetAdjudicators:

    def test_no_cache_keeps_schema_with_null_values(self, tmp_path):
        """Even without a cache the parquet must include both-adjudicator
        columns — keeps downstream readers stable across runs."""
        cand = _enriched_candidate()
        write_candidate_parquet(
            CandidateTable(candidates=[cand]),
            AttributeTable(),
            OutcomeTable(),
            str(tmp_path),
            cache=None,
        )
        df = pd.read_parquet(tmp_path / "candidate_detail.parquet")
        for col in (
            "llm_direct_outcome", "llm_direct_confidence", "llm_direct_reasoning",
            "llm_direct_evidence_sources",
            "fda_timeline_outcome", "fda_timeline_confidence", "fda_timeline_reasoning",
            "fda_timeline_evidence_sources",
            "outcomes_agree",
        ):
            assert col in df.columns
        assert df["llm_direct_outcome"].iloc[0] is None
        assert df["fda_timeline_outcome"].iloc[0] is None
        assert list(df["llm_direct_evidence_sources"].iloc[0]) == []
        assert df["outcomes_agree"].iloc[0] is None

    def test_includes_both_adjudicators_when_present(self, tmp_path, knowledge_cache):
        cand = _enriched_candidate()
        _put_llm_direct(knowledge_cache, cand, CandidateOutcome.APPROVED)
        _put_fda_timeline(knowledge_cache, cand, CandidateOutcome.APPROVED)

        write_candidate_parquet(
            CandidateTable(candidates=[cand]),
            AttributeTable(),
            OutcomeTable(),
            str(tmp_path),
            cache=knowledge_cache,
        )
        df = pd.read_parquet(tmp_path / "candidate_detail.parquet")
        row = df.iloc[0]
        assert row["llm_direct_outcome"] == "Approved"
        assert row["llm_direct_confidence"] == pytest.approx(0.81)
        assert row["fda_timeline_outcome"] == "Approved"
        assert row["fda_timeline_confidence"] == pytest.approx(0.93)
        assert list(row["fda_timeline_evidence_sources"]) == ["openfda:NDA123456"]
        assert row["outcomes_agree"] is True or row["outcomes_agree"] == True  # noqa: E712

    def test_outcomes_disagree(self, tmp_path, knowledge_cache):
        cand = _enriched_candidate()
        _put_llm_direct(knowledge_cache, cand, CandidateOutcome.APPROVED)
        _put_fda_timeline(knowledge_cache, cand, CandidateOutcome.FAILED_PHASE_3)

        write_candidate_parquet(
            CandidateTable(candidates=[cand]),
            AttributeTable(),
            OutcomeTable(),
            str(tmp_path),
            cache=knowledge_cache,
        )
        df = pd.read_parquet(tmp_path / "candidate_detail.parquet")
        assert df["outcomes_agree"].iloc[0] == False  # noqa: E712

    def test_active_run_overrides_cache_for_its_method(self, tmp_path, knowledge_cache):
        """When the active run produced an outcome via fda_timeline, the
        fda_timeline_* columns should reflect that fresh record — not
        whatever stale value the cache happens to hold."""
        cand = _enriched_candidate()
        # Cache says FAILED_PHASE_3 — but the active run produced APPROVED.
        _put_fda_timeline(knowledge_cache, cand, CandidateOutcome.FAILED_PHASE_3)
        outcome_table = OutcomeTable(outcomes={
            cand.candidate_id: CandidateOutcomeRecord(
                candidate_id=cand.candidate_id,
                outcome=CandidateOutcome.APPROVED,
                confidence=0.99,
                reasoning="fresh fda run",
                evidence_sources=["fresh"],
                approval_date=date(2024, 6, 1),
            ),
        })

        write_candidate_parquet(
            CandidateTable(candidates=[cand]),
            AttributeTable(),
            outcome_table,
            str(tmp_path),
            cache=knowledge_cache,
            adjudication_method="fda_timeline",
        )
        df = pd.read_parquet(tmp_path / "candidate_detail.parquet")
        assert df["fda_timeline_outcome"].iloc[0] == "Approved"
        assert df["fda_timeline_confidence"].iloc[0] == pytest.approx(0.99)


# ---------------------------------------------------------------------------
# write_trial_parquet
# ---------------------------------------------------------------------------

class TestTrialParquet:

    def test_basic_round_trip(self, tmp_path):
        trial = RawTrial(
            nct_id="NCT00000001",
            title="A study",
            intervention="DrugA",
            indication="T2D",
            sponsor="PharmaCo",
            phase=TrialPhase.PHASE_2,
            status=TrialStatus.COMPLETED,
            start_date=date(2020, 1, 1),
            completion_date=date(2022, 6, 30),
            last_update_submitted_date=date(2024, 5, 1),
            is_single_arm=True,
            mesh_condition_terms=["Diabetes Mellitus, Type 2"],
            mesh_condition_tree_numbers=["C18.452.394.750"],
            mesh_intervention_terms=["Aspirin"],
        )
        cand = _enriched_candidate()
        write_trial_parquet(
            CandidateTable(candidates=[cand]),
            AttributeTable(),
            OutcomeTable(),
            TrialTable(trials=[trial]),
            str(tmp_path),
        )
        df = pd.read_parquet(tmp_path / "trial_detail.parquet")
        # Two trial_ids on cand × one matching trial = two rows
        # (one matched, one unmatched with all-None trial cols).
        matched = df[df["nct_id"] == "NCT00000001"].iloc[0]
        assert list(matched["trial_mesh_condition_terms"]) == ["Diabetes Mellitus, Type 2"]
        assert list(matched["trial_mesh_intervention_terms"]) == ["Aspirin"]
        # Date should not be a string
        ts = matched["trial_start_date"]
        assert not isinstance(ts, str)
        if hasattr(ts, "date"):
            ts = ts.date()
        assert ts == date(2020, 1, 1)
        last_update = matched["trial_last_update_submitted_date"]
        assert not isinstance(last_update, str)
        if hasattr(last_update, "date"):
            last_update = last_update.date()
        assert last_update == date(2024, 5, 1)
        assert matched["trial_is_single_arm"] in (True, 1)

    def test_no_trial_table_skips_write(self, tmp_path):
        write_trial_parquet(
            CandidateTable(candidates=[]),
            AttributeTable(),
            OutcomeTable(),
            None,
            str(tmp_path),
        )
        assert not (tmp_path / "trial_detail.parquet").exists()


# ---------------------------------------------------------------------------
# write_run_manifest + manifest helpers
# ---------------------------------------------------------------------------

class TestRunManifest:

    def test_basic_round_trip(self, tmp_path):
        payload = {
            "generated_at": "2026-04-29T12:00:00+00:00",
            "git_sha": None,
            "git_dirty": None,
            "pipeline_config": {"data_source": "aact"},
            "snapshot_versions": {
                "chembl_user_version": None,
                "opentargets_user_version": None,
                "drugbank_csv_mtime": None,
                "drugbank_synonyms_csv_mtime": None,
            },
            "row_counts": {"n_candidates": 0, "n_trials": 0, "n_attributes": 0, "n_outcomes": 0},
        }
        write_run_manifest(str(tmp_path), payload)
        loaded = json.loads((tmp_path / "run_manifest.json").read_text())
        assert loaded == payload

    def test_read_user_version_returns_int_when_stamped(self, tmp_path):
        db = tmp_path / "stamped.sqlite"
        conn = sqlite3.connect(db)
        conn.execute("PRAGMA user_version = 35")
        conn.commit()
        conn.close()
        assert _read_user_version(db) == 35

    def test_read_user_version_returns_none_for_unstamped(self, tmp_path):
        db = tmp_path / "unstamped.sqlite"
        conn = sqlite3.connect(db)
        conn.execute("CREATE TABLE x (id INT)")
        conn.commit()
        conn.close()
        # Default user_version is 0 — treat as unstamped.
        assert _read_user_version(db) is None

    def test_read_user_version_handles_missing_path(self, tmp_path):
        assert _read_user_version(None) is None
        assert _read_user_version(tmp_path / "does_not_exist.sqlite") is None

    def test_file_mtime_iso(self, tmp_path):
        f = tmp_path / "drugbank.csv"
        f.write_text("col1,col2\n")
        ts = _file_mtime_iso(f)
        assert ts is not None
        # ISO-8601 with timezone suffix
        assert "T" in ts and ("+" in ts or "Z" in ts)

    def test_file_mtime_iso_handles_missing(self, tmp_path):
        assert _file_mtime_iso(None) is None
        assert _file_mtime_iso(tmp_path / "missing.csv") is None

    def test_serialize_config_includes_standardization_flag(self, tmp_path):
        cfg = PipelineConfig(enable_smiles_standardization=False)
        serialized = _serialize_config(cfg)
        assert serialized["enable_smiles_standardization"] is False

    def test_serialize_config_excludes_secrets_and_paths(self, tmp_path):
        cfg = PipelineConfig(
            cache_path="should_be_excluded.db",
            ct_cache_path="also_excluded.pkl",
            openfda_api_key="secret-key",
            fda_llm_api_key="other-secret",
            data_source="aact",
            adjudication_method="fda_timeline",
            chembl_snapshot_path=tmp_path / "chembl.sqlite",
        )
        serialized = _serialize_config(cfg)
        # Excluded fields must not appear
        for excluded in ("cache_path", "ct_cache_path", "openfda_api_key", "fda_llm_api_key"):
            assert excluded not in serialized
        # Reproducibility-relevant fields stay
        assert serialized["data_source"] == "aact"
        assert serialized["adjudication_method"] == "fda_timeline"
        # Path values become strings (JSON-safe)
        assert isinstance(serialized["chembl_snapshot_path"], str)
        # Round-trips through json.dumps without error
        json.dumps(serialized)


# ---------------------------------------------------------------------------
# SMILES standardization — parquet columns + sidecar log
# ---------------------------------------------------------------------------

class TestSmilesStandardization:

    def test_parquet_includes_smiles_canonical_columns(self, tmp_path):
        cand = _enriched_candidate(
            smiles="Cl.CCN",
            smiles_canonical="CCN",
            smiles_standardization_status="ok",
        )
        write_candidate_parquet(
            CandidateTable(candidates=[cand]),
            AttributeTable(),
            OutcomeTable(),
            str(tmp_path),
        )
        df = pd.read_parquet(tmp_path / "candidate_detail.parquet")
        assert "smiles_canonical" in df.columns
        assert "smiles_standardization_status" in df.columns
        # Raw and canonical both populated, side-by-side.
        assert df["smiles"].iloc[0] == "Cl.CCN"
        assert df["smiles_canonical"].iloc[0] == "CCN"
        assert df["smiles_standardization_status"].iloc[0] == "ok"

    def test_parquet_smiles_canonical_null_when_not_run(self, tmp_path):
        cand = _enriched_candidate()  # status defaults to None
        write_candidate_parquet(
            CandidateTable(candidates=[cand]),
            AttributeTable(),
            OutcomeTable(),
            str(tmp_path),
        )
        df = pd.read_parquet(tmp_path / "candidate_detail.parquet")
        assert df["smiles_canonical"].iloc[0] is None
        assert df["smiles_standardization_status"].iloc[0] is None

    def test_log_written_with_per_status_rows(self, tmp_path):
        cands = [
            _enriched_candidate(
                candidate_id="c_ok", drug_name="Ok",
                smiles="CC(=O)O", smiles_canonical="CC(=O)O",
                smiles_standardization_status="ok",
            ),
            _enriched_candidate(
                candidate_id="c_empty", drug_name="Empty",
                smiles=None, smiles_canonical=None,
                smiles_standardization_status="empty",
            ),
            _enriched_candidate(
                candidate_id="c_fail", drug_name="Fail",
                smiles="garbage", smiles_canonical=None,
                smiles_standardization_status="failed_parse",
            ),
            # Status None — never standardized; should NOT appear in the log.
            _enriched_candidate(candidate_id="c_skipped", drug_name="Skipped"),
        ]
        write_smiles_standardization_log(
            CandidateTable(candidates=cands),
            str(tmp_path),
        )
        df = pd.read_csv(tmp_path / "smiles_standardization_log.csv", keep_default_na=False)
        ids = list(df["candidate_id"])
        assert ids == ["c_ok", "c_empty", "c_fail"]  # c_skipped excluded
        ok_row = df.iloc[0]
        assert ok_row["status"] == "ok"
        assert ok_row["smiles_raw"] == "CC(=O)O"
        assert ok_row["smiles_canonical"] == "CC(=O)O"
        empty_row = df.iloc[1]
        assert empty_row["status"] == "empty"
        assert empty_row["smiles_raw"] == ""
        assert empty_row["smiles_canonical"] == ""
        fail_row = df.iloc[2]
        assert fail_row["status"] == "failed_parse"
        assert fail_row["smiles_raw"] == "garbage"
        assert fail_row["smiles_canonical"] == ""

    def test_log_writes_header_only_when_no_candidates_standardized(self, tmp_path):
        cands = [_enriched_candidate(candidate_id="c1")]  # status None
        write_smiles_standardization_log(
            CandidateTable(candidates=cands),
            str(tmp_path),
        )
        # File exists with header only; no data rows.
        contents = (tmp_path / "smiles_standardization_log.csv").read_text()
        lines = contents.strip().splitlines()
        assert len(lines) == 1
        assert lines[0] == "candidate_id,drug_name,status,smiles_raw,smiles_canonical"


# ---------------------------------------------------------------------------
# HINT-prep additions: ICD list on candidate parquet, per-trial refactor
# ---------------------------------------------------------------------------

class TestCandidateIcdCodes:

    def test_parquet_includes_icd10_codes_list(self, tmp_path):
        cand = _enriched_candidate(icd10_codes=["E11", "E11.9"])
        write_candidate_parquet(
            CandidateTable(candidates=[cand]),
            AttributeTable(),
            OutcomeTable(),
            str(tmp_path),
        )
        df = pd.read_parquet(tmp_path / "candidate_detail.parquet")
        assert "icd10_codes" in df.columns
        assert list(df["icd10_codes"].iloc[0]) == ["E11", "E11.9"]

    def test_parquet_icd10_codes_empty_when_unset(self, tmp_path):
        cand = _enriched_candidate()
        write_candidate_parquet(
            CandidateTable(candidates=[cand]),
            AttributeTable(),
            OutcomeTable(),
            str(tmp_path),
        )
        df = pd.read_parquet(tmp_path / "candidate_detail.parquet")
        assert list(df["icd10_codes"].iloc[0]) == []


class TestTrialDetailPerTrial:

    def _trial(self, **overrides) -> RawTrial:
        base = dict(
            nct_id="NCT00000001",
            title="A Phase 2 study",
            intervention="DrugA",
            indication="T2D",
            sponsor="PharmaCo",
            phase=TrialPhase.PHASE_2,
            status=TrialStatus.COMPLETED,
            start_date=date(2020, 1, 1),
            completion_date=date(2022, 6, 30),
            eligibility_criteria="Inclusion: adults 18-65...",
            why_stopped=None,
        )
        base.update(overrides)
        return RawTrial(**base)

    def test_parquet_one_row_per_nct_with_new_columns(self, tmp_path):
        trial = self._trial()
        cand = _enriched_candidate(trial_ids=["NCT00000001"])
        outcome = CandidateOutcomeRecord(
            candidate_id=cand.candidate_id,
            outcome=CandidateOutcome.APPROVED,
        )
        write_trial_parquet(
            CandidateTable(candidates=[cand]),
            AttributeTable(),
            OutcomeTable(outcomes={cand.candidate_id: outcome}),
            TrialTable(trials=[trial]),
            str(tmp_path),
        )
        df = pd.read_parquet(tmp_path / "trial_detail.parquet")
        assert len(df) == 1
        row = df.iloc[0]
        assert row["nct_id"] == "NCT00000001"
        assert row["trial_eligibility_criteria"] == "Inclusion: adults 18-65..."
        assert row["trial_why_stopped"] is None
        # APPROVED candidate + COMPLETED Phase 2 trial -> label 1
        assert row["trial_inferred_label"] == 1
        assert list(row["also_in_candidate_ids"]) == []

    def test_parquet_dedupes_when_same_nct_in_two_candidates(self, tmp_path):
        trial = self._trial(nct_id="NCT12345")
        c1 = _enriched_candidate(candidate_id="c1", trial_ids=["NCT12345"])
        c2 = _enriched_candidate(candidate_id="c2", trial_ids=["NCT12345"])
        write_trial_parquet(
            CandidateTable(candidates=[c1, c2]),
            AttributeTable(),
            OutcomeTable(),
            TrialTable(trials=[trial]),
            str(tmp_path),
        )
        df = pd.read_parquet(tmp_path / "trial_detail.parquet")
        assert len(df) == 1
        row = df.iloc[0]
        assert row["candidate_id"] == "c1"
        assert list(row["also_in_candidate_ids"]) == ["c2"]

    def test_csv_one_row_per_nct_with_new_columns(self, tmp_path):
        from pipeline.stages.reporting._writer import write_trial_detail

        trial = self._trial(why_stopped="Sponsor decision")
        # Two trial_ids on the candidate but only one matches the trial table
        cand = _enriched_candidate(trial_ids=["NCT00000001", "NCT00000003"])
        outcome = CandidateOutcomeRecord(
            candidate_id=cand.candidate_id,
            outcome=CandidateOutcome.FAILED_PHASE_2,
        )
        write_trial_detail(
            CandidateTable(candidates=[cand]),
            AttributeTable(),
            OutcomeTable(outcomes={cand.candidate_id: outcome}),
            TrialTable(trials=[trial]),
            str(tmp_path),
        )
        df = pd.read_csv(tmp_path / "trial_detail.csv", keep_default_na=False)
        # Two unique nct_ids -> two rows; one matched, one placeholder.
        assert list(df["nct_id"]) == ["NCT00000001", "NCT00000003"]
        matched = df[df["nct_id"] == "NCT00000001"].iloc[0]
        assert matched["trial_eligibility_criteria"] == "Inclusion: adults 18-65..."
        assert matched["trial_why_stopped"] == "Sponsor decision"
        # FAILED_PHASE_2 + COMPLETED Phase 2 -> label 0
        assert matched["trial_inferred_label"] == "0"
