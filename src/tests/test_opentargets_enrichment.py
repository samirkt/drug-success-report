"""Tests for pipeline.enrichment.opentargets.OpenTargetsEnrichment.

Exercises the name-keyed OpenTargets snapshot schema produced by
`scripts/build_opentargets_snapshot.py`. Matching is name-keyed via
`canonicalize_drug_name` so the OT stage works identically whether
DrugBank or ChEMBL IDs are available on the candidate — indication
phase matching falls back from `indication` to `mesh_indication` and
stays None if neither case-insensitively matches an OT row.
"""

from __future__ import annotations

import sqlite3
from dataclasses import replace
from pathlib import Path

import pytest

from pipeline.drugbank_norm import canonicalize_drug_name
from pipeline.enrichment.opentargets import OpenTargetsEnrichment
from pipeline.models import CandidateTable
from utils.tiered_router import CostLedger


# Tuple layout for drug rows (query_norm derived from drug_name):
# (drug_name, chembl_id, moa_text, action_type,
#  target_symbols_joined, target_ensembl_joined, pathways_joined)
DrugRow = tuple[str, str, str | None, str | None, str | None, str | None, str | None]
# (chembl_id, indication_efo_id, indication_name, max_phase)
IndRow = tuple[str, str | None, str, int | None]


def _make_ot_snapshot(
    path: Path,
    drug_rows: list[DrugRow],
    indication_rows: list[IndRow],
    user_version: int = 2503,
) -> None:
    """Build a slim OpenTargets snapshot at ``path``.

    Mirrors the schema produced by
    scripts/build_opentargets_snapshot.py — same column names, same
    indexes, same PRAGMA user_version stamp. `query_norm` is derived
    from ``drug_name`` via ``canonicalize_drug_name`` so fixtures align
    with the same normalizer the pipeline applies.
    """
    conn = sqlite3.connect(path)
    try:
        conn.execute(
            """
            CREATE TABLE name_opentargets_drug (
                query_norm        TEXT NOT NULL,
                chembl_id         TEXT NOT NULL,
                drug_name         TEXT,
                moa_text          TEXT,
                action_type       TEXT,
                target_symbols    TEXT,
                target_ensembl    TEXT,
                pathways          TEXT,
                PRIMARY KEY (query_norm, chembl_id)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE name_opentargets_indications (
                chembl_id               TEXT NOT NULL,
                indication_efo_id       TEXT,
                indication_name         TEXT,
                indication_max_phase    INTEGER,
                PRIMARY KEY (chembl_id, indication_efo_id)
            )
            """
        )
        expanded_drug = [
            (
                canonicalize_drug_name(row[0]),  # query_norm
                row[1],  # chembl_id
                row[0],  # drug_name
                row[2],  # moa_text
                row[3],  # action_type
                row[4],  # target_symbols
                row[5],  # target_ensembl
                row[6],  # pathways
            )
            for row in drug_rows
        ]
        conn.executemany(
            "INSERT INTO name_opentargets_drug VALUES "
            "(?, ?, ?, ?, ?, ?, ?, ?)",
            expanded_drug,
        )
        conn.executemany(
            "INSERT INTO name_opentargets_indications VALUES (?, ?, ?, ?)",
            indication_rows,
        )
        conn.execute(
            "CREATE INDEX idx_ot_query_norm ON name_opentargets_drug(query_norm)"
        )
        conn.execute(
            "CREATE INDEX idx_ot_ind_chembl "
            "ON name_opentargets_indications(chembl_id)"
        )
        conn.execute(f"PRAGMA user_version = {user_version}")
        conn.commit()
    finally:
        conn.close()


@pytest.fixture
def ot_snapshot(tmp_path: Path) -> Path:
    """Covers:
    - Semaglutide — single ChEMBL ID, single MoA + indication (T2D approved).
    - Insulin — two ChEMBL IDs sharing the query_norm (dedup test) with
      different target symbols + overlapping pathways.
    - Pembrolizumab — MoA, targets, pathways, and two indications
      (melanoma Phase 4 and NSCLC Phase 4) for indication-match tests.
    - Aspirin — no indications at all (so the indication-phase coverage
      can stay unmatched even when MoA/targets do match).
    """
    path = tmp_path / "opentargets_snapshot.sqlite"
    _make_ot_snapshot(
        path,
        drug_rows=[
            (
                "Semaglutide", "CHEMBL4297516",
                "GLP-1 receptor agonist",
                "AGONIST",
                "GLP1R",
                "ENSG00000112164",
                "GLP-1 signaling|Incretin regulation",
            ),
            (
                "Insulin", "CHEMBL1201247",
                "Insulin receptor agonist",
                "AGONIST",
                "INSR",
                "ENSG00000171105",
                "Insulin signaling",
            ),
            (
                "Insulin", "CHEMBL_INSULIN_ALT",
                "IGF-1 receptor agonist",
                "AGONIST",
                "IGF1R",
                "ENSG00000140443",
                "Insulin signaling|IGF signaling",
            ),
            (
                "Pembrolizumab", "CHEMBL3137343",
                "Anti-PD-1 monoclonal antibody",
                "INHIBITOR",
                "PDCD1",
                "ENSG00000188389",
                "PD-1 signaling|Immune checkpoint",
            ),
            (
                "Aspirin", "CHEMBL25",
                "Cyclooxygenase inhibitor",
                "INHIBITOR",
                "PTGS1|PTGS2",
                "ENSG00000095303|ENSG00000073756",
                "Prostaglandin synthesis",
            ),
        ],
        indication_rows=[
            # Semaglutide → type 2 diabetes mellitus, max phase 4.
            ("CHEMBL4297516", "EFO_0001360",
             "type 2 diabetes mellitus", 4),
            # Pembrolizumab indications.
            ("CHEMBL3137343", "EFO_0000756", "melanoma", 4),
            ("CHEMBL3137343", "EFO_0003060",
             "non-small cell lung carcinoma", 4),
        ],
    )
    return path


def _make_config(**overrides):
    class _Cfg:
        enable_opentargets = True
        opentargets_snapshot_path: Path | None = None

    cfg = _Cfg()
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


# ---------------------------------------------------------------------------
# is_available gating
# ---------------------------------------------------------------------------

class TestIsAvailable:
    def test_disabled_flag_returns_false(self, ot_snapshot):
        cfg = _make_config(
            enable_opentargets=False, opentargets_snapshot_path=ot_snapshot
        )
        assert OpenTargetsEnrichment().is_available(cfg) is False

    def test_missing_snapshot_returns_false(self, tmp_path):
        cfg = _make_config(opentargets_snapshot_path=tmp_path / "nope.sqlite")
        assert OpenTargetsEnrichment().is_available(cfg) is False

    def test_no_snapshot_path_returns_false(self):
        cfg = _make_config(opentargets_snapshot_path=None)
        assert OpenTargetsEnrichment().is_available(cfg) is False

    def test_valid_snapshot_returns_true(self, ot_snapshot):
        cfg = _make_config(opentargets_snapshot_path=ot_snapshot)
        assert OpenTargetsEnrichment().is_available(cfg) is True


# ---------------------------------------------------------------------------
# run() behavior
# ---------------------------------------------------------------------------

class TestRun:
    def test_moa_and_targets_attach(self, ot_snapshot, sample_candidate):
        cand = replace(sample_candidate, drug_name_raw="Semaglutide")
        table = CandidateTable(candidates=[cand])

        stage = OpenTargetsEnrichment()
        cfg = _make_config(opentargets_snapshot_path=ot_snapshot)
        assert stage.is_available(cfg) is True
        stage.run(table, ledger=CostLedger())

        assert cand.opentargets_moa == "GLP-1 receptor agonist"
        assert cand.opentargets_action_type == "AGONIST"
        assert cand.opentargets_targets == ["GLP1R"]
        assert cand.opentargets_pathways == [
            "GLP-1 signaling", "Incretin regulation"
        ]

    def test_multi_chembl_id_union_sorted_deduped(
        self, ot_snapshot, sample_candidate
    ):
        """Two Insulin ChEMBL IDs contribute INSR + IGF1R. Targets and
        pathways must union, dedup (shared "Insulin signaling"), and
        return sorted."""
        cand = replace(sample_candidate, drug_name_raw="Insulin")
        table = CandidateTable(candidates=[cand])

        stage = OpenTargetsEnrichment()
        cfg = _make_config(opentargets_snapshot_path=ot_snapshot)
        stage.is_available(cfg)
        stage.run(table, ledger=CostLedger())

        assert cand.opentargets_targets == ["IGF1R", "INSR"]
        assert cand.opentargets_pathways == [
            "IGF signaling", "Insulin signaling"
        ]

    def test_indication_exact_match(self, ot_snapshot, sample_candidate):
        cand = replace(
            sample_candidate,
            drug_name_raw="Semaglutide",
            indication="Type 2 diabetes mellitus",
        )
        table = CandidateTable(candidates=[cand])

        stage = OpenTargetsEnrichment()
        cfg = _make_config(opentargets_snapshot_path=ot_snapshot)
        stage.is_available(cfg)
        stage.run(table, ledger=CostLedger())

        assert cand.opentargets_indication_max_phase == 4

    def test_indication_no_match_stays_none(self, ot_snapshot, sample_candidate):
        """Semaglutide's only OT indication is T2D. A candidate studying
        it in a different indication gets MoA/targets, but the phase
        stays None — no fallback to max-across-indications."""
        cand = replace(
            sample_candidate,
            drug_name_raw="Semaglutide",
            indication="Obesity",  # absent from OT indications for this drug
            mesh_indication=None,
        )
        table = CandidateTable(candidates=[cand])

        stage = OpenTargetsEnrichment()
        cfg = _make_config(opentargets_snapshot_path=ot_snapshot)
        stage.is_available(cfg)
        stage.run(table, ledger=CostLedger())

        assert cand.opentargets_moa == "GLP-1 receptor agonist"
        assert cand.opentargets_indication_max_phase is None

    def test_indication_mesh_fallback(self, ot_snapshot, sample_candidate):
        """When `indication` is unfamiliar text but `mesh_indication`
        matches an OT indication row, the phase is picked up from the
        MeSH fallback."""
        cand = replace(
            sample_candidate,
            drug_name_raw="Pembrolizumab",
            indication="NSCLC (advanced)",  # free-text, wouldn't match
            mesh_indication="Non-small cell lung carcinoma",
        )
        table = CandidateTable(candidates=[cand])

        stage = OpenTargetsEnrichment()
        cfg = _make_config(opentargets_snapshot_path=ot_snapshot)
        stage.is_available(cfg)
        stage.run(table, ledger=CostLedger())

        assert cand.opentargets_indication_max_phase == 4

    def test_unknown_drug_passthrough(self, ot_snapshot, sample_candidate):
        cand = replace(sample_candidate, drug_name_raw="NovelCompoundXYZ")
        table = CandidateTable(candidates=[cand])

        stage = OpenTargetsEnrichment()
        cfg = _make_config(opentargets_snapshot_path=ot_snapshot)
        stage.is_available(cfg)
        stage.run(table, ledger=CostLedger())

        assert cand.opentargets_moa is None
        assert cand.opentargets_targets == []
        assert cand.opentargets_pathways == []
        assert cand.opentargets_indication_max_phase is None

    def test_empty_name_passthrough(self, ot_snapshot, sample_candidate):
        cand = replace(sample_candidate, drug_name_raw="")
        table = CandidateTable(candidates=[cand])

        stage = OpenTargetsEnrichment()
        cfg = _make_config(opentargets_snapshot_path=ot_snapshot)
        stage.is_available(cfg)
        stage.run(table, ledger=CostLedger())

        assert cand.opentargets_moa is None

    def test_coverage_records_four_features(
        self, ot_snapshot, sample_candidate
    ):
        """Mixed candidates exercise each counter independently:
        - c1 Semaglutide+T2D: MoA + targets + pathways + phase all hit.
        - c2 Aspirin with no matching indication: MoA/targets/pathways
          hit, phase misses (no indications in snapshot for aspirin).
        - c3 Unknown: misses everything.
        """
        c1 = replace(
            sample_candidate,
            candidate_id="c1",
            drug_name_raw="Semaglutide",
            indication="Type 2 diabetes mellitus",
        )
        c2 = replace(
            sample_candidate,
            candidate_id="c2",
            drug_name_raw="Aspirin",
            indication="Cardiovascular disease",
            mesh_indication=None,
        )
        c3 = replace(
            sample_candidate,
            candidate_id="c3",
            drug_name_raw="NovelCompoundXYZ",
        )
        table = CandidateTable(candidates=[c1, c2, c3])

        ledger = CostLedger()
        stage = OpenTargetsEnrichment()
        cfg = _make_config(opentargets_snapshot_path=ot_snapshot)
        stage.is_available(cfg)
        stage.run(table, ledger=ledger)

        assert ledger.coverage["opentargets_moa"] == (2, 3)
        assert ledger.coverage["opentargets_targets"] == (2, 3)
        assert ledger.coverage["opentargets_pathways"] == (2, 3)
        assert ledger.coverage["opentargets_indication_phase"] == (1, 3)

    def test_does_not_mutate_cache_key_inputs(
        self, ot_snapshot, sample_candidate
    ):
        """Enrichment must not touch drug_name, drug_name_raw,
        indication, highest_phase, or candidate_id — the fields
        KnowledgeCache keys derive from."""
        cand = replace(
            sample_candidate,
            drug_name_raw="Semaglutide",
            indication="Type 2 diabetes mellitus",
        )
        before = (
            cand.drug_name,
            cand.drug_name_raw,
            cand.indication,
            cand.highest_phase,
            cand.candidate_id,
        )

        stage = OpenTargetsEnrichment()
        cfg = _make_config(opentargets_snapshot_path=ot_snapshot)
        stage.is_available(cfg)
        stage.run(CandidateTable(candidates=[cand]), ledger=CostLedger())

        after = (
            cand.drug_name,
            cand.drug_name_raw,
            cand.indication,
            cand.highest_phase,
            cand.candidate_id,
        )
        assert before == after
        # Sanity-check the enrichment ran.
        assert cand.opentargets_moa

    def test_reads_ot_release_from_pragma(self, tmp_path, sample_candidate):
        path = tmp_path / "opentargets.sqlite"
        _make_ot_snapshot(
            path,
            drug_rows=[
                ("Semaglutide", "CHEMBL4297516", "GLP-1 agonist",
                 "AGONIST", "GLP1R", "ENSG00000112164", "GLP-1 signaling"),
            ],
            indication_rows=[],
            user_version=2512,
        )
        stage = OpenTargetsEnrichment()
        cfg = _make_config(opentargets_snapshot_path=path)
        stage.is_available(cfg)
        stage.run(
            CandidateTable(
                candidates=[
                    replace(sample_candidate, drug_name_raw="Semaglutide")
                ]
            ),
            ledger=CostLedger(),
        )
        assert stage._release == 2512
