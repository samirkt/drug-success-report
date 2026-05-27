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
#  target_symbols_joined, target_ensembl_joined, pathways_joined,
#  tractability_modalities_joined, tractability_labels_joined,
#  loeuf_min)
# The last three are optional — older snapshots / simpler fixtures can
# pass 7-tuples and the missing slots default to None.
DrugRow = tuple
# (chembl_id, indication_efo_id, indication_name, max_phase)
IndRow = tuple[str, str | None, str, int | None]
# (chembl_id, ensembl_id, efo_id, genetic_score)
EvidenceRow = tuple[str, str, str, float]


def _make_ot_snapshot(
    path: Path,
    drug_rows: list[DrugRow],
    indication_rows: list[IndRow],
    evidence_rows: list[EvidenceRow] | None = None,
    user_version: int = 2503,
    *,
    include_new_columns: bool = True,
) -> None:
    """Build a slim OpenTargets snapshot at ``path``.

    Mirrors the schema produced by
    scripts/build_opentargets_snapshot.py — same column names, same
    indexes, same PRAGMA user_version stamp. `query_norm` is derived
    from ``drug_name`` via ``canonicalize_drug_name`` so fixtures align
    with the same normalizer the pipeline applies.

    Pass ``include_new_columns=False`` to write the pre-genetics
    schema (without tractability columns or the target-disease evidence
    table) — used to assert backward compatibility for older snapshots.
    """
    conn = sqlite3.connect(path)
    try:
        if include_new_columns:
            conn.execute(
                """
                CREATE TABLE name_opentargets_drug (
                    query_norm                TEXT NOT NULL,
                    chembl_id                 TEXT NOT NULL,
                    drug_name                 TEXT,
                    moa_text                  TEXT,
                    action_type               TEXT,
                    target_symbols            TEXT,
                    target_ensembl            TEXT,
                    pathways                  TEXT,
                    tractability_modalities   TEXT,
                    tractability_labels       TEXT,
                    loeuf_min                 REAL,
                    PRIMARY KEY (query_norm, chembl_id)
                )
                """
            )
        else:
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
        if include_new_columns:
            conn.execute(
                """
                CREATE TABLE name_opentargets_target_disease_evidence (
                    chembl_id      TEXT NOT NULL,
                    ensembl_id     TEXT NOT NULL,
                    efo_id         TEXT NOT NULL,
                    genetic_score  REAL NOT NULL,
                    PRIMARY KEY (chembl_id, ensembl_id, efo_id)
                )
                """
            )
        if include_new_columns:
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
                    row[7] if len(row) > 7 else None,  # tractability_modalities
                    row[8] if len(row) > 8 else None,  # tractability_labels
                    row[9] if len(row) > 9 else None,  # loeuf_min
                )
                for row in drug_rows
            ]
            conn.executemany(
                "INSERT INTO name_opentargets_drug VALUES "
                "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                expanded_drug,
            )
        else:
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
        if include_new_columns and evidence_rows:
            conn.executemany(
                "INSERT INTO name_opentargets_target_disease_evidence "
                "VALUES (?, ?, ?, ?)",
                evidence_rows,
            )
        conn.execute(
            "CREATE INDEX idx_ot_query_norm ON name_opentargets_drug(query_norm)"
        )
        conn.execute(
            "CREATE INDEX idx_ot_ind_chembl "
            "ON name_opentargets_indications(chembl_id)"
        )
        if include_new_columns:
            conn.execute(
                "CREATE INDEX idx_ot_evidence_chembl_efo "
                "ON name_opentargets_target_disease_evidence(chembl_id, efo_id)"
            )
        conn.execute(f"PRAGMA user_version = {user_version}")
        conn.commit()
    finally:
        conn.close()


@pytest.fixture
def ot_snapshot(tmp_path: Path) -> Path:
    """Covers:
    - Semaglutide — single ChEMBL ID, single MoA + indication (T2D approved).
      Tractability: SM bucket; LOEUF set so the min-aggregation has a value.
      Genetic evidence: GLP1R × T2D EFO carries a score for the genetics join.
    - Insulin — two ChEMBL IDs sharing the query_norm (dedup test) with
      different target symbols, tractability buckets, and LOEUF values so
      the min-aggregation can be exercised across multiple chembl rows.
    - Pembrolizumab — MoA, targets, pathways, and two indications
      (melanoma Phase 4 and NSCLC Phase 4) for indication-match tests.
      Antibody tractability bucket only.
    - Aspirin — no indications at all (so the indication-phase coverage
      can stay unmatched even when MoA/targets do match). No tractability.
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
                "SM",                                # tractability_modalities
                "Clinical_Precedence_sm",             # tractability_labels
                0.42,                                  # loeuf_min
            ),
            (
                "Insulin", "CHEMBL1201247",
                "Insulin receptor agonist",
                "AGONIST",
                "INSR",
                "ENSG00000171105",
                "Insulin signaling",
                "AB",
                "Clinical_Precedence_ab",
                0.61,
            ),
            (
                "Insulin", "CHEMBL_INSULIN_ALT",
                "IGF-1 receptor agonist",
                "AGONIST",
                "IGF1R",
                "ENSG00000140443",
                "Insulin signaling|IGF signaling",
                "SM",
                "Predicted_Tractable_sm_High_Confidence",
                0.35,
            ),
            (
                "Pembrolizumab", "CHEMBL3137343",
                "Anti-PD-1 monoclonal antibody",
                "INHIBITOR",
                "PDCD1",
                "ENSG00000188389",
                "PD-1 signaling|Immune checkpoint",
                "AB",
                "Clinical_Precedence_ab",
                0.55,
            ),
            (
                "Aspirin", "CHEMBL25",
                "Cyclooxygenase inhibitor",
                "INHIBITOR",
                "PTGS1|PTGS2",
                "ENSG00000095303|ENSG00000073756",
                "Prostaglandin synthesis",
                None,   # no tractability
                None,
                None,   # no LOEUF
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
        evidence_rows=[
            # Semaglutide / GLP1R has strong genetic evidence for T2D
            # plus weaker evidence for an unrelated disease — exercises
            # the (chembl, matched-EFO) join and the any-indication
            # fallback (which should also pick up the T2D score).
            ("CHEMBL4297516", "ENSG00000112164", "EFO_0001360", 0.71),
            ("CHEMBL4297516", "ENSG00000112164", "EFO_0000400", 0.20),
            # Insulin's two ChEMBL rows — both with some evidence.
            ("CHEMBL1201247", "ENSG00000171105", "EFO_0001359", 0.30),
            ("CHEMBL_INSULIN_ALT", "ENSG00000140443", "EFO_0001360", 0.45),
            # Pembrolizumab — PDCD1 has melanoma evidence but no NSCLC
            # row, exercising the case where indication-match succeeds
            # but the genetic_score join is empty.
            ("CHEMBL3137343", "ENSG00000188389", "EFO_0000756", 0.62),
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


# ---------------------------------------------------------------------------
# Tractability, LOEUF, and target-disease genetic evidence
# ---------------------------------------------------------------------------

class TestTractabilityLoeufGenetics:
    """Covers the OT 25.x feature additions: tractability + LOEUF
    aggregated across the drug's targets, and the (drug × matched-EFO)
    genetic-association score plus its any-indication fallback. The
    schema is exercised against the same `ot_snapshot` fixture as the
    rest of the file so the rows under test are the canonical ones.
    """

    def test_tractability_modalities_and_labels_attach(
        self, ot_snapshot, sample_candidate
    ):
        cand = replace(sample_candidate, drug_name_raw="Semaglutide")
        table = CandidateTable(candidates=[cand])
        stage = OpenTargetsEnrichment()
        cfg = _make_config(opentargets_snapshot_path=ot_snapshot)
        stage.is_available(cfg)
        stage.run(table, ledger=CostLedger())
        assert cand.opentargets_tractability_modalities == ["SM"]
        assert cand.opentargets_tractability_labels == [
            "Clinical_Precedence_sm"
        ]

    def test_tractability_unions_across_chembl_rows(
        self, ot_snapshot, sample_candidate
    ):
        """Insulin's two ChEMBL rows contribute AB and SM modalities —
        the candidate should surface both in first-seen order."""
        cand = replace(sample_candidate, drug_name_raw="Insulin")
        table = CandidateTable(candidates=[cand])
        stage = OpenTargetsEnrichment()
        cfg = _make_config(opentargets_snapshot_path=ot_snapshot)
        stage.is_available(cfg)
        stage.run(table, ledger=CostLedger())
        assert cand.opentargets_tractability_modalities == ["AB", "SM"]
        # Both per-row labels appear in the union.
        assert set(cand.opentargets_tractability_labels) == {
            "Clinical_Precedence_ab",
            "Predicted_Tractable_sm_High_Confidence",
        }

    def test_loeuf_takes_min_across_chembl_rows(
        self, ot_snapshot, sample_candidate
    ):
        """Insulin's two ChEMBL rows carry LOEUF 0.61 and 0.35; the
        candidate aggregates as the min (most-constrained target)."""
        cand = replace(sample_candidate, drug_name_raw="Insulin")
        table = CandidateTable(candidates=[cand])
        stage = OpenTargetsEnrichment()
        cfg = _make_config(opentargets_snapshot_path=ot_snapshot)
        stage.is_available(cfg)
        stage.run(table, ledger=CostLedger())
        assert cand.opentargets_loeuf_min == pytest.approx(0.35)

    def test_tractability_absent_stays_empty(
        self, ot_snapshot, sample_candidate
    ):
        """Aspirin's drug rows have no tractability/LOEUF — the
        candidate's fields stay at their defaults."""
        cand = replace(
            sample_candidate,
            drug_name_raw="Aspirin",
            indication="Cardiovascular disease",
            mesh_indication=None,
        )
        table = CandidateTable(candidates=[cand])
        stage = OpenTargetsEnrichment()
        cfg = _make_config(opentargets_snapshot_path=ot_snapshot)
        stage.is_available(cfg)
        stage.run(table, ledger=CostLedger())
        assert cand.opentargets_tractability_modalities == []
        assert cand.opentargets_tractability_labels == []
        assert cand.opentargets_loeuf_min is None

    def test_genetic_score_indication_matched(
        self, ot_snapshot, sample_candidate
    ):
        """Semaglutide + T2D — the matched-EFO genetic score is 0.71,
        and the any-indication fallback equals or exceeds it."""
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
        assert cand.opentargets_genetic_score == pytest.approx(0.71)
        assert cand.opentargets_genetic_score_max_any_indication == pytest.approx(0.71)

    def test_genetic_score_indication_no_match_falls_back(
        self, ot_snapshot, sample_candidate
    ):
        """Semaglutide studied in an indication without OT evidence —
        the matched-indication score is None but the any-indication
        fallback still surfaces the gene's best signal."""
        cand = replace(
            sample_candidate,
            drug_name_raw="Semaglutide",
            indication="Obesity",
            mesh_indication=None,
        )
        table = CandidateTable(candidates=[cand])
        stage = OpenTargetsEnrichment()
        cfg = _make_config(opentargets_snapshot_path=ot_snapshot)
        stage.is_available(cfg)
        stage.run(table, ledger=CostLedger())
        assert cand.opentargets_genetic_score is None
        assert cand.opentargets_genetic_score_max_any_indication == pytest.approx(0.71)

    def test_genetic_score_indication_matched_but_no_evidence(
        self, ot_snapshot, sample_candidate
    ):
        """Pembrolizumab + NSCLC — indication match succeeds (phase 4)
        but the evidence table has no PDCD1 × NSCLC row, so the
        matched-EFO score stays None while the any-indication fallback
        picks up the melanoma row."""
        cand = replace(
            sample_candidate,
            drug_name_raw="Pembrolizumab",
            indication="Non-small cell lung carcinoma",
        )
        table = CandidateTable(candidates=[cand])
        stage = OpenTargetsEnrichment()
        cfg = _make_config(opentargets_snapshot_path=ot_snapshot)
        stage.is_available(cfg)
        stage.run(table, ledger=CostLedger())
        assert cand.opentargets_indication_max_phase == 4
        assert cand.opentargets_genetic_score is None
        assert cand.opentargets_genetic_score_max_any_indication == pytest.approx(0.62)

    def test_coverage_records_new_features(self, ot_snapshot, sample_candidate):
        c_sema = replace(
            sample_candidate,
            candidate_id="c1",
            drug_name_raw="Semaglutide",
            indication="Type 2 diabetes mellitus",
        )
        c_aspirin = replace(
            sample_candidate,
            candidate_id="c2",
            drug_name_raw="Aspirin",
            indication="Cardiovascular disease",
            mesh_indication=None,
        )
        c_unknown = replace(
            sample_candidate, candidate_id="c3", drug_name_raw="NovelXYZ",
        )
        table = CandidateTable(candidates=[c_sema, c_aspirin, c_unknown])

        ledger = CostLedger()
        stage = OpenTargetsEnrichment()
        cfg = _make_config(opentargets_snapshot_path=ot_snapshot)
        stage.is_available(cfg)
        stage.run(table, ledger=ledger)

        # Tractability + LOEUF: only Sema has them (Aspirin has neither;
        # Unknown isn't in the snapshot at all).
        assert ledger.coverage["opentargets_tractability"] == (1, 3)
        assert ledger.coverage["opentargets_loeuf"] == (1, 3)
        # Genetic score: only Sema's T2D match hits.
        assert ledger.coverage["opentargets_genetic_score"] == (1, 3)
        # Any-EFO fallback: Sema's GLP1R-anywhere score hits; Aspirin
        # has no genetic-evidence rows.
        assert ledger.coverage["opentargets_genetic_score_any"] == (1, 3)


class TestBackwardCompatibility:
    """An older snapshot built before the genetics/tractability schema
    addition must still load — coverage for the missing features stays
    at zero and the Candidate fields keep their defaults.
    """

    def test_old_schema_still_loads(self, tmp_path, sample_candidate):
        path = tmp_path / "old_snapshot.sqlite"
        _make_ot_snapshot(
            path,
            drug_rows=[
                ("Semaglutide", "CHEMBL4297516",
                 "GLP-1 receptor agonist", "AGONIST",
                 "GLP1R", "ENSG00000112164", "GLP-1 signaling"),
            ],
            indication_rows=[
                ("CHEMBL4297516", "EFO_0001360",
                 "type 2 diabetes mellitus", 4),
            ],
            include_new_columns=False,
        )
        cand = replace(
            sample_candidate,
            drug_name_raw="Semaglutide",
            indication="Type 2 diabetes mellitus",
        )
        table = CandidateTable(candidates=[cand])
        ledger = CostLedger()
        stage = OpenTargetsEnrichment()
        cfg = _make_config(opentargets_snapshot_path=path)
        assert stage.is_available(cfg) is True
        stage.run(table, ledger=ledger)
        # Pre-existing features still work.
        assert cand.opentargets_moa == "GLP-1 receptor agonist"
        assert cand.opentargets_indication_max_phase == 4
        # New features default to empty / None.
        assert cand.opentargets_tractability_modalities == []
        assert cand.opentargets_tractability_labels == []
        assert cand.opentargets_loeuf_min is None
        assert cand.opentargets_genetic_score is None
        assert cand.opentargets_genetic_score_max_any_indication is None
        # Coverage on the new features reads zero — not asserted as
        # missing, since the run loop unconditionally records each.
        assert ledger.coverage["opentargets_tractability"] == (0, 1)
        assert ledger.coverage["opentargets_loeuf"] == (0, 1)
        assert ledger.coverage["opentargets_genetic_score"] == (0, 1)
        assert ledger.coverage["opentargets_genetic_score_any"] == (0, 1)
