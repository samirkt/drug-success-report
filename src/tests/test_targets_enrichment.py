"""Tests for pipeline.enrichment.targets.TargetsEnrichment.

Exercises the name-keyed snapshot schema produced by
`scripts/build_chembl_targets_snapshot.py`. Matching is DrugBank-
independent: candidates are matched by `canonicalize_drug_name(drug_name_raw)`
against the snapshot's `query_norm` column.
"""

from __future__ import annotations

import sqlite3
from dataclasses import replace
from pathlib import Path

import pytest

from pipeline.drugbank_norm import canonicalize_drug_name
from pipeline.enrichment.targets import TargetsEnrichment
from pipeline.models import CandidateTable
from utils.tiered_router import CostLedger


# Tuple layout (no query_norm — the helper computes it from source_name):
# (source_name, source_kind, syn_type, chembl_id, target_chembl_id,
#  target_pref_name, target_type, uniprot_accession, action_type)
SnapshotRow = tuple[str, str, str | None, str, str, str, str, str, str]


def _make_snapshot(
    path: Path,
    rows: list[SnapshotRow],
    user_version: int = 35,
) -> None:
    """Build a slim name-keyed targets snapshot at ``path``.

    Mirrors the schema produced by
    scripts/build_chembl_targets_snapshot.py — same column names, same
    index, same PRAGMA user_version stamp. `query_norm` is derived from
    ``source_name`` via ``canonicalize_drug_name`` so fixtures stay
    aligned with whatever normalization policy the pipeline applies.
    """
    conn = sqlite3.connect(path)
    try:
        conn.execute(
            """
            CREATE TABLE name_targets (
                query_norm          TEXT NOT NULL,
                source_name         TEXT NOT NULL,
                source_kind         TEXT NOT NULL,
                syn_type            TEXT,
                chembl_id           TEXT NOT NULL,
                target_chembl_id    TEXT,
                target_pref_name    TEXT,
                target_type         TEXT,
                uniprot_accession   TEXT,
                action_type         TEXT
            )
            """
        )
        expanded = [
            (canonicalize_drug_name(source_name), *row)
            for row in rows
            for source_name in (row[0],)
        ]
        conn.executemany(
            "INSERT INTO name_targets VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            expanded,
        )
        conn.execute("CREATE INDEX idx_query_norm ON name_targets(query_norm)")
        conn.execute(f"PRAGMA user_version = {user_version}")
        conn.commit()
    finally:
        conn.close()


@pytest.fixture
def targets_snapshot(tmp_path: Path) -> Path:
    """Covers aspirin (pref + synonym), semaglutide (trade name),
    insulin (two molregnos both carrying the synonym — dedup test),
    and a protein-complex row with pipe-joined accessions."""
    path = tmp_path / "chembl_targets.sqlite"
    _make_snapshot(
        path,
        rows=[
            # Aspirin — preferred-name rows (COX-1 + COX-2, single protein)
            ("Aspirin", "pref_name", None, "CHEMBL25", "CHEMBL221",
             "Cyclooxygenase-1", "SINGLE PROTEIN", "P23219", "INHIBITOR"),
            ("Aspirin", "pref_name", None, "CHEMBL25", "CHEMBL230",
             "Cyclooxygenase-2", "SINGLE PROTEIN", "P35354", "INHIBITOR"),
            # Aspirin — synonym row (same molregno, other synonym spelling)
            ("Acetylsalicylic Acid", "synonym", "INN", "CHEMBL25", "CHEMBL221",
             "Cyclooxygenase-1", "SINGLE PROTEIN", "P23219", "INHIBITOR"),
            ("Acetylsalicylic Acid", "synonym", "INN", "CHEMBL25", "CHEMBL230",
             "Cyclooxygenase-2", "SINGLE PROTEIN", "P35354", "INHIBITOR"),
            # Semaglutide — trade name synonym (different syn_type)
            ("Ozempic", "synonym", "TRADE_NAME", "CHEMBL4297516", "CHEMBL1784",
             "Glucagon-like peptide 1 receptor", "SINGLE PROTEIN",
             "P43220", "AGONIST"),
            # Insulin — two different ChEMBL drugs share the synonym "Insulin"
            # (multi-match dedup test). Each points at a different target.
            ("Insulin", "synonym", "INN", "CHEMBL1201247", "CHEMBL1981",
             "Insulin receptor", "SINGLE PROTEIN",
             "P06213", "AGONIST"),
            ("Insulin", "synonym", "INN", "CHEMBL_INSULIN_ALT", "CHEMBL1957",
             "Insulin-like growth factor I receptor", "SINGLE PROTEIN",
             "P08069", "AGONIST"),
            # Metformin — protein-complex with pipe-joined UniProts
            ("Metformin", "pref_name", None, "CHEMBL1431", "CHEMBL2107",
             "AMP-activated protein kinase", "PROTEIN COMPLEX",
             "Q13131|Q8N2F8", "ACTIVATOR"),
        ],
    )
    return path


def _make_config(**overrides):
    class _Cfg:
        enable_targets = True
        chembl_snapshot_path: Path | None = None

    cfg = _Cfg()
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


# ---------------------------------------------------------------------------
# is_available gating (4 tests)
# ---------------------------------------------------------------------------

class TestIsAvailable:
    def test_disabled_flag_returns_false(self, targets_snapshot):
        cfg = _make_config(enable_targets=False, chembl_snapshot_path=targets_snapshot)
        assert TargetsEnrichment().is_available(cfg) is False

    def test_missing_snapshot_returns_false(self, tmp_path):
        cfg = _make_config(chembl_snapshot_path=tmp_path / "nope.sqlite")
        assert TargetsEnrichment().is_available(cfg) is False

    def test_no_snapshot_path_returns_false(self):
        cfg = _make_config(chembl_snapshot_path=None)
        assert TargetsEnrichment().is_available(cfg) is False

    def test_valid_snapshot_returns_true(self, targets_snapshot):
        cfg = _make_config(chembl_snapshot_path=targets_snapshot)
        assert TargetsEnrichment().is_available(cfg) is True


# ---------------------------------------------------------------------------
# run() behavior
# ---------------------------------------------------------------------------

class TestRun:
    def test_pref_name_match(self, targets_snapshot, sample_candidate):
        aspirin = replace(sample_candidate, drug_name_raw="Aspirin")
        table = CandidateTable(candidates=[aspirin])

        stage = TargetsEnrichment()
        cfg = _make_config(chembl_snapshot_path=targets_snapshot)
        assert stage.is_available(cfg) is True
        stage.run(table, ledger=CostLedger())

        assert set(aspirin.drug_targets) == {"P23219", "P35354"}
        assert set(aspirin.target_names) == {
            "Cyclooxygenase-1",
            "Cyclooxygenase-2",
        }

    def test_synonym_match(self, targets_snapshot, sample_candidate):
        # Acetylsalicylic Acid is a synonym for aspirin; must produce the
        # same targets as the pref-name path.
        aspirin = replace(sample_candidate, drug_name_raw="Acetylsalicylic Acid")
        table = CandidateTable(candidates=[aspirin])

        stage = TargetsEnrichment()
        cfg = _make_config(chembl_snapshot_path=targets_snapshot)
        stage.is_available(cfg)
        stage.run(table, ledger=CostLedger())

        assert set(aspirin.drug_targets) == {"P23219", "P35354"}
        assert set(aspirin.target_names) == {
            "Cyclooxygenase-1",
            "Cyclooxygenase-2",
        }

    def test_trade_name_match(self, targets_snapshot, sample_candidate):
        # Ozempic is a TRADE_NAME synonym for semaglutide.
        cand = replace(sample_candidate, drug_name_raw="Ozempic")
        table = CandidateTable(candidates=[cand])

        stage = TargetsEnrichment()
        cfg = _make_config(chembl_snapshot_path=targets_snapshot)
        stage.is_available(cfg)
        stage.run(table, ledger=CostLedger())

        assert cand.drug_targets == ["P43220"]
        assert cand.target_names == ["Glucagon-like peptide 1 receptor"]

    def test_drugbank_independence(self, targets_snapshot, sample_candidate):
        """Headline invariant: candidate with drugbank_id=None still
        gets targets when its raw name matches ChEMBL."""
        cand = replace(
            sample_candidate,
            drug_name_raw="Aspirin",
            drugbank_id=None,
        )
        table = CandidateTable(candidates=[cand])

        stage = TargetsEnrichment()
        cfg = _make_config(chembl_snapshot_path=targets_snapshot)
        stage.is_available(cfg)
        stage.run(table, ledger=CostLedger())

        assert cand.drugbank_id is None
        assert set(cand.drug_targets) == {"P23219", "P35354"}
        assert set(cand.target_names) == {
            "Cyclooxygenase-1",
            "Cyclooxygenase-2",
        }

    def test_unknown_name_passthrough(self, targets_snapshot, sample_candidate):
        cand = replace(sample_candidate, drug_name_raw="NovelCompoundXYZ")
        table = CandidateTable(candidates=[cand])

        stage = TargetsEnrichment()
        cfg = _make_config(chembl_snapshot_path=targets_snapshot)
        stage.is_available(cfg)
        stage.run(table, ledger=CostLedger())

        assert cand.drug_targets == []
        assert cand.target_names == []

    def test_multi_match_dedup_and_sorted(self, targets_snapshot, sample_candidate):
        """Two ChEMBL molregnos carry the synonym 'Insulin'; the enrichment
        should union their targets, dedup, and return sorted lists."""
        cand = replace(sample_candidate, drug_name_raw="Insulin")
        table = CandidateTable(candidates=[cand])

        stage = TargetsEnrichment()
        cfg = _make_config(chembl_snapshot_path=targets_snapshot)
        stage.is_available(cfg)
        stage.run(table, ledger=CostLedger())

        assert cand.drug_targets == ["P06213", "P08069"]  # sorted, deduped
        assert cand.target_names == [
            "Insulin receptor",
            "Insulin-like growth factor I receptor",
        ]

    def test_protein_complex_accession_split(self, targets_snapshot, sample_candidate):
        cand = replace(sample_candidate, drug_name_raw="Metformin")
        table = CandidateTable(candidates=[cand])

        stage = TargetsEnrichment()
        cfg = _make_config(chembl_snapshot_path=targets_snapshot)
        stage.is_available(cfg)
        stage.run(table, ledger=CostLedger())

        assert cand.drug_targets == ["Q13131", "Q8N2F8"]  # split + sorted
        assert cand.target_names == ["AMP-activated protein kinase"]

    def test_records_coverage_in_ledger(self, targets_snapshot, sample_candidate):
        hit = replace(sample_candidate, candidate_id="c1", drug_name_raw="Aspirin")
        miss = replace(sample_candidate, candidate_id="c2", drug_name_raw="NovelCompoundXYZ")
        empty = replace(sample_candidate, candidate_id="c3", drug_name_raw="")
        table = CandidateTable(candidates=[hit, miss, empty])

        ledger = CostLedger()
        stage = TargetsEnrichment()
        cfg = _make_config(chembl_snapshot_path=targets_snapshot)
        stage.is_available(cfg)
        stage.run(table, ledger=ledger)

        assert ledger.coverage["targets"] == (1, 3)

    def test_does_not_mutate_cache_key_inputs(
        self, targets_snapshot, sample_candidate
    ):
        """Enrichment must not touch drug_name, drug_name_raw, indication,
        highest_phase, or candidate_id — the fields KnowledgeCache keys
        derive from."""
        cand = replace(sample_candidate, drug_name_raw="Aspirin")
        before = (
            cand.drug_name,
            cand.drug_name_raw,
            cand.indication,
            cand.highest_phase,
            cand.candidate_id,
        )

        stage = TargetsEnrichment()
        cfg = _make_config(chembl_snapshot_path=targets_snapshot)
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
        assert cand.drug_targets

    def test_reads_chembl_release_from_pragma(self, tmp_path, sample_candidate):
        """Snapshot's PRAGMA user_version is surfaced as the ChEMBL release."""
        path = tmp_path / "chembl_targets.sqlite"
        _make_snapshot(
            path,
            rows=[
                ("Aspirin", "pref_name", None, "CHEMBL25", "CHEMBL221",
                 "COX-1", "SINGLE PROTEIN", "P23219", "INHIBITOR"),
            ],
            user_version=42,
        )
        stage = TargetsEnrichment()
        cfg = _make_config(chembl_snapshot_path=path)
        stage.is_available(cfg)
        stage.run(
            CandidateTable(
                candidates=[replace(sample_candidate, drug_name_raw="Aspirin")]
            ),
            ledger=CostLedger(),
        )
        assert stage._release == 42
