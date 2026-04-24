"""Tests for pipeline.enrichment.chembl_smiles.ChemblSmilesEnrichment.

Exercises the ChEMBL SMILES fallback: fills `Candidate.smiles` from the
ChEMBL targets snapshot's `canonical_smiles` column *only* when the
DrugBank stage left it empty. The lookup is name-keyed — same normalizer
(`canonicalize_drug_name`) as `TargetsEnrichment` — so a candidate with
`drugbank_id=None` can still get a SMILES string as long as its raw name
normalizes to one ChEMBL knows about. This is the headline invariant for
biologics (peptides, antibodies) that DrugBank's `canonical-smiles` is
missing.
"""

from __future__ import annotations

import sqlite3
from dataclasses import replace
from pathlib import Path

import pytest

from pipeline.drugbank_norm import canonicalize_drug_name
from pipeline.enrichment.chembl_smiles import ChemblSmilesEnrichment
from pipeline.models import CandidateTable
from utils.tiered_router import CostLedger


# Tuple layout (query_norm derived from source_name by the helper):
# (source_name, source_kind, syn_type, chembl_id, target_chembl_id,
#  target_pref_name, target_type, uniprot_accession, action_type,
#  canonical_smiles)
SnapshotRow = tuple


def _make_snapshot(
    path: Path,
    rows: list[SnapshotRow],
    user_version: int = 35,
    *,
    include_smiles_column: bool = True,
) -> None:
    """Build an on-disk ChEMBL-targets snapshot for SMILES tests.

    Mirrors the production schema produced by
    scripts/build_chembl_targets_snapshot.py. Pass
    ``include_smiles_column=False`` to exercise the missing-column
    graceful-degrade path (the legacy 10-column schema).
    """
    conn = sqlite3.connect(path)
    try:
        if include_smiles_column:
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
                    action_type         TEXT,
                    canonical_smiles    TEXT
                )
                """
            )
            expanded = [
                (canonicalize_drug_name(row[0]), *row) for row in rows
            ]
            conn.executemany(
                "INSERT INTO name_targets VALUES "
                "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                expanded,
            )
        else:
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
            # Drop the trailing canonical_smiles column for the legacy
            # schema path.
            expanded = [
                (canonicalize_drug_name(row[0]), *row[:9]) for row in rows
            ]
            conn.executemany(
                "INSERT INTO name_targets VALUES "
                "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                expanded,
            )
        conn.execute("CREATE INDEX idx_query_norm ON name_targets(query_norm)")
        conn.execute(f"PRAGMA user_version = {user_version}")
        conn.commit()
    finally:
        conn.close()


_ASPIRIN_SMILES = "CC(=O)Oc1ccccc1C(=O)O"
_SEMAGLUTIDE_SMILES_STUB = "[Fake][Semaglutide][SMILES]"
_ALT_ASPIRIN_SMILES = "CCCC-different-string"


@pytest.fixture
def smiles_snapshot(tmp_path: Path) -> Path:
    """Covers:
    - Aspirin (pref_name, has SMILES) — small molecule fill case.
    - Semaglutide (pref_name, has SMILES) — biologic fill case.
    - Metformin (pref_name, no SMILES) — row exists but SMILES null,
      must not overwrite anything and must not appear in the lookup.
    """
    path = tmp_path / "chembl_targets.sqlite"
    _make_snapshot(
        path,
        rows=[
            ("Aspirin", "pref_name", None, "CHEMBL25", "CHEMBL221",
             "Cyclooxygenase-1", "SINGLE PROTEIN", "P23219", "INHIBITOR",
             _ASPIRIN_SMILES),
            ("Semaglutide", "pref_name", None, "CHEMBL4297516", "CHEMBL1784",
             "Glucagon-like peptide 1 receptor", "SINGLE PROTEIN",
             "P43220", "AGONIST", _SEMAGLUTIDE_SMILES_STUB),
            ("Metformin", "pref_name", None, "CHEMBL1431", "CHEMBL2107",
             "AMP-activated protein kinase", "PROTEIN COMPLEX",
             "Q13131|Q8N2F8", "ACTIVATOR", None),
        ],
    )
    return path


def _make_config(**overrides):
    class _Cfg:
        enable_chembl_smiles = True
        chembl_snapshot_path: Path | None = None

    cfg = _Cfg()
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


# ---------------------------------------------------------------------------
# is_available gating
# ---------------------------------------------------------------------------

class TestIsAvailable:
    def test_disabled_flag_returns_false(self, smiles_snapshot):
        cfg = _make_config(
            enable_chembl_smiles=False, chembl_snapshot_path=smiles_snapshot
        )
        assert ChemblSmilesEnrichment().is_available(cfg) is False

    def test_missing_snapshot_returns_false(self, tmp_path):
        cfg = _make_config(chembl_snapshot_path=tmp_path / "nope.sqlite")
        assert ChemblSmilesEnrichment().is_available(cfg) is False

    def test_no_snapshot_path_returns_false(self):
        cfg = _make_config(chembl_snapshot_path=None)
        assert ChemblSmilesEnrichment().is_available(cfg) is False


# ---------------------------------------------------------------------------
# run() behavior
# ---------------------------------------------------------------------------

class TestRun:
    def test_fills_missing_smiles(self, smiles_snapshot, sample_candidate):
        cand = replace(sample_candidate, drug_name_raw="Aspirin", smiles=None)
        table = CandidateTable(candidates=[cand])

        stage = ChemblSmilesEnrichment()
        cfg = _make_config(chembl_snapshot_path=smiles_snapshot)
        assert stage.is_available(cfg) is True
        stage.run(table, ledger=CostLedger())

        assert cand.smiles == _ASPIRIN_SMILES

    def test_does_not_overwrite_drugbank(self, smiles_snapshot, sample_candidate):
        """Headline DrugBank-first invariant: a candidate that already
        has a SMILES string (from the DrugBank stage) is left untouched
        even if ChEMBL has a different string for the same name."""
        cand = replace(
            sample_candidate,
            drug_name_raw="Aspirin",
            smiles=_ALT_ASPIRIN_SMILES,
        )
        table = CandidateTable(candidates=[cand])

        stage = ChemblSmilesEnrichment()
        cfg = _make_config(chembl_snapshot_path=smiles_snapshot)
        stage.is_available(cfg)
        stage.run(table, ledger=CostLedger())

        assert cand.smiles == _ALT_ASPIRIN_SMILES

    def test_empty_name_passthrough(self, smiles_snapshot, sample_candidate):
        cand = replace(sample_candidate, drug_name_raw="", smiles=None)
        table = CandidateTable(candidates=[cand])

        stage = ChemblSmilesEnrichment()
        cfg = _make_config(chembl_snapshot_path=smiles_snapshot)
        stage.is_available(cfg)
        stage.run(table, ledger=CostLedger())

        assert cand.smiles is None

    def test_unknown_name_passthrough(self, smiles_snapshot, sample_candidate):
        cand = replace(
            sample_candidate, drug_name_raw="NovelCompoundXYZ", smiles=None
        )
        table = CandidateTable(candidates=[cand])

        stage = ChemblSmilesEnrichment()
        cfg = _make_config(chembl_snapshot_path=smiles_snapshot)
        stage.is_available(cfg)
        stage.run(table, ledger=CostLedger())

        assert cand.smiles is None

    def test_null_smiles_in_snapshot_does_not_overwrite(
        self, smiles_snapshot, sample_candidate
    ):
        """Metformin's row has canonical_smiles=NULL. A candidate that
        matches Metformin should end up with its original SMILES
        untouched — the stage must not set it to None or to the empty
        string."""
        cand = replace(
            sample_candidate,
            drug_name_raw="Metformin",
            smiles="CN(C)C(=N)NC(N)=N",
        )
        table = CandidateTable(candidates=[cand])

        stage = ChemblSmilesEnrichment()
        cfg = _make_config(chembl_snapshot_path=smiles_snapshot)
        stage.is_available(cfg)
        stage.run(table, ledger=CostLedger())

        assert cand.smiles == "CN(C)C(=N)NC(N)=N"

    def test_missing_smiles_column_graceful(
        self, tmp_path, sample_candidate
    ):
        """Old snapshot (pre-SMILES schema): stage must load cleanly and
        record zero coverage rather than raise."""
        path = tmp_path / "legacy_targets.sqlite"
        _make_snapshot(
            path,
            rows=[
                ("Aspirin", "pref_name", None, "CHEMBL25", "CHEMBL221",
                 "COX-1", "SINGLE PROTEIN", "P23219", "INHIBITOR", None),
            ],
            include_smiles_column=False,
        )

        cand = replace(sample_candidate, drug_name_raw="Aspirin", smiles=None)
        table = CandidateTable(candidates=[cand])

        ledger = CostLedger()
        stage = ChemblSmilesEnrichment()
        cfg = _make_config(chembl_snapshot_path=path)
        assert stage.is_available(cfg) is True
        stage.run(table, ledger=ledger)

        assert cand.smiles is None
        assert stage._has_smiles_column is False
        assert ledger.coverage["smiles_chembl"] == (0, 1)
        assert ledger.coverage["smiles_combined"] == (0, 1)

    def test_biologic_with_no_drugbank_still_gets_chembl(
        self, smiles_snapshot, sample_candidate
    ):
        """Headline invariant: a candidate with `drugbank_id=None` and
        no pre-existing SMILES (e.g. a biologic DrugBank couldn't supply
        a `canonical-smiles` for) gets its SMILES from ChEMBL when the
        normalized name hits the snapshot."""
        cand = replace(
            sample_candidate,
            drug_name_raw="Semaglutide",
            smiles=None,
            drugbank_id=None,
        )
        table = CandidateTable(candidates=[cand])

        stage = ChemblSmilesEnrichment()
        cfg = _make_config(chembl_snapshot_path=smiles_snapshot)
        stage.is_available(cfg)
        stage.run(table, ledger=CostLedger())

        assert cand.drugbank_id is None
        assert cand.smiles == _SEMAGLUTIDE_SMILES_STUB

    def test_records_smiles_chembl_and_combined_coverage(
        self, smiles_snapshot, sample_candidate
    ):
        """Three candidates:
        - c1: DrugBank already filled SMILES → counted in combined only.
        - c2: DrugBank empty, ChEMBL fills → counted in chembl + combined.
        - c3: DrugBank empty, name unknown → counted in neither.
        """
        drugbank_filled = replace(
            sample_candidate,
            candidate_id="c1",
            drug_name_raw="Aspirin",
            smiles="CC(=O)Oc1ccccc1C(=O)O-drugbank",
        )
        chembl_fills = replace(
            sample_candidate,
            candidate_id="c2",
            drug_name_raw="Semaglutide",
            smiles=None,
        )
        never_filled = replace(
            sample_candidate,
            candidate_id="c3",
            drug_name_raw="NovelCompoundXYZ",
            smiles=None,
        )
        table = CandidateTable(
            candidates=[drugbank_filled, chembl_fills, never_filled]
        )

        ledger = CostLedger()
        stage = ChemblSmilesEnrichment()
        cfg = _make_config(chembl_snapshot_path=smiles_snapshot)
        stage.is_available(cfg)
        stage.run(table, ledger=ledger)

        assert ledger.coverage["smiles_chembl"] == (1, 3)
        assert ledger.coverage["smiles_combined"] == (2, 3)

    def test_does_not_mutate_cache_key_inputs(
        self, smiles_snapshot, sample_candidate
    ):
        """Enrichment must not touch drug_name, drug_name_raw,
        indication, highest_phase, or candidate_id — the fields
        KnowledgeCache keys derive from."""
        cand = replace(
            sample_candidate, drug_name_raw="Semaglutide", smiles=None
        )
        before = (
            cand.drug_name,
            cand.drug_name_raw,
            cand.indication,
            cand.highest_phase,
            cand.candidate_id,
        )

        stage = ChemblSmilesEnrichment()
        cfg = _make_config(chembl_snapshot_path=smiles_snapshot)
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
        assert cand.smiles == _SEMAGLUTIDE_SMILES_STUB

    def test_reads_chembl_release_from_pragma(self, tmp_path, sample_candidate):
        """Snapshot's PRAGMA user_version is surfaced as the ChEMBL release."""
        path = tmp_path / "chembl_targets.sqlite"
        _make_snapshot(
            path,
            rows=[
                ("Aspirin", "pref_name", None, "CHEMBL25", "CHEMBL221",
                 "COX-1", "SINGLE PROTEIN", "P23219", "INHIBITOR",
                 _ASPIRIN_SMILES),
            ],
            user_version=42,
        )
        stage = ChemblSmilesEnrichment()
        cfg = _make_config(chembl_snapshot_path=path)
        stage.is_available(cfg)
        stage.run(
            CandidateTable(
                candidates=[
                    replace(sample_candidate, drug_name_raw="Aspirin", smiles=None)
                ]
            ),
            ledger=CostLedger(),
        )
        assert stage._release == 42
