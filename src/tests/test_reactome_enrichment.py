"""Tests for `pipeline.enrichment.reactome_hierarchy` and the
hierarchy-aware diagnostics that `ReactomeEnrichment` writes onto each
candidate.

Uses tiny synthetic Reactome source files so the hand-computed answers
are obvious by inspection.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from pipeline.enrichment.reactome import ReactomeEnrichment
from pipeline.enrichment.reactome_hierarchy import (
    build_hierarchy,
    load_hierarchy,
)
from pipeline.models import CandidateTable
from utils.tiered_router import CostLedger


# Synthetic Homo sapiens hierarchy:
#
#       R-HSA-1 (Metabolism)                  R-HSA-7 (Signal Transduction)
#       │                                     │
#       ├── R-HSA-2 (Metabolism of lipids)    └── R-HSA-8 (Signaling by ERBB)
#       │       │
#       │       └── R-HSA-3 (Arachidonate metabolism)
#       │               │
#       │               └── R-HSA-4 (Synthesis of Prostaglandins)
#       │
#       └── R-HSA-5 (Glucose metabolism)
#
# R-HSA-6 is an orphan (no parent, no child) — a degenerate root.

_PATHWAYS_TSV = "\n".join([
    "R-HSA-1\tMetabolism\tHomo sapiens",
    "R-HSA-2\tMetabolism of lipids\tHomo sapiens",
    "R-HSA-3\tArachidonate metabolism\tHomo sapiens",
    "R-HSA-4\tSynthesis of Prostaglandins\tHomo sapiens",
    "R-HSA-5\tGlucose metabolism\tHomo sapiens",
    "R-HSA-6\tOrphan pathway\tHomo sapiens",
    "R-HSA-7\tSignal Transduction\tHomo sapiens",
    "R-HSA-8\tSignaling by ERBB\tHomo sapiens",
    "R-MMU-99\tMouse pathway\tMus musculus",  # filtered out
    "",
])

_RELATIONS_TSV = "\n".join([
    "R-HSA-1\tR-HSA-2",
    "R-HSA-1\tR-HSA-5",
    "R-HSA-2\tR-HSA-3",
    "R-HSA-3\tR-HSA-4",
    "R-HSA-7\tR-HSA-8",
    "R-MMU-99\tR-MMU-100",  # filtered out
    "",
])

_UNIPROT_TSV = "\n".join([
    # PROT_LIPID hits the deep lipid branch (children + ancestors all
    # appear because the source file is ancestry-expanded by Reactome).
    "PROT_LIPID\tR-HSA-1\thttp://x\tMetabolism\tIEA\tHomo sapiens",
    "PROT_LIPID\tR-HSA-2\thttp://x\tMetabolism of lipids\tIEA\tHomo sapiens",
    "PROT_LIPID\tR-HSA-3\thttp://x\tArachidonate metabolism\tIEA\tHomo sapiens",
    "PROT_LIPID\tR-HSA-4\thttp://x\tSynthesis of Prostaglandins\tIEA\tHomo sapiens",
    # PROT_BROAD hits two distinct top-level branches with leaves.
    "PROT_BROAD\tR-HSA-1\thttp://x\tMetabolism\tIEA\tHomo sapiens",
    "PROT_BROAD\tR-HSA-5\thttp://x\tGlucose metabolism\tIEA\tHomo sapiens",
    "PROT_BROAD\tR-HSA-7\thttp://x\tSignal Transduction\tIEA\tHomo sapiens",
    "PROT_BROAD\tR-HSA-8\thttp://x\tSignaling by ERBB\tIEA\tHomo sapiens",
    # Non-human row (filtered out by ReactomeEnrichment).
    "PROT_MOUSE\tR-MMU-100\thttp://x\tMouse thing\tIEA\tMus musculus",
    "",
])


@pytest.fixture
def reactome_dir(tmp_path: Path) -> Path:
    (tmp_path / "ReactomePathways.txt").write_text(_PATHWAYS_TSV)
    (tmp_path / "ReactomePathwaysRelation.txt").write_text(_RELATIONS_TSV)
    (tmp_path / "UniProt2Reactome_All_Levels.txt").write_text(_UNIPROT_TSV)
    (tmp_path / "reactome_version.txt").write_text("test-1")
    return tmp_path


# ─────────────────────────────────────────────────── hierarchy builder ───


def test_build_hierarchy_shape(reactome_dir: Path) -> None:
    df = build_hierarchy(reactome_dir)
    assert (reactome_dir / "pathway_hierarchy.parquet").exists()
    assert set(df["pathway_id"]) == {f"R-HSA-{i}" for i in range(1, 9)}


def test_hierarchy_depth_and_roots(reactome_dir: Path) -> None:
    h = load_hierarchy(reactome_dir)
    # Roots: 1, 6, 7. Depths from 1: lipids 1->2->3->4 = depths 0,1,2,3.
    assert h["R-HSA-1"]["depth"] == 0
    assert h["R-HSA-2"]["depth"] == 1
    assert h["R-HSA-3"]["depth"] == 2
    assert h["R-HSA-4"]["depth"] == 3
    assert h["R-HSA-6"]["depth"] == 0          # orphan still a root
    assert h["R-HSA-7"]["depth"] == 0
    assert h["R-HSA-8"]["depth"] == 1

    assert h["R-HSA-4"]["top_level_ids"] == ["R-HSA-1"]
    assert h["R-HSA-8"]["top_level_ids"] == ["R-HSA-7"]

    assert h["R-HSA-4"]["is_leaf"] is True
    assert h["R-HSA-1"]["is_leaf"] is False


def test_hierarchy_is_cached(reactome_dir: Path) -> None:
    """Second call should hit the parquet cache (mtime-based)."""
    df1 = build_hierarchy(reactome_dir)
    parquet = reactome_dir / "pathway_hierarchy.parquet"
    mtime = parquet.stat().st_mtime
    df2 = build_hierarchy(reactome_dir)
    # parquet untouched → mtime stable
    assert parquet.stat().st_mtime == mtime
    assert len(df1) == len(df2)


# ───────────────────────────────────────── per-candidate diagnostics ────


def _make_candidate_table(candidates):
    return CandidateTable(candidates=candidates)


def test_enrichment_diagnostics_deep_branch(
    reactome_dir: Path, sample_candidate
) -> None:
    """PROT_LIPID hits one deep branch — n_pathways=4, locally-leaf=1
    (only R-HSA-4 has no child in the set), top-level set = {R-HSA-1},
    mean_depth = (0+1+2+3)/4 = 1.5."""
    cand = replace(sample_candidate, drug_targets=["PROT_LIPID"])
    table = _make_candidate_table([cand])

    stage = ReactomeEnrichment()
    # is_available wires up the data_dir; we bypass the config object by
    # invoking _build_lookup directly with the path set.
    stage._data_dir = reactome_dir
    stage.run(table, ledger=CostLedger())

    out = table.candidates[0]
    assert out.reactome_has_data is True
    assert out.reactome_n_pathways == 4
    assert out.reactome_n_top_level_pathways == 1
    assert out.reactome_n_leaf_pathways == 1
    assert out.reactome_n_leaf_global == 1
    assert out.reactome_n_internal_pathways == 2
    assert out.reactome_mean_depth == pytest.approx(1.5)
    assert out.reactome_max_depth == 3
    assert out.reactome_top_level_pathway_ids == ["R-HSA-1"]
    assert out.reactome_leaf_pathway_ids == ["R-HSA-4"]


def test_enrichment_diagnostics_broad_coverage(
    reactome_dir: Path, sample_candidate
) -> None:
    """PROT_BROAD hits two distinct top-level branches. n_pathways=4
    (R-HSA-1, 5, 7, 8). R-HSA-5 has no child in the set (locally leaf),
    R-HSA-8 has no child in the set (locally leaf) — so two local
    leaves, two top-levels, zero internal."""
    cand = replace(sample_candidate, drug_targets=["PROT_BROAD"])
    table = _make_candidate_table([cand])

    stage = ReactomeEnrichment()
    stage._data_dir = reactome_dir
    stage.run(table, ledger=CostLedger())

    out = table.candidates[0]
    assert out.reactome_n_pathways == 4
    assert out.reactome_n_top_level_pathways == 2
    assert out.reactome_n_leaf_pathways == 2
    assert sorted(out.reactome_top_level_pathway_ids) == ["R-HSA-1", "R-HSA-7"]
    assert sorted(out.reactome_leaf_pathway_ids) == ["R-HSA-5", "R-HSA-8"]
    assert out.reactome_max_depth == 1


def test_enrichment_no_targets_skips_diagnostics(
    reactome_dir: Path, sample_candidate
) -> None:
    cand = replace(sample_candidate, drug_targets=[])
    table = _make_candidate_table([cand])
    stage = ReactomeEnrichment()
    stage._data_dir = reactome_dir
    stage.run(table, ledger=CostLedger())
    out = table.candidates[0]
    assert out.reactome_has_data is False
    assert out.reactome_n_pathways is None
    assert out.reactome_n_top_level_pathways is None
    assert out.reactome_mean_depth is None
