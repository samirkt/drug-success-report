"""Tests for the killer-figure NN analog lookup."""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.baselines import BASELINES  # noqa: E402
from model.baselines.killer_figure import KillerFigureBaseline  # noqa: E402
from model.killer_figure import (  # noqa: E402
    KillerFigureRetriever,
    build_adhoc_query,
    format_text_summary,
    smiles_to_ecfp4,
    _jaccard,
)


_ECFP4_BITS = 2048
_EMB_DIM = 768


def _bits(on, n=_ECFP4_BITS) -> np.ndarray:
    a = np.zeros(n, dtype=np.int8)
    a[list(on)] = 1
    return a


def _emb(seed: int, dim: int = _EMB_DIM) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def _row(
    *,
    candidate_id: str,
    drug_name: str = "drug",
    outcome: str = "Approved",
    dt: date,
    ecfp4=None,
    embedding=None,
    drug_targets=(),
    icd10=(),
    mesh=(),
    disease_area=None,
):
    return {
        "candidate_id": candidate_id,
        "drug_name": drug_name,
        "outcome": outcome,
        "earliest_start_date": dt,
        "ecfp4": ecfp4,
        "embedding": embedding,
        "drug_targets": np.asarray(list(drug_targets), dtype=object),
        "icd10_codes": np.asarray(list(icd10), dtype=object),
        "mesh_condition_tree_numbers": np.asarray(list(mesh), dtype=object),
        "disease_area": disease_area,
    }


def _y_from_outcomes(rows: list[dict]) -> np.ndarray:
    return np.asarray(
        [1 if r["outcome"] in ("Approved", "Commercialized") else 0 for r in rows],
        dtype=np.int8,
    )


# ---------------------------------------------------------------------------
# Component-similarity primitives
# ---------------------------------------------------------------------------

def test_jaccard_returns_none_when_either_empty():
    """Empty-on-either-side means missing signal, not zero overlap."""
    assert _jaccard(set(), set()) is None
    assert _jaccard(set(), {"X"}) is None
    assert _jaccard({"X"}, set()) is None
    assert _jaccard({"X"}, {"X"}) == pytest.approx(1.0)
    assert _jaccard({"X", "Y"}, {"Y", "Z"}) == pytest.approx(1.0 / 3.0)


# ---------------------------------------------------------------------------
# Date cutoff
# ---------------------------------------------------------------------------

def test_date_cutoff_strict():
    """Pool entries dated >= query.start are excluded; <= cutoff included."""
    rows = [
        _row(candidate_id=f"c{i}", drug_name=f"d{i}", outcome="Approved",
             dt=date(2010 + i, 1, 1), drug_targets=("P1",), icd10=("C71.9",))
        for i in range(8)
    ]
    df = pd.DataFrame(rows)
    y = _y_from_outcomes(rows)

    retriever = KillerFigureRetriever(min_neighbors=3, k=5)
    retriever.fit(df, y)

    query = _row(candidate_id="q", drug_name="dq",
                 dt=date(2014, 6, 1),
                 drug_targets=("P1",), icd10=("C71.0",))
    rep = retriever.retrieve(query)
    # Pool entries are dated Jan 1 of 2010..2017. Strict-less-than 2014-06-01
    # admits 2010-01..2014-01 = 5 entries; all have shared P1 + chapter C
    # → joint_sim = 1.0 across them; k=5 fills exactly.
    assert not rep.insufficient_prior_art
    assert rep.n_neighbors == 5
    for nb in rep.neighbors:
        assert nb["earliest_start_date"] < "2014-06-01"

    # Tighten the cutoff so only 4 entries are eligible — verifies strict-less-than.
    query_jan = _row(candidate_id="q", drug_name="dq",
                     dt=date(2014, 1, 1),
                     drug_targets=("P1",), icd10=("C71.0",))
    rep_jan = retriever.retrieve(query_jan)
    assert rep_jan.n_neighbors == 4
    for nb in rep_jan.neighbors:
        assert nb["earliest_start_date"] < "2014-01-01"


# ---------------------------------------------------------------------------
# Drug-level dedup
# ---------------------------------------------------------------------------

def test_drug_level_dedup():
    """5 candidate rows under one drug_name collapse to 1 neighbor entry."""
    # 5 rows for drug_X (all approved, same target, distinct ICD codes)
    rows = []
    for i in range(5):
        rows.append(_row(
            candidate_id=f"x{i}",
            drug_name="drug_X",
            outcome="Approved",
            dt=date(2010, 1, 1 + i),
            drug_targets=("P1",),
            icd10=("C71.9",),
        ))
    # 8 rows for distinct other drugs to ensure we can fill k=10
    for i in range(8):
        rows.append(_row(
            candidate_id=f"o{i}",
            drug_name=f"drug_O{i}",
            outcome="Failed Phase 3",
            dt=date(2011, 6, 1),
            drug_targets=("P1",),
            icd10=("C71.9",),
        ))

    df = pd.DataFrame(rows)
    y = _y_from_outcomes(rows)
    retriever = KillerFigureRetriever(min_neighbors=3, k=10)
    retriever.fit(df, y)

    query = _row(candidate_id="q", drug_name="dq",
                 dt=date(2020, 1, 1),
                 drug_targets=("P1",), icd10=("C71.0",))
    rep = retriever.retrieve(query)

    drug_names = [nb["drug_name"] for nb in rep.neighbors]
    assert drug_names.count("drug_X") == 1
    # 1 drug_X + up to 8 distinct drug_O = 9 unique drugs
    assert len(drug_names) == len(set(drug_names))


# ---------------------------------------------------------------------------
# Component re-normalization
# ---------------------------------------------------------------------------

def test_component_renormalization_query_missing_targets():
    """Query with no targets → joint_sim averages molecule + indication only."""
    pool_rows = []
    for i in range(8):
        pool_rows.append(_row(
            candidate_id=f"p{i}",
            drug_name=f"drug_P{i}",
            outcome="Approved" if i % 2 == 0 else "Failed Phase 3",
            dt=date(2010 + i % 4, 1, 1),
            ecfp4=_bits(range(i, i + 50)),
            drug_targets=("P1",),  # pool entries have a target
            icd10=("C71.9",),
        ))
    df = pd.DataFrame(pool_rows)
    y = _y_from_outcomes(pool_rows)
    retriever = KillerFigureRetriever(min_neighbors=3, k=5)
    retriever.fit(df, y)

    query = _row(candidate_id="q", drug_name="dq",
                 dt=date(2020, 1, 1),
                 ecfp4=_bits(range(0, 50)),
                 drug_targets=(),  # query has no targets
                 icd10=("C71.0",))
    rep = retriever.retrieve(query)
    assert rep.n_components_query == 2  # molecule + indication only
    # Each neighbor should have target_sim=None and exactly 2 components used.
    for nb in rep.neighbors:
        assert nb["target_sim"] is None
        assert nb["n_components_used"] == 2


# ---------------------------------------------------------------------------
# Decomposition counts
# ---------------------------------------------------------------------------

def test_decomposition_counts_match_neighbors():
    """shared_targets/shared_icd10_chapters counts == intersection across neighbors."""
    rows = []
    # 6 neighbors share P1 + chapter C; 4 neighbors share P2 + chapter D.
    for i in range(6):
        rows.append(_row(
            candidate_id=f"a{i}",
            drug_name=f"drug_A{i}",
            outcome="Approved",
            dt=date(2010, 1, 1 + i),
            drug_targets=("P1", "P9"),
            icd10=("C71.9",),
            disease_area="Oncology",
        ))
    for i in range(4):
        rows.append(_row(
            candidate_id=f"b{i}",
            drug_name=f"drug_B{i}",
            outcome="Failed Phase 3",
            dt=date(2010, 6, 1 + i),
            drug_targets=("P2",),
            icd10=("D50.0",),
            disease_area="Hematology",
        ))
    df = pd.DataFrame(rows)
    y = _y_from_outcomes(rows)
    retriever = KillerFigureRetriever(min_neighbors=3, k=10)
    retriever.fit(df, y)

    # Query shares P1 + chapter C + disease_area Oncology with the A-cluster.
    query = _row(candidate_id="q", drug_name="dq",
                 dt=date(2020, 1, 1),
                 drug_targets=("P1",),
                 icd10=("C50.1",),
                 disease_area="Oncology")
    rep = retriever.retrieve(query)

    # Among the 10 neighbors, 6 share P1 with the query.
    p1_shared = next((s for s in rep.shared_targets if s["uniprot"] == "P1"), None)
    assert p1_shared is not None
    assert p1_shared["n"] == 6  # only A-cluster shares P1 with query

    # 6 neighbors share chapter C, none share chapter D with this query.
    chap_c = next(
        (c for c in rep.shared_icd10_chapters if c["chapter"] == "C"), None
    )
    assert chap_c is not None
    assert chap_c["n"] == 6

    # 6 neighbors share disease_area Oncology with query.
    onco = next(
        (a for a in rep.shared_disease_areas if a["area"] == "Oncology"), None
    )
    assert onco is not None
    assert onco["n"] == 6


# ---------------------------------------------------------------------------
# Insufficient prior art
# ---------------------------------------------------------------------------

def test_insufficient_prior_art_falls_back_to_stratum():
    """<5 unique drugs in date window → falls back, flags insufficient_prior_art."""
    # Only 2 unique drugs before 2020.
    rows = [
        _row(candidate_id="a", drug_name="drug_A", outcome="Approved",
             dt=date(2010, 1, 1), drug_targets=("P1",), icd10=("C71.9",)),
        _row(candidate_id="b", drug_name="drug_B", outcome="Failed Phase 3",
             dt=date(2011, 1, 1), drug_targets=("P1",), icd10=("C71.9",)),
    ]
    df = pd.DataFrame(rows)
    y = _y_from_outcomes(rows)
    retriever = KillerFigureRetriever(min_neighbors=5, k=10)
    retriever.fit(df, y)

    query = _row(candidate_id="q", drug_name="dq",
                 dt=date(2020, 1, 1),
                 drug_targets=("P1",), icd10=("C71.9",))
    rep = retriever.retrieve(query)
    assert rep.insufficient_prior_art is True
    # When falling back, approval_rate_neighbors == stratum_base_rate.
    assert rep.approval_rate_neighbors == pytest.approx(rep.stratum_base_rate)
    assert rep.n_neighbors == 0
    assert rep.neighbors == []


# ---------------------------------------------------------------------------
# Self-exclusion via candidate_id
# ---------------------------------------------------------------------------

def test_self_match_is_excluded():
    """An identical pool row with the same candidate_id never appears as a neighbor."""
    rows = []
    for i in range(7):
        rows.append(_row(
            candidate_id=f"p{i}",
            drug_name=f"drug_P{i}",
            outcome="Approved" if i % 2 == 0 else "Failed Phase 3",
            dt=date(2010, 1, 1 + i),
            drug_targets=("P1",),
            icd10=("C71.9",),
        ))
    # The query is also in the pool with an earlier date.
    rows.append(_row(
        candidate_id="self",
        drug_name="self_drug",
        outcome="Approved",
        dt=date(2008, 1, 1),
        drug_targets=("P1",),
        icd10=("C71.9",),
    ))
    df = pd.DataFrame(rows)
    y = _y_from_outcomes(rows)
    retriever = KillerFigureRetriever(min_neighbors=3, k=5)
    retriever.fit(df, y)

    query = _row(candidate_id="self", drug_name="self_drug",
                 dt=date(2020, 1, 1),
                 drug_targets=("P1",), icd10=("C71.9",))
    rep = retriever.retrieve(query)
    assert all(nb["candidate_id"] != "self" for nb in rep.neighbors)


# ---------------------------------------------------------------------------
# Ad-hoc query
# ---------------------------------------------------------------------------

def test_smiles_to_ecfp4_round_trip():
    arr = smiles_to_ecfp4("CC(=O)Oc1ccccc1C(=O)O")  # aspirin
    assert arr is not None
    assert arr.shape == (_ECFP4_BITS,)
    assert arr.sum() > 0


def test_smiles_to_ecfp4_invalid_returns_none():
    assert smiles_to_ecfp4("not-a-smiles") is None


def test_build_adhoc_query_yields_retrievable_row():
    rows = []
    for i in range(8):
        rows.append(_row(
            candidate_id=f"p{i}",
            drug_name=f"drug_P{i}",
            outcome="Approved" if i % 3 == 0 else "Failed Phase 3",
            dt=date(2010, 1, 1 + i),
            ecfp4=_bits(range(i, i + 50)),
            drug_targets=("P23219",),
            icd10=("I20.9",),
        ))
    df = pd.DataFrame(rows)
    y = _y_from_outcomes(rows)
    retriever = KillerFigureRetriever(min_neighbors=3, k=5)
    retriever.fit(df, y)

    q = build_adhoc_query(
        smiles="CC(=O)Oc1ccccc1C(=O)O",  # aspirin
        targets=["P23219"],
        icd10=["I20.9"],
        disease_area=None,
        start_date=date(2020, 1, 1),
    )
    rep = retriever.retrieve(q)
    assert not rep.insufficient_prior_art
    assert rep.candidate_id == "<adhoc>"
    assert rep.n_components_query == 3
    # Aspirin ECFP4 is unrelated to the synthetic bit ranges → mol_sim is
    # finite but small; what we care about is that retrieval ran.
    assert rep.n_neighbors > 0


# ---------------------------------------------------------------------------
# Baseline registration + end-to-end shape
# ---------------------------------------------------------------------------

def test_killer_figure_registered_as_baseline():
    assert "killer_figure" in BASELINES


def test_baseline_predict_proba_shape_and_artifacts(tmp_path):
    rng = np.random.default_rng(0)
    rows = []
    n = 40
    for i in range(n):
        on = sorted(rng.choice(_ECFP4_BITS, size=20, replace=False).tolist())
        rows.append({
            "candidate_id": f"c{i}",
            "drug_name": f"drug_{i}",  # unique drug names — no collisions
            "outcome": "Approved" if i % 2 == 0 else "Failed Phase 3",
            "earliest_start_date": date(2015 if i < n // 2 else 2020, 6, 1),
            "ecfp4": _bits(on),
            "embedding": _emb(seed=i + 1),
            "drug_targets": np.asarray([f"P{i % 5}"], dtype=object),
            "icd10_codes": np.asarray([f"C{(i % 10):02d}.9"], dtype=object),
            "mesh_condition_tree_numbers": np.asarray([], dtype=object),
            "disease_area": "Oncology" if i % 3 == 0 else "Other",
        })
    df = pd.DataFrame(rows)
    y = _y_from_outcomes(rows)

    train_mask = np.array(
        [r["earliest_start_date"].year <= 2018 for r in rows]
    )
    train_df = df[train_mask].reset_index(drop=True)
    test_df = df[~train_mask].reset_index(drop=True)
    y_train = y[train_mask]

    bl = KillerFigureBaseline(k=5, min_neighbors=3)
    bl.fit(train_df, y_train)
    proba = bl.predict_proba(test_df)
    assert proba.shape == (len(test_df),)
    assert np.all((proba >= 0) & (proba <= 1))

    bl.write_artifacts(tmp_path)
    jsonl_path = tmp_path / "killer_figure_report.jsonl"
    assert jsonl_path.exists()
    lines = jsonl_path.read_text().strip().split("\n")
    assert len(lines) == len(test_df)
    parsed = [json.loads(line) for line in lines]
    for r in parsed:
        # Each report must have the schema we ship to consumers.
        assert "approval_rate_neighbors" in r
        assert "stratum_base_rate" in r
        assert "pool_base_rate" in r
        assert "neighbors" in r
        assert "shared_targets" in r
        assert "insufficient_prior_art" in r


def test_format_text_summary_renders_without_error():
    """Spot-check the pretty renderer over both the full and fallback paths."""
    rows = [
        _row(candidate_id=f"p{i}", drug_name=f"drug_P{i}",
             outcome="Approved" if i % 2 == 0 else "Failed Phase 3",
             dt=date(2010, 1, 1 + i),
             drug_targets=("P1",), icd10=("C71.9",))
        for i in range(8)
    ]
    df = pd.DataFrame(rows)
    y = _y_from_outcomes(rows)
    retriever = KillerFigureRetriever(min_neighbors=3, k=5)
    retriever.fit(df, y)

    query = _row(candidate_id="q", drug_name="dq",
                 dt=date(2020, 1, 1),
                 drug_targets=("P1",), icd10=("C71.0",))
    rep = retriever.retrieve(query)
    text = format_text_summary(rep)
    assert "Killer-figure report" in text
    assert "approval" in text.lower()

    # And the insufficient-prior-art branch:
    query_early = _row(candidate_id="q", drug_name="dq",
                       dt=date(2009, 1, 1),
                       drug_targets=("P1",), icd10=("C71.0",))
    rep2 = retriever.retrieve(query_early)
    text2 = format_text_summary(rep2)
    assert rep2.insufficient_prior_art
    assert "Insufficient prior art" in text2
