"""Tests for the ex-ante nearest-approved-drug similarity feature groups."""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.features.nn_similarity import (  # noqa: E402
    MolformerNNGroup,
    TanimotoNNGroup,
    _ApprovedPool,
    _to_ordinal,
)
from model.features import FEATURE_GROUPS  # noqa: E402


_ECFP4_BITS = 2048
_EMB_DIM = 768


def _bits(on, n=_ECFP4_BITS) -> np.ndarray:
    a = np.zeros(n, dtype=np.int8)
    a[list(on)] = 1
    return a


def _emb(seed: int, dim: int = _EMB_DIM) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(dim).astype(np.float32)


def _approved_row(*, ecfp4=None, embedding=None, dt: date, outcome: str = "Approved"):
    return {
        "ecfp4": ecfp4,
        "embedding": embedding,
        "earliest_start_date": dt,
        "outcome": outcome,
    }


# ---------------------------------------------------------------------------
# Pool / date filter
# ---------------------------------------------------------------------------

def test_to_ordinal_handles_none_and_dates():
    assert _to_ordinal(None) is None
    assert _to_ordinal(float("nan")) is None
    assert _to_ordinal(date(2020, 1, 1)) == date(2020, 1, 1).toordinal()


def test_pool_strict_date_filter():
    """A query at date X excludes pool entries at X (strict <)."""
    rows = [
        _approved_row(ecfp4=_bits([0, 1]), dt=date(2010, 1, 1)),
        _approved_row(ecfp4=_bits([0, 2]), dt=date(2012, 1, 1)),
        _approved_row(ecfp4=_bits([0, 3]), dt=date(2015, 1, 1)),
    ]
    train_df = pd.DataFrame(rows)
    g = TanimotoNNGroup()
    g.fit(train_df)
    assert len(g._pool) == 3

    test_df = pd.DataFrame(
        [
            {"ecfp4": _bits([0, 1]), "earliest_start_date": date(2009, 1, 1)},
            {"ecfp4": _bits([0, 2]), "earliest_start_date": date(2012, 1, 1)},
            {"ecfp4": _bits([0, 3]), "earliest_start_date": date(2020, 1, 1)},
        ]
    )
    out = g.transform(test_df)
    # 2009: pool empty -> missing
    assert out[0, 1] == 1.0 and out[0, 0] == 0.0
    # 2012: pool = [2010 only], identical fingerprints would give 1.0
    # Query is [0,2], pool[0] is [0,1]. Tanimoto = 1/3.
    assert out[1, 1] == 0.0
    assert out[1, 0] == pytest.approx(1.0 / 3.0)
    # 2020: pool = all 3 (2010 [0,1], 2012 [0,2], 2015 [0,3]); query [0,3] matches 2015 exactly
    assert out[2, 1] == 0.0
    assert out[2, 0] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Tanimoto group
# ---------------------------------------------------------------------------

def test_tanimoto_nn_max_similarity():
    rows = [
        _approved_row(ecfp4=_bits(range(0, 100)), dt=date(2010, 1, 1)),
        _approved_row(ecfp4=_bits(range(0, 50)), dt=date(2011, 1, 1)),
        _approved_row(ecfp4=_bits(range(500, 600)), dt=date(2012, 1, 1)),
    ]
    train_df = pd.DataFrame(rows)
    g = TanimotoNNGroup()
    g.fit(train_df)

    # Query identical to first approved -> max sim should be 1.0
    test_df = pd.DataFrame(
        [{"ecfp4": _bits(range(0, 100)), "earliest_start_date": date(2020, 1, 1)}]
    )
    out = g.transform(test_df)
    assert out[0, 0] == pytest.approx(1.0)
    assert out[0, 1] == 0.0


def test_tanimoto_nn_missing_when_no_fp():
    train_df = pd.DataFrame(
        [_approved_row(ecfp4=_bits([0, 1, 2]), dt=date(2010, 1, 1))]
    )
    g = TanimotoNNGroup()
    g.fit(train_df)

    test_df = pd.DataFrame(
        [{"ecfp4": None, "earliest_start_date": date(2020, 1, 1)}]
    )
    out = g.transform(test_df)
    assert out[0, 0] == 0.0
    assert out[0, 1] == 1.0


def test_tanimoto_nn_missing_when_pool_empty():
    train_df = pd.DataFrame(
        [_approved_row(ecfp4=_bits([0, 1]), dt=date(2010, 1, 1))]
    )
    g = TanimotoNNGroup()
    g.fit(train_df)

    # Query date precedes all pool dates
    test_df = pd.DataFrame(
        [{"ecfp4": _bits([0, 1]), "earliest_start_date": date(2008, 1, 1)}]
    )
    out = g.transform(test_df)
    assert out[0, 1] == 1.0


def test_pool_only_approved_rows():
    """Pool excludes Failed and Ongoing outcomes."""
    rows = [
        _approved_row(ecfp4=_bits([0, 1]), dt=date(2010, 1, 1), outcome="Approved"),
        _approved_row(ecfp4=_bits([0, 2]), dt=date(2010, 1, 1), outcome="Failed Phase 3"),
        _approved_row(ecfp4=_bits([0, 3]), dt=date(2010, 1, 1), outcome="Ongoing"),
        _approved_row(ecfp4=_bits([0, 4]), dt=date(2010, 1, 1), outcome="Commercialized"),
    ]
    train_df = pd.DataFrame(rows)
    g = TanimotoNNGroup()
    g.fit(train_df)
    # Approved + Commercialized = 2 entries
    assert len(g._pool) == 2


def test_feature_names_and_shape():
    train_df = pd.DataFrame(
        [_approved_row(ecfp4=_bits([0, 1]), dt=date(2010, 1, 1))]
    )
    g = TanimotoNNGroup()
    g.fit(train_df)
    test_df = pd.DataFrame(
        [
            {"ecfp4": _bits([0, 1]), "earliest_start_date": date(2020, 1, 1)},
            {"ecfp4": _bits([0, 2]), "earliest_start_date": date(2020, 1, 1)},
        ]
    )
    out = g.transform(test_df)
    assert out.shape == (2, 2)
    assert g.feature_names() == ["tanimoto_nn_similarity", "tanimoto_nn_missing"]


# ---------------------------------------------------------------------------
# MolFormer group
# ---------------------------------------------------------------------------

def test_molformer_nn_cosine():
    e1 = _emb(seed=1)
    e2 = _emb(seed=2)
    e3 = _emb(seed=3)
    train_df = pd.DataFrame(
        [
            _approved_row(embedding=e1, dt=date(2010, 1, 1)),
            _approved_row(embedding=e2, dt=date(2011, 1, 1)),
            _approved_row(embedding=e3, dt=date(2012, 1, 1)),
        ]
    )
    g = MolformerNNGroup()
    g.fit(train_df)

    # Query identical to e2 → max cosine should be 1.0
    test_df = pd.DataFrame(
        [{"embedding": e2.copy(), "earliest_start_date": date(2020, 1, 1)}]
    )
    out = g.transform(test_df)
    assert out[0, 0] == pytest.approx(1.0, abs=1e-5)
    assert out[0, 1] == 0.0


def test_molformer_nn_max_against_numpy_reference():
    e1 = _emb(seed=10)
    e2 = _emb(seed=11)
    e3 = _emb(seed=12)
    q = _emb(seed=99)
    train_df = pd.DataFrame(
        [
            _approved_row(embedding=e1, dt=date(2010, 1, 1)),
            _approved_row(embedding=e2, dt=date(2011, 1, 1)),
            _approved_row(embedding=e3, dt=date(2012, 1, 1)),
        ]
    )
    g = MolformerNNGroup()
    g.fit(train_df)

    test_df = pd.DataFrame(
        [{"embedding": q, "earliest_start_date": date(2020, 1, 1)}]
    )
    out = g.transform(test_df)

    qn = q / np.linalg.norm(q)
    expected = max(
        float(np.dot(e1 / np.linalg.norm(e1), qn)),
        float(np.dot(e2 / np.linalg.norm(e2), qn)),
        float(np.dot(e3 / np.linalg.norm(e3), qn)),
    )
    assert out[0, 0] == pytest.approx(expected, abs=1e-5)


def test_molformer_nn_missing_when_no_embedding():
    train_df = pd.DataFrame(
        [_approved_row(embedding=_emb(0), dt=date(2010, 1, 1))]
    )
    g = MolformerNNGroup()
    g.fit(train_df)

    test_df = pd.DataFrame(
        [{"embedding": None, "earliest_start_date": date(2020, 1, 1)}]
    )
    out = g.transform(test_df)
    assert out[0, 0] == 0.0
    assert out[0, 1] == 1.0


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def test_groups_registered():
    assert "tanimoto_nn" in FEATURE_GROUPS
    assert "molformer_nn" in FEATURE_GROUPS


# ---------------------------------------------------------------------------
# End-to-end through train_one_run
# ---------------------------------------------------------------------------

def _fake_modeling_frame(n: int = 80) -> pd.DataFrame:
    """A frame shaped like `data_mod.build_modeling_frame` output."""
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n):
        on = sorted(rng.choice(_ECFP4_BITS, size=20, replace=False).tolist())
        rows.append(
            {
                "candidate_id": f"c{i}",
                "outcome": "Approved" if i % 2 == 0 else "Failed Phase 3",
                "earliest_start_date": date(2018 if i < n // 2 else 2020, 6, 1),
                "icd10_codes": np.array([], dtype=object),
                "drug_targets": np.array([f"P{i % 4}"], dtype=object),
                "disease_area": "oncology",
                "mesh_condition_tree_numbers": np.array([], dtype=object),
                "reactome_pathway_ids": np.array([], dtype=object),
                "reactome_n_pathways": 0,
                "ecfp4": _bits(on),
                "maccs": np.zeros(167, dtype=np.int8),
                "embedding": rng.standard_normal(_EMB_DIM).astype(np.float32),
                "y": np.int8(1 if i % 2 == 0 else 0),
            }
        )
    return pd.DataFrame(rows)


def test_full_pipeline_through_train_one_run(monkeypatch):
    from model import data as data_mod
    from model.config import FeatureConfig, ModelingConfig
    from model.train import train_one_run

    df = _fake_modeling_frame(n=80)
    monkeypatch.setattr(data_mod, "build_modeling_frame", lambda cfg: df)

    cfg = ModelingConfig(
        time_split_year=2019,
        time_split_column="earliest_start_date",
        seed=0,
        inner_val_size=0.25,
        features=FeatureConfig(enabled=("tanimoto_nn", "molformer_nn")),
    )
    result = train_one_run(cfg)
    assert set(result.groups) == {"tanimoto_nn", "molformer_nn"}
    assert result.n_features == 4  # 2 + 2
    assert "tanimoto_nn_similarity" in result.feature_names
    assert "molformer_nn_similarity" in result.feature_names
    assert np.isfinite(result.metrics["roc_auc"]) or np.isnan(result.metrics["roc_auc"])
