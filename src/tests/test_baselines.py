"""Tests for the baselines module."""

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

from model.baselines import ALL_BASELINES, BASELINES, build_baseline  # noqa: E402
from model.baselines.stratum import StratumBaseline, _first_code  # noqa: E402
from model.baselines.tanimoto import TanimotoBaseline  # noqa: E402


# ---------------------------------------------------------------------------
# Stratum baseline
# ---------------------------------------------------------------------------

def _make_train_df_for_stratum() -> tuple[pd.DataFrame, np.ndarray]:
    """3 strata: K76 has n=15 (12 pos), C50 has n=5 (5 pos), Z99 has n=12 (0 pos).
    Plus a chapter Q with n=11 (5 pos) to test fallback. One row has no code."""
    rows = []
    y = []
    for _ in range(15):
        rows.append({"icd10_codes": np.array(["K76.9"], dtype=object)})
    y.extend([1] * 12 + [0] * 3)

    for _ in range(5):
        rows.append({"icd10_codes": np.array(["C50.1"], dtype=object)})
    y.extend([1] * 5)

    for _ in range(12):
        rows.append({"icd10_codes": np.array(["Z99.0"], dtype=object)})
    y.extend([0] * 12)

    for _ in range(11):
        rows.append({"icd10_codes": np.array(["Q44.6"], dtype=object)})
    y.extend([1] * 5 + [0] * 6)

    rows.append({"icd10_codes": np.array([], dtype=object)})
    y.append(1)

    return pd.DataFrame(rows), np.asarray(y, dtype=int)


def test_first_code_handles_empty_and_none():
    assert _first_code(None) is None
    assert _first_code(np.array([], dtype=object)) is None
    assert _first_code(np.array(["A12.3"], dtype=object)) == "A12.3"
    assert _first_code(float("nan")) is None


def test_stratum_fallback_chain():
    train_df, y = _make_train_df_for_stratum()
    bl = StratumBaseline()
    bl.fit(train_df, y)

    test_df = pd.DataFrame(
        {
            "icd10_codes": [
                np.array(["K76.0"], dtype=object),  # K76 has n=15 -> 12/15 = 0.8
                np.array(["C50.5"], dtype=object),  # C50 has n=5 -> chapter C? n=5 too
                np.array(["Q44.6"], dtype=object),  # Q44 has n=11 -> 5/11
                np.array([], dtype=object),         # no code -> global
            ]
        }
    )
    pred = bl.predict_proba(test_df)
    assert pred[0] == pytest.approx(12 / 15)
    assert pred[1] == pytest.approx(float(np.mean(y)))  # both stratum and chapter < 10
    assert pred[2] == pytest.approx(5 / 11)
    assert pred[3] == pytest.approx(float(np.mean(y)))


def test_stratum_chapter_fallback_when_stratum_thin():
    """If stratum n < 10 but chapter n >= 10, use chapter."""
    rows = []
    y = []
    for code in ["F32.0"] * 12:
        rows.append({"icd10_codes": np.array([code], dtype=object)})
    y.extend([1] * 8 + [0] * 4)
    for code in ["F33.0"] * 3:
        rows.append({"icd10_codes": np.array([code], dtype=object)})
    y.extend([1, 0, 0])

    train_df = pd.DataFrame(rows)
    y_arr = np.asarray(y, dtype=int)
    bl = StratumBaseline()
    bl.fit(train_df, y_arr)

    test_df = pd.DataFrame({"icd10_codes": [np.array(["F40.0"], dtype=object)]})
    pred = bl.predict_proba(test_df)
    chapter_rate = float(np.mean(y_arr))  # all 15 rows are F-chapter
    assert pred[0] == pytest.approx(chapter_rate)
    md = bl.metadata()
    assert md["n_test_chapter_fallback"] == 1


# ---------------------------------------------------------------------------
# Tanimoto baseline
# ---------------------------------------------------------------------------

def _bits(on_indices, n=2048) -> np.ndarray:
    a = np.zeros(n, dtype=np.int8)
    a[list(on_indices)] = 1
    return a


def test_tanimoto_topk_mean():
    """Six training fingerprints with controlled overlap; one test
    fingerprint identical to train[0]. Top-5 by Tanimoto should be
    train[0..4] and the predicted prob should equal mean of their labels."""
    train_fps = [
        _bits(range(0, 100)),       # identical to test fp
        _bits(range(0, 95)),        # 95% overlap
        _bits(range(0, 90)),
        _bits(range(0, 80)),
        _bits(range(0, 70)),
        _bits(range(500, 600)),     # disjoint
    ]
    train_labels = np.asarray([1, 1, 0, 1, 0, 1], dtype=int)
    train_df = pd.DataFrame(
        {
            "candidate_id": [f"t{i}" for i in range(6)],
            "ecfp4": train_fps,
        }
    )

    test_df = pd.DataFrame(
        {
            "candidate_id": ["x"],
            "ecfp4": [_bits(range(0, 100))],
        }
    )

    bl = TanimotoBaseline(k=5)
    bl.fit(train_df, train_labels)
    pred = bl.predict_proba(test_df)
    expected = float(np.mean(train_labels[:5]))  # top-5 are the first five
    assert pred[0] == pytest.approx(expected)


def test_tanimoto_missing_fingerprint_uses_base_rate():
    train_df = pd.DataFrame(
        {
            "candidate_id": ["a", "b", "c"],
            "ecfp4": [_bits([0, 1]), _bits([0, 2]), _bits([1, 2])],
        }
    )
    train_labels = np.asarray([1, 0, 1], dtype=int)
    test_df = pd.DataFrame(
        {
            "candidate_id": ["x", "y"],
            "ecfp4": [None, _bits([0, 1])],
        }
    )
    bl = TanimotoBaseline(k=2)
    bl.fit(train_df, train_labels)
    pred = bl.predict_proba(test_df)
    base_rate = float(np.mean(train_labels))
    assert pred[0] == pytest.approx(base_rate)
    assert 0.0 <= pred[1] <= 1.0
    md = bl.metadata()
    assert md["n_test_fallback"] == 1
    assert md["n_test_with_fp"] == 1


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def test_registry_contains_lookup_baselines():
    assert "stratum" in BASELINES
    assert "tanimoto" in BASELINES
    assert "target_only" not in BASELINES  # uses train_one_run, not the protocol
    assert "stratum" in ALL_BASELINES
    assert "tanimoto" in ALL_BASELINES
    assert "target_only" in ALL_BASELINES


def test_build_baseline_dispatch():
    assert isinstance(build_baseline("stratum"), StratumBaseline)
    assert isinstance(build_baseline("tanimoto"), TanimotoBaseline)
    with pytest.raises(KeyError):
        build_baseline("nonexistent")


# ---------------------------------------------------------------------------
# Runner end-to-end on a synthetic frame
# ---------------------------------------------------------------------------

def _fake_modeling_frame(n: int = 60) -> pd.DataFrame:
    """Build a frame shaped like `data_mod.build_modeling_frame` output.

    Uses 30 train rows (year 2018) and 30 test rows (year 2020); each row
    gets an ICD-10 code, a small ECFP4 fingerprint, plus a random
    drug_targets list and a constant disease_area to keep the targets
    feature group available without other dependencies.
    """
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n):
        on = sorted(rng.choice(2048, size=20, replace=False).tolist())
        rows.append(
            {
                "candidate_id": f"c{i}",
                "outcome": "Approved" if i % 2 == 0 else "Failed Phase 3",
                "earliest_start_date": date(2018 if i < n // 2 else 2020, 6, 1),
                "icd10_codes": np.array([f"K7{i % 5}.0"], dtype=object),
                "drug_targets": np.array([f"P{i % 4}"], dtype=object),
                "disease_area": "oncology",
                "mesh_condition_tree_numbers": np.array([], dtype=object),
                "reactome_pathway_ids": np.array([], dtype=object),
                "reactome_n_pathways": 0,
                "ecfp4": _bits(on),
                "maccs": np.zeros(167, dtype=np.int8),
                "embedding": np.zeros(768, dtype=np.float32),
                "y": np.int8(1 if i % 2 == 0 else 0),
            }
        )
    return pd.DataFrame(rows)


def test_runner_writes_expected_artifacts(tmp_path: Path, monkeypatch):
    from model.baselines.runner import BaselinesConfig, run_baselines
    from model.config import ModelingConfig

    df = _fake_modeling_frame()

    # Patch build_modeling_frame so the runner doesn't touch real parquet files.
    import model.baselines.runner as runner_mod

    monkeypatch.setattr(runner_mod.data_mod, "build_modeling_frame", lambda cfg: df)

    base = ModelingConfig(
        time_split_year=2019,
        time_split_column="earliest_start_date",
        output_dir=tmp_path,
        seed=0,
    )
    cfg = BaselinesConfig(base=base, baselines=("stratum", "tanimoto"))
    result = run_baselines(cfg)

    assert (tmp_path / "stratum" / "metrics.json").exists()
    assert (tmp_path / "stratum" / "predictions.csv").exists()
    assert (tmp_path / "tanimoto" / "metrics.json").exists()
    assert (tmp_path / "tanimoto" / "predictions.csv").exists()
    assert (tmp_path / "ablation_summary.csv").exists()

    summary = pd.read_csv(tmp_path / "ablation_summary.csv")
    assert set(summary["subset_name"]) == {"stratum", "tanimoto"}
    for col in ("roc_auc", "pr_auc", "f1", "brier", "log_loss"):
        assert col in summary.columns

    metrics = json.loads((tmp_path / "stratum" / "metrics.json").read_text())
    assert metrics["groups"] == ["stratum"]
    assert "baseline" in metrics
    assert metrics["baseline"]["name"] == "stratum"
    assert "global_rate" in metrics["baseline"]["state"]


def test_target_only_uses_only_targets_group(tmp_path: Path, monkeypatch):
    from model.baselines.runner import BaselinesConfig, run_baselines
    from model.config import ModelingConfig

    df = _fake_modeling_frame(n=80)

    import model.baselines.runner as runner_mod

    monkeypatch.setattr(runner_mod.data_mod, "build_modeling_frame", lambda cfg: df)

    base = ModelingConfig(
        time_split_year=2019,
        time_split_column="earliest_start_date",
        output_dir=tmp_path,
        seed=0,
        inner_val_size=0.25,
    )
    cfg = BaselinesConfig(base=base, baselines=("target_only",))
    run_baselines(cfg)

    metrics = json.loads((tmp_path / "target_only" / "metrics.json").read_text())
    assert list(metrics["groups"]) == ["targets"]
    # 4 unique targets in the synthetic data + target_other_count + target_missing
    assert metrics["n_features"] >= 1
    assert (tmp_path / "target_only" / "predictions.csv").exists()
    assert (tmp_path / "target_only" / "feature_importances.csv").exists()
