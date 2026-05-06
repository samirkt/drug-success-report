"""Tests for the standalone ``pipeline.admet`` package.

Most tests use a stub ``ADMETModel`` so they run without the ``admet_ai``
extra (which pulls in PyTorch + Lightning). One end-to-end smoke test
imports ``admet_ai`` for real and is skipped when the package is missing.
"""

from __future__ import annotations

import math
import sqlite3
import threading
from pathlib import Path

import pandas as pd
import pytest

from pipeline.admet import (
    ADMET_COLUMNS,
    AdmetCache,
    AdmetPredictor,
    field_name,
)


def _stub_dataframe(rows: dict[str, dict[str, float]]) -> pd.DataFrame:
    """Mimic the shape of admet_ai.ADMETModel.predict output."""
    return pd.DataFrame.from_dict(rows, orient="index")


class _StubModel:
    """In-memory replacement for admet_ai.ADMETModel."""

    def __init__(self, responses: dict[str, dict[str, float]]):
        self._responses = responses
        self.call_count = 0
        self.last_input: list[str] = []

    def predict(self, smiles: list[str]) -> pd.DataFrame:
        self.call_count += 1
        self.last_input = list(smiles)
        return _stub_dataframe({s: self._responses[s] for s in smiles if s in self._responses})


def _full_row(value: float) -> dict[str, float]:
    """Synthesize a deterministic row covering every ADMET_COLUMNS key."""
    return {col: value + i * 0.01 for i, col in enumerate(ADMET_COLUMNS)}


def test_field_name_normalizes_hyphens():
    assert field_name("NR-AR") == "admet_NR_AR"
    assert field_name("NR-PPAR-gamma") == "admet_NR_PPAR_gamma"
    assert field_name("SR-p53") == "admet_SR_p53"
    assert field_name("hERG") == "admet_hERG"  # no hyphen


def test_predict_populates_all_columns(tmp_path: Path):
    cache = AdmetCache(tmp_path / "cache.db")
    predictor = AdmetPredictor(cache=cache, model_version="test-v1")
    predictor._model = _StubModel({"CCO": _full_row(1.0)})

    out = predictor.predict(["CCO"])

    assert set(out["CCO"].keys()) == set(ADMET_COLUMNS)
    assert all(isinstance(v, float) for v in out["CCO"].values())


def test_predict_uses_cache_on_second_call(tmp_path: Path):
    cache = AdmetCache(tmp_path / "cache.db")
    predictor = AdmetPredictor(cache=cache, model_version="test-v1")
    stub = _StubModel({"CCO": _full_row(1.0)})
    predictor._model = stub

    predictor.predict(["CCO"])
    predictor.predict(["CCO"])

    assert stub.call_count == 1, "cache hit should skip the model"


def test_cache_invalidated_by_model_version_bump(tmp_path: Path):
    db_path = tmp_path / "cache.db"

    cache1 = AdmetCache(db_path)
    p1 = AdmetPredictor(cache=cache1, model_version="v1")
    p1._model = _StubModel({"CCO": _full_row(1.0)})
    p1.predict(["CCO"])

    cache2 = AdmetCache(db_path)
    p2 = AdmetPredictor(cache=cache2, model_version="v2")
    stub2 = _StubModel({"CCO": _full_row(2.0)})
    p2._model = stub2
    p2.predict(["CCO"])

    assert stub2.call_count == 1, "v2 must miss because v1 rows have a different model_version"


def test_malformed_smiles_returns_none_and_is_not_cached(tmp_path: Path):
    db_path = tmp_path / "cache.db"
    cache = AdmetCache(db_path)
    predictor = AdmetPredictor(cache=cache, model_version="test-v1")
    # Stub returns an all-NaN row for the malformed SMILES.
    nan_row = {col: float("nan") for col in ADMET_COLUMNS}
    predictor._model = _StubModel({"BAD": nan_row})

    out = predictor.predict(["BAD"])
    assert out["BAD"] is None

    with sqlite3.connect(str(db_path)) as conn:
        count = conn.execute("SELECT COUNT(*) FROM admet_predictions").fetchone()[0]
    assert count == 0, "malformed predictions must not be persisted"


def test_nan_value_within_row_becomes_none(tmp_path: Path):
    cache = AdmetCache(tmp_path / "cache.db")
    predictor = AdmetPredictor(cache=cache, model_version="test-v1")
    row = _full_row(0.5)
    poisoned_col = ADMET_COLUMNS[0]
    row[poisoned_col] = float("nan")
    predictor._model = _StubModel({"CCO": row})

    out = predictor.predict(["CCO"])
    pred = out["CCO"]
    assert pred[poisoned_col] is None
    # Other columns must still round-trip as floats, never coerced to 0.0.
    assert pred[ADMET_COLUMNS[1]] is not None
    assert pred[ADMET_COLUMNS[1]] != 0.0


def test_dedup_smiles_collapses_to_single_predict_call(tmp_path: Path):
    cache = AdmetCache(tmp_path / "cache.db")
    predictor = AdmetPredictor(cache=cache, model_version="test-v1")
    stub = _StubModel({"CCO": _full_row(1.0)})
    predictor._model = stub

    out = predictor.predict(["CCO", "CCO", "CCO"])

    assert stub.call_count == 1
    assert stub.last_input == ["CCO"]
    assert "CCO" in out


def test_empty_input_returns_empty_dict(tmp_path: Path):
    cache = AdmetCache(tmp_path / "cache.db")
    predictor = AdmetPredictor(cache=cache, model_version="test-v1")
    predictor._model = _StubModel({})

    assert predictor.predict([]) == {}
    assert predictor.predict(["", "  "]) == {}


def test_cache_thread_safety_smoke(tmp_path: Path):
    cache = AdmetCache(tmp_path / "cache.db")
    rows = {f"S{i}": {col: float(i + j) for j, col in enumerate(ADMET_COLUMNS)} for i in range(20)}

    errors: list[BaseException] = []

    def worker(start: int):
        try:
            for i in range(start, start + 10):
                key = f"S{i}"
                cache.put_many({key: rows[key]}, "v1")
                cache.get_many([key], "v1")
        except BaseException as exc:  # pragma: no cover - debugging aid
            errors.append(exc)

    t1 = threading.Thread(target=worker, args=(0,))
    t2 = threading.Thread(target=worker, args=(10,))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert not errors, f"thread errors: {errors}"
    fetched = cache.get_many([f"S{i}" for i in range(20)], "v1")
    assert len(fetched) == 20


def test_real_admet_smoke():
    """End-to-end check against the actual admet_ai library."""
    pytest.importorskip("admet_ai")
    predictor = AdmetPredictor()
    out = predictor.predict(["CCO"])
    assert "CCO" in out
    pred = out["CCO"]
    assert pred is not None
    assert set(pred.keys()) == set(ADMET_COLUMNS)
    # Some columns return ints (e.g. stereo_centers) — predictor coerces all to float.
    assert all(v is None or isinstance(v, float) for v in pred.values())
