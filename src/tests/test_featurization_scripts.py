"""Smoke tests for scripts/build_molformer_embeddings.py and
scripts/build_fingerprints.py.

Both scripts live outside the test path's importable tree (``scripts/``
is a sibling of ``src/``), so we load them via ``importlib.util`` from
the repo root. Tests skip when their heavy ML deps aren't installed.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"


def _load_script(name: str):
    path = SCRIPTS_DIR / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_candidates_parquet(path: Path) -> None:
    """Three-row mock candidate_detail.parquet: 2 ok, 1 with null canonical."""
    df = pd.DataFrame({
        "candidate_id": ["c1", "c2", "c3"],
        "drug_name": ["DrugA", "DrugB", "DrugC"],
        "smiles": ["CC(=O)O", "CCO", None],
        "smiles_canonical": ["CC(=O)O", "CCO", None],
        "smiles_standardization_status": ["ok", "ok", "empty"],
    })
    df.to_parquet(path, index=False)


# ---------------------------------------------------------------------------
# Fingerprints
# ---------------------------------------------------------------------------

class TestBuildFingerprints:

    def test_writes_parquet_with_expected_schema(self, tmp_path):
        pytest.importorskip("rdkit")
        cands_path = tmp_path / "candidate_detail.parquet"
        out_path = tmp_path / "features" / "fingerprints.parquet"
        _write_candidates_parquet(cands_path)

        mod = _load_script("build_fingerprints.py")
        n = mod.build(candidates_parquet=cands_path, output_parquet=out_path)
        assert n == 2  # c3 dropped (null canonical)

        df = pd.read_parquet(out_path)
        assert list(df["candidate_id"]) == ["c1", "c2"]
        assert {"candidate_id", "ecfp4", "maccs"} <= set(df.columns)
        ecfp4_row = list(df["ecfp4"].iloc[0])
        maccs_row = list(df["maccs"].iloc[0])
        assert len(ecfp4_row) == 2048
        assert len(maccs_row) == 167
        assert all(v in (0, 1) for v in ecfp4_row)
        assert all(v in (0, 1) for v in maccs_row)

    def test_missing_smiles_canonical_column_errors(self, tmp_path):
        pytest.importorskip("rdkit")
        cands_path = tmp_path / "candidate_detail.parquet"
        # Build a parquet WITHOUT smiles_canonical
        pd.DataFrame({"candidate_id": ["c1"], "smiles": ["CCO"]}).to_parquet(
            cands_path, index=False,
        )
        out_path = tmp_path / "fingerprints.parquet"
        mod = _load_script("build_fingerprints.py")
        with pytest.raises(SystemExit):
            mod.build(candidates_parquet=cands_path, output_parquet=out_path)


# ---------------------------------------------------------------------------
# MolFormer-XL embeddings
# ---------------------------------------------------------------------------
# `transformers` + `torch` are heavy and not in CI by default; skip when
# absent. We do not actually load the real MolFormer-XL weights here —
# the test only exercises the input-parsing / output-schema scaffolding,
# which it does by monkeypatching the embedding loop. (Loading the real
# weights downloads ~700MB and runs a 768-d forward pass; out of scope
# for unit tests.)


class TestBuildMolformerEmbeddings:

    def test_writes_parquet_with_expected_schema(self, tmp_path, monkeypatch):
        # Smoke: stub the embedding loop so we don't load real weights.
        torch = pytest.importorskip("torch")  # noqa: F841
        pytest.importorskip("transformers")

        mod = _load_script("build_molformer_embeddings.py")

        cands_path = tmp_path / "candidate_detail.parquet"
        out_path = tmp_path / "features" / "molformer_embeddings.parquet"
        _write_candidates_parquet(cands_path)

        # Replace the heavy bits — model load + per-batch forward pass —
        # so the test stays cheap and offline.
        def _fake_yield(model, tokenizer, smiles_list, batch_size, device):
            for s in smiles_list:
                yield s, [0.0] * 768

        monkeypatch.setattr(mod, "_embed_with_oom_retry", _fake_yield)

        # Patch HF loaders to dummies — we never actually call them on
        # the stubbed loop, but `build` resolves device + calls `from_pretrained`.
        class _Dummy:
            def to(self, *_a, **_k): return self
            def eval(self): return self
            def __call__(self, **_k): return None  # never called via stub
        monkeypatch.setattr(mod, "_resolve_device", lambda d: "cpu")
        from transformers import AutoModel, AutoTokenizer
        monkeypatch.setattr(AutoTokenizer, "from_pretrained",
                            classmethod(lambda cls, *a, **k: _Dummy()))
        monkeypatch.setattr(AutoModel, "from_pretrained",
                            classmethod(lambda cls, *a, **k: _Dummy()))

        n = mod.build(
            candidates_parquet=cands_path,
            output_parquet=out_path,
            batch_size=4,
            device="cpu",
            model_id="dummy/model",
        )
        assert n == 2  # c3 dropped (null canonical)
        df = pd.read_parquet(out_path)
        assert list(df["candidate_id"]) == ["c1", "c2"]
        assert {"candidate_id", "embedding", "model"} <= set(df.columns)
        emb = list(df["embedding"].iloc[0])
        assert len(emb) == 768
        assert (df["model"] == "dummy/model").all()
