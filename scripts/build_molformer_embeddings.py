"""Build MolFormer-XL embeddings for the modeling-eligible candidates.

Reads ``candidate_detail.parquet`` (the per-candidate analytical snapshot
written by the pipeline), filters to rows with a non-null
``smiles_canonical``, and emits a Parquet keyed by ``candidate_id`` with
a 768-dim embedding per row.

The model defaults to ``ibm/MoLFormer-XL-both-10pct``; the script trusts
remote code on load (the model card ships custom Python). Inference
batches at ``--batch-size`` and halves on CUDA OOM until it succeeds or
hits batch=1. ``outputs.pooler_output`` is the mean-pooled
representation per the model card.

Heavy ML deps (transformers, torch) are NOT pulled in by the core
pipeline. Install them via the optional extra:

    pip install -e ".[modeling]"

Usage:

    python scripts/build_molformer_embeddings.py \
        --candidates docs/candidate_detail.parquet \
        --output docs/features/molformer_embeddings.parquet \
        [--batch-size 64] [--device auto|cpu|cuda] \
        [--model ibm/MoLFormer-XL-both-10pct]

Output schema (one row per candidate with non-null canonical SMILES):

    candidate_id : str
    embedding    : list[float]  (length 768)
    model        : str          (constant, for downstream provenance)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger("build_molformer_embeddings")

DEFAULT_MODEL = "ibm/MoLFormer-XL-both-10pct"


def _resolve_device(requested: str) -> str:
    if requested == "auto":
        try:
            import torch
        except ImportError as exc:
            raise SystemExit(
                "torch is required. Install via: pip install -e \".[modeling]\""
            ) from exc
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def _embed_batch(model, tokenizer, smiles_batch: list[str], device: str):
    """Run a forward pass and return pooler_output as a numpy array."""
    import torch
    inputs = tokenizer(
        smiles_batch,
        padding=True,
        truncation=True,
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    # MolFormer-XL exposes a `pooler_output` (mean-pooled) per its model card.
    pooled = outputs.pooler_output
    return pooled.detach().to("cpu", dtype=torch.float32).numpy()


def _embed_with_oom_retry(
    model, tokenizer, smiles_list: list[str], batch_size: int, device: str,
):
    """Yield (smiles, embedding) pairs, halving batch size on CUDA OOM."""
    try:
        import torch
    except ImportError as exc:
        raise SystemExit(
            "torch is required. Install via: pip install -e \".[modeling]\""
        ) from exc

    i = 0
    n = len(smiles_list)
    while i < n:
        chunk = smiles_list[i:i + batch_size]
        try:
            embeddings = _embed_batch(model, tokenizer, chunk, device)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if batch_size <= 1:
                raise RuntimeError(
                    "OOM at batch_size=1 — model does not fit on this GPU."
                )
            new_batch = max(1, batch_size // 2)
            logger.warning(
                "CUDA OOM at batch_size=%d — halving to %d and retrying.",
                batch_size, new_batch,
            )
            batch_size = new_batch
            continue
        for smiles, emb in zip(chunk, embeddings):
            yield smiles, emb.tolist()
        i += len(chunk)
        logger.info("  embedded %d / %d", min(i, n), n)


def build(
    candidates_parquet: Path,
    output_parquet: Path,
    batch_size: int,
    device: str,
    model_id: str,
) -> int:
    try:
        import pandas as pd
    except ImportError as exc:
        raise SystemExit(
            f"Missing dep ({exc.name}). Install with: pip install -e \".[modeling]\""
        ) from exc
    # transformers 5.x removed the `transformers.onnx` submodule, but
    # MolFormer-XL's `trust_remote_code` config imports OnnxConfig from
    # it. Shim it with an empty base class — we don't ONNX-export, so
    # the subclass is only used at definition time, never instantiated.
    import sys as _sys
    import types as _types
    if "transformers.onnx" not in _sys.modules:
        _onnx = _types.ModuleType("transformers.onnx")
        class _OnnxConfigStub:  # noqa: N801 — match transformers naming
            pass
        _onnx.OnnxConfig = _OnnxConfigStub
        _sys.modules["transformers.onnx"] = _onnx
    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            f"Missing dep ({exc.name}). Install with: pip install -e \".[modeling]\""
        ) from exc

    if not candidates_parquet.exists():
        raise SystemExit(f"--candidates {candidates_parquet} does not exist.")

    df = pd.read_parquet(candidates_parquet)
    if "smiles_canonical" not in df.columns:
        raise SystemExit(
            f"{candidates_parquet} has no `smiles_canonical` column. Re-run "
            "the pipeline with smiles standardization enabled."
        )
    rows = df[df["smiles_canonical"].notna()][["candidate_id", "smiles_canonical"]]
    rows = rows.reset_index(drop=True)
    n = len(rows)
    logger.info(
        "Embedding %d / %d candidates with non-null canonical SMILES",
        n, len(df),
    )
    if n == 0:
        # Still write an empty parquet so downstream paths exist.
        output_parquet.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"candidate_id": [], "embedding": [], "model": []}).to_parquet(
            output_parquet, index=False,
        )
        logger.info("No rows to embed; wrote empty parquet to %s", output_parquet)
        return 0

    resolved_device = _resolve_device(device)
    logger.info("Loading %s on %s", model_id, resolved_device)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
    model = model.to(resolved_device)
    model.eval()

    smiles_list = rows["smiles_canonical"].tolist()
    embeddings: list[list[float]] = []
    seen_smiles: list[str] = []
    for smiles, emb in _embed_with_oom_retry(
        model, tokenizer, smiles_list, batch_size, resolved_device,
    ):
        seen_smiles.append(smiles)
        embeddings.append(emb)

    if len(embeddings) != n:
        raise RuntimeError(
            f"Expected {n} embeddings, got {len(embeddings)} — input/output drift."
        )

    out = pd.DataFrame({
        "candidate_id": rows["candidate_id"].tolist(),
        "embedding": embeddings,
        "model": [model_id] * n,
    })

    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_parquet, index=False)
    logger.info("Wrote %d embeddings -> %s", n, output_parquet)
    return n


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    parser = argparse.ArgumentParser(
        description=(
            "Compute MolFormer-XL embeddings for candidates with non-null "
            "smiles_canonical in the pipeline's candidate_detail.parquet."
        )
    )
    parser.add_argument(
        "--candidates",
        type=Path,
        required=True,
        help="Path to candidate_detail.parquet from a pipeline run.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Where to write molformer_embeddings.parquet.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Initial batch size; halved automatically on CUDA OOM. Default: 64.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device override. Default: auto (cuda if available else cpu).",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"HF model id. Default: {DEFAULT_MODEL}",
    )
    args = parser.parse_args(argv)

    try:
        build(
            candidates_parquet=args.candidates,
            output_parquet=args.output,
            batch_size=args.batch_size,
            device=args.device,
            model_id=args.model,
        )
    except SystemExit:
        raise
    except Exception as exc:
        logger.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
