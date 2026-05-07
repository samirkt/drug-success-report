"""Batch NDC-adjudication LLM inference for the Colab offload flow.

Reads ``work_units.jsonl`` (produced locally by
``src/scripts/export_ndc_work.py``), runs each work unit through a
HuggingFace causal LM, and writes one JSON line per successful verdict
to ``verdicts.jsonl``. Failed candidates (parse error, OOM, generation
error) are omitted from the output -- the local pipeline will simply
re-issue an LLM call for them on the next run.

Usage (Colab cell or local):
    python run_inference.py \\
        --model Qwen/Qwen2.5-14B-Instruct \\
        --work work_units.jsonl \\
        --out  verdicts.jsonl

Determinism mirrors ``OpenAICompatJSONClient`` in the parent project:
``temperature=0``, ``do_sample=False`` (greedy), ``max_new_tokens=256``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from prompts import RESPONSE_SCHEMA, SYSTEM_PROMPT, USER_TEMPLATE
from json_parse import parse_json_object

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen2.5-14B-Instruct",
                   help="HF model id or local path. Use a 7B/14B for T4, "
                        "32B+ for A100. Default: Qwen/Qwen2.5-14B-Instruct.")
    p.add_argument("--work", required=True,
                   help="Input work_units.jsonl produced by export_ndc_work.py.")
    p.add_argument("--out", required=True,
                   help="Output verdicts.jsonl path.")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--device", default=None,
                   help="Override device map (default: auto).")
    p.add_argument("--dtype", default="auto",
                   choices=["auto", "float16", "bfloat16", "float32"],
                   help="Torch dtype. 'auto' picks bfloat16 on Ampere+, "
                        "float16 elsewhere.")
    return p.parse_args()


def load_model(model_id: str, dtype: str, device: str | None):
    """Lazy import so this script can be linted without torch installed."""
    import torch  # type: ignore[import]
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore[import]

    if dtype == "auto":
        # bfloat16 on Ampere+ (compute >= 8.0); float16 otherwise.
        if torch.cuda.is_available():
            major, _ = torch.cuda.get_device_capability(0)
            torch_dtype = torch.bfloat16 if major >= 8 else torch.float16
        else:
            torch_dtype = torch.float32
    else:
        torch_dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[dtype]

    logger.info("Loading tokenizer: %s", model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    logger.info("Loading model: %s (dtype=%s, device_map=%s)",
                model_id, torch_dtype, device or "auto")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        device_map=device or "auto",
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def adjudicate_one(model, tokenizer, work: dict, max_new_tokens: int) -> dict:
    """Run one work unit through the model. Raises on parse/generation failure."""
    numbered = "\n".join(
        f"{i}. {ind}" for i, ind in enumerate(work["label_indications"], start=1)
    )
    user = USER_TEMPLATE.format(
        trial_indication=work["trial_indication"],
        mesh_indication=work.get("mesh_indication") or "(none)",
        matched_synonym=work["matched_synonym"],
        numbered_label_indications=numbered,
    )
    system_with_schema = (
        SYSTEM_PROMPT
        + "\n\nReturn ONLY a JSON object matching this schema, "
          "no code fence, no commentary:\n"
        + json.dumps(RESPONSE_SCHEMA)
    )
    messages = [
        {"role": "system", "content": system_with_schema},
        {"role": "user", "content": user},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    import torch  # type: ignore[import]
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            pad_token_id=tokenizer.eos_token_id,
        )
    completion_ids = out[0][inputs.input_ids.shape[1]:]
    completion = tokenizer.decode(completion_ids, skip_special_tokens=True)

    payload = parse_json_object(completion)
    if not payload:
        raise ValueError(f"Could not parse JSON from completion: {completion[:200]!r}")

    approved = bool(payload.get("approved", False))
    confidence = float(payload.get("confidence", 0.0))
    matched = payload.get("matched_indication")
    reasoning = str(payload.get("reasoning", "")).strip() or "(no reasoning)"

    return {
        "candidate_id": work["candidate_id"],
        "approved": approved,
        "confidence": max(0.0, min(1.0, confidence)),
        "matched_indication": matched if isinstance(matched, str) and matched else None,
        "reasoning": reasoning,
        "matched_synonym": work["matched_synonym"],
    }


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        stream=sys.stdout,
    )
    # Colab runs this as a subprocess (`!python run_inference.py ...`) which
    # by default block-buffers stdout — no live progress until the buffer
    # fills. Force line buffering so tqdm and log lines stream to the cell.
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    from tqdm.auto import tqdm

    args = parse_args()

    work_path = Path(args.work)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    work_units = []
    with work_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            work_units.append(json.loads(line))
    logger.info("Loaded %d work units from %s", len(work_units), work_path)

    if not work_units:
        out_path.write_text("")
        logger.warning("No work units to process; wrote empty %s", out_path)
        return

    model, tokenizer = load_model(args.model, args.dtype, args.device)
    logger.info("Model loaded; starting inference over %d work units", len(work_units))

    n_ok = 0
    n_err = 0
    n_approved = 0
    t_start = time.monotonic()
    pbar = tqdm(
        work_units,
        total=len(work_units),
        unit="cand",
        desc="adjudicate",
        dynamic_ncols=True,
        file=sys.stdout,
        mininterval=0.5,
    )
    with out_path.open("w") as f:
        for work in pbar:
            try:
                verdict = adjudicate_one(model, tokenizer, work, args.max_new_tokens)
            except Exception as e:
                n_err += 1
                # One-line warning per error — tqdm.write keeps the bar intact.
                pbar.write(
                    f"[err] {work.get('matched_synonym') or work.get('drug_name', '?')}: {e}",
                    file=sys.stdout,
                )
                pbar.set_postfix(ok=n_ok, err=n_err, approved=n_approved, refresh=False)
                continue
            f.write(json.dumps(verdict) + "\n")
            f.flush()  # so a Colab disconnect doesn't lose progress
            n_ok += 1
            if verdict["approved"]:
                n_approved += 1
            pbar.set_postfix(ok=n_ok, err=n_err, approved=n_approved, refresh=False)
    pbar.close()

    total = time.monotonic() - t_start
    logger.info(
        "Done: %d ok, %d errors, %d approved in %.1fs (%.2fs/call). Output: %s",
        n_ok, n_err, n_approved, total, total / max(1, n_ok + n_err), out_path,
    )


if __name__ == "__main__":
    main()
