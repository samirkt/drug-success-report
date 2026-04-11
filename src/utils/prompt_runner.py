"""Prompt-agnostic Anthropic request helpers."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Callable, Mapping

import anthropic

#DEFAULT_MODEL = "claude-sonnet-4-20250514"
DEFAULT_MODEL = "claude-opus-4-6"
DEFAULT_API_KEY_ENV = "CLAUDE_API_KEY_UCD"
DEFAULT_MAX_TOKENS = 10_000
BATCH_SIZE = 50
_RESERVED_MODEL_PARAMS = {"model", "system", "messages"}


def load_prompt_from_txt(path: str | Path) -> str:
    """Load a prompt from a UTF-8 text file and strip leading/trailing whitespace."""
    return Path(path).read_text(encoding="utf-8").strip()


def load_system_prompt_from_txt(path: str | Path) -> str:
    """Load a system prompt from a text file."""
    return load_prompt_from_txt(path)


def load_user_prompt_from_txt(path: str | Path) -> str:
    """Load a user prompt from a text file."""
    return load_prompt_from_txt(path)


def load_prompts_from_txt(
    system_prompt_path: str | Path,
    user_prompt_path: str | Path,
) -> tuple[str, str]:
    """Load system and user prompts from text files."""
    return (
        load_system_prompt_from_txt(system_prompt_path),
        load_user_prompt_from_txt(user_prompt_path),
    )


def create_client(
    *,
    api_key: str | None = None,
    api_key_env: str = DEFAULT_API_KEY_ENV,
) -> anthropic.Anthropic:
    """Create an Anthropic client using an explicit API key or environment fallback."""
    return anthropic.Anthropic(api_key=api_key or os.getenv(api_key_env))


def process_prompt_request(
    *,
    system_prompt: str,
    user_prompt: str,
    model: str = DEFAULT_MODEL,
    model_params: Mapping[str, Any] | None = None,
    api_key: str | None = None,
    api_key_env: str = DEFAULT_API_KEY_ENV,
    client: anthropic.Anthropic | None = None,
) -> anthropic.types.Message:
    """Send a generic prompt request and return the raw Anthropic response."""
    params = dict(model_params or {})
    reserved = _RESERVED_MODEL_PARAMS.intersection(params)
    if reserved:
        joined = ", ".join(sorted(reserved))
        raise ValueError(f"model_params cannot contain reserved keys: {joined}")

    request: dict[str, Any] = {
        "model": model,
        "system": [{"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}}],
        "messages": [{"role": "user", "content": user_prompt}],
        "temperature": 0.0,  # default to deterministic responses; can be overridden by model_params
        **params,
    }
    request.setdefault("max_tokens", DEFAULT_MAX_TOKENS)

    active_client = client or create_client(api_key=api_key, api_key_env=api_key_env)
    return active_client.messages.create(**request)


def extract_text_response(message: anthropic.types.Message) -> str:
    """Extract joined text blocks from an Anthropic message."""
    text_blocks: list[str] = []
    for block in message.content:
        if getattr(block, "type", None) == "text":
            text_blocks.append(block.text)
    return "\n".join(text_blocks).strip()


def build_candidates_block(
    candidates: list,
    fields_fn: Callable[[int, Any], list[str]],
) -> str:
    """
    Build a numbered [N] candidate block for batch LLM prompts.

    Args:
        candidates: ordered list of candidate objects (any type)
        fields_fn:  callable(1-based-index, candidate) -> list of "Label: value"
                    strings for that candidate's block (no [N] header or blank line)

    Returns:
        Multi-line string with each candidate preceded by a [N] header and
        separated by a blank line, ready to substitute into {{CANDIDATES_BLOCK}}.
    """
    lines: list[str] = []
    for i, candidate in enumerate(candidates, start=1):
        lines.append(f"[{i}]")
        lines.extend(fields_fn(i, candidate))
        lines.append("")
    return "\n".join(lines).rstrip()


def parse_batch_response(
    raw: str,
    logger: logging.Logger,
    context: str = "",
) -> dict[int, dict]:
    """
    Parse a JSON array response from a batch LLM call.

    Args:
        raw:     raw text from extract_text_response()
        logger:  caller's logger for warnings on malformed input
        context: short label for log messages (e.g. "Adjudication", "Classification")

    Returns:
        dict mapping candidate_index (int, 1-based) -> response dict.
        Returns {} on total parse failure so callers can apply their own defaults.
    """
    prefix = f"{context}: " if context else ""
    if raw.startswith("```"):
        raw = raw.removeprefix("```json").removeprefix("```").strip()
        if raw.endswith("```"):
            raw = raw[:-3].strip()

    try:
        data_list = json.loads(raw)
    except json.JSONDecodeError:
        data_list = _decode_first_json_array(raw)
        if data_list is None:
            logger.warning("%sbatch response parse failed: Response was not valid JSON array", prefix)
            print(raw)
            return {}
    except Exception as exc:
        logger.warning("%sbatch response parse failed: %s", prefix, exc)
        print(raw)
        return {}

    if not isinstance(data_list, list):
        logger.warning(
            "%sbatch response parse failed: Expected JSON array, got %s",
            prefix,
            type(data_list).__name__,
        )
        print(raw)
        return {}

    by_index: dict[int, dict] = {}
    for item in data_list:
        if isinstance(item, dict):
            idx = item.get("candidate_index")
            if isinstance(idx, int):
                by_index[idx] = item
    return by_index


def _decode_first_json_array(raw: str) -> list[Any] | None:
    decoder = json.JSONDecoder()
    search_from = 0

    while True:
        start = raw.find("[", search_from)
        if start == -1:
            return None
        try:
            parsed, _ = decoder.raw_decode(raw[start:])
        except json.JSONDecodeError:
            search_from = start + 1
            continue
        if isinstance(parsed, list):
            return parsed
        search_from = start + 1


__all__ = [
    "BATCH_SIZE",
    "DEFAULT_API_KEY_ENV",
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_MODEL",
    "build_candidates_block",
    "create_client",
    "extract_text_response",
    "load_prompt_from_txt",
    "load_prompts_from_txt",
    "load_system_prompt_from_txt",
    "load_user_prompt_from_txt",
    "parse_batch_response",
    "process_prompt_request",
]
