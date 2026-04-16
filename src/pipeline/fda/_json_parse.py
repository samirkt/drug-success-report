"""Shared JSON response parser for LLMClient implementations.

Both `AnthropicJSONClient` and `OpenAICompatJSONClient` need to pull a single
JSON object out of a free-form model response. Models sometimes wrap their
output in ```json fences despite instructions to the contrary; hosted and
local JSON-mode implementations vary in how strictly they obey. This parser
is deliberately forgiving — strip fences, try a straight `json.loads`, and
fall back to extracting the outermost `{...}` substring. Returns `{}` on
unrecoverable failures so callers don't need to catch exceptions.
"""

from __future__ import annotations

import json
import logging
import re

logger = logging.getLogger(__name__)

_JSON_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.MULTILINE)


def parse_json_object(raw: str) -> dict:
    """Parse a JSON object, tolerating optional ```json fences."""
    cleaned = _JSON_FENCE.sub("", raw).strip()
    try:
        obj = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            logger.warning("JSON parse: could not locate JSON object in response")
            return {}
        try:
            obj = json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError as e:
            logger.warning("JSON parse: failed to decode extracted object: %s", e)
            return {}
    if not isinstance(obj, dict):
        logger.warning("JSON parse: expected object, got %s", type(obj).__name__)
        return {}
    return obj
