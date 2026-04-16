"""OpenAI-compatible HTTP backend for the LLMClient protocol.

Targets any server that implements the OpenAI Chat Completions API shape,
covering local runtimes (Ollama `/v1`, vLLM, llama.cpp server, LM Studio)
and hosted aggregators (Together, Groq, Fireworks, DeepInfra, OpenAI).

Structured output is requested via `response_format={"type": "json_object"}`.
Backends that silently ignore that field fall through the shared soft
parser (`_json_parse.parse_json_object`), which tolerates code fences and
stray prose. Temperature is forced to zero for determinism.

Cost ledger entries are recorded under the model string verbatim; unknown
models silently price at $0 in `CostLedger`, so local runs still log
token volume under `stage="adjudication_fda"` without synthetic pricing.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

from ._json_parse import parse_json_object

logger = logging.getLogger(__name__)


def _get_httpx():
    """Lazy httpx import so tests that inject fakes don't require it."""
    import httpx
    return httpx


class OpenAICompatJSONClient:
    """Concrete LLMClient: OpenAI-compatible HTTP + JSON response parsing.

    Implements `complete_json(system, user, schema) -> dict` from
    `pipeline.fda.llm_adjudicator.LLMClient` by POSTing to
    `{base_url}/chat/completions`.
    """

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: Optional[str] = None,
        timeout: float = 120.0,
        client: Any = None,
        cache: Any = None,  # pipeline.knowledge_cache.KnowledgeCache | None
        ledger: Any = None,  # utils.tiered_router.CostLedger | None
        stage_label: str = "adjudication_fda",
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key or None
        self.timeout = timeout
        self.cache = cache
        self.ledger = ledger
        self.stage_label = stage_label
        if client is not None:
            self._client = client
            self._owns_client = False
        else:
            httpx = _get_httpx()
            self._client = httpx.Client(timeout=timeout)
            self._owns_client = True

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def complete_json(self, system: str, user: str, schema: dict) -> dict:
        if self.cache is not None:
            key = self.cache.make_llm_json_key(
                self.model, system, user, json.dumps(schema, sort_keys=True)
            )
            cached = self.cache.get_llm_json(key)
            if cached is not None:
                return cached
        else:
            key = None

        system_with_schema = (
            f"{system.rstrip()}\n\n"
            "Return ONLY a single JSON object matching this schema — no prose, "
            "no code fence, no commentary:\n"
            f"{json.dumps(schema)}"
        )

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_with_schema},
                {"role": "user", "content": user},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0,
        }

        response = self._client.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=body,
        )
        response.raise_for_status()
        data = response.json()

        self._record_usage(data)

        content = _extract_content(data)
        payload = parse_json_object(content)

        if key is not None:
            self.cache.put_llm_json(key, payload)
        return payload

    def _record_usage(self, data: dict) -> None:
        if self.ledger is None:
            return
        usage = data.get("usage") or {}
        self.ledger.record(
            stage=self.stage_label,
            model=self.model,
            input_tokens=int(usage.get("prompt_tokens") or 0),
            output_tokens=int(usage.get("completion_tokens") or 0),
            n_candidates=1,
        )


def _extract_content(data: dict) -> str:
    choices = data.get("choices") or []
    if not choices:
        logger.warning("OpenAICompatJSONClient: response contained no choices")
        return ""
    message = choices[0].get("message") or {}
    content = message.get("content")
    if content is None:
        logger.warning("OpenAICompatJSONClient: choice[0].message.content was null")
        return ""
    return str(content)
