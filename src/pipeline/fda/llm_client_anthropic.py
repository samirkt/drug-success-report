"""Anthropic-backed implementation of the LLMClient protocol.

Used by the FDA-timeline adjudication stage for indication extraction
(from label text / approval letters) and indication matching (trial
vs. approved). Wraps `utils.prompt_runner` so Sonnet/Opus spend flows
into the shared `CostLedger` and repeated calls can be served from the
shared `KnowledgeCache`.
"""

from __future__ import annotations

import json
import logging
from typing import Optional

import anthropic

from utils.prompt_runner import (
    create_client,
    extract_text_response,
    process_prompt_request,
)
from utils.tiered_router import MODEL_SONNET

from ._json_parse import parse_json_object

logger = logging.getLogger(__name__)


class AnthropicJSONClient:
    """Concrete LLMClient: Anthropic SDK + JSON response parsing.

    Implements the `complete_json(system, user, schema) -> dict` protocol
    from `pipeline.fda.llm_adjudicator.LLMClient`. Temperature is forced
    to zero; the schema is appended to the system prompt so the model
    returns a single JSON object that we parse.
    """

    def __init__(
        self,
        *,
        model: str = MODEL_SONNET,
        client: Optional[anthropic.Anthropic] = None,
        cache=None,  # pipeline.knowledge_cache.KnowledgeCache | None
        ledger=None,  # utils.tiered_router.CostLedger | None
        stage_label: str = "adjudication_fda",
    ):
        self.model = model
        self.client = client or create_client()
        self.cache = cache
        self.ledger = ledger
        self.stage_label = stage_label
        self.cache_hits = 0
        self.cache_misses = 0

    def complete_json(self, system: str, user: str, schema: dict) -> dict:
        if self.cache is not None:
            key = self.cache.make_llm_json_key(
                self.model, system, user, json.dumps(schema, sort_keys=True)
            )
            cached = self.cache.get_llm_json(key)
            if cached is not None:
                self.cache_hits += 1
                return cached
            self.cache_misses += 1
        else:
            key = None

        system_with_schema = (
            f"{system.rstrip()}\n\n"
            "Return ONLY a single JSON object matching this schema — no prose, "
            "no code fence, no commentary:\n"
            f"{json.dumps(schema)}"
        )

        message = process_prompt_request(
            system_prompt=system_with_schema,
            user_prompt=user,
            model=self.model,
            client=self.client,
        )
        self._record_usage(message)
        raw = extract_text_response(message)
        payload = parse_json_object(raw)

        if key is not None:
            self.cache.put_llm_json(key, payload)
        return payload

    def _record_usage(self, message: anthropic.types.Message) -> None:
        if self.ledger is None:
            return
        usage = getattr(message, "usage", None)
        if usage is None:
            return
        self.ledger.record(
            stage=self.stage_label,
            model=self.model,
            input_tokens=getattr(usage, "input_tokens", 0) or 0,
            output_tokens=getattr(usage, "output_tokens", 0) or 0,
            n_candidates=1,
            cache_creation_input_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0,
            cache_read_input_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
        )
