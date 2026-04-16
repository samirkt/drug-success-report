"""Tests for the OpenAI-compatible JSON LLM client used by FDA adjudication.

Tests inject a fake httpx-style client so no network is involved. The
client's contract is: POST to `{base_url}/chat/completions` with a JSON
body specifying `response_format={"type":"json_object"}`, pass an
Authorization header iff an api_key is configured, parse the response,
record token usage through a CostLedger, and share results with the
KnowledgeCache on subsequent identical calls.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from pipeline.fda.llm_client_openai_compat import OpenAICompatJSONClient
from pipeline.knowledge_cache import KnowledgeCache


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeResponse:
    def __init__(self, payload: dict, status_code: int = 200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self) -> dict:
        return self._payload


class FakeHTTPClient:
    """Captures POST calls and returns pre-seeded responses."""

    def __init__(self, response: dict | None = None):
        self.calls: list[dict] = []
        self._response = response or _chat_completion_response(
            content='{"ok": true}',
            prompt_tokens=10,
            completion_tokens=3,
        )

    def post(self, url: str, *, headers: dict, json: dict) -> FakeResponse:
        self.calls.append({"url": url, "headers": headers, "body": json})
        return FakeResponse(self._response)

    def close(self) -> None:
        pass


class FakeLedger:
    def __init__(self):
        self.entries: list[dict] = []

    def record(self, **kwargs):
        self.entries.append(kwargs)


def _chat_completion_response(
    *,
    content: str,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
) -> dict:
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


SAMPLE_SCHEMA = {
    "type": "object",
    "properties": {"ok": {"type": "boolean"}},
}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestOpenAICompatRequestShape:
    def test_round_trip_parses_json_from_content(self):
        http = FakeHTTPClient(
            _chat_completion_response(content='{"verdict": "APPROVED"}')
        )
        client = OpenAICompatJSONClient(
            base_url="http://localhost:11434/v1",
            model="qwen2.5:32b-instruct",
            client=http,
        )

        result = client.complete_json("sys", "user", SAMPLE_SCHEMA)

        assert result == {"verdict": "APPROVED"}

    def test_posts_to_chat_completions_with_expected_body(self):
        http = FakeHTTPClient()
        client = OpenAICompatJSONClient(
            base_url="http://localhost:11434/v1",
            model="qwen2.5:32b-instruct",
            client=http,
        )
        client.complete_json("sys", "user", SAMPLE_SCHEMA)

        assert len(http.calls) == 1
        call = http.calls[0]
        assert call["url"] == "http://localhost:11434/v1/chat/completions"
        body = call["body"]
        assert body["model"] == "qwen2.5:32b-instruct"
        assert body["temperature"] == 0
        assert body["response_format"] == {"type": "json_object"}
        roles = [m["role"] for m in body["messages"]]
        assert roles == ["system", "user"]
        # schema is appended to the system prompt so even backends that
        # ignore response_format see it as instructions
        assert "json_object" in body["response_format"]["type"]
        assert "ok" in body["messages"][0]["content"]

    def test_trailing_slash_in_base_url_is_normalized(self):
        http = FakeHTTPClient()
        client = OpenAICompatJSONClient(
            base_url="http://localhost:11434/v1/",
            model="qwen2.5:32b-instruct",
            client=http,
        )
        client.complete_json("sys", "user", SAMPLE_SCHEMA)

        assert http.calls[0]["url"] == "http://localhost:11434/v1/chat/completions"

    def test_authorization_header_present_iff_api_key_given(self):
        http_with_key = FakeHTTPClient()
        OpenAICompatJSONClient(
            base_url="https://api.together.xyz/v1",
            model="meta-llama/Llama-3-70b",
            api_key="sk-test",
            client=http_with_key,
        ).complete_json("sys", "user", SAMPLE_SCHEMA)
        assert http_with_key.calls[0]["headers"].get("Authorization") == "Bearer sk-test"

        http_no_key = FakeHTTPClient()
        OpenAICompatJSONClient(
            base_url="http://localhost:11434/v1",
            model="qwen2.5:32b-instruct",
            client=http_no_key,
        ).complete_json("sys", "user", SAMPLE_SCHEMA)
        assert "Authorization" not in http_no_key.calls[0]["headers"]

    def test_empty_string_api_key_is_treated_as_absent(self):
        http = FakeHTTPClient()
        OpenAICompatJSONClient(
            base_url="http://localhost:11434/v1",
            model="qwen2.5:32b-instruct",
            api_key="",
            client=http,
        ).complete_json("sys", "user", SAMPLE_SCHEMA)
        assert "Authorization" not in http.calls[0]["headers"]


class TestOpenAICompatCache:
    def test_cache_hit_skips_http(self, tmp_path):
        cache = KnowledgeCache(tmp_path / "c.db")
        key = cache.make_llm_json_key(
            "qwen2.5:32b-instruct", "sys", "user", json.dumps(SAMPLE_SCHEMA, sort_keys=True)
        )
        cache.put_llm_json(key, {"cached": True})

        http = FakeHTTPClient()
        client = OpenAICompatJSONClient(
            base_url="http://localhost:11434/v1",
            model="qwen2.5:32b-instruct",
            client=http,
            cache=cache,
        )
        result = client.complete_json("sys", "user", SAMPLE_SCHEMA)

        assert result == {"cached": True}
        assert http.calls == []
        assert client.cache_hits == 1
        assert client.cache_misses == 0

    def test_cache_miss_writes_result(self, tmp_path):
        cache = KnowledgeCache(tmp_path / "c.db")
        http = FakeHTTPClient(
            _chat_completion_response(content='{"v": 42}', prompt_tokens=5, completion_tokens=1)
        )
        client = OpenAICompatJSONClient(
            base_url="http://localhost:11434/v1",
            model="qwen2.5:32b-instruct",
            client=http,
            cache=cache,
        )
        client.complete_json("sys", "user", SAMPLE_SCHEMA)

        key = cache.make_llm_json_key(
            "qwen2.5:32b-instruct", "sys", "user", json.dumps(SAMPLE_SCHEMA, sort_keys=True)
        )
        assert cache.get_llm_json(key) == {"v": 42}
        assert client.cache_hits == 0
        assert client.cache_misses == 1

        # Second identical call should now be a cache hit and skip HTTP.
        client.complete_json("sys", "user", SAMPLE_SCHEMA)
        assert client.cache_hits == 1
        assert client.cache_misses == 1
        assert len(http.calls) == 1

    def test_different_models_get_independent_cache_entries(self, tmp_path):
        cache = KnowledgeCache(tmp_path / "c.db")

        http_a = FakeHTTPClient(_chat_completion_response(content='{"m": "a"}'))
        OpenAICompatJSONClient(
            base_url="http://localhost:11434/v1",
            model="model-a",
            client=http_a,
            cache=cache,
        ).complete_json("sys", "user", SAMPLE_SCHEMA)

        # Different model → different cache key → new HTTP call expected
        http_b = FakeHTTPClient(_chat_completion_response(content='{"m": "b"}'))
        result_b = OpenAICompatJSONClient(
            base_url="http://localhost:11434/v1",
            model="model-b",
            client=http_b,
            cache=cache,
        ).complete_json("sys", "user", SAMPLE_SCHEMA)

        assert result_b == {"m": "b"}
        assert len(http_b.calls) == 1


class TestOpenAICompatLedger:
    def test_records_prompt_and_completion_tokens(self):
        http = FakeHTTPClient(
            _chat_completion_response(content='{}', prompt_tokens=137, completion_tokens=42)
        )
        ledger = FakeLedger()
        client = OpenAICompatJSONClient(
            base_url="http://localhost:11434/v1",
            model="qwen2.5:32b-instruct",
            client=http,
            ledger=ledger,
            stage_label="adjudication_fda",
        )
        client.complete_json("sys", "user", SAMPLE_SCHEMA)

        assert len(ledger.entries) == 1
        entry = ledger.entries[0]
        assert entry["stage"] == "adjudication_fda"
        assert entry["model"] == "qwen2.5:32b-instruct"
        assert entry["input_tokens"] == 137
        assert entry["output_tokens"] == 42
        assert entry["n_candidates"] == 1

    def test_no_ledger_configured_does_not_raise(self):
        http = FakeHTTPClient()
        client = OpenAICompatJSONClient(
            base_url="http://localhost:11434/v1",
            model="qwen2.5:32b-instruct",
            client=http,
        )
        # Must not raise when ledger is None
        client.complete_json("sys", "user", SAMPLE_SCHEMA)

    def test_missing_usage_field_defaults_to_zero(self):
        response_without_usage = {
            "choices": [{"message": {"content": '{}'}}],
        }
        http = FakeHTTPClient(response_without_usage)
        ledger = FakeLedger()
        client = OpenAICompatJSONClient(
            base_url="http://localhost:11434/v1",
            model="qwen2.5:32b-instruct",
            client=http,
            ledger=ledger,
        )
        client.complete_json("sys", "user", SAMPLE_SCHEMA)

        entry = ledger.entries[0]
        assert entry["input_tokens"] == 0
        assert entry["output_tokens"] == 0


class TestOpenAICompatParsing:
    def test_fenced_json_response_is_tolerated(self):
        fenced = "```json\n{\"verdict\": \"APPROVED\"}\n```"
        http = FakeHTTPClient(_chat_completion_response(content=fenced))
        client = OpenAICompatJSONClient(
            base_url="http://localhost:11434/v1",
            model="qwen2.5:32b-instruct",
            client=http,
        )
        assert client.complete_json("sys", "user", SAMPLE_SCHEMA) == {"verdict": "APPROVED"}

    def test_malformed_json_returns_empty_dict(self):
        http = FakeHTTPClient(_chat_completion_response(content="not json at all"))
        client = OpenAICompatJSONClient(
            base_url="http://localhost:11434/v1",
            model="qwen2.5:32b-instruct",
            client=http,
        )
        assert client.complete_json("sys", "user", SAMPLE_SCHEMA) == {}

    def test_empty_choices_returns_empty_dict(self):
        http = FakeHTTPClient({"choices": [], "usage": {}})
        client = OpenAICompatJSONClient(
            base_url="http://localhost:11434/v1",
            model="qwen2.5:32b-instruct",
            client=http,
        )
        assert client.complete_json("sys", "user", SAMPLE_SCHEMA) == {}

    def test_prose_wrapped_json_is_extracted(self):
        # Some local models prefix their response with explanation.
        content = "Here is the JSON you asked for:\n{\"verdict\": \"APPROVED\"}\nHope this helps!"
        http = FakeHTTPClient(_chat_completion_response(content=content))
        client = OpenAICompatJSONClient(
            base_url="http://localhost:11434/v1",
            model="qwen2.5:32b-instruct",
            client=http,
        )
        assert client.complete_json("sys", "user", SAMPLE_SCHEMA) == {"verdict": "APPROVED"}


class TestOpenAICompatHTTPErrors:
    def test_non_200_status_raises(self):
        class ErrResponse(FakeResponse):
            def __init__(self):
                super().__init__({}, status_code=500)

        class ErrClient:
            def __init__(self):
                self.calls = []

            def post(self, url, *, headers, json):
                self.calls.append((url, headers, json))
                return ErrResponse()

            def close(self):
                pass

        client = OpenAICompatJSONClient(
            base_url="http://localhost:11434/v1",
            model="qwen2.5:32b-instruct",
            client=ErrClient(),
        )
        with pytest.raises(RuntimeError):
            client.complete_json("sys", "user", SAMPLE_SCHEMA)
