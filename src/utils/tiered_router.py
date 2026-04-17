"""
Tiered model routing for LLM classification and adjudication.

Runs a batch on a cheaper model (Sonnet) first, identifies low-confidence
results via a caller-supplied predicate, re-runs those on a more capable
model (Opus), and merges the results back by original index.

Also provides CostLedger for tracking cumulative token usage and cost.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Callable

from utils.prompt_runner import (
    extract_text_response,
    parse_batch_response,
    process_prompt_request,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model constants and pricing (per 1M tokens, USD)
# ---------------------------------------------------------------------------

MODEL_SONNET = "claude-sonnet-4-6"
MODEL_OPUS = "claude-opus-4-6"

# Pricing per 1M tokens (input / output)
_PRICING: dict[str, tuple[float, float]] = {
    MODEL_SONNET: (3.00, 15.00),
    MODEL_OPUS: (5.00, 25.00),
}

# Prompt caching multipliers (relative to base input price)
_CACHE_CREATION_MULTIPLIER = 1.25
_CACHE_READ_MULTIPLIER = 0.10

# Default confidence threshold — results at or below this are escalated
CONFIDENCE_THRESHOLD = 0.25  # maps to "LOW" in the pipeline's confidence scheme


# ---------------------------------------------------------------------------
# Cost ledger
# ---------------------------------------------------------------------------

@dataclass
class CostLedger:
    """Accumulates token counts and dollar costs across a full pipeline run."""

    entries: list[dict[str, Any]] = field(default_factory=list)
    _lock: threading.Lock = field(
        default_factory=threading.Lock, repr=False, compare=False
    )

    @property
    def total_input_tokens(self) -> int:
        """Total input tokens across all buckets (uncached + cache creation + cache read)."""
        return sum(
            e["input_tokens"] + e.get("cache_creation_input_tokens", 0) + e.get("cache_read_input_tokens", 0)
            for e in self.entries
        )

    @property
    def total_output_tokens(self) -> int:
        return sum(e["output_tokens"] for e in self.entries)

    @property
    def total_cost_usd(self) -> float:
        return sum(e["cost_usd"] for e in self.entries)

    def record(
        self,
        *,
        stage: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        n_candidates: int,
        cache_creation_input_tokens: int = 0,
        cache_read_input_tokens: int = 0,
    ) -> None:
        """Record a single LLM call's token usage.

        The Anthropic API splits input tokens into three buckets when prompt
        caching is active:
          - input_tokens: uncached input (billed at base input rate)
          - cache_creation_input_tokens: written to cache (billed at 1.25x)
          - cache_read_input_tokens: read from cache (billed at 0.10x)
        """
        input_price, output_price = _PRICING.get(model, (0.0, 0.0))
        cost = (
            input_tokens * input_price
            + cache_creation_input_tokens * input_price * _CACHE_CREATION_MULTIPLIER
            + cache_read_input_tokens * input_price * _CACHE_READ_MULTIPLIER
            + output_tokens * output_price
        ) / 1_000_000
        with self._lock:
            self.entries.append(
                {
                    "stage": stage,
                    "model": model,
                    "input_tokens": input_tokens,
                    "cache_creation_input_tokens": cache_creation_input_tokens,
                    "cache_read_input_tokens": cache_read_input_tokens,
                    "output_tokens": output_tokens,
                    "cost_usd": cost,
                    "n_candidates": n_candidates,
                }
            )

    def record_cache_hits(self, *, stage: str, n_hits: int) -> None:
        """Record candidates resolved from cache or deterministically (zero LLM cost).

        Kept separate from record() so cost_per_candidate_summary() can correctly
        include these in the denominator without inflating token counts.
        """
        with self._lock:
            self.entries.append(
                {
                    "stage": stage,
                    "model": None,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cost_usd": 0.0,
                    "n_candidates": n_hits,
                    "cache_hit": True,
                }
            )

    def cost_per_candidate_summary(self) -> list[dict]:
        """Return per-stage cost-per-candidate breakdown.

        Two metrics per stage:
          - cost_per_total: cost / (llm_candidates + cache_hits)
          - cost_per_llm:   cost / llm_candidates  (excludes cache hits)
          - cost_per_llm_tier1: tier1_cost / tier1_llm_candidates
          - cost_per_llm_tier2: tier2_cost / tier2_llm_candidates
        """
        from collections import defaultdict

        stage_cost: dict[str, float] = defaultdict(float)
        stage_llm: dict[str, int] = defaultdict(int)
        stage_cache: dict[str, int] = defaultdict(int)
        stage_tier1_cost: dict[str, float] = defaultdict(float)
        stage_tier2_cost: dict[str, float] = defaultdict(float)
        stage_tier1_llm: dict[str, int] = defaultdict(int)
        stage_tier2_llm: dict[str, int] = defaultdict(int)

        for e in self.entries:
            s = e["stage"]
            if e.get("cache_hit"):
                stage_cache[s] += e["n_candidates"]
            else:
                stage_cost[s] += e["cost_usd"]
                stage_llm[s] += e["n_candidates"]
                if e.get("model") == MODEL_SONNET:
                    stage_tier1_cost[s] += e["cost_usd"]
                    stage_tier1_llm[s] += e["n_candidates"]
                elif e.get("model") == MODEL_OPUS:
                    stage_tier2_cost[s] += e["cost_usd"]
                    stage_tier2_llm[s] += e["n_candidates"]

        rows = []
        for s in sorted(set(stage_llm) | set(stage_cache)):
            llm_n = stage_llm[s]
            cache_n = stage_cache[s]
            total_n = llm_n + cache_n
            cost = stage_cost[s]
            tier1_llm_n = stage_tier1_llm[s]
            tier2_llm_n = stage_tier2_llm[s]
            tier1_cost = stage_tier1_cost[s]
            tier2_cost = stage_tier2_cost[s]
            rows.append(
                {
                    "stage": s,
                    "total_cost_usd": cost,
                    "llm_candidates": llm_n,
                    "cache_hits": cache_n,
                    "total_candidates": total_n,
                    "cost_per_total": cost / total_n if total_n > 0 else None,
                    "cost_per_llm": cost / llm_n if llm_n > 0 else None,
                    "tier1_cost_usd": tier1_cost,
                    "tier2_cost_usd": tier2_cost,
                    "tier1_llm_candidates": tier1_llm_n,
                    "tier2_llm_candidates": tier2_llm_n,
                    "cost_per_llm_tier1": (
                        tier1_cost / tier1_llm_n if tier1_llm_n > 0 else None
                    ),
                    "cost_per_llm_tier2": (
                        tier2_cost / tier2_llm_n if tier2_llm_n > 0 else None
                    ),
                }
            )
        return rows

    def log(self) -> None:
        """Emit a summary to the logger."""
        logger.info(
            "Cost summary: %d calls, %d input tokens, %d output tokens, $%.4f USD",
            len(self.entries),
            self.total_input_tokens,
            self.total_output_tokens,
            self.total_cost_usd,
        )
        for row in self.cost_per_candidate_summary():
            logger.info(
                "  %s: $%.4f | %d LLM + %d cache = %d total | "
                "$%.6f/candidate (all) | $%.6f/candidate (LLM-only) | "
                "tier1: $%.6f/candidate (%d) | tier2: $%.6f/candidate (%d)",
                row["stage"],
                row["total_cost_usd"],
                row["llm_candidates"],
                row["cache_hits"],
                row["total_candidates"],
                row["cost_per_total"] or 0.0,
                row["cost_per_llm"] or 0.0,
                row["cost_per_llm_tier1"] or 0.0,
                row["tier1_llm_candidates"],
                row["cost_per_llm_tier2"] or 0.0,
                row["tier2_llm_candidates"],
            )
        for entry in self.entries:
            cache_create = entry.get("cache_creation_input_tokens", 0)
            cache_read = entry.get("cache_read_input_tokens", 0)
            logger.debug(
                "  %s [%s]: %d in + %d cache_create + %d cache_read / %d out, %d candidates, $%.4f",
                entry["stage"],
                entry["model"],
                entry["input_tokens"],
                cache_create,
                cache_read,
                entry["output_tokens"],
                entry["n_candidates"],
                entry["cost_usd"],
            )


# ---------------------------------------------------------------------------
# Escalation predicates
# ---------------------------------------------------------------------------

def confidence_below_threshold(
    result: dict, threshold: float = CONFIDENCE_THRESHOLD
) -> bool:
    """Return True if the result's confidence is at or below the threshold.

    Works for both classification (modality_confidence) and adjudication
    (confidence) response schemas.
    """
    _CONFIDENCE_MAP = {"HIGH": 1.0, "MEDIUM": 0.5, "LOW": 0.25}
    # Adjudication uses "confidence", classification uses "modality_confidence"
    conf_str = result.get("confidence") or result.get("modality_confidence") or ""
    conf_val = _CONFIDENCE_MAP.get(conf_str, 0.0)
    return conf_val <= threshold


def outcome_is_unknown(result: dict) -> bool:
    """Return True if the adjudicated outcome is UNKNOWN."""
    return result.get("outcome", "UNKNOWN") == "UNKNOWN"


def default_escalation_predicate(result: dict) -> bool:
    """Default: escalate if confidence is LOW or outcome is UNKNOWN."""
    return confidence_below_threshold(result) or outcome_is_unknown(result)


# ---------------------------------------------------------------------------
# Tiered batch call
# ---------------------------------------------------------------------------

def tiered_batch_call(
    *,
    system_prompt: str,
    user_template: str,
    candidates: list,
    fields_fn: Callable[[int, Any], list[str]],
    parse_fn: Callable[[dict, Any], Any],
    default_fn: Callable[[Any], Any],
    escalation_predicate: Callable[[dict], bool] = default_escalation_predicate,
    stage_name: str = "",
    ledger: CostLedger | None = None,
) -> list:
    """
    Run a batch LLM call with tiered Sonnet → Opus routing.

    Args:
        system_prompt:  the system prompt text
        user_template:  user prompt template with {{CANDIDATES_BLOCK}} placeholder
        candidates:     ordered list of candidate objects
        fields_fn:      callable(1-based-index, candidate) → list of field strings
        parse_fn:       callable(response_dict, candidate) → parsed result object
        default_fn:     callable(candidate) → default result when LLM fails
        escalation_predicate:
                        callable(response_dict) → True if this result should be
                        re-run on Opus
        stage_name:     label for logging and cost ledger entries
        ledger:         optional CostLedger to record token costs

    Returns:
        list of parsed results, one per candidate, in the same order as input.
    """
    from utils.prompt_runner import build_candidates_block

    n = len(candidates)
    prefix = f"{stage_name}: " if stage_name else ""

    # --- Pass 1: Sonnet ---
    user_prompt = user_template.replace(
        "{{CANDIDATES_BLOCK}}",
        build_candidates_block(candidates, fields_fn),
    )
    try:
        message = process_prompt_request(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=MODEL_SONNET,
        )
        raw = extract_text_response(message)
        by_index = parse_batch_response(raw, logger, context=f"{prefix}Sonnet")
        if ledger is not None:
            ledger.record(
                stage=stage_name,
                model=MODEL_SONNET,
                input_tokens=getattr(message.usage, "input_tokens", 0),
                output_tokens=getattr(message.usage, "output_tokens", 0),
                cache_creation_input_tokens=getattr(message.usage, "cache_creation_input_tokens", 0) or 0,
                cache_read_input_tokens=getattr(message.usage, "cache_read_input_tokens", 0) or 0,
                n_candidates=n,
            )
    except Exception as exc:
        logger.warning("%sSonnet pass failed: %s — falling back to Opus for all", prefix, exc)
        by_index = {}

    # Identify candidates that need escalation
    escalation_indices: list[int] = []  # 1-based
    for i in range(1, n + 1):
        data = by_index.get(i)
        if data is None or escalation_predicate(data):
            escalation_indices.append(i)

    n_escalated = len(escalation_indices)
    logger.info(
        "%s%d / %d candidates escalated to Opus",
        prefix, n_escalated, n,
    )
    if n_escalated > 0:
        # Build sub-batch for escalated candidates only
        escalated_candidates = [candidates[i - 1] for i in escalation_indices]

        # Remap indices: escalated candidates get new 1-based indices in their sub-batch
        def _remapped_fields(new_i: int, c: Any) -> list[str]:
            # Use the original fields_fn; new_i is the sub-batch index
            return fields_fn(new_i, c)

        user_prompt_opus = user_template.replace(
            "{{CANDIDATES_BLOCK}}",
            build_candidates_block(escalated_candidates, _remapped_fields),
        )

        try:
            message_opus = process_prompt_request(
                system_prompt=system_prompt,
                user_prompt=user_prompt_opus,
                model=MODEL_OPUS,
            )
            raw_opus = extract_text_response(message_opus)
            by_index_opus = parse_batch_response(
                raw_opus, logger, context=f"{prefix}Opus"
            )
            if ledger is not None:
                ledger.record(
                    stage=stage_name,
                    model=MODEL_OPUS,
                    input_tokens=getattr(message_opus.usage, "input_tokens", 0),
                    output_tokens=getattr(message_opus.usage, "output_tokens", 0),
                    cache_creation_input_tokens=getattr(message_opus.usage, "cache_creation_input_tokens", 0) or 0,
                    cache_read_input_tokens=getattr(message_opus.usage, "cache_read_input_tokens", 0) or 0,
                    n_candidates=n_escalated,
                )

            # Merge Opus results back into the main by_index using original indices
            for sub_i, orig_i in enumerate(escalation_indices, start=1):
                opus_data = by_index_opus.get(sub_i)
                if opus_data is not None:
                    by_index[orig_i] = opus_data

        except Exception as exc:
            logger.warning("%sOpus escalation pass failed: %s", prefix, exc)

    # --- Assemble final results ---
    results: list = []
    for i, candidate in enumerate(candidates, start=1):
        data = by_index.get(i)
        if data is not None:
            try:
                results.append(parse_fn(data, candidate))
            except Exception:
                results.append(default_fn(candidate))
        else:
            results.append(default_fn(candidate))

    return results


__all__ = [
    "CONFIDENCE_THRESHOLD",
    "CostLedger",
    "MODEL_OPUS",
    "MODEL_SONNET",
    "confidence_below_threshold",
    "cost_per_candidate_summary",
    "default_escalation_predicate",
    "outcome_is_unknown",
    "record_cache_hits",
    "tiered_batch_call",
]
