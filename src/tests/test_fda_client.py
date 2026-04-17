"""Tests for FDAClient that don't require network access.

The thread-safety test guards the throttle lock added so candidate-level
parallelism in AdjudicationStage doesn't bypass openFDA's rate limit.
"""

from __future__ import annotations

import threading
import time
from unittest.mock import patch

import pytest

from pipeline.fda.fda_client import FDAClient


class TestThrottleConcurrency:
    """Confirm `_throttle()` enforces the min-interval across concurrent threads.

    Without `self._throttle_lock`, two threads can both read a stale
    `_last_request_at`, both pass the elapsed check, and both fire
    immediately — silently violating openFDA's 2 req/sec limit.
    """

    def test_throttle_serializes_concurrent_calls(self, tmp_path):
        client = FDAClient(
            cache_dir=tmp_path,
            requests_per_second=4.0,  # min_interval = 0.25s
        )

        fire_times: list[float] = []
        lock = threading.Lock()

        def call_throttle():
            client._throttle()
            with lock:
                fire_times.append(time.monotonic())

        threads = [threading.Thread(target=call_throttle) for _ in range(8)]
        start = time.monotonic()
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # 8 calls with 0.25s min-interval should take >= 7 * 0.25s = 1.75s
        # in total; allow generous slop for thread scheduling variance.
        elapsed = time.monotonic() - start
        assert elapsed >= 1.6, f"Throttle violated: 8 calls completed in {elapsed:.2f}s"

        # Pairwise gaps should each respect the min interval. Sort because
        # threads append in fire order, but allow a tiny tolerance for
        # scheduling jitter.
        fire_times.sort()
        gaps = [fire_times[i + 1] - fire_times[i] for i in range(len(fire_times) - 1)]
        # All gaps must be >= ~0.24s (0.25s min minus small scheduling slop)
        assert all(g >= 0.20 for g in gaps), f"Gaps too tight: {gaps}"

        client.close()

    def test_serial_calls_still_throttled(self, tmp_path):
        """Sanity: non-concurrent calls still respect the interval (no regression)."""
        client = FDAClient(
            cache_dir=tmp_path,
            requests_per_second=10.0,  # min_interval = 0.10s
        )

        start = time.monotonic()
        for _ in range(5):
            client._throttle()
        elapsed = time.monotonic() - start

        # 5 calls with 0.10s min-interval: first is free, then 4 * 0.10s
        assert elapsed >= 0.35

        client.close()
