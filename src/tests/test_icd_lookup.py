"""Tests for the NLM ICD-10-CM lookup wrapper + on-disk cache."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from pipeline.icd_lookup import IcdCache, get_icd_cached, get_icd_from_nih


@patch("pipeline.icd_lookup.requests.get")
def test_get_icd_returns_codes(mock_get):
    resp = MagicMock()
    resp.json.return_value = [3, ["K70.30", "K70.31", "K70.32"], None, []]
    mock_get.return_value = resp

    codes = get_icd_from_nih("alcoholic cirrhosis of liver")
    assert codes == ["K70.30", "K70.31", "K70.32"]


@patch("pipeline.icd_lookup.requests.get")
def test_get_icd_returns_none_on_no_matches(mock_get):
    resp = MagicMock()
    resp.json.return_value = [0, [], None, []]
    mock_get.return_value = resp

    assert get_icd_from_nih("zzz nonsense disease") is None


def test_cache_round_trips_positive_and_negative(tmp_path):
    cache = IcdCache(tmp_path / "icd.sqlite")
    assert cache.get("foo") == (False, None)

    cache.put("foo", ["A00", "A01"])
    assert cache.get("foo") == (True, ["A00", "A01"])

    cache.put("bar", None)  # cached negative — distinct from miss
    assert cache.get("bar") == (True, None)


@patch("pipeline.icd_lookup.requests.get")
def test_get_icd_cached_uses_cache_on_second_call(mock_get, tmp_path):
    resp = MagicMock()
    resp.json.return_value = [1, ["A00"], None, []]
    mock_get.return_value = resp

    cache = IcdCache(tmp_path / "icd.sqlite")
    first = get_icd_cached("cholera", cache)
    second = get_icd_cached("cholera", cache)

    assert first == ["A00"]
    assert second == ["A00"]
    # Network call only happens once thanks to the cache.
    assert mock_get.call_count == 1
