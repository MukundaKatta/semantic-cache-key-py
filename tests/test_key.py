"""Tests for ``semantic_cache_key.key`` and helpers."""

from __future__ import annotations

import pytest

from semantic_cache_key import key, normalize_text, semantic_cache_key, stable_value


def test_normalize_text_lowercases_and_collapses_whitespace():
    assert normalize_text("  HELLO   World\n") == "hello world"


def test_normalize_text_handles_none_and_non_strings():
    assert normalize_text(None) == ""
    assert normalize_text(42) == "42"


def test_key_invariant_to_whitespace_and_casing():
    a = key(prompt="Hello,   world!", model="gpt-5")
    b = key(prompt="hello, world!", model="gpt-5")
    c = key(prompt="hello,\tworld!\n", model="gpt-5")
    assert a == b == c


def test_key_changes_when_model_changes():
    a = key(prompt="hello", model="gpt-5")
    b = key(prompt="hello", model="claude-sonnet-4-5")
    assert a != b


def test_key_invariant_to_tool_order():
    tools_a = [{"name": "search", "type": "function"}, {"name": "calc"}]
    tools_b = [{"name": "calc"}, {"type": "function", "name": "search"}]
    assert key("p", "m", tools=tools_a) == key("p", "m", tools=tools_b)


def test_key_changes_when_tool_set_changes():
    a = key("p", "m", tools=[{"name": "search"}])
    b = key("p", "m", tools=[{"name": "search"}, {"name": "calc"}])
    assert a != b


def test_key_changes_when_retrieval_context_changes():
    a = key("p", "m", retrieval_context=["doc-a", "doc-b"])
    b = key("p", "m", retrieval_context=["doc-a", "doc-c"])
    assert a != b


def test_key_changes_when_system_prompt_changes():
    a = key("p", "m", system="You are helpful.")
    b = key("p", "m", system="You are terse.")
    assert a != b


def test_key_version_bump_invalidates_cache():
    a = key("p", "m", version="v1")
    b = key("p", "m", version="v2")
    assert a != b


def test_key_default_length_is_16_hex_chars():
    k = key("hello", "gpt-5")
    assert len(k) == 16
    int(k, 16)  # parses as hex


def test_key_length_can_be_overridden_up_to_64():
    full = key("hello", "gpt-5", length=64)
    assert len(full) == 64
    assert full.startswith(key("hello", "gpt-5", length=16))


def test_key_length_out_of_range_rejected():
    with pytest.raises(ValueError):
        key("p", "m", length=2)
    with pytest.raises(ValueError):
        key("p", "m", length=65)


def test_stable_value_sorts_dict_keys_recursively():
    out = stable_value({"b": {"y": 2, "x": 1}, "a": 1})
    assert list(out.keys()) == ["a", "b"]
    assert list(out["b"].keys()) == ["x", "y"]


def test_semantic_cache_key_js_compat_returns_full_64_char_digest():
    h = semantic_cache_key({"prompt": "hello", "model": "gpt-5"})
    assert len(h) == 64
