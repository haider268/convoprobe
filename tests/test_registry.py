"""LLMRegistry: shared limiters, replay without providers, usage."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from pydantic import SecretStr

from convoprobe.config import Settings
from convoprobe.models import factory
from convoprobe.models.cache import SqliteLLMCache
from convoprobe.models.registry import LLMRegistry


@pytest.fixture
def cache(tmp_path: Path) -> Iterator[SqliteLLMCache]:
    with SqliteLLMCache(tmp_path / "cache.db") as opened:
        yield opened


def settings() -> Settings:
    return Settings(
        _env_file=None,  # type: ignore[call-arg]
        groq_api_key=SecretStr("test-key"),
        google_api_key=SecretStr("test-key"),
    )


def test_same_provider_shares_limiter(cache: SqliteLLMCache) -> None:
    registry = LLMRegistry(settings(), cache)
    sim = registry.client("ollama:qwen2.5:3b", 0.7)
    judge = registry.client("ollama:llama3.2:3b", 0.0)
    groq = registry.client("groq:llama-3.1-8b-instant", 0.7)
    assert sim.limiter is judge.limiter
    assert groq.limiter is not sim.limiter
    assert sim.limiter is not None and sim.limiter.requests is None
    assert groq.limiter is not None and groq.limiter.requests is not None
    assert groq.limiter.requests.capacity == 30.0
    assert groq.limiter.tokens is not None and groq.limiter.tokens.capacity == 6000.0


def test_gemini_limiter_from_settings(cache: SqliteLLMCache) -> None:
    gemini = LLMRegistry(settings(), cache).client("gemini:some-model", 0.0)
    assert gemini.limiter is not None
    assert gemini.limiter.requests is not None and gemini.limiter.requests.capacity == 10.0
    assert gemini.limiter.tokens is None


def test_replay_only_never_builds_models(
    cache: SqliteLLMCache, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("make_chat_model must not be called in replay mode")

    monkeypatch.setattr(factory, "make_chat_model", refuse)
    registry = LLMRegistry(Settings(_env_file=None), cache, replay_only=True, run_id="r1")  # type: ignore[call-arg]
    client = registry.client("groq:llama-3.1-8b-instant", 0.7)
    assert client.chat_model is None
    assert client.replay_only is True
    assert client.run_id == "r1"


def test_usage_lists_all_clients(cache: SqliteLLMCache) -> None:
    registry = LLMRegistry(settings(), cache)
    registry.client("ollama:qwen2.5:3b", 0.7)
    registry.client("ollama:llama3.2:3b", 0.0)
    assert [u.spec for u in registry.usage()] == ["ollama:qwen2.5:3b", "ollama:llama3.2:3b"]
    assert all(u.calls == 0 for u in registry.usage())
