"""LLMClient with a scripted chat model and a real SQLite cache."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel

from convoprobe.errors import ConvoprobeError
from convoprobe.models.cache import SqliteLLMCache
from convoprobe.models.client import (
    LLMClient,
    ReplayMissError,
    StructuredOutputError,
    _content_text,
)
from convoprobe.models.ratelimit import ProviderLimiter
from convoprobe.models.retry import RetryPolicy
from tests.llm_doubles import FakeChatModel

MESSAGES = [SystemMessage(content="You are a test."), HumanMessage(content="Say hi.")]


class Verdict(BaseModel):
    verdict: str
    score: int


class Other(BaseModel):
    verdict: str


class ServerDown(Exception):
    status_code = 503


async def no_sleep(_seconds: float) -> None:
    return None


@pytest.fixture
def cache(tmp_path: Path) -> Iterator[SqliteLLMCache]:
    with SqliteLLMCache(tmp_path / "cache.db") as opened:
        yield opened


def client(
    cache: SqliteLLMCache, outputs: list[Any], temperature: float = 0.0, **kwargs: Any
) -> tuple[LLMClient, FakeChatModel]:
    model = FakeChatModel(outputs=outputs)
    return (
        LLMClient(
            "ollama:fake",
            cache=cache,
            chat_model=model,
            temperature=temperature,
            sleep=no_sleep,
            **kwargs,
        ),
        model,
    )


def cache_rows(cache: SqliteLLMCache, table: str) -> int:
    conn = sqlite3.connect(cache.path)
    try:
        return int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
    finally:
        conn.close()


async def test_text_miss_then_hit(cache: SqliteLLMCache) -> None:
    llm, model = client(cache, ["hello"])
    assert await llm.text(MESSAGES) == "hello"
    assert await llm.text(MESSAGES) == "hello"
    assert model.calls == 1
    assert llm.usage.calls == 1
    assert llm.usage.cache_hits == 1


async def test_salt_gives_independent_samples(cache: SqliteLLMCache) -> None:
    llm, model = client(cache, ["first", "second"])
    assert await llm.text(MESSAGES, salt="c1-r0:1") == "first"
    assert await llm.text(MESSAGES, salt="c1-r1:1") == "second"
    assert model.calls == 2


def test_key_depends_on_temperature_schema_and_kind(cache: SqliteLLMCache) -> None:
    cold, _ = client(cache, [], temperature=0.0)
    warm, _ = client(cache, [], temperature=0.7)
    assert cold.cache_key("text", MESSAGES, None, "") != warm.cache_key("text", MESSAGES, None, "")
    assert cold.cache_key("structured", MESSAGES, Verdict, "") != cold.cache_key(
        "structured", MESSAGES, Other, ""
    )
    assert cold.cache_key("text", MESSAGES, None, "") != cold.cache_key(
        "structured", MESSAGES, None, ""
    )
    assert cold.cache_key("text", MESSAGES, None, "") == cold.cache_key("text", MESSAGES, None, "")


async def test_different_temperature_calls_again(cache: SqliteLLMCache) -> None:
    cold, cold_model = client(cache, ["a"], temperature=0.0)
    warm, warm_model = client(cache, ["b"], temperature=0.7)
    assert await cold.text(MESSAGES) == "a"
    assert await warm.text(MESSAGES) == "b"
    assert (cold_model.calls, warm_model.calls) == (1, 1)


async def test_structured_success_and_cached(cache: SqliteLLMCache) -> None:
    llm, model = client(cache, [{"verdict": "yes", "score": 4}])
    first = await llm.structured(MESSAGES, Verdict)
    second = await llm.structured(MESSAGES, Verdict)
    assert first == Verdict(verdict="yes", score=4)
    assert isinstance(second, Verdict)
    assert second == first
    assert model.calls == 1


async def test_structured_different_schema_calls_again(cache: SqliteLLMCache) -> None:
    llm, model = client(cache, [{"verdict": "yes", "score": 1}, {"verdict": "no"}])
    await llm.structured(MESSAGES, Verdict)
    assert await llm.structured(MESSAGES, Other) == Other(verdict="no")
    assert model.calls == 2


async def test_structured_bad_output_not_cached(cache: SqliteLLMCache) -> None:
    llm, model = client(cache, ["BAD", {"verdict": "yes", "score": 2}])
    with pytest.raises(StructuredOutputError) as info:
        await llm.structured(MESSAGES, Verdict)
    assert str(info.value) == ("ollama:fake returned output that does not match Verdict: not json")
    assert cache_rows(cache, "llm_cache") == 0
    assert (await llm.structured(MESSAGES, Verdict)).score == 2
    assert model.calls == 2


async def test_structured_parsed_but_invalid_for_schema(cache: SqliteLLMCache) -> None:
    llm, _ = client(cache, [{"verdict": "yes"}])
    with pytest.raises(StructuredOutputError, match="does not match Verdict"):
        await llm.structured(MESSAGES, Verdict)
    assert cache_rows(cache, "llm_cache") == 0


async def test_replay_only(cache: SqliteLLMCache) -> None:
    recorder, _ = client(cache, ["recorded"])
    await recorder.text(MESSAGES)
    replay = LLMClient(
        "ollama:fake", cache=cache, chat_model=None, temperature=0.0, replay_only=True
    )
    assert await replay.text(MESSAGES) == "recorded"
    with pytest.raises(ReplayMissError) as info:
        await replay.text(MESSAGES, salt="never recorded")
    assert str(info.value) == (
        "no recorded response for ollama:fake (text); record it with a live model first"
    )


async def test_usage_tokens_summed(cache: SqliteLLMCache) -> None:
    llm, _ = client(cache, ["a", {"verdict": "y", "score": 1}])
    await llm.text(MESSAGES)
    await llm.structured(MESSAGES, Verdict)
    await llm.text(MESSAGES)  # cache hit adds no tokens
    assert (llm.usage.input_tokens, llm.usage.output_tokens) == (20, 10)
    assert llm.usage.calls == 2


async def test_mark_used_rows(cache: SqliteLLMCache) -> None:
    llm, _ = client(cache, ["a", "b"], run_id="run-1")
    await llm.text(MESSAGES)
    await llm.text(MESSAGES)
    await llm.text(MESSAGES, salt="other")
    assert cache_rows(cache, "llm_cache_usage") == 2


async def test_transient_then_success(cache: SqliteLLMCache) -> None:
    llm, model = client(cache, [ServerDown("down"), ServerDown("down"), "finally"])
    assert await llm.text(MESSAGES) == "finally"
    assert llm.usage.retries == 2
    assert llm.usage.calls == 1
    assert model.calls == 3


async def test_goes_through_limiter(cache: SqliteLLMCache) -> None:
    limiter = ProviderLimiter(60, 10_000, 1)
    llm, _ = client(cache, ["a"], limiter=limiter, retry=RetryPolicy(max_attempts=1))
    await llm.text(MESSAGES)
    assert limiter.requests is not None
    assert limiter.requests.tokens < 60


def test_needs_chat_model_unless_replay(cache: SqliteLLMCache) -> None:
    with pytest.raises(ConvoprobeError, match="chat model is required"):
        LLMClient("ollama:fake", cache=cache, chat_model=None, temperature=0.0)


def test_content_text() -> None:
    assert _content_text("plain") == "plain"
    parts = ["a", {"type": "text", "text": "b"}, {"type": "image", "url": "x"}]
    assert _content_text(parts) == "ab"
    assert _content_text(AIMessage(content="x").content) == "x"
    assert _content_text(42) == "42"
