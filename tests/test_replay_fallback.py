"""Replay follows a recorded retry when the first attempt was never cached.

Invalid model output is never cached, so a live run whose first attempt failed
leaves only the ":retry1" answer. Replay must reach it the same way, and a second
miss must raise the original ReplayMissError. Live behaviour is unchanged.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from convoprobe.datamodels import Conversation, LLMJudgeItem, LLMPersona, Message, Scenario
from convoprobe.models.cache import SqliteLLMCache
from convoprobe.models.client import LLMClient, ReplayMissError
from convoprobe.scoring.judge import judge_item
from convoprobe.simulator.llm import LLMSimulator
from convoprobe.simulator.personas import generate_personas
from tests.llm_doubles import fake_client

SPEC = "ollama:fake"
SCENARIO = Scenario.model_validate(
    {
        "name": "t",
        "agent_description": "Shop agent.",
        "personas": [{"id": "pat", "description": "Pat Example.", "goal": "Refund."}],
        "rubric": [{"id": "r", "type": "max_turns", "value": 5}],
    }
)
PAT = LLMPersona(id="pat", description="Pat Example.", goal="Refund.")
ITEM = LLMJudgeItem(id="polite", type="llm_judge", question="Polite?", pass_if="yes")
CONVERSATION = Conversation(
    id="c1",
    persona_id="pat",
    messages=[Message(role="user", text="hi"), Message(role="agent", text="hello")],
)
VERDICT = {"reasoning": "fine", "verdict": "yes", "evidence": [{"turn": 1, "quote": "hello"}]}
TURN = {"goal_status": "in_progress", "message": "where is my order"}
PEOPLE = {"personas": [{"name": "Grumpy", "description": "Short.", "goal": "Refund."}]}


@pytest.fixture
def cache(tmp_path: Path) -> Iterator[SqliteLLMCache]:
    with SqliteLLMCache(tmp_path / "cache.db") as opened:
        yield opened


def replay(cache: SqliteLLMCache) -> LLMClient:
    return LLMClient(SPEC, cache=cache, chat_model=None, temperature=0.0, replay_only=True)


def spy_salts(client: LLMClient) -> list[str]:
    seen: list[str] = []
    real = client.structured

    async def spy(messages: Any, schema: Any, *, salt: str = "") -> Any:
        seen.append(salt)
        return await real(messages, schema, salt=salt)

    client.structured = spy  # type: ignore[method-assign]
    return seen


def always_miss(client: LLMClient) -> list[str]:
    """Every call misses, with the salt in the message, so the raised one is identifiable."""
    seen: list[str] = []

    async def miss(messages: Any, schema: Any, *, salt: str = "") -> Any:
        seen.append(salt)
        raise ReplayMissError(f"miss {salt}")

    client.structured = miss  # type: ignore[method-assign]
    return seen


# Judge


async def test_judge_replays_recorded_retry(cache: SqliteLLMCache) -> None:
    live, model = fake_client(cache, ["BAD", VERDICT], SPEC)
    recorded = await judge_item(ITEM, CONVERSATION, SCENARIO, live)
    assert model.calls == 2 and recorded.status == "pass"
    client = replay(cache)
    salts = spy_salts(client)
    replayed = await judge_item(ITEM, CONVERSATION, SCENARIO, client)
    assert replayed == recorded
    assert salts == ["judge:c1:polite", "judge:c1:polite:retry1"]


async def test_judge_double_miss_raises_original(cache: SqliteLLMCache) -> None:
    client = replay(cache)
    salts = always_miss(client)
    with pytest.raises(ReplayMissError, match=r"^miss judge:c1:polite$"):
        await judge_item(ITEM, CONVERSATION, SCENARIO, client)
    assert salts == ["judge:c1:polite", "judge:c1:polite:retry1"]


async def test_judge_live_reads_retry_only_after_failure(cache: SqliteLLMCache) -> None:
    live, _ = fake_client(cache, [VERDICT], SPEC)
    salts = spy_salts(live)
    await judge_item(ITEM, CONVERSATION, SCENARIO, live)
    assert salts == ["judge:c1:polite"]
    other, _ = fake_client(cache, [], SPEC)
    missed = always_miss(other)
    with pytest.raises(ReplayMissError, match=r"^miss judge:c1:polite$"):
        await judge_item(ITEM, CONVERSATION, SCENARIO, other)
    assert missed == ["judge:c1:polite"]  # a live client never falls back


# Simulator


async def test_simulator_replays_recorded_retry(cache: SqliteLLMCache) -> None:
    live, model = fake_client(cache, ["BAD", TURN], SPEC)
    recorded = await LLMSimulator(live).next_message(SCENARIO, PAT, [], conversation_id="c1")
    assert model.calls == 2 and recorded == "where is my order"
    client = replay(cache)
    salts = spy_salts(client)
    sim = LLMSimulator(client)
    assert await sim.next_message(SCENARIO, PAT, [], conversation_id="c1") == recorded
    assert salts == ["sim:c1:0", "sim:c1:0:retry1"]


async def test_simulator_double_miss_raises_original(cache: SqliteLLMCache) -> None:
    client = replay(cache)
    salts = always_miss(client)
    with pytest.raises(ReplayMissError, match=r"^miss sim:c1:0$"):
        await LLMSimulator(client).next_message(SCENARIO, PAT, [], conversation_id="c1")
    assert salts == ["sim:c1:0", "sim:c1:0:retry1"]


async def test_simulator_live_reads_retry_only_after_failure(cache: SqliteLLMCache) -> None:
    live, _ = fake_client(cache, [TURN], SPEC)
    salts = spy_salts(live)
    await LLMSimulator(live).next_message(SCENARIO, PAT, [], conversation_id="c1")
    assert salts == ["sim:c1:0"]
    other, _ = fake_client(cache, [], SPEC)
    missed = always_miss(other)
    with pytest.raises(ReplayMissError, match=r"^miss sim:c1:0$"):
        await LLMSimulator(other).next_message(SCENARIO, PAT, [], conversation_id="c1")
    assert missed == ["sim:c1:0"]


# Personas


async def test_personas_replay_recorded_retry(cache: SqliteLLMCache) -> None:
    live, model = fake_client(cache, ["BAD", PEOPLE], SPEC)
    recorded = await generate_personas(SCENARIO, 1, live, salt="p")
    assert model.calls == 2 and [p.id for p in recorded[0]] == ["gen_grumpy"]
    client = replay(cache)
    salts = spy_salts(client)
    assert await generate_personas(SCENARIO, 1, client, salt="p") == recorded
    assert salts == ["p", "p:retry1"]


async def test_personas_double_miss_raises_original(cache: SqliteLLMCache) -> None:
    client = replay(cache)
    salts = always_miss(client)
    with pytest.raises(ReplayMissError, match=r"^miss p$"):
        await generate_personas(SCENARIO, 1, client, salt="p")
    assert salts == ["p", "p:retry1"]


async def test_personas_live_reads_retry_only_after_failure(cache: SqliteLLMCache) -> None:
    live, _ = fake_client(cache, [PEOPLE], SPEC)
    salts = spy_salts(live)
    await generate_personas(SCENARIO, 1, live, salt="p")
    assert salts == ["p"]
    other, _ = fake_client(cache, [], SPEC)
    missed = always_miss(other)
    with pytest.raises(ReplayMissError, match=r"^miss p$"):
        await generate_personas(SCENARIO, 1, other, salt="p")
    assert missed == ["p"]
