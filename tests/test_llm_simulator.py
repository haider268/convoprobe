"""LLMSimulator: prompt, role flipping, stop signal, salts."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from convoprobe.datamodels import LLMPersona, Message, Scenario, ScriptedPersona
from convoprobe.errors import ConvoprobeError
from convoprobe.models.cache import SqliteLLMCache
from convoprobe.models.client import StructuredOutputError
from convoprobe.simulator.llm import OPENING, SIMULATOR_SYSTEM, LLMSimulator, SimTurn
from convoprobe.simulator.simulator import UserSimulator
from tests.llm_doubles import fake_client

SCENARIO = Scenario.model_validate(
    {
        "name": "t",
        "agent_description": "Support agent for shop.example.\n",
        "personas": [{"id": "pat", "description": "Pat Example, impatient.", "goal": "Refund."}],
        "rubric": [{"id": "r", "type": "max_turns", "value": 5}],
    }
)
PAT = LLMPersona(id="pat", description="Pat Example, impatient.", goal="Refund.")


def turn(status: str, message: str) -> dict[str, str]:
    return {"goal_status": status, "message": message}


@pytest.fixture
def cache(tmp_path: Path) -> Iterator[SqliteLLMCache]:
    with SqliteLLMCache(tmp_path / "cache.db") as opened:
        yield opened


async def test_messages_sent_with_flipped_roles(cache: SqliteLLMCache) -> None:
    client, model = fake_client(cache, [turn("in_progress", "  still waiting  ")])
    history = [
        Message(role="user", text="I want a refund."),
        Message(role="agent", text="What is your order number?"),
    ]
    text = await LLMSimulator(client).next_message(SCENARIO, PAT, history, conversation_id="c1")
    assert text == "still waiting"
    sent = model.seen[0]
    assert [type(m) for m in sent] == [SystemMessage, HumanMessage, AIMessage, HumanMessage]
    assert sent[0].content == SIMULATOR_SYSTEM.format(
        agent_description="Support agent for shop.example.",
        description="Pat Example, impatient.",
        goal="Refund.",
    )
    assert sent[1].content == OPENING
    assert sent[2].content == "I want a refund."
    assert sent[3].content == "What is your order number?"
    assert "You are NOT the agent." in str(sent[0].content)


async def test_first_message(cache: SqliteLLMCache) -> None:
    client, model = fake_client(cache, [turn("in_progress", "Hi, my jacket arrived torn.")])
    text = await LLMSimulator(client).next_message(SCENARIO, PAT, [], conversation_id="c1")
    assert text == "Hi, my jacket arrived torn."
    assert [type(m) for m in model.seen[0]] == [SystemMessage, HumanMessage]


@pytest.mark.parametrize("status", ["achieved", "impossible"])
async def test_stop_after_agent_reply(cache: SqliteLLMCache, status: str) -> None:
    client, _ = fake_client(cache, [turn(status, "thanks bye")])
    history = [Message(role="user", text="refund 1042"), Message(role="agent", text="Done.")]
    sim = LLMSimulator(client)
    assert await sim.next_message(SCENARIO, PAT, history, conversation_id="c1") is None


async def test_achieved_before_any_agent_reply_is_ignored(cache: SqliteLLMCache) -> None:
    client, _ = fake_client(cache, [turn("achieved", "Hello, I need a refund.")])
    text = await LLMSimulator(client).next_message(SCENARIO, PAT, [], conversation_id="c1")
    assert text == "Hello, I need a refund."


AFTER_REPLY = [Message(role="user", text="hi"), Message(role="agent", text="Hello.")]


async def test_blank_message_after_agent_reply_stops(cache: SqliteLLMCache) -> None:
    client, model = fake_client(cache, [turn("in_progress", "   ")])
    sim = LLMSimulator(client)
    assert await sim.next_message(SCENARIO, PAT, AFTER_REPLY, conversation_id="c1") is None
    assert model.calls == 1


async def test_empty_first_message_retried_once(cache: SqliteLLMCache) -> None:
    client, model = fake_client(cache, [turn("in_progress", " "), turn("in_progress", "Hi.")])
    text = await LLMSimulator(client).next_message(SCENARIO, PAT, [], conversation_id="c1")
    assert text == "Hi."
    assert model.calls == 2
    # The retry is a separate cache entry under its own salt.
    replay, replay_model = fake_client(cache, [])
    assert await LLMSimulator(replay).next_message(SCENARIO, PAT, [], conversation_id="c1") == "Hi."
    assert replay_model.calls == 0


async def test_empty_first_message_twice_raises(cache: SqliteLLMCache) -> None:
    client, model = fake_client(cache, [turn("in_progress", ""), turn("achieved", "  ")])
    with pytest.raises(StructuredOutputError) as info:
        await LLMSimulator(client).next_message(SCENARIO, PAT, [], conversation_id="c1")
    assert str(info.value) == "ollama:fake produced an empty first user message"
    assert model.calls == 2


async def test_salt_differs_by_conversation_and_turn(cache: SqliteLLMCache) -> None:
    outputs = [turn("in_progress", f"line {i}") for i in range(3)]
    client, model = fake_client(cache, outputs)
    sim = LLMSimulator(client)
    first = await sim.next_message(SCENARIO, PAT, [], conversation_id="pat-r0")
    other_repeat = await sim.next_message(SCENARIO, PAT, [], conversation_id="pat-r1")
    again = await sim.next_message(SCENARIO, PAT, [], conversation_id="pat-r0")
    assert (first, other_repeat, again) == ("line 0", "line 1", "line 0")
    assert model.calls == 2

    history = [Message(role="user", text="line 0"), Message(role="agent", text="ok")]
    key_turn0 = client.cache_key("structured", [], SimTurn, "sim:pat-r0:0")
    key_turn2 = client.cache_key("structured", [], SimTurn, "sim:pat-r0:2")
    assert key_turn0 != key_turn2
    assert await sim.next_message(SCENARIO, PAT, history, conversation_id="pat-r0") == "line 2"
    assert model.calls == 3


async def test_retry_once_on_bad_output(cache: SqliteLLMCache) -> None:
    client, model = fake_client(cache, ["BAD", turn("in_progress", "second try")])
    text = await LLMSimulator(client).next_message(SCENARIO, PAT, [], conversation_id="c1")
    assert text == "second try"
    assert model.calls == 2
    assert model.seen[0] == model.seen[1]


async def test_second_bad_output_propagates(cache: SqliteLLMCache) -> None:
    client, model = fake_client(cache, ["BAD", "BAD"])
    with pytest.raises(StructuredOutputError):
        await LLMSimulator(client).next_message(SCENARIO, PAT, [], conversation_id="c1")
    assert model.calls == 2


async def test_scripted_persona_raises(cache: SqliteLLMCache) -> None:
    client, _ = fake_client(cache, [])
    persona = ScriptedPersona(id="s", script=["hi"])
    with pytest.raises(ConvoprobeError, match="persona s is scripted"):
        await LLMSimulator(client).next_message(SCENARIO, persona, [], conversation_id="c1")


def test_prompt_pins_reserved_fake_contact_details() -> None:
    paragraph = (
        "If the agent asks for your name, email, or phone number, use only these made-up"
        " details: name Pat Example, email pat@example.com, phone 555-0123. Never invent any"
        " other name, email address, or phone number."
    )
    assert SIMULATOR_SYSTEM.endswith("\n\n" + paragraph)


def test_is_user_simulator(cache: SqliteLLMCache) -> None:
    client, _ = fake_client(cache, [])
    assert isinstance(LLMSimulator(client), UserSimulator)
