"""ScriptedSimulator."""

from __future__ import annotations

import pytest

from convoprobe.datamodels import LLMPersona, Message, Scenario, ScriptedPersona
from convoprobe.errors import ConvoprobeError
from convoprobe.simulator.simulator import ScriptedSimulator, UserSimulator

SCENARIO = Scenario.model_validate(
    {
        "name": "t",
        "agent_description": "x",
        "personas": [{"id": "s", "script": ["one", "two"]}],
        "rubric": [{"id": "r", "type": "max_turns", "value": 5}],
    }
)


async def test_scripted_sequence_then_none() -> None:
    sim = ScriptedSimulator()
    persona = ScriptedPersona(id="s", script=["one", "two"])
    messages: list[Message] = []
    sent = []
    while (text := await sim.next_message(SCENARIO, persona, messages)) is not None:
        sent.append(text)
        messages.append(Message(role="user", text=text))
        messages.append(Message(role="agent", text="reply"))
    assert sent == ["one", "two"]


async def test_llm_persona_raises() -> None:
    persona = LLMPersona(id="p", description="Pat Example", goal="ask")
    with pytest.raises(ConvoprobeError, match="persona p has no script"):
        await ScriptedSimulator().next_message(SCENARIO, persona, [])


def test_is_user_simulator() -> None:
    assert isinstance(ScriptedSimulator(), UserSimulator)
