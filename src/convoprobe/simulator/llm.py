"""LLM user simulator.

Roles are flipped on purpose: the simulator model is the assistant in its own
chat, so its past user lines are AIMessages and the agent's replies are
HumanMessages. Small models confuse themselves with the agent otherwise.
"""

from __future__ import annotations

from typing import Literal

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from convoprobe.datamodels import LLMPersona, Message, Scenario, ScriptedPersona
from convoprobe.errors import ConvoprobeError
from convoprobe.models.client import LLMClient

SIMULATOR_SYSTEM = """You are role-playing a USER talking to an AI agent. You are NOT the agent. Never offer help, never answer as the agent.

What the agent does:
{agent_description}

Who you are:
{description}

Your goal:
{goal}

Stay in character, including any difficult behaviour described above. Write like a real person typing: short, natural, no labels, no quotes around your message.
First decide goal_status from the agent's LAST message. Then write your next message."""  # noqa: E501

OPENING = "(The conversation starts. Send your first message to the agent.)"


class SimTurn(BaseModel):
    goal_status: Literal["in_progress", "achieved", "impossible"] = Field(
        description=(
            "Judge from the agent's LAST message: achieved if your goal is done,"
            " impossible if the agent clearly cannot do it, otherwise in_progress."
        )
    )
    message: str = Field(
        description=(
            "Your next message as the user, 1 to 2 short sentences."
            " Empty if achieved or impossible."
        )
    )


def build_messages(
    scenario: Scenario, persona: LLMPersona, messages: list[Message]
) -> list[BaseMessage]:
    """System prompt, opening cue, then the transcript with roles flipped.

    Tool calls are not shown, only agent text.
    """
    system = SIMULATOR_SYSTEM.format(
        agent_description=scenario.agent_description.strip(),
        description=persona.description,
        goal=persona.goal,
    )
    chat: list[BaseMessage] = [SystemMessage(content=system), HumanMessage(content=OPENING)]
    for message in messages:
        if message.role == "user":
            chat.append(AIMessage(content=message.text))
        else:
            chat.append(HumanMessage(content=message.text))
    return chat


class LLMSimulator:
    """Plays an LLM persona with a chat model."""

    def __init__(self, client: LLMClient) -> None:
        self.client = client

    async def next_message(
        self,
        scenario: Scenario,
        persona: LLMPersona | ScriptedPersona,
        messages: list[Message],
        *,
        conversation_id: str,
    ) -> str | None:
        if isinstance(persona, ScriptedPersona):
            raise ConvoprobeError(f"persona {persona.id} is scripted, not an LLM persona")
        turn = await self.client.structured(
            build_messages(scenario, persona, messages),
            SimTurn,
            salt=f"sim:{conversation_id}:{len(messages)}",
        )
        agent_replied = any(m.role == "agent" for m in messages)
        if agent_replied and turn.goal_status != "in_progress":
            return None
        text = turn.message.strip()
        return text or None
