"""User simulators produce the next user message for a persona."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from convoprobe.datamodels import LLMPersona, Message, Scenario, ScriptedPersona
from convoprobe.errors import ConvoprobeError


@runtime_checkable
class UserSimulator(Protocol):
    """Returns the next user message, or None to end the conversation."""

    async def next_message(
        self,
        scenario: Scenario,
        persona: LLMPersona | ScriptedPersona,
        messages: list[Message],
    ) -> str | None:
        """Next user turn, or None to stop."""


class ScriptedSimulator:
    """Replays a scripted persona's turns in order. No model involved."""

    async def next_message(
        self,
        scenario: Scenario,
        persona: LLMPersona | ScriptedPersona,
        messages: list[Message],
    ) -> str | None:
        if not isinstance(persona, ScriptedPersona):
            raise ConvoprobeError(f"persona {persona.id} has no script")
        sent = sum(1 for m in messages if m.role == "user")
        return persona.script[sent] if sent < len(persona.script) else None
