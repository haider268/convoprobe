"""The AgentAdapter protocol: how convoprobe talks to an agent under test."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from convoprobe.datamodels import AgentTurn


@runtime_checkable
class AgentAdapter(Protocol):
    """One adapter serves many conversations, keyed by conversation id."""

    async def start(self, conversation_id: str) -> None:
        """Prepare a new conversation."""

    async def send(self, conversation_id: str, user_message: str) -> AgentTurn:
        """Send one user message and return the agent's reply."""

    async def end(self, conversation_id: str) -> None:
        """Release anything held for the conversation."""
