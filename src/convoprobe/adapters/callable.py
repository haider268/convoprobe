"""Adapter for a plain (history) -> reply function, sync or async."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable
from typing import Union

from convoprobe.datamodels import AgentTurn, Message
from convoprobe.errors import ConvoprobeError

Reply = Union[str, AgentTurn]  # noqa: UP007
AgentFn = Callable[[list[Message]], Union[Reply, Awaitable[Reply]]]  # noqa: UP007


class CallableAdapter:
    """Wraps a function that takes the full history and returns a reply.

    The function sees a copy of the history with the new user message last.
    Sync functions run in a worker thread so they do not block the event loop.
    """

    def __init__(self, fn: AgentFn) -> None:
        self.fn = fn
        self._histories: dict[str, list[Message]] = {}

    async def start(self, conversation_id: str) -> None:
        self._histories[conversation_id] = []

    async def send(self, conversation_id: str, user_message: str) -> AgentTurn:
        history = self._histories.get(conversation_id)
        if history is None:
            raise ConvoprobeError(f"conversation {conversation_id} not started")
        history.append(Message(role="user", text=user_message))
        if inspect.iscoroutinefunction(self.fn):
            result = await self.fn(list(history))
        else:
            result = await asyncio.to_thread(self.fn, list(history))
        if inspect.isawaitable(result):
            result = await result
        turn = AgentTurn(text=result) if isinstance(result, str) else result
        history.append(Message(role="agent", text=turn.text, tool_calls=turn.tool_calls))
        return turn

    async def end(self, conversation_id: str) -> None:
        self._histories.pop(conversation_id, None)
