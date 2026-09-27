"""Adapter for a compiled LangGraph graph.

One thread per conversation. Interrupts (for example a human approval step)
are resumed automatically with a configured value or callable. langgraph is
imported only inside methods, so the core package works without the extra.
"""

from __future__ import annotations

import builtins
import json
import uuid
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from convoprobe.datamodels import AgentTurn, ToolCallRecord
from convoprobe.errors import ConvoprobeError

NO_CHECKPOINTER = (
    "LangGraphAdapter needs a graph compiled with a checkpointer,"
    " for example graph.compile(checkpointer=InMemorySaver())"
)


def _text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and part.get("type") == "text":
                parts.append(str(part.get("text", "")))
        return "".join(parts)
    return ""


def _tool_result(content: Any) -> Any:
    text = _text_of(content)
    try:
        return json.loads(text)
    except ValueError:
        return text


def turn_from_messages(new: list[BaseMessage]) -> tuple[str, list[ToolCallRecord]]:
    """Reply text and tool calls (with results) from the messages a turn added."""
    texts: list[str] = []
    records: list[ToolCallRecord] = []
    by_id: dict[str, ToolCallRecord] = {}
    for message in new:
        if isinstance(message, AIMessage):
            text = _text_of(message.content)
            if text:
                texts.append(text)
            for call in message.tool_calls:
                record = ToolCallRecord(name=call["name"], args=dict(call.get("args") or {}))
                records.append(record)
                if call.get("id"):
                    by_id[str(call["id"])] = record
    for message in new:
        if isinstance(message, ToolMessage) and message.tool_call_id in by_id:
            by_id[message.tool_call_id].result = _tool_result(message.content)
    return "\n".join(texts), records


class LangGraphAdapter:
    """Drives a compiled LangGraph graph as the agent under test.

    resume: the value sent back when the graph interrupts, or a callable that
    gets the interrupt payload and returns the value. None means an interrupt
    is an error. max_resumes caps interrupts per user turn.
    """

    def __init__(
        self,
        graph: Any,
        *,
        resume: Any = None,
        max_resumes: int = 5,
        input_key: str = "messages",
        thread_prefix: str = "convoprobe",
    ) -> None:
        if getattr(graph, "checkpointer", None) is None:
            raise ConvoprobeError(NO_CHECKPOINTER)
        self.graph = graph
        self.resume = resume
        self.max_resumes = max_resumes
        self.input_key = input_key
        self.thread_prefix = thread_prefix
        self._threads: dict[str, str] = {}

    async def start(self, conversation_id: str) -> None:
        # A fresh thread per start. The checkpointer keeps old threads, so reusing
        # "<prefix>:<id>" would continue a previous attempt's messages on a resumed
        # run, or a previous run's when one adapter serves several runs.
        suffix = uuid.uuid4().hex[:8]
        self._threads[conversation_id] = f"{self.thread_prefix}:{conversation_id}:{suffix}"

    async def end(self, conversation_id: str) -> None:
        # Checkpoint data stays in the saver; only the mapping is dropped.
        self._threads.pop(conversation_id, None)

    def _resume_value(self, value: Any) -> Any:
        # builtins.callable: the adapters package has a submodule named callable.
        return self.resume(value) if builtins.callable(self.resume) else self.resume

    async def send(self, conversation_id: str, user_message: str) -> AgentTurn:
        from langgraph.types import Command

        thread_id = self._threads.get(conversation_id)
        if thread_id is None:
            raise ConvoprobeError(f"conversation {conversation_id} not started")
        config = {"configurable": {"thread_id": thread_id}}
        snapshot = await self.graph.aget_state(config)
        before = len((snapshot.values or {}).get(self.input_key, []))

        payload: Any = {self.input_key: [HumanMessage(content=user_message)]}
        node_path: list[str] = []
        interrupt_values: list[Any] = []
        resumes = 0
        while True:
            pending: list[Any] = []
            async for chunk in self.graph.astream(payload, config, stream_mode="updates"):
                for name, update in chunk.items():
                    if name == "__interrupt__":
                        pending.extend(update)
                    else:
                        node_path.append(name)
            if not pending:
                break
            interrupt_values.extend(i.value for i in pending)
            if self.resume is None:
                raise ConvoprobeError(
                    f"graph interrupted with {pending[0].value!r} but no resume value is"
                    " configured; pass resume= to LangGraphAdapter"
                )
            if resumes >= self.max_resumes:
                raise ConvoprobeError(
                    f"graph interrupted more than {self.max_resumes} times in one turn"
                )
            if len(pending) == 1:
                payload = Command(resume=self._resume_value(pending[0].value))
            else:
                payload = Command(resume={i.id: self._resume_value(i.value) for i in pending})
            resumes += 1

        snapshot = await self.graph.aget_state(config)
        new = list((snapshot.values or {}).get(self.input_key, []))[before:]
        text, tool_calls = turn_from_messages(new)
        return AgentTurn(
            text=text,
            tool_calls=tool_calls,
            metadata={"node_path": node_path, "interrupts": interrupt_values, "resumes": resumes},
        )
