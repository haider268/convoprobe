"""LangGraphAdapter on tiny purpose-built graphs. No model involved."""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("langgraph")

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import interrupt

from convoprobe.adapters.langgraph_adapter import (
    NO_CHECKPOINTER,
    LangGraphAdapter,
    turn_from_messages,
)
from convoprobe.errors import ConvoprobeError


def compile_single(node: Any, *, checkpointer: bool = True) -> Any:
    builder = StateGraph(MessagesState)
    builder.add_node("node", node)
    builder.add_edge(START, "node")
    builder.add_edge("node", END)
    return builder.compile(checkpointer=InMemorySaver() if checkpointer else None)


def echo(state: MessagesState) -> dict[str, Any]:
    humans = [m for m in state["messages"] if isinstance(m, HumanMessage)]
    return {"messages": [AIMessage(content=f"{len(humans)}: {humans[-1].content}")]}


def ask_once(state: MessagesState) -> dict[str, Any]:
    value = interrupt({"question": "approve?"})
    return {"messages": [AIMessage(content=f"got {value}")]}


async def started(adapter: LangGraphAdapter, *cids: str) -> LangGraphAdapter:
    for cid in cids:
        await adapter.start(cid)
    return adapter


async def test_memory_and_thread_isolation() -> None:
    adapter = await started(LangGraphAdapter(compile_single(echo)), "a", "b")
    assert (await adapter.send("a", "one")).text == "1: one"
    assert (await adapter.send("a", "two")).text == "2: two"
    assert (await adapter.send("b", "other")).text == "1: other"
    assert (await adapter.send("a", "three")).text == "3: three"


async def test_restart_opens_a_fresh_thread() -> None:
    adapter = await started(LangGraphAdapter(compile_single(echo)), "a")
    await adapter.send("a", "one")
    await adapter.send("a", "two")
    await adapter.end("a")
    await adapter.start("a")  # a resumed run starts the same conversation id again
    assert (await adapter.send("a", "fresh")).text == "1: fresh"


async def test_end_forgets_thread_and_send_before_start() -> None:
    adapter = await started(LangGraphAdapter(compile_single(echo)), "a")
    await adapter.send("a", "one")
    await adapter.end("a")
    with pytest.raises(ConvoprobeError, match="conversation a not started"):
        await adapter.send("a", "two")
    with pytest.raises(ConvoprobeError, match="conversation never not started"):
        await adapter.send("never", "x")


async def test_resume_with_string() -> None:
    adapter = await started(LangGraphAdapter(compile_single(ask_once), resume="yes"), "a")
    turn = await adapter.send("a", "hi")
    assert turn.text == "got yes"
    assert turn.metadata == {
        "node_path": ["node"],
        "interrupts": [{"question": "approve?"}],
        "resumes": 1,
    }


async def test_resume_with_callable_gets_payload() -> None:
    seen: list[Any] = []

    def decide(payload: Any) -> str:
        seen.append(payload)
        return "from callable"

    adapter = await started(LangGraphAdapter(compile_single(ask_once), resume=decide), "a")
    assert (await adapter.send("a", "hi")).text == "got from callable"
    assert seen == [{"question": "approve?"}]


async def test_interrupt_without_resume_value() -> None:
    adapter = await started(LangGraphAdapter(compile_single(ask_once)), "a")
    with pytest.raises(ConvoprobeError) as info:
        await adapter.send("a", "hi")
    assert str(info.value) == (
        "graph interrupted with {'question': 'approve?'} but no resume value is configured;"
        " pass resume= to LangGraphAdapter"
    )


async def test_max_resumes_exceeded() -> None:
    def nag(state: MessagesState) -> dict[str, Any]:
        interrupt("again?")
        return {"messages": [AIMessage(content="asked")]}

    builder = StateGraph(MessagesState)
    builder.add_node("nag", nag)
    builder.add_edge(START, "nag")
    builder.add_edge("nag", "nag")  # loops forever, interrupting every time
    adapter = LangGraphAdapter(
        builder.compile(checkpointer=InMemorySaver()), resume="yes", max_resumes=2
    )
    await adapter.start("a")
    with pytest.raises(ConvoprobeError, match="graph interrupted more than 2 times in one turn"):
        await adapter.send("a", "hi")


async def test_parallel_interrupts_resumed_by_id() -> None:
    def left(state: MessagesState) -> dict[str, Any]:
        return {"messages": [AIMessage(content=f"left {interrupt('L')}")]}

    def right(state: MessagesState) -> dict[str, Any]:
        return {"messages": [AIMessage(content=f"right {interrupt('R')}")]}

    builder = StateGraph(MessagesState)
    builder.add_node("left", left)
    builder.add_node("right", right)
    builder.add_edge(START, "left")
    builder.add_edge(START, "right")
    builder.add_edge("left", END)
    builder.add_edge("right", END)
    adapter = LangGraphAdapter(
        builder.compile(checkpointer=InMemorySaver()), resume=lambda v: v.lower()
    )
    await adapter.start("a")
    turn = await adapter.send("a", "hi")
    assert sorted(turn.text.splitlines()) == ["left l", "right r"]
    assert sorted(turn.metadata["interrupts"]) == ["L", "R"]
    assert turn.metadata["resumes"] == 1


def test_graph_without_checkpointer() -> None:
    with pytest.raises(ConvoprobeError) as info:
        LangGraphAdapter(compile_single(echo, checkpointer=False))
    assert str(info.value) == NO_CHECKPOINTER


async def test_tool_results_json_and_raw_string() -> None:
    def tools(state: MessagesState) -> dict[str, Any]:
        return {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "lookup", "args": {"id": 1}, "id": "t1"},
                        {"name": "notes", "args": {}, "id": "t2"},
                    ],
                ),
                ToolMessage(content='{"found": true}', tool_call_id="t1"),
                ToolMessage(content="plain text result", tool_call_id="t2"),
                AIMessage(content=[{"type": "text", "text": "all done"}, {"type": "image"}]),
            ]
        }

    adapter = await started(LangGraphAdapter(compile_single(tools)), "a")
    turn = await adapter.send("a", "hi")
    assert turn.text == "all done"
    assert [(c.name, c.args, c.result) for c in turn.tool_calls] == [
        ("lookup", {"id": 1}, {"found": True}),
        ("notes", {}, "plain text result"),
    ]


def test_turn_from_messages_skips_empty_text_and_unknown_tool_ids() -> None:
    text, calls = turn_from_messages(
        [
            HumanMessage(content="user text is not agent text"),
            AIMessage(content="", tool_calls=[{"name": "x", "args": {}, "id": "a"}]),
            ToolMessage(content="orphan", tool_call_id="unknown"),
            AIMessage(content="first"),
            AIMessage(content=["second"]),
        ]
    )
    assert text == "first\nsecond"
    assert [(c.name, c.result) for c in calls] == [("x", None)]
