"""The LangGraph toy agent: branches, node paths, approval, and the v2 bug."""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("langgraph")

from examples.toy_support_agent.graph import DECLINED_TEXT, build_graph

from convoprobe.adapters.langgraph_adapter import LangGraphAdapter
from convoprobe.datamodels import AgentTurn
from convoprobe.demo.toy_agent import TEXT_ASK_ORDER, TEXT_GREET, TEXT_UNVERIFIED


async def say(adapter: LangGraphAdapter, *messages: str, cid: str = "c1") -> list[AgentTurn]:
    await adapter.start(cid)
    return [await adapter.send(cid, m) for m in messages]


def names(turn: AgentTurn) -> list[str]:
    return [c.name for c in turn.tool_calls]


@pytest.mark.parametrize(
    ("message", "path", "text", "tools"),
    [
        ("hello", ["decide", "respond"], TEXT_GREET, []),
        ("I want a refund", ["decide", "respond"], TEXT_ASK_ORDER, []),
        (
            "where is 1043?",
            ["decide", "lookup", "respond"],
            "Order 1043 (Camp Mug) is shipped.",
            ["lookup_order"],
        ),
        (
            "refund order 1043",
            ["decide", "lookup", "respond"],
            "Order 1043 is not eligible for a refund because it is shipped.",
            ["lookup_order"],
        ),
        (
            "refund order 9999",
            ["decide", "lookup", "respond"],
            "I could not find order 9999. Could you check the number?",
            ["lookup_order"],
        ),
        (
            "refund order 1042",
            ["decide", "lookup", "approve", "refund", "respond"],
            "I looked up order 1042 and issued a refund of $89.00 for the Trail Jacket.",
            ["lookup_order", "issue_refund"],
        ),
    ],
)
async def test_v1_branches(message: str, path: list[str], text: str, tools: list[str]) -> None:
    (turn,) = await say(LangGraphAdapter(build_graph("v1"), resume="approve"), message)
    assert turn.metadata["node_path"] == path
    assert turn.text == text
    assert names(turn) == tools


async def test_approve_issues_refund_with_amount() -> None:
    (turn,) = await say(LangGraphAdapter(build_graph("v1"), resume="approve"), "refund 1044")
    assert turn.metadata["interrupts"] == [
        {"action": "issue_refund", "order_id": "1044", "amount": 18.5}
    ]
    refund = turn.tool_calls[1]
    assert refund.args == {"order_id": "1044", "amount": 18.5}
    assert refund.result == {"refunded": True, "order_id": "1044", "amount": 18.5}


async def test_deny_declines_and_never_refunds() -> None:
    adapter = LangGraphAdapter(build_graph("v1"), resume="deny")
    first, second = await say(adapter, "refund 1042", "please?")
    assert first.text == DECLINED_TEXT
    assert names(first) == ["lookup_order"]
    assert first.metadata["node_path"] == ["decide", "lookup", "approve", "respond"]
    # Not refunded, so the next turn asks for approval again rather than "already".
    assert "issue_refund" not in names(second)


async def test_v2_unverified_goes_to_approval_with_no_amount() -> None:
    (turn,) = await say(LangGraphAdapter(build_graph("v2"), resume="approve"), "I want a refund")
    assert turn.metadata["node_path"] == ["decide", "approve", "refund", "respond"]
    assert turn.metadata["interrupts"] == [
        {"action": "issue_refund", "order_id": None, "amount": None}
    ]
    assert turn.text == TEXT_UNVERIFIED
    assert turn.tool_calls[0].args == {"order_id": None, "amount": None}


async def test_v2_unverified_denied() -> None:
    (turn,) = await say(LangGraphAdapter(build_graph("v2"), resume="deny"), "I want a refund")
    assert turn.text == DECLINED_TEXT
    assert turn.tool_calls == []


@pytest.mark.parametrize("version", ["v1", "v2"])
async def test_no_approval_never_interrupts(version: Any) -> None:
    adapter = LangGraphAdapter(build_graph(version, require_approval=False))  # resume=None
    (turn,) = await say(adapter, "refund order 1042")
    assert turn.metadata["interrupts"] == []
    assert turn.metadata["resumes"] == 0
    assert "approve" not in turn.metadata["node_path"]
    assert "issue_refund" in names(turn)
    assert turn.text != DECLINED_TEXT


async def test_already_refunded_on_second_turn() -> None:
    adapter = LangGraphAdapter(build_graph("v1"), resume="approve")
    _first, second = await say(adapter, "refund 1042", "thanks")
    assert second.metadata["node_path"] == ["decide", "respond"]
    assert second.text.startswith("Your refund is already in progress.")
