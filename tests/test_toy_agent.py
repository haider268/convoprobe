"""Toy shop agent: every v1 branch, and the v2 bug."""

from __future__ import annotations

from typing import Any

import pytest

from convoprobe.datamodels import AgentTurn, Message, ToolCallRecord
from convoprobe.demo.toy_agent import (
    ORDERS,
    issue_refund,
    lookup_order,
    toy_agent,
    toy_agent_v1,
    toy_agent_v2,
)

GREETING = "Hi, I can help with order status and refunds. What do you need?"
ASK_ORDER = "I can help with a refund. What is your order number?"
IN_PROGRESS = "Your refund is already in progress. Is there anything else I can help with?"


def history(*items: str | Message) -> list[Message]:
    """Plain strings are user messages; Message objects are used as given."""
    return [Message(role="user", text=i) if isinstance(i, str) else i for i in items]


def calls(turn: AgentTurn) -> list[tuple[str, dict[str, Any]]]:
    return [(c.name, c.args) for c in turn.tool_calls]


def refunded_agent_message() -> Message:
    return Message(
        role="agent",
        text="refunded",
        tool_calls=[ToolCallRecord(name="issue_refund", args={"order_id": "1042"})],
    )


def test_tools() -> None:
    assert lookup_order("9999") == {"found": False}
    assert lookup_order("1042") == {"found": True, **ORDERS["1042"]}
    assert issue_refund("1042", 89.0) == {"refunded": True, "order_id": "1042", "amount": 89.0}


# v1 branches


def test_v1_a_already_refunded() -> None:
    turn = toy_agent_v1(history("refund 1042", refunded_agent_message(), "thanks"))
    assert turn.text == IN_PROGRESS
    assert turn.tool_calls == []


def test_v1_b_refund_without_order() -> None:
    turn = toy_agent_v1(history("I want a refund now."))
    assert turn.text == ASK_ORDER
    assert turn.tool_calls == []


def test_v1_c_refund_issued() -> None:
    turn = toy_agent_v1(history("My order 1042 arrived damaged, I want a refund."))
    assert turn.text == "I looked up order 1042 and issued a refund of $89.00 for the Trail Jacket."
    assert calls(turn) == [
        ("lookup_order", {"order_id": "1042"}),
        ("issue_refund", {"order_id": "1042", "amount": 89.0}),
    ]
    assert turn.tool_calls[0].result == lookup_order("1042")
    assert turn.tool_calls[1].result == {"refunded": True, "order_id": "1042", "amount": 89.0}


def test_v1_c_money_back_and_last_order_id_wins() -> None:
    turn = toy_agent_v1(history("I want my money back for 1042", "sorry, it was 1044"))
    assert turn.text == "I looked up order 1044 and issued a refund of $18.50 for the Wool Socks."
    assert calls(turn)[1] == ("issue_refund", {"order_id": "1044", "amount": 18.5})


def test_v1_c_not_refundable() -> None:
    turn = toy_agent_v1(history("Refund order 1043 please"))
    assert turn.text == "Order 1043 is not eligible for a refund because it is shipped."
    assert calls(turn) == [("lookup_order", {"order_id": "1043"})]


def test_v1_c_not_found() -> None:
    turn = toy_agent_v1(history("Refund order 9999 please"))
    assert turn.text == "I could not find order 9999. Could you check the number?"
    assert calls(turn) == [("lookup_order", {"order_id": "9999"})]


def test_v1_d_order_status() -> None:
    turn = toy_agent_v1(history("Where is order 1043?"))
    assert turn.text == "Order 1043 (Camp Mug) is shipped."
    assert calls(turn) == [("lookup_order", {"order_id": "1043"})]


def test_v1_d_not_found() -> None:
    turn = toy_agent_v1(history("Where is order 9999?"))
    assert turn.text == "I could not find order 9999. Could you check the number?"
    assert calls(turn) == [("lookup_order", {"order_id": "9999"})]


def test_v1_e_greeting() -> None:
    turn = toy_agent_v1(history("hello"))
    assert turn.text == GREETING
    assert turn.tool_calls == []


def test_default_version_is_v1() -> None:
    h = history("Refund order 1042")
    assert toy_agent(h) == toy_agent_v1(h)


# v2 bug


@pytest.mark.parametrize(
    ("text", "order_id"),
    [("I want a refund now.", None), ("My order 1042 arrived damaged, I want a refund.", "1042")],
)
def test_v2_refunds_without_lookup(text: str, order_id: str | None) -> None:
    turn = toy_agent_v2(history(text))
    assert turn.text == "Done, I have issued your refund."
    assert calls(turn) == [("issue_refund", {"order_id": order_id, "amount": None})]


@pytest.mark.parametrize(
    "h",
    [
        history("hello"),
        history("Where is order 1043?"),
        history("Where is order 9999?"),
        history("refund 1042", refunded_agent_message(), "thanks"),
    ],
)
def test_v2_non_refund_paths_match_v1(h: list[Message]) -> None:
    assert toy_agent_v2(h) == toy_agent_v1(h)
