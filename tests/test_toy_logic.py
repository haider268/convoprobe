"""decide() and the reply texts shared by every toy agent shape."""

from __future__ import annotations

import pytest

from convoprobe.datamodels import Message, ToolCallRecord
from convoprobe.demo.toy_agent import (
    ORDERS,
    TEXT_ALREADY,
    TEXT_ASK_ORDER,
    TEXT_GREET,
    TEXT_UNVERIFIED,
    Decision,
    decide,
    not_found_text,
    not_refundable_text,
    refunded_text,
    status_text,
)


def users(*texts: str) -> list[Message]:
    return [Message(role="user", text=t) for t in texts]


REFUNDED = Message(
    role="agent", text="done", tool_calls=[ToolCallRecord(name="issue_refund", args={})]
)


@pytest.mark.parametrize(
    ("history", "v1", "v2"),
    [
        (
            [*users("refund 1042"), REFUNDED, *users("thanks")],
            Decision("already_refunded", "1042"),
            Decision("already_refunded", "1042"),
        ),
        (
            users("I want a refund"),
            Decision("ask_order", None),
            Decision("refund_unverified", None),
        ),
        (
            users("refund order 1042"),
            Decision("refund", "1042"),
            Decision("refund_unverified", "1042"),
        ),
        (
            users("money back for 1042", "actually 1044"),
            Decision("refund", "1044"),
            Decision("refund_unverified", "1044"),
        ),
        (users("where is 1043?"), Decision("status", "1043"), Decision("status", "1043")),
        (users("hello"), Decision("greet", None), Decision("greet", None)),
        # Kept limitation: only whole 4-digit numbers are order ids.
        (users("where is order 10420?"), Decision("greet", None), Decision("greet", None)),
    ],
)
def test_decide(history: list[Message], v1: Decision, v2: Decision) -> None:
    assert decide(history, "v1") == v1
    assert decide(history, "v2") == v2


def test_texts() -> None:
    assert TEXT_ALREADY == (
        "Your refund is already in progress. Is there anything else I can help with?"
    )
    assert TEXT_UNVERIFIED == "Done, I have issued your refund."
    assert TEXT_ASK_ORDER == "I can help with a refund. What is your order number?"
    assert TEXT_GREET == "Hi, I can help with order status and refunds. What do you need?"
    assert not_found_text("9999") == "I could not find order 9999. Could you check the number?"
    assert not_refundable_text("1043", ORDERS["1043"]) == (
        "Order 1043 is not eligible for a refund because it is shipped."
    )
    assert refunded_text("1044", ORDERS["1044"]) == (
        "I looked up order 1044 and issued a refund of $18.50 for the Wool Socks."
    )
    assert status_text("1042", ORDERS["1042"]) == "Order 1042 (Trail Jacket) is delivered."
