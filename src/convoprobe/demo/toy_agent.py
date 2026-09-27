"""Toy support agent for the fictional shop at shop.example.

Decisions are made in code and replies are templates, so it runs offline with
no model. It is stateless: every reply is computed from the full history.
"""

from __future__ import annotations

import re
from typing import Any, Literal

from convoprobe.datamodels import AgentTurn, Message, ToolCallRecord

ORDERS: dict[str, dict[str, Any]] = {
    "1042": {"item": "Trail Jacket", "status": "delivered", "amount": 89.0, "refundable": True},
    "1043": {"item": "Camp Mug", "status": "shipped", "amount": 12.0, "refundable": False},
    "1044": {"item": "Wool Socks", "status": "delivered", "amount": 18.5, "refundable": True},
}

_ORDER_ID = re.compile(r"\b(\d{4})\b")
_REFUND = re.compile(r"(?i)\b(refund|money back)\b")


def lookup_order(order_id: str) -> dict[str, Any]:
    """Look up an order in the fake order table."""
    if order_id not in ORDERS:
        return {"found": False}
    return {"found": True, **ORDERS[order_id]}


def issue_refund(order_id: str | None, amount: float | None) -> dict[str, Any]:
    """Pretend to issue a refund. Nothing leaves the process."""
    return {"refunded": True, "order_id": order_id, "amount": amount}


def toy_agent(history: list[Message], version: Literal["v1", "v2"] = "v1") -> AgentTurn:
    """Reply to the last user message. v2 skips the order lookup on refunds."""
    user_texts = [m.text for m in history if m.role == "user"]
    order_id: str | None = None
    for text in user_texts:
        for match in _ORDER_ID.finditer(text):
            order_id = match.group(1)
    wants_refund = any(_REFUND.search(text) for text in user_texts)
    already_refunded = any(
        call.name == "issue_refund" for m in history if m.role == "agent" for call in m.tool_calls
    )
    calls: list[ToolCallRecord] = []

    def lookup(oid: str) -> dict[str, Any]:
        result = lookup_order(oid)
        calls.append(ToolCallRecord(name="lookup_order", args={"order_id": oid}, result=result))
        return result

    def refund(oid: str | None, amount: float | None) -> dict[str, Any]:
        result = issue_refund(oid, amount)
        calls.append(
            ToolCallRecord(
                name="issue_refund", args={"order_id": oid, "amount": amount}, result=result
            )
        )
        return result

    def reply(text: str) -> AgentTurn:
        return AgentTurn(text=text, tool_calls=calls)

    if already_refunded:
        return reply("Your refund is already in progress. Is there anything else I can help with?")

    if wants_refund and version == "v2":
        refund(order_id, None)
        return reply("Done, I have issued your refund.")

    if wants_refund and order_id is None:
        return reply("I can help with a refund. What is your order number?")

    if wants_refund and order_id is not None:
        order = lookup(order_id)
        if not order["found"]:
            return reply(f"I could not find order {order_id}. Could you check the number?")
        if not order["refundable"]:
            return reply(
                f"Order {order_id} is not eligible for a refund because it is {order['status']}."
            )
        refund(order_id, order["amount"])
        return reply(
            f"I looked up order {order_id} and issued a refund of ${order['amount']:.2f} "
            f"for the {order['item']}."
        )

    if order_id is not None:
        order = lookup(order_id)
        if not order["found"]:
            return reply(f"I could not find order {order_id}. Could you check the number?")
        return reply(f"Order {order_id} ({order['item']}) is {order['status']}.")

    return reply("Hi, I can help with order status and refunds. What do you need?")


def toy_agent_v1(history: list[Message]) -> AgentTurn:
    """The correct agent: looks up the order before any refund."""
    return toy_agent(history, version="v1")


def toy_agent_v2(history: list[Message]) -> AgentTurn:
    """Contains a deliberate bug for the regression demo: refunds without a lookup."""
    return toy_agent(history, version="v2")
