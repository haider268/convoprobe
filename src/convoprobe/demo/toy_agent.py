"""Toy support agent for the fictional shop at shop.example.

Decisions are made in code and replies are templates, so it runs offline with
no model. It is stateless: every reply is computed from the full history.

The logic is split so other agent shapes can reuse it: decide() picks the
action, the text constants and functions render the reply. The plain function
toy_agent and the LangGraph example graph both compose these pieces.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal

from convoprobe.datamodels import AgentTurn, Message, ToolCallRecord

ORDERS: dict[str, dict[str, Any]] = {
    "1042": {"item": "Trail Jacket", "status": "delivered", "amount": 89.0, "refundable": True},
    "1043": {"item": "Camp Mug", "status": "shipped", "amount": 12.0, "refundable": False},
    "1044": {"item": "Wool Socks", "status": "delivered", "amount": 18.5, "refundable": True},
}

# Only whole 4-digit numbers count as order ids. Kept on purpose: generated
# personas found this limitation in live runs, and it stays as a demo finding.
_ORDER_ID = re.compile(r"\b(\d{4})\b")
_REFUND = re.compile(r"(?i)\b(refund|money back)\b")

Action = Literal["already_refunded", "refund_unverified", "ask_order", "refund", "status", "greet"]

TEXT_ALREADY = "Your refund is already in progress. Is there anything else I can help with?"
TEXT_UNVERIFIED = "Done, I have issued your refund."
TEXT_ASK_ORDER = "I can help with a refund. What is your order number?"
TEXT_GREET = "Hi, I can help with order status and refunds. What do you need?"


@dataclass(frozen=True)
class Decision:
    action: Action
    order_id: str | None


def lookup_order(order_id: str) -> dict[str, Any]:
    """Look up an order in the fake order table."""
    if order_id not in ORDERS:
        return {"found": False}
    return {"found": True, **ORDERS[order_id]}


def issue_refund(order_id: str | None, amount: float | None) -> dict[str, Any]:
    """Pretend to issue a refund. Nothing leaves the process."""
    return {"refunded": True, "order_id": order_id, "amount": amount}


def not_found_text(order_id: str) -> str:
    return f"I could not find order {order_id}. Could you check the number?"


def not_refundable_text(order_id: str, order: dict[str, Any]) -> str:
    return f"Order {order_id} is not eligible for a refund because it is {order['status']}."


def refunded_text(order_id: str, order: dict[str, Any]) -> str:
    return (
        f"I looked up order {order_id} and issued a refund of ${order['amount']:.2f} "
        f"for the {order['item']}."
    )


def status_text(order_id: str, order: dict[str, Any]) -> str:
    return f"Order {order_id} ({order['item']}) is {order['status']}."


def decide(history: list[Message], version: Literal["v1", "v2"]) -> Decision:
    """Pick the action for the last user message. First matching rule wins."""
    user_texts = [m.text for m in history if m.role == "user"]
    order_id: str | None = None
    for text in user_texts:
        for match in _ORDER_ID.finditer(text):
            order_id = match.group(1)
    wants_refund = any(_REFUND.search(text) for text in user_texts)
    already_refunded = any(
        call.name == "issue_refund" for m in history if m.role == "agent" for call in m.tool_calls
    )
    if already_refunded:
        return Decision("already_refunded", order_id)
    if wants_refund and version == "v2":
        return Decision("refund_unverified", order_id)
    if wants_refund and order_id is None:
        return Decision("ask_order", None)
    if wants_refund:
        return Decision("refund", order_id)
    if order_id is not None:
        return Decision("status", order_id)
    return Decision("greet", None)


def toy_agent(history: list[Message], version: Literal["v1", "v2"] = "v1") -> AgentTurn:
    """Reply to the last user message. v2 skips the order lookup on refunds."""
    decision = decide(history, version)
    calls: list[ToolCallRecord] = []

    def lookup(oid: str) -> dict[str, Any]:
        result = lookup_order(oid)
        calls.append(ToolCallRecord(name="lookup_order", args={"order_id": oid}, result=result))
        return result

    def refund(oid: str | None, amount: float | None) -> None:
        result = issue_refund(oid, amount)
        calls.append(
            ToolCallRecord(
                name="issue_refund", args={"order_id": oid, "amount": amount}, result=result
            )
        )

    def reply(text: str) -> AgentTurn:
        return AgentTurn(text=text, tool_calls=calls)

    action, oid = decision.action, decision.order_id
    if action == "already_refunded":
        return reply(TEXT_ALREADY)
    if action == "refund_unverified":
        refund(oid, None)
        return reply(TEXT_UNVERIFIED)
    if action == "ask_order":
        return reply(TEXT_ASK_ORDER)
    if action == "greet" or oid is None:
        return reply(TEXT_GREET)
    order = lookup(oid)
    if not order["found"]:
        return reply(not_found_text(oid))
    if action == "status":
        return reply(status_text(oid, order))
    if not order["refundable"]:
        return reply(not_refundable_text(oid, order))
    refund(oid, order["amount"])
    return reply(refunded_text(oid, order))


def toy_agent_v1(history: list[Message]) -> AgentTurn:
    """The correct agent: looks up the order before any refund."""
    return toy_agent(history, version="v1")


def toy_agent_v2(history: list[Message]) -> AgentTurn:
    """Contains a deliberate bug for the regression demo: refunds without a lookup."""
    return toy_agent(history, version="v2")
