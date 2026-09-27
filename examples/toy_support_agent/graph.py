"""The toy support agent rebuilt as a LangGraph graph, with a human approval step.

It reuses decide(), the tools, and the reply texts from convoprobe.demo.toy_agent,
so it behaves exactly like the plain function version. Needs the [langgraph] extra.

Point convoprobe at one of the module-level adapters, for example:
    --adapter examples.toy_support_agent.graph:adapter_v1
"""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any, Literal

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import interrupt

from convoprobe.adapters.langgraph_adapter import LangGraphAdapter
from convoprobe.datamodels import Message, ToolCallRecord
from convoprobe.demo.toy_agent import (
    TEXT_ALREADY,
    TEXT_ASK_ORDER,
    TEXT_GREET,
    TEXT_UNVERIFIED,
    decide,
    issue_refund,
    lookup_order,
    not_found_text,
    not_refundable_text,
    refunded_text,
    status_text,
)

DECLINED_TEXT = (
    "A staff member did not approve this refund. I have passed your request to the team."
)


class ToyState(MessagesState):
    decision: dict[str, Any] | None
    order: dict[str, Any] | None
    approved: bool | None


def to_history(messages: list[AnyMessage]) -> list[Message]:
    """Graph messages as convoprobe messages. Tool results are not needed to decide."""
    history: list[Message] = []
    for message in messages:
        if isinstance(message, HumanMessage):
            history.append(Message(role="user", text=str(message.content)))
        elif isinstance(message, AIMessage):
            calls = [
                ToolCallRecord(name=c["name"], args=c.get("args") or {}) for c in message.tool_calls
            ]
            history.append(Message(role="agent", text=str(message.content), tool_calls=calls))
    return history


def _tool_step(state: ToyState, name: str, args: dict[str, Any], result: Any) -> list[AnyMessage]:
    call_id = f"call_{len(state['messages'])}"
    return [
        AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}]),
        ToolMessage(content=json.dumps(result, sort_keys=True), tool_call_id=call_id),
    ]


def _refund_amount(state: ToyState) -> float | None:
    decision = state["decision"] or {}
    if decision.get("action") == "refund_unverified":
        return None
    return (state.get("order") or {}).get("amount")


def _refundable(state: ToyState) -> bool:
    order = state.get("order") or {}
    return bool(order.get("found") and order.get("refundable"))


def build_graph(
    version: Literal["v1", "v2"], *, require_approval: bool = True, checkpointer: Any = None
) -> Any:
    """Compile the toy agent graph. The checkpointer defaults to InMemorySaver."""

    def decide_node(state: ToyState) -> dict[str, Any]:
        decision = decide(to_history(state["messages"]), version)
        return {"decision": asdict(decision), "order": None, "approved": None}

    def lookup_node(state: ToyState) -> dict[str, Any]:
        oid = (state["decision"] or {})["order_id"]
        result = lookup_order(oid)
        messages = _tool_step(state, "lookup_order", {"order_id": oid}, result)
        return {"messages": messages, "order": result}

    def approve_node(state: ToyState) -> dict[str, Any]:
        oid = (state["decision"] or {}).get("order_id")
        value = interrupt(
            {"action": "issue_refund", "order_id": oid, "amount": _refund_amount(state)}
        )
        return {"approved": value == "approve"}

    def refund_node(state: ToyState) -> dict[str, Any]:
        oid = (state["decision"] or {}).get("order_id")
        amount = _refund_amount(state)
        result = issue_refund(oid, amount)
        args = {"order_id": oid, "amount": amount}
        return {"messages": _tool_step(state, "issue_refund", args, result)}

    def respond_node(state: ToyState) -> dict[str, Any]:
        decision = state["decision"] or {}
        action, oid = decision.get("action"), decision.get("order_id")
        order = state.get("order") or {}
        # None means no approval step ran (require_approval=False): not a decline.
        approved = state.get("approved") is not False
        if action == "already_refunded":
            text = TEXT_ALREADY
        elif action == "ask_order":
            text = TEXT_ASK_ORDER
        elif action == "greet":
            text = TEXT_GREET
        elif action == "refund_unverified":
            text = TEXT_UNVERIFIED if approved else DECLINED_TEXT
        elif not order.get("found"):
            text = not_found_text(oid)
        elif action == "status":
            text = status_text(oid, order)
        elif not order.get("refundable"):
            text = not_refundable_text(oid, order)
        else:
            text = refunded_text(oid, order) if approved else DECLINED_TEXT
        return {"messages": [AIMessage(content=text)]}

    refund_gate = "approve" if require_approval else "refund"

    def after_decide(state: ToyState) -> str:
        action = (state["decision"] or {}).get("action")
        if action in ("refund", "status"):
            return "lookup"
        if action == "refund_unverified":
            return refund_gate
        return "respond"

    def after_lookup(state: ToyState) -> str:
        action = (state["decision"] or {}).get("action")
        return refund_gate if action == "refund" and _refundable(state) else "respond"

    def after_approve(state: ToyState) -> str:
        return "refund" if state.get("approved") else "respond"

    builder = StateGraph(ToyState)
    builder.add_node("decide", decide_node)
    builder.add_node("lookup", lookup_node)
    builder.add_node("approve", approve_node)
    builder.add_node("refund", refund_node)
    builder.add_node("respond", respond_node)
    builder.add_edge(START, "decide")
    builder.add_conditional_edges(
        "decide", after_decide, ["lookup", "approve", "refund", "respond"]
    )
    builder.add_conditional_edges("lookup", after_lookup, ["approve", "refund", "respond"])
    builder.add_conditional_edges("approve", after_approve, ["refund", "respond"])
    builder.add_edge("refund", "respond")
    builder.add_edge("respond", END)
    return builder.compile(
        checkpointer=checkpointer if checkpointer is not None else InMemorySaver()
    )


adapter_v1 = LangGraphAdapter(build_graph("v1"), resume="approve")
adapter_v2 = LangGraphAdapter(build_graph("v2"), resume="approve")
adapter_v1_deny = LangGraphAdapter(build_graph("v1"), resume="deny")
