"""Deterministic checks: pass, fail, skip, and error for every check type."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import TypeAdapter

from convoprobe.datamodels import (
    CheckResult,
    Conversation,
    Message,
    RubricItem,
    ToolCalledItem,
    ToolCallRecord,
)
from convoprobe.scoring import deterministic
from convoprobe.scoring.deterministic import run_deterministic_checks

rubric_adapter: TypeAdapter[Any] = TypeAdapter(RubricItem)


def item(**data: Any) -> Any:
    data.setdefault("id", "check")
    return rubric_adapter.validate_python(data)


def call(name: str, **args: Any) -> ToolCallRecord:
    return ToolCallRecord(name=name, args=args)


def agent(text: str = "ok", *calls: ToolCallRecord, latency: float | None = None) -> Message:
    return Message(role="agent", text=text, tool_calls=list(calls), latency_ms=latency)


def user(text: str = "hi") -> Message:
    return Message(role="user", text=text)


def conv(*messages: Message, persona: str = "pat") -> Conversation:
    return Conversation(id="c1", persona_id=persona, messages=list(messages))


def check(rubric_item: Any, conversation: Conversation) -> CheckResult:
    results = run_deterministic_checks([rubric_item], conversation)
    assert len(results) == 1
    return results[0]


def assert_result(result: CheckResult, status: str, detail: str) -> None:
    assert (result.status, result.detail) == (status, detail)


# tool_called


def test_tool_called_pass() -> None:
    c = conv(user(), agent("a"), user(), agent("b", call("lookup_order")))
    assert_result(
        check(item(type="tool_called", tool="lookup_order"), c),
        "pass",
        "lookup_order called at turn 2",
    )


def test_tool_called_fail() -> None:
    c = conv(user(), agent("a", call("other")))
    assert_result(
        check(item(type="tool_called", tool="lookup_order"), c),
        "fail",
        "lookup_order was never called",
    )


# tool_not_called


def test_tool_not_called_pass() -> None:
    c = conv(user(), agent("a", call("lookup_order")))
    assert_result(
        check(item(type="tool_not_called", tool="issue_refund"), c),
        "pass",
        "issue_refund never called",
    )


def test_tool_not_called_fail() -> None:
    c = conv(user(), agent("a"), user(), agent("b"), user(), agent("c", call("issue_refund")))
    assert_result(
        check(item(type="tool_not_called", tool="issue_refund"), c),
        "fail",
        "issue_refund called at turn 3",
    )


# tool_order

ORDER = {"type": "tool_order", "before": "lookup_order", "after": "issue_refund"}


def test_tool_order_pass_across_turns() -> None:
    c = conv(user(), agent("a", call("lookup_order")), user(), agent("b", call("issue_refund")))
    assert_result(check(item(**ORDER), c), "pass", "lookup_order called before issue_refund")


def test_tool_order_pass_same_message() -> None:
    c = conv(user(), agent("a", call("lookup_order"), call("issue_refund")))
    assert_result(check(item(**ORDER), c), "pass", "lookup_order called before issue_refund")


def test_tool_order_after_not_called() -> None:
    c = conv(user(), agent("a", call("lookup_order")))
    assert_result(check(item(**ORDER), c), "pass", "issue_refund not called")


def test_tool_order_fail_same_message_wrong_order() -> None:
    c = conv(user(), agent("a", call("issue_refund"), call("lookup_order")))
    assert_result(
        check(item(**ORDER), c), "fail", "issue_refund called at turn 1 before any lookup_order"
    )


def test_tool_order_fail_never_looked_up() -> None:
    c = conv(user(), agent("a"), user(), agent("b", call("issue_refund")))
    assert_result(
        check(item(**ORDER), c), "fail", "issue_refund called at turn 2 before any lookup_order"
    )


# regex_present


def test_regex_present_pass_ignore_case() -> None:
    c = conv(user(), agent("hello"), user(), agent("Your ORDER is on its way"))
    assert_result(
        check(item(type="regex_present", pattern="order"), c),
        "pass",
        "agent turn 2 matched /order/",
    )


def test_regex_present_fail_case_sensitive() -> None:
    c = conv(user(), agent("Your ORDER is on its way"))
    assert_result(
        check(item(type="regex_present", pattern="order", ignore_case=False), c),
        "fail",
        "no agent message matched /order/",
    )


def test_regex_present_user_speaker() -> None:
    c = conv(user("hello"), agent("order info"), user("my order 1042"))
    assert_result(
        check(item(type="regex_present", pattern=r"\d{4}", speaker="user"), c),
        "pass",
        r"user turn 2 matched /\d{4}/",
    )


# regex_absent

GUARANTEE = r"\bguarantee(d)?\b"


def test_regex_absent_pass() -> None:
    c = conv(user("is it guaranteed?"), agent("It should arrive soon."))
    assert_result(
        check(item(type="regex_absent", pattern=GUARANTEE), c),
        "pass",
        rf"no agent message matched /{GUARANTEE}/",
    )


def test_regex_absent_fail() -> None:
    c = conv(user(), agent("fine"), user(), agent("Delivery is Guaranteed tomorrow."))
    assert_result(
        check(item(type="regex_absent", pattern=GUARANTEE), c),
        "fail",
        rf"agent turn 2 matched /{GUARANTEE}/: 'Guaranteed'",
    )


# max_turns


def test_max_turns_pass() -> None:
    c = conv(user(), agent(), user(), agent())
    assert_result(check(item(type="max_turns", value=2), c), "pass", "2 agent turns (limit 2)")


def test_max_turns_fail() -> None:
    c = conv(user(), agent(), user(), agent(), user(), agent())
    assert_result(check(item(type="max_turns", value=2), c), "fail", "3 agent turns (limit 2)")


# max_latency_ms


def test_max_latency_pass() -> None:
    c = conv(user(), agent(latency=120.0), user(), agent(latency=480.5))
    assert_result(
        check(item(type="max_latency_ms", value=500), c),
        "pass",
        "slowest turn 2 took 480.5 ms (limit 500)",
    )


def test_max_latency_fail() -> None:
    c = conv(user(), agent(latency=100.0), user(), agent(latency=2500.0), user(), agent())
    assert_result(
        check(item(type="max_latency_ms", value=1000), c),
        "fail",
        "turn 2 took 2500 ms (limit 1000)",
    )


def test_max_latency_skip_without_latency() -> None:
    c = conv(user(), agent(), user(), agent())
    assert_result(check(item(type="max_latency_ms", value=1000), c), "skip", "no latency recorded")


# tool_args

ARGS = {"type": "tool_args", "tool": "issue_refund", "required": ["order_id", "amount"]}


def test_tool_args_pass() -> None:
    c = conv(user(), agent("a", call("issue_refund", order_id="1042", amount=20)))
    assert_result(check(item(**ARGS), c), "pass", "every issue_refund call has order_id, amount")


def test_tool_args_not_called() -> None:
    c = conv(user(), agent("a", call("lookup_order", order_id="1042")))
    assert_result(check(item(**ARGS), c), "pass", "issue_refund not called")


@pytest.mark.parametrize("bad", [None, ""])
def test_tool_args_fail_empty_value(bad: Any) -> None:
    c = conv(
        user(),
        agent("a", call("issue_refund", order_id="1042", amount=5)),
        user(),
        agent("b", call("issue_refund", order_id=bad, amount=5)),
    )
    assert_result(check(item(**ARGS), c), "fail", "turn 2: issue_refund missing order_id")


def test_tool_args_fail_missing_key() -> None:
    c = conv(user(), agent("a", call("issue_refund", order_id="1042")))
    assert_result(check(item(**ARGS), c), "fail", "turn 1: issue_refund missing amount")


# applies_to, llm_judge, error isolation


def test_applies_to_skips_other_personas() -> None:
    rubric_item = item(type="tool_not_called", tool="issue_refund", applies_to=["other"])
    c = conv(user(), agent("a", call("issue_refund")), persona="pat")
    assert_result(check(rubric_item, c), "skip", "not applicable to persona pat")


def test_applies_to_runs_for_listed_persona() -> None:
    rubric_item = item(type="tool_not_called", tool="issue_refund", applies_to=["pat"])
    c = conv(user(), agent("a", call("issue_refund")), persona="pat")
    assert check(rubric_item, c).status == "fail"


def test_llm_judge_ignored() -> None:
    rubric = [
        item(id="judge", type="llm_judge", question="Polite?", pass_if="yes"),
        item(id="turns", type="max_turns", value=5),
    ]
    results = run_deterministic_checks(rubric, conv(user(), agent()))
    assert [r.rubric_id for r in results] == ["turns"]


def test_result_ids_filled() -> None:
    result = check(item(id="turns", type="max_turns", value=5), conv(user(), agent()))
    assert (result.rubric_id, result.persona_id, result.conversation_id) == ("turns", "pat", "c1")


def test_error_isolation(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(_item: Any, _conv: Conversation) -> Any:
        raise RuntimeError("check exploded")

    monkeypatch.setitem(deterministic._CHECKS, ToolCalledItem, boom)
    rubric = [
        item(id="first", type="max_turns", value=5),
        item(id="broken", type="tool_called", tool="x"),
        item(id="last", type="tool_not_called", tool="x"),
    ]
    results = run_deterministic_checks(rubric, conv(user(), agent()))
    assert [(r.rubric_id, r.status, r.detail) for r in results] == [
        ("first", "pass", "1 agent turns (limit 5)"),
        ("broken", "error", "check exploded"),
        ("last", "pass", "x never called"),
    ]
