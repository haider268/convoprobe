"""Deterministic rubric checks. Plain Python, no LLM.

"Turn N" in details is the 1-based index of the message among messages from
the same speaker (for tool checks, always the agent).
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any, Literal

from convoprobe.datamodels import (
    CheckResult,
    Conversation,
    LLMJudgeItem,
    MaxLatencyItem,
    MaxTurnsItem,
    RegexAbsentItem,
    RegexPresentItem,
    RubricItem,
    ToolArgsItem,
    ToolCalledItem,
    ToolNotCalledItem,
    ToolOrderItem,
)

Status = Literal["pass", "fail", "skip", "error"]
Outcome = tuple[Status, str]


def _fmt(number: float) -> str:
    return str(int(number)) if float(number).is_integer() else f"{number:.1f}"


def _tool_called(item: ToolCalledItem, conv: Conversation) -> Outcome:
    for index, call in conv.tool_calls_in_order():
        if call.name == item.tool:
            return "pass", f"{item.tool} called at turn {index + 1}"
    return "fail", f"{item.tool} was never called"


def _tool_not_called(item: ToolNotCalledItem, conv: Conversation) -> Outcome:
    for index, call in conv.tool_calls_in_order():
        if call.name == item.tool:
            return "fail", f"{item.tool} called at turn {index + 1}"
    return "pass", f"{item.tool} never called"


def _tool_order(item: ToolOrderItem, conv: Conversation) -> Outcome:
    seen_before = False
    after_called = False
    for index, call in conv.tool_calls_in_order():
        if call.name == item.before:
            seen_before = True
        elif call.name == item.after:
            after_called = True
            if not seen_before:
                return "fail", f"{item.after} called at turn {index + 1} before any {item.before}"
    if not after_called:
        return "pass", f"{item.after} not called"
    return "pass", f"{item.before} called before {item.after}"


def _speaker_matches(
    item: RegexPresentItem | RegexAbsentItem, conv: Conversation
) -> tuple[int, re.Match[str]] | None:
    """First (1-based speaker turn, match) for the item's pattern, or None."""
    flags = re.IGNORECASE if item.ignore_case else 0
    regex = re.compile(item.pattern, flags)
    messages = [m for m in conv.messages if m.role == item.speaker]
    for index, message in enumerate(messages):
        match = regex.search(message.text)
        if match:
            return index + 1, match
    return None


def _regex_present(item: RegexPresentItem, conv: Conversation) -> Outcome:
    found = _speaker_matches(item, conv)
    if found is None:
        return "fail", f"no {item.speaker} message matched /{item.pattern}/"
    return "pass", f"{item.speaker} turn {found[0]} matched /{item.pattern}/"


def _regex_absent(item: RegexAbsentItem, conv: Conversation) -> Outcome:
    found = _speaker_matches(item, conv)
    if found is None:
        return "pass", f"no {item.speaker} message matched /{item.pattern}/"
    turn, match = found
    return "fail", f"{item.speaker} turn {turn} matched /{item.pattern}/: '{match.group(0)}'"


def _max_turns(item: MaxTurnsItem, conv: Conversation) -> Outcome:
    count = len(conv.agent_messages())
    status: Status = "pass" if count <= item.value else "fail"
    return status, f"{count} agent turns (limit {item.value})"


def _max_latency(item: MaxLatencyItem, conv: Conversation) -> Outcome:
    timed = [
        (index, m.latency_ms)
        for index, m in enumerate(conv.agent_messages())
        if m.latency_ms is not None
    ]
    if not timed:
        return "skip", "no latency recorded"
    index, slowest = max(timed, key=lambda pair: pair[1])
    took = f"turn {index + 1} took {_fmt(slowest)} ms (limit {_fmt(item.value)})"
    if slowest <= item.value:
        return "pass", f"slowest {took}"
    return "fail", took


def _tool_args(item: ToolArgsItem, conv: Conversation) -> Outcome:
    calls = [(i, c) for i, c in conv.tool_calls_in_order() if c.name == item.tool]
    if not calls:
        return "pass", f"{item.tool} not called"
    for index, call in calls:
        for key in item.required:
            if call.args.get(key) in (None, ""):
                return "fail", f"turn {index + 1}: {item.tool} missing {key}"
    return "pass", f"every {item.tool} call has {', '.join(item.required)}"


_CHECKS: dict[type, Callable[[Any, Conversation], Outcome]] = {
    ToolCalledItem: _tool_called,
    ToolNotCalledItem: _tool_not_called,
    ToolOrderItem: _tool_order,
    RegexPresentItem: _regex_present,
    RegexAbsentItem: _regex_absent,
    MaxTurnsItem: _max_turns,
    MaxLatencyItem: _max_latency,
    ToolArgsItem: _tool_args,
}


def run_deterministic_checks(
    rubric: list[RubricItem], conversation: Conversation
) -> list[CheckResult]:
    """Run every non-LLM rubric item against one conversation."""
    results: list[CheckResult] = []
    for item in rubric:
        if isinstance(item, LLMJudgeItem):
            continue
        status: Status
        if item.applies_to is not None and conversation.persona_id not in item.applies_to:
            status, detail = "skip", f"not applicable to persona {conversation.persona_id}"
        else:
            try:
                status, detail = _CHECKS[type(item)](item, conversation)
            except Exception as exc:
                status, detail = "error", str(exc)
        results.append(
            CheckResult(
                rubric_id=item.id,
                persona_id=conversation.persona_id,
                conversation_id=conversation.id,
                status=status,
                detail=detail,
            )
        )
    return results
