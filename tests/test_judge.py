"""LLM judge: transcript format, verdicts, evidence validation, retries."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from pydantic import TypeAdapter

from convoprobe.datamodels import Conversation, Message, RubricItem, Scenario, ToolCallRecord
from convoprobe.errors import TransientError
from convoprobe.models.cache import SqliteLLMCache
from convoprobe.scoring.judge import (
    JUDGE_HUMAN,
    JUDGE_SYSTEM,
    ScaleVerdict,
    YesNoVerdict,
    format_transcript,
    judge_item,
    run_judge_checks,
)
from tests.llm_doubles import ServerDown, fake_client

rubric_adapter: TypeAdapter[Any] = TypeAdapter(RubricItem)

SCENARIO = Scenario.model_validate(
    {
        "name": "t",
        "agent_description": "Support agent for shop.example.",
        "personas": [{"id": "pat", "script": ["hi"]}],
        "rubric": [{"id": "r", "type": "max_turns", "value": 5}],
    }
)

CONVERSATION = Conversation(
    id="pat-r0",
    persona_id="pat",
    messages=[
        Message(role="user", text="I want a refund for 1042"),
        Message(
            role="agent",
            text="Done, refund   issued.",
            tool_calls=[
                ToolCallRecord(
                    name="issue_refund",
                    args={"order_id": "1042", "amount": 89.0},
                    result={"refunded": True},
                )
            ],
        ),
        Message(role="user", text="thanks"),
    ],
)

TRANSCRIPT = (
    "turn 1 USER: I want a refund for 1042\n"
    "turn 1 AGENT: Done, refund   issued.\n"
    'turn 1 AGENT TOOL CALL: issue_refund({"amount": 89.0, "order_id": "1042"})'
    ' -> {"refunded": true}\n'
    "turn 2 USER: thanks"
)

YES_NO = rubric_adapter.validate_python(
    {"id": "polite", "type": "llm_judge", "question": "Was the agent polite?", "pass_if": "yes"}
)
SCALE = rubric_adapter.validate_python(
    {"id": "tone", "type": "llm_judge", "question": "Tone?", "scale": "1-5", "min_pass": 3}
)


def yes_no(verdict: str, *evidence: tuple[int, str], reasoning: str = "clear") -> dict[str, Any]:
    return {
        "reasoning": reasoning,
        "verdict": verdict,
        "evidence": [{"turn": t, "quote": q} for t, q in evidence],
    }


def scale(score: int, *evidence: tuple[int, str]) -> dict[str, Any]:
    return {
        "reasoning": "ok",
        "score": score,
        "evidence": [{"turn": t, "quote": q} for t, q in evidence],
    }


@pytest.fixture
def cache(tmp_path: Path) -> Iterator[SqliteLLMCache]:
    with SqliteLLMCache(tmp_path / "cache.db") as opened:
        yield opened


def test_format_transcript() -> None:
    assert format_transcript(CONVERSATION) == TRANSCRIPT


def test_format_transcript_empty() -> None:
    assert format_transcript(Conversation(id="x", persona_id="p")) == ""


def test_schemas_put_reasoning_first() -> None:
    assert next(iter(YesNoVerdict.model_fields)) == "reasoning"
    assert next(iter(ScaleVerdict.model_fields)) == "reasoning"
    assert "quote_found" not in str(YesNoVerdict.model_json_schema())


async def test_yes_no_pass_and_prompt(cache: SqliteLLMCache) -> None:
    client, model = fake_client(cache, [yes_no("yes", (1, "refund issued"))])
    result = await judge_item(YES_NO, CONVERSATION, SCENARIO, client)
    assert (result.status, result.source, result.verdict) == ("pass", "judge", "yes")
    assert result.detail == "verdict yes (pass if yes): clear"
    assert result.evidence[0].quote_found is True
    system, human = model.seen[0]
    assert system.content == JUDGE_SYSTEM
    assert human.content == JUDGE_HUMAN.format(
        agent_description="Support agent for shop.example.",
        transcript=TRANSCRIPT,
        question="Was the agent polite?",
        instruction="Answer verdict yes or no.",
    )


async def test_yes_no_fail(cache: SqliteLLMCache) -> None:
    client, _ = fake_client(cache, [yes_no("no", (2, "thanks"))])
    result = await judge_item(YES_NO, CONVERSATION, SCENARIO, client)
    assert result.status == "fail"
    assert result.detail == "verdict no (pass if yes): clear"


async def test_scale_pass_fail_and_instruction(cache: SqliteLLMCache) -> None:
    client, model = fake_client(cache, [scale(4), scale(2)])
    passed = await judge_item(SCALE, CONVERSATION, SCENARIO, client)
    assert (passed.status, passed.score, passed.detail) == ("pass", 4, "score 4/5 (min 3): ok")
    assert str(model.seen[0][1].content).endswith(
        "Give a score from 1 (very poor) to 5 (excellent)."
    )
    other = CONVERSATION.model_copy(update={"id": "pat-r1"})
    failed = await judge_item(SCALE, other, SCENARIO, client)
    assert (failed.status, failed.detail) == ("fail", "score 2/5 (min 3): ok")


async def test_invalid_turn_then_valid_uses_retry(cache: SqliteLLMCache) -> None:
    client, model = fake_client(cache, [yes_no("yes", (5, "x")), yes_no("yes", (1, "Done"))])
    result = await judge_item(YES_NO, CONVERSATION, SCENARIO, client)
    assert result.status == "pass"
    assert model.calls == 2
    assert model.seen[0] == model.seen[1]


async def test_two_invalid_turns_is_error(cache: SqliteLLMCache) -> None:
    client, _ = fake_client(cache, [yes_no("yes", (9, "x")), yes_no("yes", (0, "x"))])
    result = await judge_item(YES_NO, CONVERSATION, SCENARIO, client)
    assert result.status == "error"
    assert result.detail == "judge output invalid: evidence turn 0 is outside 1..2"
    assert result.source == "judge"


async def test_structured_error_twice_is_error(cache: SqliteLLMCache) -> None:
    client, _ = fake_client(cache, ["BAD", "BAD"])
    result = await judge_item(YES_NO, CONVERSATION, SCENARIO, client)
    assert result.status == "error"
    assert result.detail == (
        "judge output invalid: ollama:fake returned output that does not match"
        " YesNoVerdict: not json"
    )


async def test_quote_found_and_missing_suffix(cache: SqliteLLMCache) -> None:
    output = yes_no(
        "yes",
        (1, "DONE,  refund issued"),
        (1, "issue_refund"),
        (2, "I love this shop"),
    )
    client, _ = fake_client(cache, [output])
    result = await judge_item(YES_NO, CONVERSATION, SCENARIO, client)
    assert [e.quote_found for e in result.evidence] == [True, True, False]
    assert result.detail == "verdict yes (pass if yes): clear [1 quote(s) not found in transcript]"


async def test_reasoning_truncated(cache: SqliteLLMCache) -> None:
    client, _ = fake_client(cache, [yes_no("yes", reasoning="x" * 500)])
    result = await judge_item(YES_NO, CONVERSATION, SCENARIO, client)
    assert result.detail == "verdict yes (pass if yes): " + "x" * 300


async def test_transient_error_propagates(cache: SqliteLLMCache) -> None:
    client, _ = fake_client(cache, [ServerDown("down")] * 4)
    with pytest.raises(TransientError):
        await judge_item(YES_NO, CONVERSATION, SCENARIO, client)


async def test_run_judge_checks_skips(cache: SqliteLLMCache) -> None:
    only_other = rubric_adapter.validate_python(
        {
            "id": "other_only",
            "type": "llm_judge",
            "question": "Q?",
            "pass_if": "yes",
            "applies_to": ["someone_else"],
        }
    )
    deterministic = rubric_adapter.validate_python({"id": "t", "type": "max_turns", "value": 3})
    rubric = [deterministic, only_other, YES_NO]

    none_client = await run_judge_checks(rubric, CONVERSATION, SCENARIO, None)
    assert [(r.rubric_id, r.status, r.detail, r.source) for r in none_client] == [
        ("other_only", "skip", "not applicable to persona pat", "judge"),
        ("polite", "skip", "no judge model configured", "judge"),
    ]

    client, model = fake_client(cache, [yes_no("yes")])
    judged = await run_judge_checks(rubric, CONVERSATION, SCENARIO, client)
    assert [(r.rubric_id, r.status) for r in judged] == [("other_only", "skip"), ("polite", "pass")]
    assert model.calls == 1
