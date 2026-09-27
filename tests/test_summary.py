"""summarize: pass rates and the llm_judge placeholder row."""

from __future__ import annotations

from typing import Any

from pydantic import TypeAdapter

from convoprobe.datamodels import CheckResult, RubricItem
from convoprobe.summary import summarize

rubric_adapter: TypeAdapter[Any] = TypeAdapter(list[RubricItem])

RUBRIC = rubric_adapter.validate_python(
    [
        {"id": "mixed", "type": "max_turns", "value": 3},
        {"id": "all_skip", "type": "tool_called", "tool": "x"},
        {"id": "judge", "type": "llm_judge", "question": "Polite?", "pass_if": "yes"},
    ]
)


def result(rubric_id: str, status: str, n: int) -> CheckResult:
    return CheckResult.model_validate(
        {
            "rubric_id": rubric_id,
            "persona_id": "p",
            "conversation_id": f"p-r{n}",
            "status": status,
            "detail": "d",
        }
    )


def test_pass_rate_counts_errors_not_skips() -> None:
    checks = [
        result("mixed", "pass", 0),
        result("mixed", "pass", 1),
        result("mixed", "fail", 2),
        result("mixed", "error", 3),
        result("mixed", "skip", 4),
        result("all_skip", "skip", 0),
    ]
    summaries, overall = summarize(RUBRIC, checks)
    mixed, all_skip, judge = summaries
    assert (mixed.passed, mixed.failed, mixed.errored, mixed.skipped) == (2, 1, 1, 1)
    assert mixed.pass_rate == 0.5
    assert all_skip.skipped == 1
    assert all_skip.pass_rate is None
    assert judge.rubric_id == "judge"
    assert (judge.passed, judge.failed, judge.errored, judge.skipped) == (0, 0, 0, 0)
    assert judge.pass_rate is None
    assert overall == 0.5


def test_order_follows_rubric() -> None:
    summaries, _ = summarize(RUBRIC, [])
    assert [s.rubric_id for s in summaries] == ["mixed", "all_skip", "judge"]


def test_overall_none_without_scored_checks() -> None:
    _, overall = summarize(RUBRIC, [result("all_skip", "skip", 0)])
    assert overall is None
    assert summarize(RUBRIC, [])[1] is None


def test_overall_across_items() -> None:
    checks = [
        result("mixed", "pass", 0),
        result("all_skip", "pass", 0),
        result("all_skip", "fail", 1),
    ]
    _, overall = summarize(RUBRIC, checks)
    assert overall == 2 / 3
