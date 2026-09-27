"""Per-rubric pass rates, repeat variance, and per-persona rates."""

from __future__ import annotations

import statistics
from collections import defaultdict

from convoprobe.datamodels import CheckResult, Conversation, RubricItem, RubricSummary

SCORED = ("pass", "fail", "error")


def _rate(passed: int, denominator: int) -> float | None:
    return passed / denominator if denominator else None


def _group_rate(statuses: list[str]) -> float | None:
    return _rate(statuses.count("pass"), sum(statuses.count(s) for s in SCORED))


def summarize(
    rubric: list[RubricItem],
    checks: list[CheckResult],
    conversations: list[Conversation] | None = None,
) -> tuple[list[RubricSummary], float | None]:
    """One summary per rubric item, in rubric order, plus the overall pass rate.

    Pass rate is passed / (passed + failed + errored). Skips do not count.
    pass_rate_stdev is the population stdev of per-repeat pass rates (repeats
    with at least one scored check), None with fewer than 2 such repeats.
    Repeat numbers come from conversations; without them there is no stdev.
    """
    repeat_of = {c.id: c.repeat_index for c in conversations or []}
    summaries: list[RubricSummary] = []
    for item in rubric:
        rows = [c for c in checks if c.rubric_id == item.id]
        statuses = [c.status for c in rows]
        passed = statuses.count("pass")
        failed = statuses.count("fail")
        errored = statuses.count("error")

        by_repeat: dict[int, list[str]] = defaultdict(list)
        by_persona: dict[str, list[str]] = defaultdict(list)
        for row in rows:
            by_persona[row.persona_id].append(row.status)
            if row.conversation_id in repeat_of:
                by_repeat[repeat_of[row.conversation_id]].append(row.status)
        repeat_rates = [
            rate
            for _, group in sorted(by_repeat.items())
            if (rate := _group_rate(group)) is not None
        ]
        stdev = statistics.pstdev(repeat_rates) if len(repeat_rates) >= 2 else None

        summaries.append(
            RubricSummary(
                rubric_id=item.id,
                passed=passed,
                failed=failed,
                errored=errored,
                skipped=statuses.count("skip"),
                pass_rate=_rate(passed, passed + failed + errored),
                pass_rate_stdev=stdev,
                persona_rates={pid: _group_rate(group) for pid, group in by_persona.items()},
            )
        )
    total_passed = sum(s.passed for s in summaries)
    total_scored = sum(s.passed + s.failed + s.errored for s in summaries)
    return summaries, _rate(total_passed, total_scored)
