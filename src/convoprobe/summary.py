"""Per-rubric pass rates from check results."""

from __future__ import annotations

from convoprobe.datamodels import CheckResult, RubricItem, RubricSummary


def _rate(passed: int, denominator: int) -> float | None:
    return passed / denominator if denominator else None


def summarize(
    rubric: list[RubricItem], checks: list[CheckResult]
) -> tuple[list[RubricSummary], float | None]:
    """One summary per rubric item, in rubric order, plus the overall pass rate.

    Pass rate is passed / (passed + failed + errored). Skips do not count.
    llm_judge items are listed with zeros until the judge is added.
    """
    summaries: list[RubricSummary] = []
    for item in rubric:
        statuses = [c.status for c in checks if c.rubric_id == item.id]
        passed = statuses.count("pass")
        failed = statuses.count("fail")
        errored = statuses.count("error")
        summaries.append(
            RubricSummary(
                rubric_id=item.id,
                passed=passed,
                failed=failed,
                errored=errored,
                skipped=statuses.count("skip"),
                pass_rate=_rate(passed, passed + failed + errored),
            )
        )
    total_passed = sum(s.passed for s in summaries)
    total_scored = sum(s.passed + s.failed + s.errored for s in summaries)
    return summaries, _rate(total_passed, total_scored)
