"""Baselines and comparison: the CI gate.

A baseline freezes one run's pass rates. compare() checks a later run against
it. A rubric item regresses when its pass rate drops by more than the threshold
(pass_rate_drop, 0..1), or when it has no results at all in the new run.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ValidationError

from convoprobe.datamodels import RunResult
from convoprobe.errors import ConvoprobeError

EPSILON = 1e-9

Status = Literal["ok", "regression", "improved", "new", "missing"]


class RubricBaseline(BaseModel):
    pass_rate: float | None
    passed: int
    failed: int
    errored: int
    pass_rate_stdev: float | None = None


class Baseline(BaseModel):
    format: Literal[1] = 1
    scenario_name: str
    scenario_hash: str
    run_id: str
    created_at: datetime
    pass_rate_drop: float
    overall_pass_rate: float | None
    rubric: dict[str, RubricBaseline]
    persona_rates: dict[str, dict[str, float | None]]


class RubricComparison(BaseModel):
    rubric_id: str
    baseline: float | None
    current: float | None
    delta_points: float | None
    status: Status
    persona_drops: list[tuple[str, float | None, float | None]]


class Comparison(BaseModel):
    threshold: float
    scenario_changed: bool
    rubric: list[RubricComparison]
    overall_baseline: float | None
    overall_current: float | None
    overall_regressed: bool
    regressed: bool


def _persona_rates(result: RunResult) -> dict[str, dict[str, float | None]]:
    """persona_id -> rubric_id -> pass rate."""
    rates: dict[str, dict[str, float | None]] = {}
    for row in result.summary:
        for persona_id, rate in row.persona_rates.items():
            rates.setdefault(persona_id, {})[row.rubric_id] = rate
    return rates


def baseline_from(result: RunResult, pass_rate_drop: float) -> Baseline:
    """Freeze a run's rates as a baseline."""
    return Baseline(
        scenario_name=result.scenario_name,
        scenario_hash=result.scenario_hash,
        run_id=result.run_id,
        created_at=datetime.now(UTC),
        pass_rate_drop=pass_rate_drop,
        overall_pass_rate=result.overall_pass_rate,
        rubric={
            row.rubric_id: RubricBaseline(
                pass_rate=row.pass_rate,
                passed=row.passed,
                failed=row.failed,
                errored=row.errored,
                pass_rate_stdev=row.pass_rate_stdev,
            )
            for row in result.summary
        },
        persona_rates=_persona_rates(result),
    )


def save_baseline(baseline: Baseline, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(baseline.model_dump_json(indent=2) + "\n", encoding="utf-8", newline="\n")


def load_baseline(path: Path) -> Baseline:
    path = Path(path)
    if not path.is_file():
        raise ConvoprobeError(f"no baseline at {path}. Save one with: convoprobe baseline save")
    try:
        return Baseline.model_validate_json(path.read_text(encoding="utf-8"))
    except ValidationError as exc:
        first = exc.errors()[0]
        where = ".".join(str(p) for p in first["loc"])
        detail = f"{where}: {first['msg']}" if where else first["msg"]
        raise ConvoprobeError(f"{path}: not a convoprobe baseline: {detail}") from None


def _points(old: float | None, new: float | None) -> float | None:
    return None if old is None or new is None else 100 * (new - old)


def compare(baseline: Baseline, current: RunResult, threshold: float | None = None) -> Comparison:
    """Compare a run with a baseline. Only drops beyond the threshold count."""
    t = baseline.pass_rate_drop if threshold is None else threshold
    current_rates = {row.rubric_id: row.pass_rate for row in current.summary}
    current_personas = _persona_rates(current)
    rows: list[RubricComparison] = []

    for rubric_id, old in baseline.rubric.items():
        old_rate = old.pass_rate
        new_rate = current_rates.get(rubric_id)
        status: Status
        if old_rate is None:
            status = "ok" if new_rate is None else "new"
        elif new_rate is None:
            status = "missing"
        elif old_rate - new_rate > t + EPSILON:
            status = "regression"
        elif new_rate - old_rate > t + EPSILON:
            status = "improved"
        else:
            status = "ok"
        drops: list[tuple[str, float | None, float | None]] = []
        if status == "regression":
            for persona_id in sorted(baseline.persona_rates):
                before = baseline.persona_rates[persona_id].get(rubric_id)
                after = current_personas.get(persona_id, {}).get(rubric_id)
                if before is not None and (after is None or before > after):
                    drops.append((persona_id, before, after))
        rows.append(
            RubricComparison(
                rubric_id=rubric_id,
                baseline=old_rate,
                current=new_rate,
                delta_points=_points(old_rate, new_rate),
                status=status,
                persona_drops=drops,
            )
        )

    for row in current.summary:
        if row.rubric_id not in baseline.rubric:
            rows.append(
                RubricComparison(
                    rubric_id=row.rubric_id,
                    baseline=None,
                    current=row.pass_rate,
                    delta_points=None,
                    status="new",
                    persona_drops=[],
                )
            )

    old_overall, new_overall = baseline.overall_pass_rate, current.overall_pass_rate
    overall_regressed = (
        old_overall is not None
        and new_overall is not None
        and old_overall - new_overall > t + EPSILON
    )
    regressed = overall_regressed or any(r.status in ("regression", "missing") for r in rows)
    return Comparison(
        threshold=t,
        scenario_changed=baseline.scenario_hash != current.scenario_hash,
        rubric=rows,
        overall_baseline=old_overall,
        overall_current=new_overall,
        overall_regressed=overall_regressed,
        regressed=regressed,
    )
