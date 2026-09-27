"""Baseline and compare: every gating rule, with exact statuses and deltas."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from convoprobe.baseline import (
    Baseline,
    RubricComparison,
    baseline_from,
    compare,
    load_baseline,
    save_baseline,
)
from convoprobe.datamodels import RubricSummary, RunResult
from convoprobe.errors import ConvoprobeError

Rates = dict[str, float | None]


def run(
    rates: Rates,
    personas: dict[str, Rates] | None = None,
    overall: float | None = None,
    scenario_hash: str = "h1",
) -> RunResult:
    """A RunResult with the given per-rubric and per-persona pass rates."""
    personas = personas or {}
    summary = []
    for rubric_id, rate in rates.items():
        passed = 0 if rate is None else round(rate * 10)
        failed = 0 if rate is None else 10 - passed
        summary.append(
            RubricSummary(
                rubric_id=rubric_id,
                passed=passed,
                failed=failed,
                errored=0,
                skipped=10 if rate is None else 0,
                pass_rate=rate,
                persona_rates={p: r[rubric_id] for p, r in personas.items() if rubric_id in r},
            )
        )
    return RunResult(
        run_id="run",
        scenario_name="s",
        adapter="a",
        started_at=datetime(2026, 9, 28, tzinfo=UTC),
        finished_at=None,
        conversation_status={},
        conversations=[],
        checks=[],
        summary=summary,
        overall_pass_rate=overall,
        notes=[],
        scenario_hash=scenario_hash,
    )


def base(rates: Rates, **kwargs: object) -> Baseline:
    return baseline_from(run(rates, **kwargs), pass_rate_drop=0.05)  # type: ignore[arg-type]


def row(comparison_rows: list[RubricComparison], rubric_id: str) -> RubricComparison:
    return next(r for r in comparison_rows if r.rubric_id == rubric_id)


def test_statuses_and_deltas() -> None:
    old = base({"drop": 1.0, "up": 0.5, "same": 0.8, "none_none": None, "none_new": None})
    new = run({"drop": 0.0, "up": 0.7, "same": 0.8, "none_none": None, "none_new": 0.4})
    result = compare(old, new)
    assert row(result.rubric, "drop").status == "regression"
    assert row(result.rubric, "drop").delta_points == pytest.approx(-100.0)
    assert row(result.rubric, "up").status == "improved"
    assert row(result.rubric, "up").delta_points == pytest.approx(20.0)
    assert row(result.rubric, "same").status == "ok"
    assert row(result.rubric, "same").delta_points == pytest.approx(0.0)
    assert row(result.rubric, "none_none").status == "ok"
    assert row(result.rubric, "none_new").status == "new"
    assert row(result.rubric, "none_new").delta_points is None
    assert result.regressed is True
    assert result.threshold == 0.05


def test_threshold_boundary_exactly_t_is_ok() -> None:
    old = base({"r": 0.8})
    assert compare(old, run({"r": 0.75})).rubric[0].status == "ok"
    assert compare(old, run({"r": 0.85})).rubric[0].status == "ok"
    assert compare(old, run({"r": 0.7499})).rubric[0].status == "regression"
    assert compare(old, run({"r": 0.8501})).rubric[0].status == "improved"
    assert compare(old, run({"r": 0.7499})).regressed is True
    assert compare(old, run({"r": 0.75})).regressed is False


def test_explicit_threshold_overrides_baseline() -> None:
    old = base({"r": 1.0})
    assert compare(old, run({"r": 0.0}), threshold=1.0).regressed is False
    assert compare(old, run({"r": 0.85}), threshold=0.1).rubric[0].status == "regression"


def test_missing_counts_as_regression() -> None:
    old = base({"gone": 1.0, "no_data": 0.9})
    result = compare(old, run({"no_data": None}))
    assert row(result.rubric, "gone").status == "missing"
    assert row(result.rubric, "no_data").status == "missing"
    assert row(result.rubric, "gone").persona_drops == []
    assert result.regressed is True


def test_rubric_only_in_current_is_new() -> None:
    result = compare(base({"a": 1.0}), run({"a": 1.0, "added": 0.3}))
    assert [(r.rubric_id, r.status) for r in result.rubric] == [("a", "ok"), ("added", "new")]
    assert result.regressed is False


def test_persona_drops() -> None:
    personas_old = {"p1": {"r": 1.0}, "p2": {"r": 1.0}, "p3": {"r": 0.5}, "p4": {"r": 1.0}}
    personas_new = {"p1": {"r": 0.0}, "p2": {"r": 1.0}, "p3": {"r": 0.2}}  # p4 gone
    result = compare(base({"r": 1.0}, personas=personas_old), run({"r": 0.4}, personas_new))
    assert result.rubric[0].persona_drops == [
        ("p1", 1.0, 0.0),
        ("p3", 0.5, 0.2),
        ("p4", 1.0, None),
    ]


def test_overall_regression_alone() -> None:
    old = base({"r": 1.0}, overall=0.9)
    result = compare(old, run({"r": 1.0}, overall=0.5))
    assert result.rubric[0].status == "ok"
    assert result.overall_regressed is True
    assert result.regressed is True
    assert compare(old, run({"r": 1.0}, overall=None)).overall_regressed is False


def test_scenario_changed_is_reported_not_gating() -> None:
    result = compare(base({"r": 1.0}), run({"r": 1.0}, scenario_hash="h2"))
    assert result.scenario_changed is True
    assert result.regressed is False
    assert compare(base({"r": 1.0}), run({"r": 1.0})).scenario_changed is False


def test_save_load_round_trip(tmp_path: Path) -> None:
    old = base({"r": 0.8, "none": None}, personas={"p": {"r": 0.8}}, overall=0.8)
    path = tmp_path / "deep" / "baseline.json"
    save_baseline(old, path)
    assert load_baseline(path) == old
    assert old.rubric["r"].passed == 8
    assert old.persona_rates == {"p": {"r": 0.8}}


def test_load_errors(tmp_path: Path) -> None:
    missing = tmp_path / "none.json"
    with pytest.raises(ConvoprobeError, match=f"no baseline at {missing}".replace("\\", "\\\\")):
        load_baseline(missing)
    bad = tmp_path / "bad.json"
    bad.write_text('{"format": 2}', encoding="utf-8")
    with pytest.raises(ConvoprobeError) as info:
        load_baseline(bad)
    assert str(info.value).startswith(f"{bad}: not a convoprobe baseline: format:")
    broken = tmp_path / "broken.json"
    broken.write_text("{nope", encoding="utf-8")
    with pytest.raises(ConvoprobeError, match="not a convoprobe baseline: Invalid JSON"):
        load_baseline(broken)
