"""report, run --html, baseline save, compare, and the regression demo. All --no-llm."""

from __future__ import annotations

import io
import re
from importlib.resources import as_file, files
from pathlib import Path

import pytest
from rich.console import Console
from typer.testing import CliRunner

from convoprobe import cli
from convoprobe.baseline import Comparison
from convoprobe.cli import app, format_points
from convoprobe.report.json_report import print_run_report
from convoprobe.report.load import load_results

runner = CliRunner()

REGRESSION_LINE = (
    "REGRESSION: 4 item(s) dropped more than 5 pts: verified_before_refund,"
    " no_refund_without_order, refund_has_order_id, overall"
)


def scenario_path() -> str:
    with as_file(files("convoprobe.demo") / "scenario.yaml") as path:
        return str(path)


def run_agent(tmp_path: Path, version: str, *extra: str) -> Path:
    args = [
        "run",
        scenario_path(),
        "--adapter",
        f"convoprobe.demo.toy_agent:toy_agent_{version}",
        "--no-llm",
        "--db",
        str(tmp_path / "state.db"),
        "--out",
        str(tmp_path / "out"),
        "--run-id",
        f"m4-{version}",
        *extra,
    ]
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    return tmp_path / "out" / f"m4-{version}"


@pytest.fixture
def baseline_file(tmp_path: Path) -> Path:
    path = tmp_path / "baseline.json"
    result = runner.invoke(
        app, ["baseline", "save", str(run_agent(tmp_path, "v1")), "--path", str(path)]
    )
    assert result.exit_code == 0, result.output
    assert result.output.strip() == f"baseline saved: {path} (run m4-v1, 6 rubric items)"
    return path


def compare(results: Path, baseline: Path, *extra: str) -> tuple[int, list[str]]:
    result = runner.invoke(app, ["compare", str(results), "--baseline", str(baseline), *extra])
    return result.exit_code, result.output.rstrip().splitlines()


def test_regression_demo(tmp_path: Path, baseline_file: Path) -> None:
    code, lines = compare(run_agent(tmp_path, "v2"), baseline_file)
    assert code == 1
    assert "verified_before_refund: 100% -> 0% (-100 pts, threshold 5 pts)" in lines
    personas = lines[
        lines.index("verified_before_refund: 100% -> 0% (-100 pts, threshold 5 pts)") + 1
    ]
    assert personas == (
        "  personas: scripted_happy_path 100% -> 0%, scripted_skip_verification 100% -> 0%"
    )
    assert "refund_has_order_id: 100% -> 50% (-50 pts, threshold 5 pts)" in lines
    assert "overall: 100% -> 56% (-44.4 pts)" in lines
    assert lines[-1] == REGRESSION_LINE
    assert "note: the scenario changed since the baseline was saved" not in lines


def test_same_run_no_regression(tmp_path: Path, baseline_file: Path) -> None:
    code, lines = compare(tmp_path / "out" / "m4-v1", baseline_file)
    assert code == 0
    assert lines[-1] == "no regression"


def test_threshold_one_passes(tmp_path: Path, baseline_file: Path) -> None:
    code, lines = compare(run_agent(tmp_path, "v2"), baseline_file, "--threshold", "1.0")
    assert code == 0
    assert lines[-1] == "no regression"


def test_json_output(tmp_path: Path, baseline_file: Path) -> None:
    result = runner.invoke(
        app, ["compare", str(run_agent(tmp_path, "v2")), "--baseline", str(baseline_file), "--json"]
    )
    assert result.exit_code == 1
    comparison = Comparison.model_validate_json(result.output)
    assert comparison.regressed is True
    assert comparison.overall_regressed is True
    statuses = {r.rubric_id: r.status for r in comparison.rubric}
    assert statuses["verified_before_refund"] == "regression"
    assert statuses["honest_about_refund"] == "ok"


def test_scenario_changed_note(tmp_path: Path, baseline_file: Path) -> None:
    changed = tmp_path / "changed.yaml"
    changed.write_text(
        Path(scenario_path()).read_text("utf-8").replace("value: 8", "value: 9"), encoding="utf-8"
    )
    args = ["run", str(changed), "--adapter", "convoprobe.demo.toy_agent:toy_agent_v1", "--no-llm"]
    args += ["--db", str(tmp_path / "s2.db"), "--out", str(tmp_path / "out"), "--run-id", "changed"]
    assert runner.invoke(app, args).exit_code == 0
    code, lines = compare(tmp_path / "out" / "changed", baseline_file)
    assert code == 0
    assert "note: the scenario changed since the baseline was saved" in lines


def test_missing_baseline_and_bad_threshold(tmp_path: Path, baseline_file: Path) -> None:
    results = tmp_path / "out" / "m4-v1"
    code, lines = compare(results, tmp_path / "none.json")
    assert code == 2
    assert lines[0].startswith("error: no baseline at")
    for bad in ("1.5", "-0.1"):
        code, lines = compare(results, baseline_file, "--threshold", bad)
        assert code == 2
        assert lines == ["error: --threshold must be between 0 and 1"]
    code, lines = compare(tmp_path / "nothing", baseline_file)
    assert code == 2
    assert lines[0].startswith("error: no results.json at")


def test_baseline_save_bad_results(tmp_path: Path) -> None:
    result = runner.invoke(app, ["baseline", "save", str(tmp_path / "none")])
    assert result.exit_code == 2
    assert "error: no results.json at" in result.output


def test_report_command_and_html(tmp_path: Path) -> None:
    run_dir = run_agent(tmp_path, "v2")
    result = runner.invoke(app, ["report", str(run_dir), "--html"])
    assert result.exit_code == 0, result.output
    assert result.output.splitlines()[0] == "convoprobe report m4-v2"
    assert (run_dir / "report.html").is_file()
    assert f"html: {run_dir / 'report.html'}" in result.output
    custom = tmp_path / "custom.html"
    result = runner.invoke(
        app, ["report", str(run_dir / "results.json"), "--html", "--out", str(custom)]
    )
    assert result.exit_code == 0
    assert custom.is_file()
    missing = runner.invoke(app, ["report", str(tmp_path / "none")])
    assert missing.exit_code == 2


def test_run_html_flag(tmp_path: Path) -> None:
    run_dir = run_agent(tmp_path, "v1", "--html")
    assert (run_dir / "report.html").is_file()


def test_demo_out_writes_both_reports(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "_demo_cassette", lambda: None)
    result = runner.invoke(app, ["demo", "--out", str(tmp_path / "demo")])
    assert result.exit_code == 0, result.output
    reports = sorted((tmp_path / "demo").glob("*/report.html"))
    assert len(reports) == 2
    assert sum(line.startswith("html v") for line in result.output.splitlines()) == 2
    assert result.output.rstrip().splitlines()[-1].startswith("v2 regressed on")


def test_run_records_mode_and_rubric(tmp_path: Path) -> None:
    result = load_results(run_agent(tmp_path, "v1"))
    assert result.mode == "no-llm"
    assert [item.id for item in result.rubric] == [
        "verified_before_refund",
        "no_refund_without_order",
        "refund_has_order_id",
        "turn_limit",
        "no_guarantee_language",
        "honest_about_refund",
    ]


@pytest.mark.parametrize("with_cassette", [True, False])
def test_demo_records_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, with_cassette: bool
) -> None:
    if not with_cassette:
        monkeypatch.setattr(cli, "_demo_cassette", lambda: None)
    result = runner.invoke(app, ["demo", "--out", str(tmp_path / "demo")])
    assert result.exit_code == 0, result.output
    modes = {load_results(p.parent).mode for p in (tmp_path / "demo").glob("*/results.json")}
    assert modes == {"replay" if with_cassette else "no-llm"}


def test_format_points() -> None:
    assert format_points(-100.0) == "-100 pts"
    assert format_points(12.5) == "+12.5 pts"
    assert format_points(-44.444) == "-44.4 pts"
    assert format_points(0.0) == "0 pts"
    assert format_points(None) == "-"


def test_terminal_persona_matrix(tmp_path: Path) -> None:
    result = load_results(run_agent(tmp_path, "v2"))
    out = io.StringIO()
    print_run_report(result, Console(file=out, width=100, color_system=None), "t")
    lines = out.getvalue().splitlines()
    assert any("pass rate by persona" in line for line in lines)
    row = next(line for line in lines if "scripted_skip_verification" in line)
    cells = [c.strip() for c in re.split(r"[|│┃]", row) if c.strip()]
    # persona, then rubric 1..6: verified, no_refund, has_order_id, turns, guarantee, judge
    assert cells == ["scripted_skip_verification", "0%", "0%", "0%", "100%", "100%", "-"]
    assert (
        "columns: 1 verified_before_refund, 2 no_refund_without_order, 3 refund_has_order_id,"
        " 4 turn_limit, 5 no_guarantee_language, 6 honest_about_refund"
    ) in lines
