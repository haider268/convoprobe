"""CLI: demo, run, and exit codes."""

from __future__ import annotations

from importlib.resources import as_file, files
from pathlib import Path

from typer.testing import CliRunner

from convoprobe.cli import DEMO_HEADER, app

runner = CliRunner()


def demo_scenario_path() -> str:
    with as_file(files("convoprobe.demo") / "scenario.yaml") as path:
        return str(path)


def test_demo() -> None:
    result = runner.invoke(app, ["demo"])
    assert result.exit_code == 0, result.output
    assert DEMO_HEADER == "DEMO: toy shop agent, scripted personas, no model calls, no API keys."
    assert result.output.splitlines()[0] == DEMO_HEADER
    assert "v1 (correct agent)" in result.output
    assert "v2 (prompt change with a bug)" in result.output
    assert "REGRESSION" in result.output
    assert result.output.rstrip().splitlines()[-1] == (
        "v2 regressed on 3 rubric item(s): "
        "verified_before_refund, no_refund_without_order, refund_has_order_id"
    )


def test_demo_with_out(tmp_path: Path) -> None:
    result = runner.invoke(app, ["demo", "--out", str(tmp_path / "keep")])
    assert result.exit_code == 0, result.output
    assert len(list((tmp_path / "keep").glob("*/results.json"))) == 2


def test_run(tmp_path: Path) -> None:
    args = [
        "run",
        demo_scenario_path(),
        "--adapter",
        "convoprobe.demo.toy_agent:toy_agent_v1",
        "--db",
        str(tmp_path / "state.db"),
        "--out",
        str(tmp_path / "out"),
        "--run-id",
        "cli-run",
        "--no-llm",
    ]
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    assert "convoprobe run cli-run" in result.output
    assert "overall pass rate: 100%" in result.output
    assert (tmp_path / "out" / "cli-run" / "results.json").is_file()
    assert "results.json" in result.output


def test_run_bad_adapter(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        ["run", demo_scenario_path(), "--adapter", "no_colon", "--db", str(tmp_path / "s.db")],
    )
    assert result.exit_code == 2
    assert "error: adapter 'no_colon'" in result.output


def test_run_missing_scenario(tmp_path: Path) -> None:
    missing = tmp_path / "missing.yaml"
    result = runner.invoke(
        app,
        ["run", str(missing), "--adapter", "convoprobe.demo.toy_agent:toy_agent_v1"],
    )
    assert result.exit_code == 2
    assert f"error: {missing}: file not found" in result.output


def test_run_scenario_mismatch(tmp_path: Path) -> None:
    other = tmp_path / "other.yaml"
    other.write_text(
        Path(demo_scenario_path()).read_text("utf-8").replace("toy-support", "other"),
        encoding="utf-8",
    )
    common = [
        "--adapter",
        "convoprobe.demo.toy_agent:toy_agent_v1",
        "--db",
        str(tmp_path / "s.db"),
        "--out",
        str(tmp_path / "out"),
        "--run-id",
        "same",
        "--no-llm",
    ]
    assert runner.invoke(app, ["run", demo_scenario_path(), *common]).exit_code == 0
    result = runner.invoke(app, ["run", str(other), *common])
    assert result.exit_code == 2
    assert "run same was created from a different scenario" in result.output


def test_version() -> None:
    assert runner.invoke(app, ["version"]).stdout.strip() == "0.0.1"
