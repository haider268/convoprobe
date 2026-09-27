"""Command line interface."""

from __future__ import annotations

import asyncio
import tempfile
from importlib.resources import as_file, files
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from convoprobe import __version__
from convoprobe.adapters import CallableAdapter, load_adapter
from convoprobe.datamodels import RunResult
from convoprobe.demo.toy_agent import toy_agent_v1, toy_agent_v2
from convoprobe.errors import ConvoprobeError
from convoprobe.report.json_report import format_rate, print_run_report
from convoprobe.runner import run_scenario
from convoprobe.scenario import ScenarioError, load_scenario
from convoprobe.store import RunStore

app = typer.Typer(
    help="Stress-test conversational agents with simulated users.", no_args_is_help=True
)

DEMO_HEADER = "DEMO: toy shop agent, scripted personas, no model calls, no API keys."


def _fail(message: str) -> typer.Exit:
    typer.echo(f"error: {message}", err=True)
    return typer.Exit(code=2)


@app.callback()
def _root() -> None:
    """convoprobe command line."""


@app.command()
def version() -> None:
    """Print the installed convoprobe version."""
    typer.echo(__version__)


@app.command()
def run(
    scenario_path: Annotated[Path, typer.Argument(metavar="SCENARIO", help="Scenario YAML file.")],
    adapter: Annotated[str, typer.Option(help="Agent adapter as module.path:attr.")],
    out: Annotated[Path, typer.Option(help="Output folder for results.")] = Path("results"),
    run_id: Annotated[str | None, typer.Option(help="Run id. Reuse one to resume.")] = None,
    db: Annotated[Path, typer.Option(help="SQLite state file.")] = Path(
        ".convoprobe/convoprobe.db"
    ),
    concurrency: Annotated[int, typer.Option(min=1, help="Conversations at once.")] = 4,
) -> None:
    """Run a scenario against an agent and write results."""
    try:
        scenario = load_scenario(scenario_path)
        agent = load_adapter(adapter)
        with RunStore(db) as store:
            result = asyncio.run(
                run_scenario(
                    scenario,
                    agent,
                    store=store,
                    out_dir=out,
                    adapter_label=adapter,
                    run_id=run_id,
                    concurrency=concurrency,
                )
            )
    except (ScenarioError, ConvoprobeError) as exc:
        raise _fail(str(exc)) from None
    console = Console()
    print_run_report(result, console, f"convoprobe run {result.run_id}")
    console.print(f"results: {out / result.run_id / 'results.json'}", markup=False, soft_wrap=True)


@app.command()
def demo(
    out: Annotated[
        Path | None, typer.Option(help="Keep results here instead of a temp folder.")
    ] = None,
) -> None:
    """Run the offline demo: toy agent v1, then a buggy v2, then compare."""
    console = Console()
    console.print(DEMO_HEADER, markup=False, soft_wrap=True)
    with (
        tempfile.TemporaryDirectory(prefix="convoprobe-demo-") as tmp,
        as_file(files("convoprobe.demo") / "scenario.yaml") as scenario_file,
    ):
        scenario = load_scenario(scenario_file)
        out_dir = out if out is not None else Path(tmp) / "results"
        with RunStore(Path(tmp) / "demo.db") as store:
            v1 = asyncio.run(
                run_scenario(
                    scenario,
                    CallableAdapter(toy_agent_v1),
                    store=store,
                    out_dir=out_dir,
                    adapter_label="demo:toy_agent_v1",
                )
            )
            v2 = asyncio.run(
                run_scenario(
                    scenario,
                    CallableAdapter(toy_agent_v2),
                    store=store,
                    out_dir=out_dir,
                    adapter_label="demo:toy_agent_v2",
                )
            )
    print_run_report(v1, console, "v1 (correct agent)")
    print_run_report(v2, console, "v2 (prompt change with a bug)")
    regressed = _print_comparison(v1, v2, console)
    console.print(
        f"v2 regressed on {len(regressed)} rubric item(s): {', '.join(regressed) or 'none'}",
        markup=False,
        soft_wrap=True,
    )


def _print_comparison(old: RunResult, new: RunResult, console: Console) -> list[str]:
    """Print the v1 vs v2 table and return the regressed rubric ids."""
    new_rates = {row.rubric_id: row.pass_rate for row in new.summary}
    table = Table(title="v1 vs v2")
    for column in ("rubric", "v1", "v2", "change"):
        table.add_column(column)
    regressed: list[str] = []
    for row in old.summary:
        old_rate, new_rate = row.pass_rate, new_rates.get(row.rubric_id)
        worse = old_rate is not None and new_rate is not None and new_rate < old_rate
        if worse:
            regressed.append(row.rubric_id)
        table.add_row(
            row.rubric_id,
            format_rate(old_rate),
            format_rate(new_rate),
            "REGRESSION" if worse else "",
        )
    console.print(table)
    return regressed


def main() -> None:
    """Console script entry point."""
    app()
