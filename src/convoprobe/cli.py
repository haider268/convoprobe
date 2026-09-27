"""Command line interface."""

from __future__ import annotations

import asyncio
import tempfile
from contextlib import ExitStack
from importlib.resources import as_file, files
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from convoprobe import __version__
from convoprobe.adapters import CallableAdapter, load_adapter
from convoprobe.config import HOSTED_JUDGE, HOSTED_SIMULATOR, Settings, load_settings
from convoprobe.datamodels import RunResult, Scenario
from convoprobe.demo.toy_agent import toy_agent_v1, toy_agent_v2
from convoprobe.errors import ConvoprobeError
from convoprobe.models.cache import JsonCassette, LLMCache, SqliteLLMCache
from convoprobe.models.client import LLMClient
from convoprobe.models.factory import parse_spec
from convoprobe.models.preflight import check_ollama
from convoprobe.models.registry import LLMRegistry
from convoprobe.report.json_report import format_rate, print_run_report
from convoprobe.runner import new_run_id, run_scenario
from convoprobe.scenario import ScenarioError, load_scenario, parse_scenario
from convoprobe.simulator.llm import LLMSimulator
from convoprobe.store import RunStore

app = typer.Typer(
    help="Stress-test conversational agents with simulated users.", no_args_is_help=True
)

DEMO_HEADER = "DEMO: toy shop agent, scripted personas, no model calls, no API keys."
DEFAULT_DB = Path(".convoprobe/convoprobe.db")
SIMULATOR_TEMPERATURE = 0.7
JUDGE_TEMPERATURE = 0.0


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


def _open_cassette(path: Path) -> JsonCassette:
    if not path.is_file():
        raise ConvoprobeError(f"cassette not found: {path}")
    try:
        return JsonCassette(path)
    except ValueError as exc:
        raise ConvoprobeError(f"cassette {path} is not valid JSON: {exc}") from exc


def _pick_spec(explicit: str | None, hosted: bool, hosted_spec: str, default: str) -> str:
    if explicit:
        return explicit
    return hosted_spec if hosted else default


async def _preflight(settings: Settings, specs: list[str]) -> None:
    """Check Ollama is up and has the models, when any spec uses Ollama."""
    models: list[str] = []
    for spec in specs:
        provider, model = parse_spec(spec)
        if provider == "ollama" and model not in models:
            models.append(model)
    if models:
        await check_ollama(settings.ollama_base_url, models)


def _with_repeats(scenario: Scenario, repeats: int, source: str) -> Scenario:
    updated = scenario.model_copy(update={"repeats": repeats})
    return parse_scenario(updated.model_dump(mode="json"), source=source)


@app.command()
def run(
    scenario_path: Annotated[Path, typer.Argument(metavar="SCENARIO", help="Scenario YAML file.")],
    adapter: Annotated[str, typer.Option(help="Agent adapter as module.path:attr.")],
    out: Annotated[Path, typer.Option(help="Output folder for results.")] = Path("results"),
    run_id: Annotated[str | None, typer.Option(help="Run id. Reuse one to resume.")] = None,
    db: Annotated[Path, typer.Option(help="SQLite state and LLM cache file.")] = DEFAULT_DB,
    concurrency: Annotated[int, typer.Option(min=1, help="Conversations at once.")] = 4,
    simulator: Annotated[
        str | None, typer.Option(help="Simulator model as provider:model.")
    ] = None,
    judge: Annotated[str | None, typer.Option(help="Judge model as provider:model.")] = None,
    hosted: Annotated[
        bool, typer.Option("--hosted", help="Use the hosted Groq defaults for unset models.")
    ] = False,
    replay: Annotated[
        Path | None, typer.Option(help="Replay LLM calls from a cassette. No keys needed.")
    ] = None,
    no_llm: Annotated[
        bool, typer.Option("--no-llm", help="No simulator, judge, or persona generation.")
    ] = False,
    repeats: Annotated[int | None, typer.Option(help="Override the scenario's repeats.")] = None,
) -> None:
    """Run a scenario against an agent and write results."""
    if no_llm and (replay is not None or hosted or simulator or judge):
        raise _fail("--no-llm cannot be combined with model options")
    try:
        scenario = load_scenario(scenario_path)
        if repeats is not None:
            scenario = _with_repeats(scenario, repeats, str(scenario_path))
        agent = load_adapter(adapter)
        run_id = run_id or new_run_id()
        with ExitStack() as stack:
            store = stack.enter_context(RunStore(db))
            sim_client: LLMClient | None = None
            judge_client: LLMClient | None = None
            if no_llm:
                typer.echo("mode: no-llm")
            else:
                settings = load_settings()
                sim_spec = _pick_spec(
                    simulator, hosted, HOSTED_SIMULATOR, settings.convoprobe_simulator
                )
                judge_spec = _pick_spec(judge, hosted, HOSTED_JUDGE, settings.convoprobe_judge)
                cache: LLMCache = (
                    _open_cassette(replay)
                    if replay is not None
                    else stack.enter_context(SqliteLLMCache(db))
                )
                registry = LLMRegistry(
                    settings, cache, replay_only=replay is not None, run_id=run_id
                )
                sim_client = registry.client(sim_spec, SIMULATOR_TEMPERATURE)
                judge_client = registry.client(judge_spec, JUDGE_TEMPERATURE)
                if replay is None:
                    asyncio.run(_preflight(settings, [sim_spec, judge_spec]))
                    typer.echo("mode: live")
                else:
                    typer.echo(f"mode: replay ({replay})")
                typer.echo(f"models: simulator {sim_spec}, judge {judge_spec}")
            result = asyncio.run(
                run_scenario(
                    scenario,
                    agent,
                    store=store,
                    out_dir=out,
                    adapter_label=adapter,
                    run_id=run_id,
                    concurrency=concurrency,
                    llm_simulator=LLMSimulator(sim_client) if sim_client else None,
                    judge_client=judge_client,
                    persona_client=sim_client,
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
