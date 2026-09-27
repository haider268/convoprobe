"""Command line interface."""

from __future__ import annotations

import asyncio
import tempfile
from contextlib import ExitStack
from datetime import UTC, datetime
from importlib.resources import as_file, files
from importlib.resources.abc import Traversable
from pathlib import Path
from typing import Annotated

import typer
import yaml
from rich.console import Console
from rich.table import Table

from convoprobe import __version__
from convoprobe import demo as demo_package
from convoprobe.adapters import CallableAdapter, load_adapter
from convoprobe.config import HOSTED_JUDGE, HOSTED_SIMULATOR, Settings, load_settings
from convoprobe.datamodels import RunResult, Scenario
from convoprobe.demo import CASSETTE_NAME, DEMO_JUDGE, DEMO_SIMULATOR
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
from convoprobe.simulator.personas import generate_personas
from convoprobe.store import RunStore

app = typer.Typer(
    help="Stress-test conversational agents with simulated users.", no_args_is_help=True
)

DEMO_HEADER = "DEMO: toy shop agent, scripted personas, no model calls, no API keys."
DEMO_REPLAY_HEADER = (
    "DEMO: toy shop agent. Scripted personas run live; LLM personas and judge verdicts"
    " replay a cassette recorded on {recorded_at} (simulator {simulator}, judge {judge})."
    " No API keys, no network."
)
DEFAULT_DB = Path(".convoprobe/convoprobe.db")
RECORD_DB = Path(".convoprobe/demo-record.db")
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


def _warn_small_judge(judge_spec: str) -> None:
    """Live runs only: say plainly that a local 3B judge is not reliable."""
    provider, _model = parse_spec(judge_spec)
    if provider == "ollama":
        typer.echo(
            f"note: judge {judge_spec} is a small local model. In review testing a 3B local"
            " judge gave wrong verdicts. Use --hosted or --replay for results you rely on."
        )


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
                if replay is None:
                    _warn_small_judge(judge_spec)
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


cassette_app = typer.Typer(help="Record and inspect replay cassettes.", no_args_is_help=True)
app.add_typer(cassette_app, name="cassette")
personas_app = typer.Typer(help="Generate personas with the simulator model.", no_args_is_help=True)
app.add_typer(personas_app, name="personas")


def _today() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%d")


@cassette_app.command("export")
def cassette_export(
    run_ids: Annotated[
        list[str], typer.Option("--run-id", help="Run whose LLM calls to export. Repeatable.")
    ],
    out: Annotated[Path, typer.Option(help="Cassette file to write.")],
    db: Annotated[Path, typer.Option(help="SQLite file holding the LLM cache.")] = DEFAULT_DB,
) -> None:
    """Export the cached LLM calls of one or more runs to a replay cassette."""
    with SqliteLLMCache(db) as cache:
        meta = {"recorded_at": _today(), "run_ids": ",".join(run_ids)}
        count = cache.export_cassette(run_ids, out, meta=meta)
    if count == 0:
        out.unlink(missing_ok=True)
        raise _fail("no cached LLM calls found for those run ids")
    typer.echo(f"wrote {count} entries to {out}")


@personas_app.command("generate")
def personas_generate(
    scenario_path: Annotated[Path, typer.Argument(metavar="SCENARIO", help="Scenario YAML file.")],
    n: Annotated[int, typer.Option("-n", min=1, max=50, help="How many personas.")],
    simulator: Annotated[
        str | None, typer.Option(help="Simulator model as provider:model.")
    ] = None,
    hosted: Annotated[
        bool, typer.Option("--hosted", help="Use the hosted Groq simulator unless set.")
    ] = False,
    replay: Annotated[Path | None, typer.Option(help="Replay from a cassette.")] = None,
    db: Annotated[Path, typer.Option(help="SQLite file for the LLM cache.")] = DEFAULT_DB,
) -> None:
    """Generate personas for a scenario and print them as YAML."""
    try:
        scenario = load_scenario(scenario_path)
        settings = load_settings()
        spec = _pick_spec(simulator, hosted, HOSTED_SIMULATOR, settings.convoprobe_simulator)
        with ExitStack() as stack:
            cache: LLMCache = (
                _open_cassette(replay)
                if replay is not None
                else stack.enter_context(SqliteLLMCache(db))
            )
            registry = LLMRegistry(settings, cache, replay_only=replay is not None)
            client = registry.client(spec, SIMULATOR_TEMPERATURE)
            if replay is None:
                asyncio.run(_preflight(settings, [spec]))
            salt = f"personas-cli:{RunStore.scenario_hash(scenario)}:{n}"
            personas, notes = asyncio.run(generate_personas(scenario, n, client, salt=salt))
    except (ScenarioError, ConvoprobeError) as exc:
        raise _fail(str(exc)) from None
    rows = [{"id": p.id, "description": p.description, "goal": p.goal} for p in personas]
    typer.echo(yaml.safe_dump(rows, sort_keys=False, allow_unicode=True), nl=False)
    for note in notes:
        typer.echo(f"note: {note}", err=True)


def _demo_cassette() -> Traversable | None:
    """The packaged demo cassette, or None when it is not shipped."""
    resource = files("convoprobe.demo") / CASSETTE_NAME
    return resource if resource.is_file() else None


def _demo_clients(
    settings: Settings, cache: LLMCache, *, replay_only: bool, run_id: str | None
) -> tuple[LLMClient, LLMClient]:
    """Fresh simulator and judge clients, so each run reports its own usage."""
    registry = LLMRegistry(settings, cache, replay_only=replay_only, run_id=run_id)
    return (
        registry.client(DEMO_SIMULATOR, SIMULATOR_TEMPERATURE),
        registry.client(DEMO_JUDGE, JUDGE_TEMPERATURE),
    )


async def _demo_run(
    scenario: Scenario,
    version: str,
    store: RunStore,
    out_dir: Path,
    clients: tuple[LLMClient, LLMClient] | None,
    run_id: str | None = None,
) -> RunResult:
    fn = toy_agent_v1 if version == "v1" else toy_agent_v2
    sim, judge = clients if clients else (None, None)
    return await run_scenario(
        scenario,
        CallableAdapter(fn),
        store=store,
        out_dir=out_dir,
        adapter_label=f"demo:toy_agent_{version}",
        run_id=run_id,
        llm_simulator=LLMSimulator(sim) if sim else None,
        judge_client=judge,
        persona_client=sim,
    )


@app.command()
def demo(
    out: Annotated[
        Path | None, typer.Option(help="Keep results here instead of a temp folder.")
    ] = None,
    record: Annotated[
        bool,
        typer.Option("--record", help="Record a new demo cassette live. Needs GROQ_API_KEY."),
    ] = False,
) -> None:
    """Run the demo: toy agent v1, then a buggy v2, then compare."""
    if record:
        _demo_record(out)
        return
    console = Console()
    resource = _demo_cassette()
    with ExitStack() as stack:
        tmp = stack.enter_context(tempfile.TemporaryDirectory(prefix="convoprobe-demo-"))
        scenario_file = stack.enter_context(as_file(files("convoprobe.demo") / "scenario.yaml"))
        scenario = load_scenario(scenario_file)
        out_dir = out if out is not None else Path(tmp) / "results"
        cassette: JsonCassette | None = None
        header = DEMO_HEADER
        if resource is not None:
            cassette = JsonCassette(stack.enter_context(as_file(resource)))
            header = DEMO_REPLAY_HEADER.format(
                recorded_at=cassette.meta.get("recorded_at", "an unknown date"),
                simulator=DEMO_SIMULATOR,
                judge=DEMO_JUDGE,
            )
        console.print(header, markup=False, soft_wrap=True)
        settings = load_settings()
        store = stack.enter_context(RunStore(Path(tmp) / "demo.db"))
        results = []
        for version in ("v1", "v2"):
            clients = (
                _demo_clients(settings, cassette, replay_only=True, run_id=None)
                if cassette is not None
                else None
            )
            results.append(asyncio.run(_demo_run(scenario, version, store, out_dir, clients)))
    v1, v2 = results
    print_run_report(v1, console, "v1 (correct agent)")
    print_run_report(v2, console, "v2 (prompt change with a bug)")
    regressed = _print_comparison(v1, v2, console)
    console.print(
        f"v2 regressed on {len(regressed)} rubric item(s): {', '.join(regressed) or 'none'}",
        markup=False,
        soft_wrap=True,
    )


def _demo_record(out: Path | None) -> None:
    """Run v1 and v2 live, then write the cassette if every conversation finished."""
    console = Console()
    settings = load_settings()
    with ExitStack() as stack:
        try:
            _demo_clients(settings, _NullCache(), replay_only=False, run_id=None)
        except ConvoprobeError as exc:
            raise _fail(str(exc)) from None
        _warn_small_judge(DEMO_JUDGE)
        RECORD_DB.unlink(missing_ok=True)
        tmp = stack.enter_context(tempfile.TemporaryDirectory(prefix="convoprobe-record-"))
        scenario_file = stack.enter_context(as_file(files("convoprobe.demo") / "scenario.yaml"))
        scenario = load_scenario(scenario_file)
        out_dir = out if out is not None else Path(tmp) / "results"
        store = stack.enter_context(RunStore(RECORD_DB))
        cache = stack.enter_context(SqliteLLMCache(RECORD_DB))
        results: list[RunResult] = []
        for version in ("v1", "v2"):
            run_id = f"demo-{version}"
            clients = _demo_clients(settings, cache, replay_only=False, run_id=run_id)
            results.append(
                asyncio.run(_demo_run(scenario, version, store, out_dir, clients, run_id))
            )
        failures = [
            (cid, next((c.error for c in result.conversations if c.id == cid), None))
            for result in results
            for cid, status in result.conversation_status.items()
            if status != "done"
        ]
        if failures:
            for cid, error in failures:
                typer.echo(f"{cid}: {error or 'no error recorded'}", err=True)
            typer.echo("cassette not written: not every conversation finished", err=True)
            raise typer.Exit(code=1)
        cassette_path = Path(demo_package.__file__).parent / CASSETTE_NAME
        meta = {"recorded_at": _today(), "simulator": DEMO_SIMULATOR, "judge": DEMO_JUDGE}
        count = cache.export_cassette(["demo-v1", "demo-v2"], cassette_path, meta=meta)
    typer.echo(f"wrote {count} entries to {cassette_path}")
    print_run_report(results[0], console, "v1 (correct agent)")
    print_run_report(results[1], console, "v2 (prompt change with a bug)")


class _NullCache:
    """Only used to check model settings and keys before a recording starts."""

    def get(self, key: str) -> None:
        return None

    def put(self, key: str, spec: str, kind: str, response: dict[str, object]) -> None:
        return None

    def mark_used(self, run_id: str, key: str) -> None:
        return None


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
