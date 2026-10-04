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
from convoprobe.baseline import Comparison, baseline_from, load_baseline, save_baseline
from convoprobe.baseline import compare as compare_runs
from convoprobe.calibration import (
    CalibrationReport,
    format_kappa,
    format_share,
    judge_slug,
    load_report,
    render_table,
    score_judge,
)
from convoprobe.calibration import prepare as prepare_calibration
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
from convoprobe.report.html import write_html
from convoprobe.report.json_report import format_rate, print_run_report
from convoprobe.report.load import load_results, results_file
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
REPORT_FILE = "report.html"
DEFAULT_BASELINE = Path(".convoprobe/baseline.json")
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
    html: Annotated[
        bool, typer.Option("--html", help="Also write report.html next to results.json.")
    ] = False,
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
            mode = "no-llm" if no_llm else "replay" if replay is not None else "live"
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
                    mode=mode,
                )
            )
    except (ScenarioError, ConvoprobeError) as exc:
        raise _fail(str(exc)) from None
    console = Console()
    print_run_report(result, console, f"convoprobe run {result.run_id}")
    console.print(f"results: {out / result.run_id / 'results.json'}", markup=False, soft_wrap=True)
    if html:
        path = write_html(result, out / result.run_id / REPORT_FILE)
        console.print(f"html: {path}", markup=False, soft_wrap=True)


@app.command()
def report(
    results: Annotated[
        Path, typer.Argument(metavar="RESULTS", help="A run folder or its results.json.")
    ],
    html: Annotated[bool, typer.Option("--html", help="Also write an HTML report.")] = False,
    out: Annotated[
        Path | None, typer.Option(help="HTML path. Default: report.html next to results.json.")
    ] = None,
) -> None:
    """Print the report for a finished run, and optionally write it as HTML."""
    try:
        result = load_results(results)
    except ConvoprobeError as exc:
        raise _fail(str(exc)) from None
    console = Console()
    print_run_report(result, console, f"convoprobe report {result.run_id}")
    if html:
        target = out if out is not None else results_file(results).parent / REPORT_FILE
        path = write_html(result, target)
        console.print(f"html: {path}", markup=False, soft_wrap=True)


baseline_app = typer.Typer(help="Save the pass rates that compare checks against.")
app.add_typer(baseline_app, name="baseline")

_STATUS_TEXT = {
    "ok": "OK",
    "regression": "REGRESSION",
    "improved": "IMPROVED",
    "new": "NEW",
    "missing": "MISSING",
}


def format_points(delta: float | None) -> str:
    """A change in percentage points: "-100 pts", "+12.5 pts", or "-"."""
    if delta is None:
        return "-"
    rounded = round(delta, 1)
    if rounded == 0:
        return "0 pts"
    if rounded == int(rounded):
        return f"{int(rounded):+d} pts"
    return f"{rounded:+.1f} pts"


def _threshold_text(threshold: float) -> str:
    return f"{threshold * 100:g}"


@baseline_app.command("save")
def baseline_save(
    results: Annotated[
        Path, typer.Argument(metavar="RESULTS", help="A run folder or its results.json.")
    ],
    path: Annotated[Path, typer.Option(help="Where to write the baseline.")] = DEFAULT_BASELINE,
) -> None:
    """Save a run's pass rates as the baseline."""
    try:
        result = load_results(results)
    except ConvoprobeError as exc:
        raise _fail(str(exc)) from None
    baseline = baseline_from(result, result.pass_rate_drop)
    save_baseline(baseline, path)
    typer.echo(f"baseline saved: {path} (run {result.run_id}, {len(baseline.rubric)} rubric items)")


@app.command()
def compare(
    results: Annotated[
        Path, typer.Argument(metavar="RESULTS", help="A run folder or its results.json.")
    ],
    baseline: Annotated[Path, typer.Option(help="Baseline file.")] = DEFAULT_BASELINE,
    threshold: Annotated[
        float | None,
        typer.Option(help="Allowed pass rate drop, 0 to 1. Default: from the baseline."),
    ] = None,
    as_json: Annotated[bool, typer.Option("--json", help="Print the comparison as JSON.")] = False,
) -> None:
    """Compare a run with the baseline. Exit 0 no regression, 1 regression, 2 error."""
    if threshold is not None and not 0 <= threshold <= 1:
        raise _fail("--threshold must be between 0 and 1")
    try:
        saved = load_baseline(baseline)
        result = load_results(results)
    except ConvoprobeError as exc:
        raise _fail(str(exc)) from None
    comparison = compare_runs(saved, result, threshold)
    if as_json:
        typer.echo(comparison.model_dump_json(indent=2))
        raise typer.Exit(code=1 if comparison.regressed else 0)
    _print_comparison_report(comparison)
    raise typer.Exit(code=1 if comparison.regressed else 0)


def _print_comparison_report(comparison: Comparison) -> None:
    console = Console()
    table = Table()
    for column in ("rubric", "baseline", "current", "change", "status"):
        table.add_column(column, justify="left" if column in ("rubric", "status") else "right")
    for row in comparison.rubric:
        table.add_row(
            row.rubric_id,
            format_rate(row.baseline),
            format_rate(row.current),
            format_points(row.delta_points),
            _STATUS_TEXT[row.status],
        )
    console.print(table)
    if comparison.scenario_changed:
        typer.echo("note: the scenario changed since the baseline was saved")
    t = _threshold_text(comparison.threshold)
    failed: list[str] = []
    for row in comparison.rubric:
        if row.status == "regression":
            failed.append(row.rubric_id)
            typer.echo(
                f"{row.rubric_id}: {format_rate(row.baseline)} -> {format_rate(row.current)}"
                f" ({format_points(row.delta_points)}, threshold {t} pts)"
            )
            if row.persona_drops:
                drops = ", ".join(
                    f"{pid} {format_rate(old)} -> {format_rate(new)}"
                    for pid, old, new in row.persona_drops
                )
                typer.echo(f"  personas: {drops}")
        elif row.status == "missing":
            failed.append(row.rubric_id)
            typer.echo(f"{row.rubric_id}: no results in this run")
    if comparison.overall_regressed:
        failed.append("overall")
        delta = format_points(
            None
            if comparison.overall_baseline is None or comparison.overall_current is None
            else 100 * (comparison.overall_current - comparison.overall_baseline)
        )
        typer.echo(
            f"overall: {format_rate(comparison.overall_baseline)} ->"
            f" {format_rate(comparison.overall_current)} ({delta})"
        )
    if failed:
        typer.echo(
            f"REGRESSION: {len(failed)} item(s) dropped more than {t} pts: {', '.join(failed)}"
        )
    else:
        typer.echo("no regression")


calibrate_app = typer.Typer(help="Build judge calibration sets.", no_args_is_help=True)
app.add_typer(calibrate_app, name="calibrate")


@calibrate_app.command("prepare")
def calibrate_prepare(
    sources: Annotated[
        list[Path],
        typer.Option("--from", help="A run folder or results.json. Repeatable, in order."),
    ],
    out: Annotated[Path, typer.Option(help="Folder for the labelling set.")],
    max_items: Annotated[
        int, typer.Option("--max", min=1, help="Most conversations to include.")
    ] = 30,
) -> None:
    """Build a blind labelling set: transcripts and empty labels, no judge verdicts."""
    try:
        prepared = prepare_calibration(sources, out, max_items)
    except ConvoprobeError as exc:
        raise _fail(str(exc)) from None
    for source, count in prepared.per_source:
        typer.echo(f"{source}: {count} item(s)")
    typer.echo(f"prepared {prepared.items} item(s), {prepared.labels} label(s) to fill: {out}")


@calibrate_app.command("score")
def calibrate_score(
    set_dir: Annotated[Path, typer.Option("--set", help="Calibration set folder.")],
    scenarios: Annotated[
        list[Path],
        typer.Option("--scenario", help="Scenario file defining the judge items. Repeatable."),
    ],
    judge: Annotated[str, typer.Option(help="Judge model as provider:model.")],
    out: Annotated[Path, typer.Option(help="Report JSON to write.")],
    replay: Annotated[
        Path | None, typer.Option(help="Replay judge calls from a cassette. No keys needed.")
    ] = None,
    db: Annotated[Path, typer.Option(help="SQLite file for the LLM cache.")] = DEFAULT_DB,
) -> None:
    """Run one judge over the labelled set and measure its agreement with the labels."""
    try:
        settings = load_settings()
        with ExitStack() as stack:
            cache: LLMCache = (
                _open_cassette(replay)
                if replay is not None
                else stack.enter_context(SqliteLLMCache(db))
            )
            registry = LLMRegistry(
                settings,
                cache,
                replay_only=replay is not None,
                run_id=f"calib-{judge_slug(judge)}",
            )
            client = registry.client(judge, JUDGE_TEMPERATURE)
            if replay is None:
                asyncio.run(_preflight(settings, [judge]))
            report = asyncio.run(score_judge(set_dir, scenarios, judge, client))
    except (ScenarioError, ConvoprobeError) as exc:
        raise _fail(str(exc)) from None
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8", newline="\n")
    _print_calibration(report, Console())
    typer.echo(f"report: {out}")


def _print_calibration(report: CalibrationReport, console: Console) -> None:
    table = Table(title=f"judge {report.judge}")
    for column in ("rubric", "n", "agreement", "kappa", "errors"):
        table.add_column(column, justify="left" if column == "rubric" else "right")
    for row in report.per_rubric:
        agreement = (
            format_share(row.agreement)
            if row.answer_type == "yes_no"
            else f"exact {format_share(row.exact)}, within 1 {format_share(row.within1)}"
        )
        table.add_row(
            row.rubric_id,
            str(row.n),
            agreement,
            format_kappa(row.decision_kappa),
            str(row.n_errors),
        )
    console.print(table)
    console.print(
        f"overall decisions: {format_share(report.overall_decision_agreement)} agree,"
        f" kappa {format_kappa(report.overall_decision_kappa)}"
        f" ({report.n_labels} labels, {report.n_errors} errors)",
        markup=False,
        soft_wrap=True,
    )
    console.print(f"labels: {report.labeled_by}", markup=False, soft_wrap=True)


@calibrate_app.command("table")
def calibrate_table(
    reports: Annotated[
        list[Path], typer.Argument(metavar="FILE", help="Calibration report JSON files.")
    ],
    out: Annotated[Path, typer.Option(help="Markdown file to write.")] = Path(
        "calibration/RESULTS.md"
    ),
    title: Annotated[str | None, typer.Option(help="Section heading above the table.")] = None,
    append: Annotated[
        bool, typer.Option("--append", help="Add to the existing --out file.")
    ] = False,
    footer: Annotated[
        bool,
        typer.Option(
            "--footer", help="End with sample size, labels, date, reproduce commands, explanation."
        ),
    ] = False,
    notes: Annotated[
        list[str] | None, typer.Option("--note", help="Paragraph after the table. Repeatable.")
    ] = None,
    no_table: Annotated[
        bool, typer.Option("--no-table", help="Write the heading and notes only.")
    ] = False,
) -> None:
    """Write a Markdown table comparing judges. Build a multi-section file with --append."""
    if append and not out.is_file():
        raise _fail(f"--append needs an existing file: {out}")
    try:
        loaded = [load_report(path) for path in reports]
        text = render_table(
            loaded,
            _today(),
            title=title,
            notes=notes or [],
            table=not no_table,
            footer=footer,
            heading=not append,
        )
    except ConvoprobeError as exc:
        raise _fail(str(exc)) from None
    if append:
        text = out.read_text(encoding="utf-8").rstrip("\n") + "\n\n" + text
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8", newline="\n")
    typer.echo(f"{'appended to' if append else 'wrote'} {out} ({len(loaded)} judge(s))")


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
    mode: str = "",
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
        mode=mode,
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
            mode = "replay" if cassette is not None else "no-llm"
            results.append(
                asyncio.run(_demo_run(scenario, version, store, out_dir, clients, mode=mode))
            )
    v1, v2 = results
    print_run_report(v1, console, "v1 (correct agent)")
    print_run_report(v2, console, "v2 (prompt change with a bug)")
    if out is not None:
        for version, result in (("v1", v1), ("v2", v2)):
            path = write_html(result, out / result.run_id / REPORT_FILE)
            console.print(f"html {version}: {path}", markup=False, soft_wrap=True)
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
                asyncio.run(
                    _demo_run(scenario, version, store, out_dir, clients, run_id, mode="live")
                )
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
