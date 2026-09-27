"""results.json writer and the run report printed at the end of every run."""

from __future__ import annotations

from pathlib import Path

from rich.console import Console
from rich.markup import escape
from rich.table import Table

from convoprobe.datamodels import RunResult


def format_rate(rate: float | None) -> str:
    """A pass rate as a whole percentage, or "-" when there is nothing to rate."""
    return "-" if rate is None else f"{rate * 100:.0f}%"


def rate_band(rate: float | None) -> str | None:
    """ "pass" at 100%, "fail" at 0%, "mixed" in between, None with no data."""
    if rate is None:
        return None
    if rate >= 1.0:
        return "pass"
    if rate <= 0.0:
        return "fail"
    return "mixed"


_BAND_STYLE = {"pass": "green", "fail": "red", "mixed": "yellow"}


def persona_matrix(
    result: RunResult,
) -> tuple[list[str], list[str], dict[str, dict[str, float | None]]]:
    """(persona ids sorted, rubric ids in rubric order, persona -> rubric -> rate)."""
    rubric_ids = [row.rubric_id for row in result.summary]
    rates: dict[str, dict[str, float | None]] = {}
    for row in result.summary:
        for persona_id, rate in row.persona_rates.items():
            rates.setdefault(persona_id, {})[row.rubric_id] = rate
    return sorted(rates), rubric_ids, rates


def write_results(result: RunResult, out_dir: Path) -> Path:
    """Write out_dir/<run_id>/results.json and return its path."""
    path = Path(out_dir) / result.run_id / "results.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    return path


def print_run_report(result: RunResult, console: Console, title: str) -> None:
    """Rubric table, conversation counts, overall pass rate, and notes."""
    console.print(title, markup=False, soft_wrap=True)
    show_stdev = any(row.pass_rate_stdev is not None for row in result.summary)
    columns = ["rubric", "pass", "fail", "error", "skip", "rate"] + (["±"] if show_stdev else [])
    table = Table()
    for column in columns:
        table.add_column(column, justify="left" if column == "rubric" else "right")
    for row in result.summary:
        cells = [
            row.rubric_id,
            str(row.passed),
            str(row.failed),
            str(row.errored),
            str(row.skipped),
            format_rate(row.pass_rate),
        ]
        if show_stdev:
            stdev = row.pass_rate_stdev
            cells.append("-" if stdev is None else f"{stdev:.2f}")
        table.add_row(*cells)
    console.print(table)
    _print_persona_matrix(result, console)
    if result.models:
        console.print(
            f"models: simulator {result.models.get('simulator', 'none')},"
            f" judge {result.models.get('judge', 'none')}",
            markup=False,
            soft_wrap=True,
        )
    for usage in result.llm_usage:
        console.print(
            f"llm {usage.spec}: {usage.calls} calls, {usage.cache_hits} cache hits,"
            f" {usage.retries} retries, {usage.input_tokens} in / {usage.output_tokens} out tokens",
            markup=False,
            soft_wrap=True,
        )
    statuses = list(result.conversation_status.values())
    line = (
        f"conversations: {statuses.count('done')} done, {statuses.count('failed')} failed,"
        f" {statuses.count('failed_transient')} transient, {statuses.count('skipped')} skipped"
    )
    sim_errors = sum(1 for c in result.conversations if c.ended_reason == "simulator_error")
    if sim_errors:
        line += f", {sim_errors} simulator errors"
    console.print(line, soft_wrap=True)
    console.print(f"overall pass rate: {format_rate(result.overall_pass_rate)}", soft_wrap=True)
    for note in result.notes:
        console.print(f"note: {note}", markup=False, soft_wrap=True)


def _print_persona_matrix(result: RunResult, console: Console) -> None:
    """Pass rate per persona and rubric item. Colour is only on the cells."""
    personas, rubric_ids, rates = persona_matrix(result)
    if not personas:
        return
    # Numbered rubric columns with a legend below: full ids do not fit 80 columns.
    table = Table(title="pass rate by persona")
    table.add_column("persona", no_wrap=True)
    for number in range(1, len(rubric_ids) + 1):
        table.add_column(str(number), justify="right", no_wrap=True)
    for persona_id in personas:
        cells = [escape(persona_id)]
        for rubric_id in rubric_ids:
            rate = rates[persona_id].get(rubric_id)
            band = rate_band(rate)
            text = format_rate(rate)
            cells.append(f"[{_BAND_STYLE[band]}]{text}[/]" if band else text)
        table.add_row(*cells)
    console.print(table)
    legend = ", ".join(f"{n} {rubric_id}" for n, rubric_id in enumerate(rubric_ids, start=1))
    console.print(f"columns: {legend}", markup=False, soft_wrap=True)
