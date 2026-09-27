"""results.json writer and the run report printed at the end of every run."""

from __future__ import annotations

from pathlib import Path

from rich.console import Console
from rich.table import Table

from convoprobe.datamodels import RunResult


def format_rate(rate: float | None) -> str:
    """A pass rate as a whole percentage, or "-" when there is nothing to rate."""
    return "-" if rate is None else f"{rate * 100:.0f}%"


def write_results(result: RunResult, out_dir: Path) -> Path:
    """Write out_dir/<run_id>/results.json and return its path."""
    path = Path(out_dir) / result.run_id / "results.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    return path


def print_run_report(result: RunResult, console: Console, title: str) -> None:
    """Rubric table, conversation counts, overall pass rate, and notes."""
    console.print(title, markup=False, soft_wrap=True)
    table = Table()
    for column in ("rubric", "pass", "fail", "error", "skip", "rate"):
        table.add_column(column, justify="left" if column == "rubric" else "right")
    for row in result.summary:
        table.add_row(
            row.rubric_id,
            str(row.passed),
            str(row.failed),
            str(row.errored),
            str(row.skipped),
            format_rate(row.pass_rate),
        )
    console.print(table)
    statuses = list(result.conversation_status.values())
    console.print(
        f"conversations: {statuses.count('done')} done, {statuses.count('failed')} failed,"
        f" {statuses.count('failed_transient')} transient, {statuses.count('skipped')} skipped",
        soft_wrap=True,
    )
    console.print(f"overall pass rate: {format_rate(result.overall_pass_rate)}", soft_wrap=True)
    for note in result.notes:
        console.print(f"note: {note}", markup=False, soft_wrap=True)
