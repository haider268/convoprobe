"""Load a results.json written by a run."""

from __future__ import annotations

from pathlib import Path

from pydantic import ValidationError

from convoprobe.datamodels import RunResult
from convoprobe.errors import ConvoprobeError

RESULTS_FILE = "results.json"


def results_file(path: str | Path) -> Path:
    """The results.json for a run directory or a results.json path."""
    path = Path(path)
    return path / RESULTS_FILE if path.is_dir() else path


def load_results(path: str | Path) -> RunResult:
    """Read a run directory or results.json file into a RunResult."""
    file = results_file(path)
    if not file.is_file():
        raise ConvoprobeError(f"no results.json at {path}")
    try:
        return RunResult.model_validate_json(file.read_text(encoding="utf-8"))
    except ValidationError as exc:
        first = exc.errors()[0]
        where = ".".join(str(p) for p in first["loc"])
        detail = f"{where}: {first['msg']}" if where else first["msg"]
        raise ConvoprobeError(f"{file}: not a convoprobe results file: {detail}") from None
