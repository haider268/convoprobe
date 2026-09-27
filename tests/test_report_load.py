"""load_results: run folders, results.json files, and bad input."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from convoprobe.errors import ConvoprobeError
from convoprobe.report.load import load_results, results_file

OLD_STYLE = {
    "run_id": "old",
    "scenario_name": "toy-support",
    "adapter": "x",
    "started_at": "2026-09-01T10:00:00+00:00",
    "finished_at": None,
    "conversation_status": {},
    "conversations": [],
    "checks": [],
    "summary": [],
    "overall_pass_rate": None,
    "notes": [],
}


def write(folder: Path, data: object) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "results.json"
    path.write_text(data if isinstance(data, str) else json.dumps(data), encoding="utf-8")
    return path


def test_folder_and_file_inputs(tmp_path: Path) -> None:
    file = write(tmp_path / "run", OLD_STYLE)
    assert load_results(tmp_path / "run").run_id == "old"
    assert load_results(file).run_id == "old"
    assert results_file(tmp_path / "run") == file


def test_old_style_results_get_defaults(tmp_path: Path) -> None:
    result = load_results(write(tmp_path / "run", OLD_STYLE))
    assert result.scenario_hash == ""
    assert result.pass_rate_drop == 0.05
    assert result.mode == ""
    assert result.rubric == []
    assert result.agent_description == ""


def test_missing(tmp_path: Path) -> None:
    with pytest.raises(ConvoprobeError) as info:
        load_results(tmp_path / "nothing")
    assert str(info.value) == f"no results.json at {tmp_path / 'nothing'}"


def test_invalid_json(tmp_path: Path) -> None:
    file = write(tmp_path / "run", "{not json")
    with pytest.raises(ConvoprobeError) as info:
        load_results(file)
    assert str(info.value).startswith(f"{file}: not a convoprobe results file: Invalid JSON")


def test_invalid_schema(tmp_path: Path) -> None:
    file = write(tmp_path / "run", {**OLD_STYLE, "run_id": 5})
    with pytest.raises(ConvoprobeError) as info:
        load_results(file)
    assert str(info.value) == (
        f"{file}: not a convoprobe results file: run_id: Input should be a valid string"
    )
