"""Scenario YAML loading and validation with field-level error messages."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from convoprobe.datamodels import Scenario

_PERSONA_TAGS = {"llm", "scripted", "generate"}
_RUBRIC_TAGS = {
    "tool_called",
    "tool_not_called",
    "tool_order",
    "regex_present",
    "regex_absent",
    "max_turns",
    "max_latency_ms",
    "tool_args",
    "llm_judge",
}
_TAGS_BY_LIST = {"personas": _PERSONA_TAGS, "rubric": _RUBRIC_TAGS}


class ScenarioError(Exception):
    """A scenario file could not be loaded or is invalid."""


def _dotted(loc: tuple[int | str, ...]) -> str:
    """Join a pydantic loc with dots, dropping union tag parts."""
    parts: list[str] = []
    for i, part in enumerate(loc):
        is_tag = (
            i >= 2
            and isinstance(loc[i - 1], int)
            and part in _TAGS_BY_LIST.get(str(loc[i - 2]), set())
        )
        if not is_tag:
            parts.append(str(part))
    return ".".join(parts)


def parse_scenario(data: dict[str, Any], source: str = "<memory>") -> Scenario:
    """Validate an already-parsed mapping into a Scenario."""
    if not isinstance(data, dict):
        raise ScenarioError(f"{source}: expected a mapping at the top level")
    try:
        return Scenario.model_validate(data)
    except ValidationError as exc:
        lines = []
        for error in exc.errors():
            loc = _dotted(error["loc"])
            prefix = f"{source}: {loc}: " if loc else f"{source}: "
            lines.append(prefix + error["msg"])
        raise ScenarioError("\n".join(lines)) from None


def load_scenario(path: str | Path) -> Scenario:
    """Load and validate a scenario YAML file."""
    path = Path(path)
    if not path.is_file():
        raise ScenarioError(f"{path}: file not found")
    text = path.read_text(encoding="utf-8")
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        mark = getattr(exc, "problem_mark", None)
        problem = getattr(exc, "problem", None) or str(exc)
        if mark is None:
            raise ScenarioError(f"{path}: YAML error: {problem}") from None
        raise ScenarioError(f"{path}: YAML error at line {mark.line + 1}: {problem}") from None
    if not isinstance(data, dict):
        raise ScenarioError(f"{path}: expected a mapping at the top level")
    return parse_scenario(data, source=str(path))
