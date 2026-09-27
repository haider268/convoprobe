"""Scenario loading and error messages."""

from __future__ import annotations

from importlib.resources import as_file, files
from pathlib import Path

import pytest

from convoprobe.datamodels import RegexAbsentItem
from convoprobe.scenario import ScenarioError, load_scenario, parse_scenario

VALID = """\
name: t
agent_description: A test agent.
personas:
  - id: pat
    script: ["hi"]
rubric:
  - id: turns
    type: max_turns
    value: 3
"""


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "scenario.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_demo_scenario_loads() -> None:
    with as_file(files("convoprobe.demo") / "scenario.yaml") as path:
        scenario = load_scenario(path)
    assert scenario.name == "toy-support"
    assert len(scenario.personas) == 4
    assert len(scenario.concrete_personas()) == 3
    assert len(scenario.rubric) == 6
    regex = scenario.rubric[4]
    assert isinstance(regex, RegexAbsentItem)
    assert regex.pattern == r"\bguarantee(d)?\b"


def test_valid_file(tmp_path: Path) -> None:
    assert load_scenario(str(write(tmp_path, VALID))).name == "t"


def test_missing_file(tmp_path: Path) -> None:
    path = tmp_path / "nope.yaml"
    with pytest.raises(ScenarioError) as info:
        load_scenario(path)
    assert str(info.value) == f"{path}: file not found"


def test_empty_file(tmp_path: Path) -> None:
    path = write(tmp_path, "")
    with pytest.raises(ScenarioError) as info:
        load_scenario(path)
    assert str(info.value) == f"{path}: expected a mapping at the top level"


def test_top_level_list(tmp_path: Path) -> None:
    path = write(tmp_path, "- a\n- b\n")
    with pytest.raises(ScenarioError) as info:
        load_scenario(path)
    assert str(info.value) == f"{path}: expected a mapping at the top level"


def test_yaml_syntax_error_line(tmp_path: Path) -> None:
    path = write(tmp_path, "name: t\nagent_description: x\npersonas: [\n  - id: a\n")
    with pytest.raises(ScenarioError) as info:
        load_scenario(path)
    message = str(info.value)
    assert message.startswith(f"{path}: YAML error at line ")
    # The broken flow sequence is reported on line 4 (1-based).
    assert "line 4:" in message


def test_yaml_error_on_first_line(tmp_path: Path) -> None:
    path = write(tmp_path, "name: [unclosed\n")
    with pytest.raises(ScenarioError, match=r"YAML error at line 2"):
        load_scenario(path)


def test_validation_error_dotted_path(tmp_path: Path) -> None:
    text = VALID + "  - id: bad_regex\n    type: regex_present\n    pattern: '(unclosed'\n"
    path = write(tmp_path, text)
    with pytest.raises(ScenarioError) as info:
        load_scenario(path)
    message = str(info.value)
    assert message.startswith(f"{path}: rubric.1.pattern: ")
    assert "invalid regex" in message
    assert "regex_present" not in message.split(":")[1]


def test_validation_error_persona_path_strips_tag() -> None:
    data = {
        "name": "t",
        "agent_description": "x",
        "personas": [{"id": "Bad-ID", "description": "d", "goal": "g"}],
        "rubric": [{"id": "r", "type": "max_turns", "value": 1}],
    }
    with pytest.raises(ScenarioError) as info:
        parse_scenario(data, source="mem.yaml")
    assert str(info.value).startswith("mem.yaml: personas.0.id: ")


def test_generate_field_name_kept() -> None:
    data = {
        "name": "t",
        "agent_description": "x",
        "personas": [{"generate": 0}],
        "rubric": [{"id": "r", "type": "max_turns", "value": 1}],
    }
    with pytest.raises(ScenarioError) as info:
        parse_scenario(data)
    assert str(info.value).startswith("<memory>: personas.0.generate: ")


def test_one_line_per_error() -> None:
    with pytest.raises(ScenarioError) as info:
        parse_scenario({"name": "", "agent_description": ""})
    lines = str(info.value).splitlines()
    assert len(lines) == 4
    assert all(line.startswith("<memory>: ") for line in lines)
    assert {line.split(": ")[1] for line in lines} == {
        "name",
        "agent_description",
        "personas",
        "rubric",
    }


def test_model_level_error_has_no_loc() -> None:
    data = {
        "name": "t",
        "agent_description": "x",
        "personas": [{"id": "a", "script": ["x"]}, {"id": "a", "script": ["y"]}],
        "rubric": [{"id": "r", "type": "max_turns", "value": 1}],
    }
    with pytest.raises(ScenarioError) as info:
        parse_scenario(data)
    assert str(info.value) == "<memory>: Value error, duplicate persona id 'a'"


def test_parse_rejects_non_mapping() -> None:
    with pytest.raises(ScenarioError, match="expected a mapping"):
        parse_scenario(["not", "a", "dict"])  # type: ignore[arg-type]
