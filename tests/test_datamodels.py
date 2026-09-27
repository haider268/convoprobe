"""Data model parsing and validation."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import TypeAdapter, ValidationError

from convoprobe.datamodels import (
    Conversation,
    GeneratePersonas,
    LLMJudgeItem,
    LLMPersona,
    Message,
    PersonaEntry,
    RubricItem,
    Scenario,
    ScriptedPersona,
    ToolCallRecord,
    ToolOrderItem,
)

persona_adapter: TypeAdapter[Any] = TypeAdapter(PersonaEntry)
rubric_adapter: TypeAdapter[Any] = TypeAdapter(RubricItem)


def scenario_dict(**overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "name": "test",
        "agent_description": "A test agent for shop.example.",
        "personas": [
            {"id": "pat", "description": "Pat Example, polite.", "goal": "Ask a question."},
            {"id": "scripted_one", "script": ["hello"]},
        ],
        "rubric": [{"id": "turns", "type": "max_turns", "value": 5}],
    }
    data.update(overrides)
    return data


# Personas


def test_llm_persona_parses() -> None:
    persona = persona_adapter.validate_python({"id": "a", "description": "d", "goal": "g"})
    assert isinstance(persona, LLMPersona)


def test_scripted_persona_parses() -> None:
    persona = persona_adapter.validate_python({"id": "b", "script": ["hi", "bye"]})
    assert isinstance(persona, ScriptedPersona)
    assert persona.script == ["hi", "bye"]
    assert persona.description == ""


def test_generate_persona_parses() -> None:
    persona = persona_adapter.validate_python({"generate": 4})
    assert isinstance(persona, GeneratePersonas)
    assert persona.generate == 4


def test_persona_discriminator_accepts_instances() -> None:
    for instance in (
        LLMPersona(id="a", description="d", goal="g"),
        ScriptedPersona(id="b", script=["x"]),
        GeneratePersonas(generate=2),
    ):
        assert persona_adapter.validate_python(instance) is not None
        assert type(persona_adapter.validate_python(instance)) is type(instance)


@pytest.mark.parametrize("count", [0, 51])
def test_generate_bounds(count: int) -> None:
    with pytest.raises(ValidationError):
        persona_adapter.validate_python({"generate": count})


def test_scripted_persona_needs_turns() -> None:
    with pytest.raises(ValidationError):
        persona_adapter.validate_python({"id": "b", "script": []})


def test_llm_persona_needs_goal() -> None:
    with pytest.raises(ValidationError):
        persona_adapter.validate_python({"id": "a", "description": "d", "goal": ""})


def test_bad_persona_id_pattern() -> None:
    with pytest.raises(ValidationError, match="pattern"):
        persona_adapter.validate_python({"id": "Bad-ID", "description": "d", "goal": "g"})


def test_bad_rubric_id_pattern() -> None:
    with pytest.raises(ValidationError, match="pattern"):
        rubric_adapter.validate_python({"id": "Bad-ID", "type": "max_turns", "value": 1})


# Rubric items


def test_rubric_discriminator_picks_class() -> None:
    item = rubric_adapter.validate_python(
        {"id": "o", "type": "tool_order", "before": "a", "after": "b"}
    )
    assert isinstance(item, ToolOrderItem)


def test_unknown_rubric_type_rejected() -> None:
    with pytest.raises(ValidationError, match="does not match any of the expected tags"):
        rubric_adapter.validate_python({"id": "x", "type": "made_up"})


def test_tool_order_before_equals_after() -> None:
    with pytest.raises(ValidationError, match="before and after must differ"):
        rubric_adapter.validate_python(
            {"id": "o", "type": "tool_order", "before": "a", "after": "a"}
        )


@pytest.mark.parametrize("kind", ["regex_present", "regex_absent"])
def test_bad_regex_rejected(kind: str) -> None:
    with pytest.raises(ValidationError, match="invalid regex"):
        rubric_adapter.validate_python({"id": "r", "type": kind, "pattern": "(unclosed"})


def test_regex_defaults() -> None:
    item = rubric_adapter.validate_python({"id": "r", "type": "regex_present", "pattern": "hi"})
    assert item.speaker == "agent"
    assert item.ignore_case is True


def test_llm_judge_yes_no_and_scale() -> None:
    yes_no = rubric_adapter.validate_python(
        {"id": "j", "type": "llm_judge", "question": "Polite?", "pass_if": "yes"}
    )
    scale = rubric_adapter.validate_python(
        {"id": "k", "type": "llm_judge", "question": "Tone?", "scale": "1-5", "min_pass": 3}
    )
    assert isinstance(yes_no, LLMJudgeItem)
    assert isinstance(scale, LLMJudgeItem)
    assert scale.min_pass == 3


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        ({"pass_if": "yes", "scale": "1-5", "min_pass": 3}, "exactly one of pass_if or scale"),
        ({}, "exactly one of pass_if or scale"),
        ({"scale": "1-5"}, "min_pass is required"),
        ({"pass_if": "no", "min_pass": 2}, "min_pass is only allowed with scale"),
    ],
)
def test_llm_judge_validator(extra: dict[str, Any], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        rubric_adapter.validate_python({"id": "j", "type": "llm_judge", "question": "Q?", **extra})


def test_extra_field_rejected() -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        rubric_adapter.validate_python({"id": "t", "type": "max_turns", "value": 1, "bogus": 1})
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        Scenario.model_validate(scenario_dict(unknown_key=True))


# Scenario


def test_scenario_parses_and_concrete_personas() -> None:
    data = scenario_dict()
    data["personas"].append({"generate": 3})
    scenario = Scenario.model_validate(data)
    assert len(scenario.personas) == 3
    assert [p.id for p in scenario.concrete_personas()] == ["pat", "scripted_one"]
    assert scenario.max_turns == 10
    assert scenario.repeats == 1
    assert scenario.thresholds.pass_rate_drop == 0.05


def test_duplicate_persona_id() -> None:
    personas = [{"id": "pat", "script": ["a"]}, {"id": "pat", "script": ["b"]}]
    with pytest.raises(ValidationError, match="duplicate persona id 'pat'"):
        Scenario.model_validate(scenario_dict(personas=personas))


def test_duplicate_rubric_id() -> None:
    rubric = [
        {"id": "same", "type": "max_turns", "value": 1},
        {"id": "same", "type": "tool_called", "tool": "x"},
    ]
    with pytest.raises(ValidationError, match="duplicate rubric id 'same'"):
        Scenario.model_validate(scenario_dict(rubric=rubric))


def test_applies_to_unknown_persona() -> None:
    rubric = [{"id": "t", "type": "max_turns", "value": 1, "applies_to": ["ghost"]}]
    with pytest.raises(ValidationError, match="unknown persona 'ghost'"):
        Scenario.model_validate(scenario_dict(rubric=rubric))


def test_applies_to_known_persona() -> None:
    rubric = [{"id": "t", "type": "max_turns", "value": 1, "applies_to": ["scripted_one"]}]
    scenario = Scenario.model_validate(scenario_dict(rubric=rubric))
    assert scenario.rubric[0].applies_to == ["scripted_one"]


def test_thresholds_bounds() -> None:
    with pytest.raises(ValidationError):
        Scenario.model_validate(scenario_dict(thresholds={"pass_rate_drop": 1.5}))


# Conversation


def test_tool_calls_in_order_indices() -> None:
    conv = Conversation(
        id="c1",
        persona_id="pat",
        messages=[
            Message(role="user", text="hi"),
            Message(role="agent", text="one", tool_calls=[ToolCallRecord(name="a")]),
            Message(role="user", text="more"),
            Message(role="agent", text="two"),
            Message(role="user", text="again"),
            Message(
                role="agent",
                text="three",
                tool_calls=[ToolCallRecord(name="b"), ToolCallRecord(name="c")],
            ),
        ],
    )
    assert len(conv.agent_messages()) == 3
    assert [(i, c.name) for i, c in conv.tool_calls_in_order()] == [(0, "a"), (2, "b"), (2, "c")]
