"""Persona ids and LLM persona generation."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from convoprobe.datamodels import Scenario
from convoprobe.models.cache import SqliteLLMCache
from convoprobe.simulator.personas import PERSONA_PROMPT, generate_personas, persona_id
from tests.llm_doubles import fake_client

SCENARIO = Scenario.model_validate(
    {
        "name": "t",
        "agent_description": "Support agent for shop.example.",
        "personas": [
            {"id": "cooperative", "description": "Answers clearly.", "goal": "Refund."},
            {"id": "gen_pat", "script": ["I want a refund now.", "now!"]},
            {"generate": 3},
        ],
        "rubric": [{"id": "r", "type": "max_turns", "value": 5}],
    }
)


def generated(*names: str) -> dict[str, Any]:
    return {
        "personas": [
            {"name": n, "description": f"{n} behaves oddly.", "goal": "Get a refund."}
            for n in names
        ]
    }


@pytest.fixture
def cache(tmp_path: Path) -> Iterator[SqliteLLMCache]:
    with SqliteLLMCache(tmp_path / "cache.db") as opened:
        yield opened


def test_persona_id_basic() -> None:
    taken: set[str] = set()
    assert persona_id("Confused Elderly Caller!", taken) == "gen_confused_elderly_caller"
    assert taken == {"gen_confused_elderly_caller"}


def test_persona_id_collision() -> None:
    taken: set[str] = set()
    assert persona_id("Pat", taken) == "gen_pat"
    assert persona_id("pat", taken) == "gen_pat_2"
    assert persona_id("PAT!", taken) == "gen_pat_3"


def test_persona_id_empty_name() -> None:
    assert persona_id("!!!", set()) == "gen_persona"
    assert persona_id("", set()) == "gen_persona"


def test_persona_id_truncated() -> None:
    taken: set[str] = set()
    long_id = persona_id("a" * 60, taken)
    assert long_id == "gen_" + "a" * 36
    assert len(long_id) == 40
    second = persona_id("a" * 60, taken)
    assert second == "gen_" + "a" * 34 + "_2"
    assert len(second) == 40


async def test_generate_keeps_first_n_and_avoids_scenario_ids(cache: SqliteLLMCache) -> None:
    client, model = fake_client(cache, [generated("Pat", "Hostile Harry", "Off Topic", "Extra")])
    personas, notes = await generate_personas(SCENARIO, 3, client, salt="s")
    assert [p.id for p in personas] == ["gen_pat_2", "gen_hostile_harry", "gen_off_topic"]
    assert personas[1].description == "Hostile Harry behaves oddly."
    assert notes == []
    prompt = str(model.seen[0][0].content)
    assert prompt.startswith("Create 3 different simulated users")
    assert "- cooperative: Answers clearly.\n- gen_pat: I want a refund now." in prompt


async def test_prompt_lists_none_when_no_concrete_personas(cache: SqliteLLMCache) -> None:
    only_generate = SCENARIO.model_copy(update={"personas": [SCENARIO.personas[2]]})
    client, model = fake_client(cache, [generated("One")])
    await generate_personas(only_generate, 1, client, salt="s")
    assert model.seen[0][0].content == PERSONA_PROMPT.format(
        n=1, agent_description="Support agent for shop.example.", existing="- none"
    )


async def test_fewer_than_n_note(cache: SqliteLLMCache) -> None:
    client, _ = fake_client(cache, [generated("Only One")])
    personas, notes = await generate_personas(SCENARIO, 3, client, salt="s")
    assert len(personas) == 1
    assert notes == ["asked for 3 personas, model returned 1"]


async def test_retry_then_success(cache: SqliteLLMCache) -> None:
    client, model = fake_client(cache, ["BAD", generated("A", "B")])
    personas, notes = await generate_personas(SCENARIO, 2, client, salt="s")
    assert [p.id for p in personas] == ["gen_a", "gen_b"]
    assert notes == []
    assert model.calls == 2


async def test_double_failure_note(cache: SqliteLLMCache) -> None:
    client, _ = fake_client(cache, ["BAD", "BAD"])
    personas, notes = await generate_personas(SCENARIO, 2, client, salt="s")
    assert personas == []
    assert notes == [
        "persona generation failed: ollama:fake returned output that does not match"
        " GeneratedPersonas: not json"
    ]


async def test_empty_goal_dropped(cache: SqliteLLMCache) -> None:
    output = {"personas": [{"name": "Blank", "description": "d", "goal": ""}]}
    client, _ = fake_client(cache, [output])
    personas, notes = await generate_personas(SCENARIO, 1, client, salt="s")
    assert personas == []
    assert notes == ["generated persona 'Blank' dropped: empty description or goal"]


async def test_taken_ids_respected(cache: SqliteLLMCache) -> None:
    client, _ = fake_client(cache, [generated("Grumpy")])
    personas, _ = await generate_personas(SCENARIO, 1, client, salt="s", taken={"gen_grumpy"})
    assert personas[0].id == "gen_grumpy_2"
