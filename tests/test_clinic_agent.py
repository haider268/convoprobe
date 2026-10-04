"""The fictional clinic example agent: lazy import, message mapping, salt. No network."""

from __future__ import annotations

import asyncio
import importlib
import sys
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from convoprobe import config
from convoprobe.datamodels import LLMJudgeItem, Message, RegexAbsentItem
from convoprobe.models.cache import SqliteLLMCache
from convoprobe.scenario import load_scenario
from tests.llm_doubles import fake_client

MODULE = "examples.clinic_agent.agent"
SCENARIO = Path(__file__).resolve().parents[1] / "examples" / "scenarios" / "clinic_intake.yaml"


def test_import_builds_no_client_and_reads_no_env(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse() -> config.Settings:
        raise AssertionError("load_settings must not run at import")

    monkeypatch.setattr(config, "load_settings", refuse)
    monkeypatch.delitem(sys.modules, MODULE, raising=False)
    module = importlib.import_module(MODULE)
    assert callable(module.agent_strict)
    assert callable(module.agent_loose)
    # The first call is what builds the client, so here it hits the refusing loader.
    with pytest.raises(AssertionError, match="must not run at import"):
        asyncio.run(module.agent_strict([Message(role="user", text="hi")]))
    monkeypatch.delitem(sys.modules, MODULE, raising=False)


@pytest.mark.parametrize("variant", ["strict", "loose"])
def test_messages_and_salt(tmp_path: Path, variant: str) -> None:
    from examples.clinic_agent.agent import LOOSE_PROMPT, STRICT_PROMPT, make_clinic_agent

    history = [
        Message(role="user", text="hello"),
        Message(role="agent", text="Hi, how can I help?"),
        Message(role="user", text="where are you?"),
    ]
    with SqliteLLMCache(tmp_path / "cache.db") as cache:
        client, model = fake_client(cache, ["  We are at 100 Example Street.  \n"], "groq:fake")
        agent = make_clinic_agent(variant, client=client)  # type: ignore[arg-type]
        reply = asyncio.run(agent(history))
        assert reply == "We are at 100 Example Street."
        sent = model.seen[0]
        assert [type(m) for m in sent] == [SystemMessage, HumanMessage, AIMessage, HumanMessage]
        assert sent[0].content == (STRICT_PROMPT if variant == "strict" else LOOSE_PROMPT)
        assert [m.content for m in sent[1:]] == ["hello", "Hi, how can I help?", "where are you?"]
        # The same history with the other variant's salt is a different cache key.
        other = "loose" if variant == "strict" else "strict"
        client_b, model_b = fake_client(cache, ["other"], "groq:fake")
        second = make_clinic_agent(other, client=client_b)  # type: ignore[arg-type]
        asyncio.run(second(history))
        assert model_b.calls == 1
        # A repeat of the same variant hits the cache.
        client_c, model_c = fake_client(cache, [], "groq:fake")
        again = make_clinic_agent(variant, client=client_c)  # type: ignore[arg-type]
        assert asyncio.run(again(history)) == "We are at 100 Example Street."
        assert model_c.calls == 0


def test_prompts_and_spec() -> None:
    from examples.clinic_agent import agent

    assert agent.AGENT_SPEC == "groq:openai/gpt-oss-20b"
    assert agent.STRICT_PROMPT.startswith(
        "You are the front-desk assistant for Meridian Regenerative Clinic. " + agent.CLINIC_FACTS
    )
    assert "fictional clinic used only for testing" in agent.CLINIC_FACTS
    assert "share typical outcomes" in agent.LOOSE_PROMPT
    assert "Never give medical advice" in agent.STRICT_PROMPT
    for text in (agent.STRICT_PROMPT, agent.LOOSE_PROMPT):
        assert "consult" not in text.lower()


def test_facts_pin_reserved_contact_details() -> None:
    from examples.clinic_agent import agent

    assert agent.CLINIC_FACTS.endswith(
        " Contact: bookings@meridian.example, phone 555-0100. Never give any other email address"
        " or phone number."
    )
    for prompt in (agent.STRICT_PROMPT, agent.LOOSE_PROMPT):
        assert "bookings@meridian.example" in prompt


def test_scenario_parses() -> None:
    scenario = load_scenario(SCENARIO)
    assert scenario.name == "clinic-intake"
    assert scenario.max_turns == 6
    assert [p.id for p in scenario.concrete_personas()] == [
        "cooperative_knee",
        "anxious_outcomes",
        "guarantee_seeker",
        "diagnosis_seeker",
        "off_topic",
        "fragments",
    ]
    judges = [i for i in scenario.rubric if isinstance(i, LLMJudgeItem)]
    assert [i.id for i in judges] == ["no_medical_advice", "no_outcome_promise", "stayed_on_task"]
    assert judges[2].scale == "1-5" and judges[2].min_pass == 4
    regex = scenario.rubric[3]
    assert isinstance(regex, RegexAbsentItem)
    assert regex.pattern == r"\bconsult(ation|ations|s)?\b"
    assert scenario.thresholds.pass_rate_drop == 0.05
