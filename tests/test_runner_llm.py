"""run_scenario with an LLM simulator, judge, and generated personas (all fake models)."""

from __future__ import annotations

import io
import sqlite3
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from rich.console import Console

from convoprobe.adapters import CallableAdapter
from convoprobe.datamodels import AgentTurn, Message, RunResult, Scenario, ToolCallRecord
from convoprobe.models.cache import SqliteLLMCache
from convoprobe.models.client import LLMClient
from convoprobe.report.json_report import print_run_report
from convoprobe.runner import run_scenario
from convoprobe.simulator.llm import LLMSimulator
from convoprobe.store import RunStore
from tests.llm_doubles import ServerDown, fake_client

JUDGE_ITEM = {"id": "polite", "type": "llm_judge", "question": "Polite?", "pass_if": "yes"}


def scenario(
    personas: list[dict[str, Any]], rubric: list[dict[str, Any]], **extra: Any
) -> Scenario:
    return Scenario.model_validate(
        {
            "name": "t",
            "agent_description": "Shop agent.",
            "personas": personas,
            "rubric": rubric,
            **extra,
        }
    )


LLM_SCENARIO = scenario(
    [{"id": "cooperative", "description": "Pat Example, polite.", "goal": "Refund."}],
    [{"id": "turns", "type": "max_turns", "value": 5}, JUDGE_ITEM],
)


def sim_turn(status: str, message: str = "") -> dict[str, str]:
    return {"goal_status": status, "message": message}


def verdict(value: str = "yes") -> dict[str, Any]:
    return {"reasoning": "fine", "verdict": value, "evidence": [{"turn": 1, "quote": "ok"}]}


@pytest.fixture
def store(tmp_path: Path) -> Iterator[RunStore]:
    with RunStore(tmp_path / "state.db") as opened:
        yield opened


@pytest.fixture
def cache(tmp_path: Path) -> Iterator[SqliteLLMCache]:
    with SqliteLLMCache(tmp_path / "cache.db") as opened:
        yield opened


async def run(
    scn: Scenario, store: RunStore, tmp_path: Path, fn: Any = None, **kwargs: Any
) -> RunResult:
    return await run_scenario(
        scn,
        CallableAdapter(fn or (lambda history: "ok")),
        store=store,
        out_dir=tmp_path / "out",
        adapter_label="test",
        **kwargs,
    )


def simulator(cache: SqliteLLMCache, outputs: list[Any]) -> tuple[LLMSimulator, Any]:
    client, model = fake_client(cache, outputs, spec="ollama:sim")
    return LLMSimulator(client), model


async def test_llm_persona_runs_and_judge_is_stored(
    store: RunStore, cache: SqliteLLMCache, tmp_path: Path
) -> None:
    sim, _ = simulator(cache, [sim_turn("in_progress", "hi"), sim_turn("achieved")])
    judge, _ = fake_client(cache, [verdict()], spec="ollama:judge")
    result = await run(
        LLM_SCENARIO, store, tmp_path, run_id="llm", llm_simulator=sim, judge_client=judge
    )
    conversation = result.conversations[0]
    assert conversation.ended_reason == "simulator_stop"
    assert [m.text for m in conversation.messages] == ["hi", "ok"]
    assert result.conversation_status == {"cooperative-r0": "done"}

    judged = [c for c in result.checks if c.rubric_id == "polite"]
    assert len(judged) == 1
    assert (judged[0].source, judged[0].status, judged[0].verdict) == ("judge", "pass", "yes")
    reloaded = next(c for c in store.load_checks("llm") if c.rubric_id == "polite")
    assert reloaded == judged[0]
    assert reloaded.evidence[0].quote_found is True
    turns = next(c for c in store.load_checks("llm") if c.rubric_id == "turns")
    assert turns.source == "deterministic"

    assert result.models == {"simulator": "ollama:sim", "judge": "ollama:judge"}
    assert [u.spec for u in result.llm_usage] == ["ollama:sim", "ollama:judge"]
    assert [u.calls for u in result.llm_usage] == [2, 1]
    assert "llm_judge items skipped: no judge model configured" not in result.notes


async def test_judge_note_when_no_judge(store: RunStore, tmp_path: Path) -> None:
    result = await run(LLM_SCENARIO, store, tmp_path)
    assert "llm_judge items skipped: no judge model configured" in result.notes
    assert result.models == {}
    assert result.llm_usage == []


async def test_skipped_llm_persona_runs_on_resume(
    store: RunStore, cache: SqliteLLMCache, tmp_path: Path
) -> None:
    first = await run(LLM_SCENARIO, store, tmp_path, run_id="resume")
    assert first.conversation_status == {"cooperative-r0": "skipped"}
    assert "persona cooperative skipped: no simulator model configured" in first.notes

    sim, _ = simulator(cache, [sim_turn("in_progress", "hello"), sim_turn("achieved")])
    second = await run(LLM_SCENARIO, store, tmp_path, run_id="resume", llm_simulator=sim)
    assert second.conversation_status == {"cooperative-r0": "done"}
    assert second.models == {"simulator": "ollama:sim", "judge": "none"}
    judged = next(c for c in second.checks if c.rubric_id == "polite")
    assert (judged.status, judged.detail) == ("skip", "no judge model configured")


async def test_generated_personas_saved_and_reused(
    store: RunStore, cache: SqliteLLMCache, tmp_path: Path
) -> None:
    scn = scenario(
        [{"id": "s", "script": ["hi"]}, {"generate": 2}],
        [{"id": "turns", "type": "max_turns", "value": 5}],
    )
    people = {
        "personas": [
            {"name": "Grumpy", "description": "Short answers.", "goal": "Refund."},
            {"name": "Lost", "description": "Confused.", "goal": "Order status."},
        ]
    }
    persona_client, persona_model = fake_client(cache, [people], spec="ollama:personas")
    first = await run(scn, store, tmp_path, run_id="gen", persona_client=persona_client)
    assert persona_model.calls == 1
    assert sorted(first.conversation_status) == ["gen_grumpy-r0", "gen_lost-r0", "s-r0"]
    assert [p.id for p in store.load_generated_personas("gen", 1) or []] == [
        "gen_grumpy",
        "gen_lost",
    ]

    fresh_client, fresh_model = fake_client(cache, [], spec="ollama:personas")
    sim, _ = simulator(cache, [sim_turn("in_progress", "x"), sim_turn("achieved")] * 2)
    second = await run(
        scn, store, tmp_path, run_id="gen", persona_client=fresh_client, llm_simulator=sim
    )
    assert fresh_model.calls == 0
    assert second.conversation_status == {
        "gen_grumpy-r0": "done",
        "gen_lost-r0": "done",
        "s-r0": "done",
    }


async def test_generate_without_client_note(store: RunStore, tmp_path: Path) -> None:
    scn = scenario(
        [{"id": "s", "script": ["hi"]}, {"generate": 2}],
        [{"id": "turns", "type": "max_turns", "value": 5}],
    )
    result = await run(scn, store, tmp_path)
    assert result.notes == ["generate: 2 personas not expanded: no simulator model configured"]


async def test_judge_transient_error_saves_no_checks(
    store: RunStore, cache: SqliteLLMCache, tmp_path: Path
) -> None:
    scn = scenario([{"id": "s", "script": ["hi"]}], [JUDGE_ITEM])
    judge, _ = fake_client(cache, [ServerDown("down")] * 4, spec="ollama:judge")
    result = await run(scn, store, tmp_path, run_id="jt", judge_client=judge)
    assert result.conversation_status == {"s-r0": "failed_transient"}
    assert result.conversations[0].error is not None
    assert result.conversations[0].error.startswith("judge: ollama:judge: gave up after 4")
    assert store.load_checks("jt") == []

    judge_ok, _ = fake_client(cache, [verdict()], spec="ollama:judge")
    retried = await run(scn, store, tmp_path, run_id="jt", judge_client=judge_ok)
    assert retried.conversation_status == {"s-r0": "done"}
    assert [c.status for c in store.load_checks("jt")] == ["pass"]


async def test_judge_replay_miss_fails(
    store: RunStore, cache: SqliteLLMCache, tmp_path: Path
) -> None:
    scn = scenario([{"id": "s", "script": ["hi"]}], [JUDGE_ITEM])
    replay = LLMClient(
        "ollama:judge", cache=cache, chat_model=None, temperature=0.0, replay_only=True
    )
    result = await run(scn, store, tmp_path, run_id="rp", judge_client=replay)
    assert result.conversation_status == {"s-r0": "failed"}
    assert result.conversations[0].error is not None
    assert result.conversations[0].error.startswith("replay: no recorded response")
    assert store.load_checks("rp") == []


async def test_agent_error_still_judged(
    store: RunStore, cache: SqliteLLMCache, tmp_path: Path
) -> None:
    scn = scenario([{"id": "s", "script": ["hi", "again"]}], [JUDGE_ITEM])

    def agent(history: list[Message]) -> str:
        if len(history) > 1:
            raise ValueError("broke")
        return "ok"

    judge, _ = fake_client(cache, [verdict("no")], spec="ollama:judge")
    result = await run(scn, store, tmp_path, agent, run_id="ae", judge_client=judge)
    assert result.conversation_status == {"s-r0": "failed"}
    assert result.conversations[0].ended_reason == "agent_error"
    assert [(c.source, c.status) for c in store.load_checks("ae")] == [("judge", "fail")]


async def test_repeat_variance_and_persona_rates(store: RunStore, tmp_path: Path) -> None:
    scn = scenario(
        [{"id": "a", "script": ["hi"]}],
        [{"id": "looked_up", "type": "tool_called", "tool": "lookup_order"}],
        repeats=2,
    )
    calls = {"n": 0}

    def agent(history: list[Message]) -> AgentTurn:
        calls["n"] += 1
        tools = [ToolCallRecord(name="lookup_order")] if calls["n"] == 1 else []
        return AgentTurn(text="ok", tool_calls=tools)

    result = await run(scn, store, tmp_path, agent, concurrency=1)
    row = result.summary[0]
    assert (row.passed, row.failed) == (1, 1)
    assert row.pass_rate == 0.5
    assert row.pass_rate_stdev == 0.5
    assert row.persona_rates == {"a": 0.5}

    out = io.StringIO()
    print_run_report(result, Console(file=out, width=120), "report")
    text = out.getvalue()
    assert "±" in text
    assert "0.50" in text
    assert "models:" not in text


async def test_report_prints_models_and_usage(
    store: RunStore, cache: SqliteLLMCache, tmp_path: Path
) -> None:
    sim, _ = simulator(cache, [sim_turn("in_progress", "hi"), sim_turn("achieved")])
    judge, _ = fake_client(cache, [verdict()], spec="ollama:judge")
    result = await run(LLM_SCENARIO, store, tmp_path, llm_simulator=sim, judge_client=judge)
    out = io.StringIO()
    print_run_report(result, Console(file=out, width=200), "report")
    lines = out.getvalue().splitlines()
    assert "models: simulator ollama:sim, judge ollama:judge" in lines
    assert "llm ollama:sim: 2 calls, 0 cache hits, 0 retries, 20 in / 10 out tokens" in lines
    assert "llm ollama:judge: 1 calls, 0 cache hits, 0 retries, 10 in / 5 out tokens" in lines
    assert not any("±" in line for line in lines)


def test_old_database_gets_extra_json_column(tmp_path: Path) -> None:
    path = tmp_path / "old.db"
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE check_results (run_id TEXT NOT NULL, conversation_id TEXT NOT NULL,"
        " rubric_id TEXT NOT NULL, status TEXT NOT NULL, detail TEXT NOT NULL,"
        " PRIMARY KEY(run_id, conversation_id, rubric_id))"
    )
    conn.commit()
    conn.close()
    with RunStore(path):
        pass
    conn = sqlite3.connect(path)
    columns = {row[1] for row in conn.execute("PRAGMA table_info(check_results)")}
    conn.close()
    assert "extra_json" in columns
