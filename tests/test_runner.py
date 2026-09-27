"""run_scenario: demo results, ended reasons, resume, failures, concurrency, output files."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator
from importlib.resources import as_file, files
from pathlib import Path
from typing import Any

import pytest

from convoprobe.adapters import CallableAdapter
from convoprobe.datamodels import Message, RunResult, Scenario
from convoprobe.demo.toy_agent import toy_agent_v1, toy_agent_v2
from convoprobe.errors import TransientError
from convoprobe.runner import new_run_id, run_scenario
from convoprobe.scenario import load_scenario
from convoprobe.store import RunStore


@pytest.fixture
def store(tmp_path: Path) -> Iterator[RunStore]:
    with RunStore(tmp_path / "state.db") as opened:
        yield opened


@pytest.fixture
def demo_scenario() -> Scenario:
    with as_file(files("convoprobe.demo") / "scenario.yaml") as path:
        return load_scenario(path)


def scripted(personas: dict[str, list[str]], max_turns: int = 10) -> Scenario:
    return Scenario.model_validate(
        {
            "name": "test",
            "agent_description": "x",
            "max_turns": max_turns,
            "personas": [{"id": pid, "script": script} for pid, script in personas.items()],
            "rubric": [{"id": "turns", "type": "max_turns", "value": 20}],
        }
    )


async def run(
    scenario: Scenario, fn: Any, store: RunStore, tmp_path: Path, **kwargs: Any
) -> RunResult:
    return await run_scenario(
        scenario,
        CallableAdapter(fn),
        store=store,
        out_dir=tmp_path / "out",
        adapter_label="test",
        **kwargs,
    )


def summary_counts(result: RunResult) -> dict[str, tuple[int, int, int, int]]:
    return {s.rubric_id: (s.passed, s.failed, s.errored, s.skipped) for s in result.summary}


def test_new_run_id_format() -> None:
    run_id = new_run_id()
    date, clock, suffix = run_id.split("-")
    assert (len(date), len(clock), len(suffix)) == (8, 6, 4)


async def test_demo_v1_all_pass(demo_scenario: Scenario, store: RunStore, tmp_path: Path) -> None:
    result = await run(demo_scenario, toy_agent_v1, store, tmp_path)
    assert result.checks
    assert {c.status for c in result.checks} <= {"pass", "skip"}
    assert result.overall_pass_rate == 1.0
    assert result.conversation_status == {
        "cooperative-r0": "skipped",
        "scripted_happy_path-r0": "done",
        "scripted_skip_verification-r0": "done",
    }
    assert "persona cooperative skipped: no simulator model configured" in result.notes
    assert "generate: 4 personas not expanded: no simulator model configured" in result.notes
    assert "llm_judge items skipped: no judge model configured" in result.notes


async def test_demo_v2_summary(demo_scenario: Scenario, store: RunStore, tmp_path: Path) -> None:
    result = await run(demo_scenario, toy_agent_v2, store, tmp_path)
    # (passed, failed, errored, skipped). Only the two scripted conversations
    # run checks, so no_refund_without_order skips exactly one: happy path.
    assert summary_counts(result) == {
        "verified_before_refund": (0, 2, 0, 0),
        "no_refund_without_order": (0, 1, 0, 1),
        "refund_has_order_id": (1, 1, 0, 0),
        "turn_limit": (2, 0, 0, 0),
        "no_guarantee_language": (2, 0, 0, 0),
        "honest_about_refund": (0, 0, 0, 2),
    }
    judge_rows = [c for c in result.checks if c.rubric_id == "honest_about_refund"]
    assert {(c.status, c.detail, c.source) for c in judge_rows} == {
        ("skip", "no judge model configured", "judge")
    }
    assert "llm_judge items skipped: no judge model configured" in result.notes


async def test_ended_reasons(store: RunStore, tmp_path: Path) -> None:
    scenario = scripted({"short": ["a"], "exact": ["a", "b"], "long": ["1", "2", "3"]}, max_turns=2)
    result = await run(scenario, lambda h: "ok", store, tmp_path)
    by_id = {c.persona_id: c for c in result.conversations}
    assert by_id["short"].ended_reason == "script_end"
    # A script exactly max_turns long uses every turn, so the loop ends on the limit.
    assert by_id["exact"].ended_reason == "max_turns"
    assert by_id["long"].ended_reason == "max_turns"
    assert len(by_id["long"].agent_messages()) == 2
    assert all(m.latency_ms is not None for m in by_id["long"].agent_messages())


async def test_transient_failure_resumes(store: RunStore, tmp_path: Path) -> None:
    scenario = scripted({"a": ["hello from a", "bye"], "b": ["hello from b"]})
    counts = {"a": 0, "b": 0}
    failed_once = False

    def agent(history: list[Message]) -> str:
        nonlocal failed_once
        persona = "a" if history[0].text.endswith("a") else "b"
        counts[persona] += 1
        if persona == "a" and len(history) == 1 and not failed_once:
            failed_once = True
            raise TransientError("rate limited")
        return "ok"

    first = await run(scenario, agent, store, tmp_path, run_id="resume-me")
    assert first.conversation_status == {"a-r0": "failed_transient", "b-r0": "done"}
    b_calls = counts["b"]

    second = await run(scenario, agent, store, tmp_path, run_id="resume-me")
    assert second.conversation_status == {"a-r0": "done", "b-r0": "done"}
    assert store.conversation_attempts("resume-me", "a-r0") == 2
    assert store.conversation_attempts("resume-me", "b-r0") == 1
    assert counts["b"] == b_calls
    assert {c.persona_id for c in second.conversations} == {"a", "b"}
    assert {c.conversation_id for c in second.checks} == {"a-r0", "b-r0"}


async def test_non_transient_failure(store: RunStore, tmp_path: Path) -> None:
    scenario = scripted({"a": ["one", "two", "three"]})

    def agent(history: list[Message]) -> str:
        if len(history) > 1:
            raise ValueError("agent broke")
        return "ok"

    result = await run(scenario, agent, store, tmp_path, run_id="broken")
    assert result.conversation_status == {"a-r0": "failed"}
    conversation = result.conversations[0]
    assert conversation.ended_reason == "agent_error"
    assert conversation.error == "ValueError: agent broke"
    assert len(conversation.agent_messages()) == 1
    assert [(c.rubric_id, c.status) for c in store.load_checks("broken")] == [("turns", "pass")]

    again = await run(scenario, agent, store, tmp_path, run_id="broken")
    assert store.conversation_attempts("broken", "a-r0") == 1
    assert again.conversation_status == {"a-r0": "failed"}


async def test_concurrency_limit(store: RunStore, tmp_path: Path) -> None:
    scenario = scripted({f"p{i}": ["hi", "again"] for i in range(6)})
    in_flight = 0
    peak = 0

    async def agent(history: list[Message]) -> str:
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        in_flight -= 1
        return "ok"

    result = await run(scenario, agent, store, tmp_path, concurrency=2)
    assert set(result.conversation_status.values()) == {"done"}
    assert len(result.conversation_status) == 6
    assert 1 <= peak <= 2


async def test_repeats_make_separate_conversations(store: RunStore, tmp_path: Path) -> None:
    scenario = scripted({"a": ["hi"]}).model_copy(update={"repeats": 3})
    result = await run(scenario, lambda h: "ok", store, tmp_path)
    assert sorted(result.conversation_status) == ["a-r0", "a-r1", "a-r2"]
    assert sorted(c.repeat_index for c in result.conversations) == [0, 1, 2]


async def test_output_files(store: RunStore, tmp_path: Path) -> None:
    scenario = scripted({"a": ["hi"]})
    result = await run(scenario, lambda h: "ok", store, tmp_path, run_id="files")
    run_dir = tmp_path / "out" / "files"
    parsed = RunResult.model_validate_json((run_dir / "results.json").read_text("utf-8"))
    assert parsed == result
    assert parsed.finished_at is not None
    events = [json.loads(line) for line in (run_dir / "log.jsonl").read_text("utf-8").splitlines()]
    names = [e["event"] for e in events]
    assert names[0] == "run_started"
    assert names[-1] == "run_finished"
    assert {"conversation_started", "turn", "conversation_finished"} <= set(names)
    assert all(e["run_id"] == "files" for e in events)
    turn = next(e for e in events if e["event"] == "turn")
    assert turn["conversation_id"] == "a-r0"
    assert turn["tool_calls"] == []


async def test_log_records_skip_and_failure(
    demo_scenario: Scenario, store: RunStore, tmp_path: Path
) -> None:
    def agent(history: list[Message]) -> str:
        raise TransientError("try later")

    await run(demo_scenario, agent, store, tmp_path, run_id="logged")
    lines = (tmp_path / "out" / "logged" / "log.jsonl").read_text("utf-8").splitlines()
    events = [json.loads(line) for line in lines]
    skipped = [e for e in events if e["event"] == "conversation_skipped"]
    failed = [e for e in events if e["event"] == "conversation_failed"]
    assert skipped == [
        {
            **skipped[0],
            "conversation_id": "cooperative-r0",
            "reason": "no simulator model configured",
        }
    ]
    assert len(failed) == 2
    assert all(e["transient"] is True and e["error"] == "try later" for e in failed)
