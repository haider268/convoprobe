"""RunStore: runs, conversations, check results, resume."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from convoprobe.datamodels import CheckResult, Conversation, Message, Scenario
from convoprobe.errors import ConvoprobeError
from convoprobe.store import RunStore


def scenario(name: str = "t") -> Scenario:
    return Scenario.model_validate(
        {
            "name": name,
            "agent_description": "x",
            "personas": [{"id": "s", "script": ["hi"]}],
            "rubric": [{"id": "r", "type": "max_turns", "value": 5}],
        }
    )


def check(cid: str, rubric_id: str, status: str = "pass") -> CheckResult:
    return CheckResult.model_validate(
        {
            "rubric_id": rubric_id,
            "persona_id": "s",
            "conversation_id": cid,
            "status": status,
            "detail": f"{rubric_id} {status}",
        }
    )


@pytest.fixture
def store(tmp_path: Path) -> Iterator[RunStore]:
    with RunStore(tmp_path / "nested" / "state.db") as opened:
        yield opened


def test_creates_parent_dirs_and_context_manager(tmp_path: Path) -> None:
    path = tmp_path / "a" / "b" / "state.db"
    with RunStore(path) as store:
        assert store.create_or_resume_run("r1", scenario(), "label") is False
    assert path.is_file()


def test_create_then_resume(store: RunStore) -> None:
    assert store.create_or_resume_run("r1", scenario(), "label") is False
    assert store.create_or_resume_run("r1", scenario(), "label") is True
    started, finished = store.run_times("r1")
    assert started.endswith("+00:00")
    assert finished is None
    store.finish_run("r1")
    assert store.run_times("r1")[1] is not None


def test_hash_mismatch(store: RunStore) -> None:
    store.create_or_resume_run("r1", scenario("one"), "label")
    with pytest.raises(ConvoprobeError, match="run r1 was created from a different scenario"):
        store.create_or_resume_run("r1", scenario("two"), "label")


def test_scenario_hash_stable() -> None:
    assert RunStore.scenario_hash(scenario()) == RunStore.scenario_hash(scenario())
    assert RunStore.scenario_hash(scenario("a")) != RunStore.scenario_hash(scenario("b"))


def test_unknown_run_and_conversation(store: RunStore) -> None:
    with pytest.raises(ConvoprobeError, match="run nope not found"):
        store.run_times("nope")
    store.create_or_resume_run("r1", scenario(), "label")
    with pytest.raises(ConvoprobeError, match="conversation x not found"):
        store.conversation_status("r1", "x")
    with pytest.raises(ConvoprobeError, match="conversation x not found"):
        store.conversation_attempts("r1", "x")


def test_ensure_conversation_idempotent(store: RunStore) -> None:
    store.create_or_resume_run("r1", scenario(), "label")
    store.ensure_conversation("r1", "s-r0", "s", 0)
    conv = Conversation(id="s-r0", persona_id="s")
    store.save_conversation("r1", conv, "done")
    store.ensure_conversation("r1", "s-r0", "s", 0)
    assert store.conversation_status("r1", "s-r0") == "done"
    assert store.conversation_statuses("r1") == {"s-r0": "done"}


def test_pending_loads_as_none(store: RunStore) -> None:
    store.create_or_resume_run("r1", scenario(), "label")
    store.ensure_conversation("r1", "s-r0", "s", 0)
    assert store.load_conversations("r1") == [(None, "pending", None)]
    assert store.conversation_attempts("r1", "s-r0") == 0


def test_conversation_round_trip_and_attempts(store: RunStore) -> None:
    store.create_or_resume_run("r1", scenario(), "label")
    store.ensure_conversation("r1", "s-r0", "s", 0)
    partial = Conversation(id="s-r0", persona_id="s", messages=[Message(role="user", text="hi")])
    store.save_conversation("r1", partial, "failed_transient", "timeout")
    assert store.load_conversations("r1") == [(partial, "failed_transient", "timeout")]
    full = partial.model_copy(update={"ended_reason": "script_end"})
    store.save_conversation("r1", full, "done")
    assert store.load_conversations("r1") == [(full, "done", None)]
    assert store.conversation_attempts("r1", "s-r0") == 2


def test_conversations_ordered_by_id(store: RunStore) -> None:
    store.create_or_resume_run("r1", scenario(), "label")
    for cid in ("b-r0", "a-r0"):
        store.save_conversation("r1", Conversation(id=cid, persona_id=cid[0]), "done")
    assert [c.id for c, _s, _e in store.load_conversations("r1") if c] == ["a-r0", "b-r0"]


def test_checks_round_trip_and_replace(store: RunStore) -> None:
    store.create_or_resume_run("r1", scenario(), "label")
    store.save_conversation("r1", Conversation(id="s-r0", persona_id="s"), "done")
    store.save_checks("r1", "s-r0", [check("s-r0", "z"), check("s-r0", "a", "fail")])
    assert store.load_checks("r1") == [check("s-r0", "a", "fail"), check("s-r0", "z")]
    store.save_checks("r1", "s-r0", [check("s-r0", "a")])
    assert store.load_checks("r1") == [check("s-r0", "a")]


def test_foreign_key_enforced(store: RunStore) -> None:
    with pytest.raises(sqlite3.IntegrityError):
        store.ensure_conversation("no_such_run", "s-r0", "s", 0)
