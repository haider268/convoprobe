"""CLI model flags, replay, cassette export, persona generation, demo. All offline."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator
from importlib.resources import as_file, files
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml
from typer.testing import CliRunner

from convoprobe import cli
from convoprobe.adapters import CallableAdapter
from convoprobe.cli import DEMO_HEADER, app
from convoprobe.datamodels import LLMPersona, RunResult
from convoprobe.demo.toy_agent import toy_agent_v1
from convoprobe.errors import ConvoprobeError
from convoprobe.models.cache import SqliteLLMCache
from convoprobe.runner import run_scenario
from convoprobe.scenario import load_scenario
from convoprobe.simulator.llm import LLMSimulator
from convoprobe.simulator.personas import generate_personas
from convoprobe.store import RunStore
from tests.llm_doubles import fake_client

runner = CliRunner()
AGENT = "convoprobe.demo.toy_agent:toy_agent_v1"

SCENARIO_YAML = """\
name: replay-test
agent_description: Support agent for a fictional shop at shop.example.
max_turns: 4
personas:
  - id: caller
    description: Pat Example, wants a refund for order 1042.
    goal: Get a refund for order 1042.
  - id: scripted_refund
    script:
      - "My order 1042 arrived damaged, I want a refund."
rubric:
  - id: verified_before_refund
    type: tool_order
    before: lookup_order
    after: issue_refund
  - id: honest_about_refund
    type: llm_judge
    question: Did the agent claim a refund without checking the order?
    pass_if: "no"
"""

VERDICT = {"reasoning": "It looked up the order.", "verdict": "no", "evidence": []}


def demo_scenario_path() -> str:
    with as_file(files("convoprobe.demo") / "scenario.yaml") as path:
        return str(path)


@pytest.fixture
def scenario_file(tmp_path: Path) -> Path:
    path = tmp_path / "scenario.yaml"
    path.write_text(SCENARIO_YAML, encoding="utf-8")
    return path


@pytest.fixture
def cache(tmp_path: Path) -> Iterator[SqliteLLMCache]:
    with SqliteLLMCache(tmp_path / "record.db") as opened:
        yield opened


def run_args(tmp_path: Path, scenario: str, *extra: str) -> list[str]:
    return [
        "run",
        scenario,
        "--adapter",
        AGENT,
        "--db",
        str(tmp_path / "state.db"),
        "--out",
        str(tmp_path / "out"),
        "--run-id",
        "cli",
        *extra,
    ]


def results(tmp_path: Path) -> RunResult:
    path = tmp_path / "out" / "cli" / "results.json"
    return RunResult.model_validate_json(path.read_text(encoding="utf-8"))


async def record_cassette(scenario_file: Path, cache: SqliteLLMCache, out: Path) -> int:
    """Run the scenario on fake models with the CLI's temperatures and methods, then export."""
    sim, _ = fake_client(
        cache,
        [
            {"goal_status": "in_progress", "message": "Refund order 1042 please."},
            {"goal_status": "achieved", "message": ""},
        ],
        spec="ollama:fake-sim",
        run_id="rec",
        structured_method="json_schema",
    )
    sim.temperature = cli.SIMULATOR_TEMPERATURE
    judge, _ = fake_client(
        cache,
        [VERDICT, VERDICT],
        spec="ollama:fake-judge",
        run_id="rec",
        structured_method="json_schema",
    )
    with RunStore(out.parent / "record-state.db") as store:
        result = await run_scenario(
            load_scenario(scenario_file),
            CallableAdapter(toy_agent_v1),
            store=store,
            out_dir=out.parent / "record-out",
            adapter_label="record",
            run_id="rec",
            concurrency=1,
            llm_simulator=LLMSimulator(sim),
            judge_client=judge,
            persona_client=sim,
        )
    assert set(result.conversation_status.values()) == {"done"}
    return cache.export_cassette(["rec"], out, meta={"recorded_at": "2026-09-28"})


# run


def test_run_no_llm_matches_m1(tmp_path: Path) -> None:
    result = runner.invoke(app, run_args(tmp_path, demo_scenario_path(), "--no-llm"))
    assert result.exit_code == 0, result.output
    lines = result.output.splitlines()
    assert lines[0] == "mode: no-llm"
    assert not any(line.startswith("models:") for line in lines)
    assert "note: llm_judge items skipped: no judge model configured" in lines
    assert "note: persona cooperative skipped: no simulator model configured" in lines
    judge_rows = [c for c in results(tmp_path).checks if c.rubric_id == "honest_about_refund"]
    assert {c.status for c in judge_rows} == {"skip"}


@pytest.mark.parametrize(
    "extra",
    [["--hosted"], ["--judge", "ollama:x"], ["--simulator", "ollama:x"], ["--replay", "c.json"]],
)
def test_no_llm_with_model_options(tmp_path: Path, extra: list[str]) -> None:
    result = runner.invoke(app, run_args(tmp_path, demo_scenario_path(), "--no-llm", *extra))
    assert result.exit_code == 2
    assert "error: --no-llm cannot be combined with model options" in result.output


def test_live_mode_ollama_unreachable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("langchain_ollama")

    async def down(base_url: str, models: list[str], transport: Any = None) -> None:
        assert models == ["qwen2.5:3b", "llama3.2:3b"]
        raise ConvoprobeError(f"Ollama is not reachable at {base_url}. Start Ollama.")

    monkeypatch.setattr(cli, "check_ollama", down)
    result = runner.invoke(app, run_args(tmp_path, demo_scenario_path()))
    assert result.exit_code == 2
    assert "error: Ollama is not reachable at http://localhost:11434" in result.output


def test_live_mode_blocked_network_is_a_clean_error(tmp_path: Path) -> None:
    pytest.importorskip("langchain_ollama")
    result = runner.invoke(app, run_args(tmp_path, demo_scenario_path()))
    assert result.exit_code == 2
    assert "error: Ollama is not reachable" in result.output


SMALL_JUDGE_NOTE = (
    "note: judge ollama:llama3.2:3b is a small local model. In review testing a 3B local"
    " judge gave wrong verdicts. Use --hosted or --replay for results you rely on."
)


def test_live_ollama_judge_warning(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("langchain_ollama")

    async def ok(base_url: str, models: list[str], transport: Any = None) -> None:
        return None

    monkeypatch.setattr(cli, "check_ollama", ok)
    # Scripted persona and no judge item: the run makes no model call at all.
    scripted = tmp_path / "scripted.yaml"
    scripted.write_text(
        "name: s\nagent_description: x\npersonas:\n  - id: s\n    script: ['hi']\n"
        "rubric:\n  - id: t\n    type: max_turns\n    value: 3\n",
        encoding="utf-8",
    )
    result = runner.invoke(app, run_args(tmp_path, str(scripted)))
    assert result.exit_code == 0, result.output
    lines = result.output.splitlines()
    assert lines[:3] == [
        "mode: live",
        "models: simulator ollama:qwen2.5:3b, judge ollama:llama3.2:3b",
        SMALL_JUDGE_NOTE,
    ]
    assert lines.count(SMALL_JUDGE_NOTE) == 1


def test_no_judge_warning_for_hosted_judge(capsys: pytest.CaptureFixture[str]) -> None:
    cli._warn_small_judge("groq:openai/gpt-oss-120b")
    assert capsys.readouterr().out == ""
    cli._warn_small_judge("ollama:llama3.2:3b")
    assert capsys.readouterr().out.strip() == SMALL_JUDGE_NOTE


def test_hosted_without_key(tmp_path: Path) -> None:
    result = runner.invoke(app, run_args(tmp_path, demo_scenario_path(), "--hosted"))
    assert result.exit_code == 2
    assert "error: GROQ_API_KEY is not set. Put it in .env (never in code or chat)." in (
        result.output
    )


def test_bad_spec_and_missing_cassette(tmp_path: Path) -> None:
    bad = runner.invoke(app, run_args(tmp_path, demo_scenario_path(), "--simulator", "gpt4"))
    assert bad.exit_code == 2
    assert "model spec must look like provider:model" in bad.output
    missing = runner.invoke(
        app, run_args(tmp_path, demo_scenario_path(), "--replay", str(tmp_path / "none.json"))
    )
    assert missing.exit_code == 2
    assert "error: cassette not found" in missing.output


def test_repeats_override(tmp_path: Path) -> None:
    result = runner.invoke(
        app, run_args(tmp_path, demo_scenario_path(), "--no-llm", "--repeats", "2")
    )
    assert result.exit_code == 0, result.output
    assert "scripted_happy_path-r1" in results(tmp_path).conversation_status
    bad = runner.invoke(app, run_args(tmp_path, demo_scenario_path(), "--no-llm", "--repeats", "0"))
    assert bad.exit_code == 2
    assert "repeats" in bad.output


def test_replay_round_trip(tmp_path: Path, scenario_file: Path, cache: SqliteLLMCache) -> None:
    cassette = tmp_path / "cassette.json"
    assert asyncio.run(record_cassette(scenario_file, cache, cassette)) == 4
    extra = ["--replay", str(cassette), "--simulator", "ollama:fake-sim"]
    extra += ["--judge", "ollama:fake-judge"]
    result = runner.invoke(app, run_args(tmp_path, str(scenario_file), *extra))
    assert result.exit_code == 0, result.output
    lines = result.output.splitlines()
    assert lines[0] == f"mode: replay ({cassette})"
    assert lines[1] == "models: simulator ollama:fake-sim, judge ollama:fake-judge"
    # Replay mode: the verdicts are recorded, so no small-judge warning.
    assert not any("is a small local model" in line for line in lines)
    replayed = results(tmp_path)
    assert set(replayed.conversation_status.values()) == {"done"}
    judged = [c for c in replayed.checks if c.rubric_id == "honest_about_refund"]
    assert [(c.source, c.status, c.verdict) for c in judged] == [
        ("judge", "pass", "no"),
        ("judge", "pass", "no"),
    ]
    assert replayed.llm_usage[0].cache_hits == 2
    assert replayed.llm_usage[0].calls == 0


def test_replay_missing_entry_fails_conversation(
    tmp_path: Path, scenario_file: Path, cache: SqliteLLMCache
) -> None:
    cassette = tmp_path / "cassette.json"
    asyncio.run(record_cassette(scenario_file, cache, cassette))
    data = json.loads(cassette.read_text(encoding="utf-8"))
    data["entries"] = {k: v for k, v in data["entries"].items() if v["spec"] != "ollama:fake-judge"}
    cassette.write_text(json.dumps(data), encoding="utf-8")
    extra = ["--replay", str(cassette), "--simulator", "ollama:fake-sim"]
    extra += ["--judge", "ollama:fake-judge"]
    result = runner.invoke(app, run_args(tmp_path, str(scenario_file), *extra))
    assert result.exit_code == 0, result.output
    replayed = results(tmp_path)
    assert set(replayed.conversation_status.values()) == {"failed"}
    assert all(c.error and c.error.startswith("replay: ") for c in replayed.conversations)


# cassette export


def test_cassette_export(tmp_path: Path) -> None:
    db = tmp_path / "state.db"
    with SqliteLLMCache(db) as store_cache:
        store_cache.put("k1", "ollama:m", "text", {"text": "a"})
        store_cache.put("k2", "ollama:m", "text", {"text": "b"})
        store_cache.mark_used("r1", "k1")
        store_cache.mark_used("r2", "k2")
    out = tmp_path / "c.json"
    args = ["cassette", "export", "--run-id", "r1", "--run-id", "r2", "--db", str(db)]
    result = runner.invoke(app, [*args, "--out", str(out)])
    assert result.exit_code == 0, result.output
    assert result.output.strip() == f"wrote 2 entries to {out}"
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["meta"]["run_ids"] == "r1,r2"
    assert len(data["meta"]["recorded_at"]) == 10


def test_cassette_export_nothing(tmp_path: Path) -> None:
    out = tmp_path / "c.json"
    args = ["cassette", "export", "--run-id", "nope", "--db", str(tmp_path / "s.db")]
    result = runner.invoke(app, [*args, "--out", str(out)])
    assert result.exit_code == 2
    assert "error: no cached LLM calls found for those run ids" in result.output
    assert not out.exists()


# personas generate


def test_personas_generate_replay(
    tmp_path: Path, scenario_file: Path, cache: SqliteLLMCache
) -> None:
    scenario = load_scenario(scenario_file)
    client, _ = fake_client(
        cache,
        [
            {
                "personas": [
                    {"name": "Hurried Hana", "description": "Types fast.", "goal": "Refund."},
                    {"name": "Lost Leo", "description": "Confused.", "goal": "Order status."},
                ]
            }
        ],
        spec="ollama:fake-sim",
        run_id="p",
        structured_method="json_schema",
    )
    client.temperature = cli.SIMULATOR_TEMPERATURE
    salt = f"personas-cli:{RunStore.scenario_hash(scenario)}:3"
    asyncio.run(generate_personas(scenario, 3, client, salt=salt))
    cassette = tmp_path / "personas.json"
    cache.export_cassette(["p"], cassette)

    args = ["personas", "generate", str(scenario_file), "-n", "3"]
    args += ["--replay", str(cassette), "--simulator", "ollama:fake-sim"]
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    rows = yaml.safe_load(result.stdout)
    personas = [LLMPersona.model_validate(row) for row in rows]
    assert [p.id for p in personas] == ["gen_hurried_hana", "gen_lost_leo"]
    assert "note: asked for 3 personas, model returned 2" in result.output


def test_personas_generate_errors(tmp_path: Path, scenario_file: Path) -> None:
    args = ["personas", "generate", str(scenario_file), "-n", "2", "--hosted"]
    args += ["--db", str(tmp_path / "p.db")]
    result = runner.invoke(app, args)
    assert result.exit_code == 2
    assert "GROQ_API_KEY is not set" in result.output


# demo


RECORDED_FINAL_LINE = (
    "v2 regressed on 4 rubric item(s): verified_before_refund, no_refund_without_order,"
    " refund_has_order_id, honest_about_refund"
)


def test_demo_replays_committed_cassette() -> None:
    # The conftest guard blocks DNS, remote connects, and the Ollama port here.
    assert cli._demo_cassette() is not None
    result = runner.invoke(app, ["demo"])
    assert result.exit_code == 0, result.output
    lines = result.output.rstrip().splitlines()
    assert lines[0] == (
        "DEMO: toy shop agent. Scripted personas run live; LLM personas and judge verdicts"
        " replay a cassette recorded on 2026-09-27 (simulator groq:openai/gpt-oss-20b,"
        " judge groq:openai/gpt-oss-120b). No API keys, no network."
    )
    assert lines[-1] == RECORDED_FINAL_LINE
    assert lines.count("conversations: 7 done, 0 failed, 0 transient, 0 skipped") == 2
    assert (
        "llm groq:openai/gpt-oss-20b: 0 calls, 43 cache hits, 0 retries, 0 in / 0 out tokens"
        in (lines)
    )


def test_demo_without_cassette(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "_demo_cassette", lambda: None)
    result = runner.invoke(app, ["demo"])
    assert result.exit_code == 0, result.output
    assert result.output.splitlines()[0] == DEMO_HEADER
    assert "llm " not in result.output


def _fake_recording(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fail_v2: bool) -> Path:
    """Replace live models with fakes; return where the cassette would be written."""
    monkeypatch.chdir(tmp_path)
    package_dir = tmp_path / "pkg"
    package_dir.mkdir()
    monkeypatch.setattr(cli, "demo_package", SimpleNamespace(__file__=str(package_dir / "x.py")))

    def clients(settings: Any, cache: Any, *, replay_only: bool, run_id: str | None) -> Any:
        sim, _ = fake_client(cache, [], spec=cli.DEMO_SIMULATOR, run_id=run_id)
        judge, _ = fake_client(cache, [], spec=cli.DEMO_JUDGE, run_id=run_id)
        return sim, judge

    async def fake_run(
        scenario: Any, version: str, store: Any, out_dir: Path, pair: Any, run_id: Any = None
    ) -> RunResult:
        sim, _judge = pair
        sim.cache.put(f"key-{version}", sim.spec, "structured", {"data": {}})
        sim.cache.mark_used(run_id, f"key-{version}")
        result = await run_scenario(
            scenario,
            CallableAdapter(toy_agent_v1),
            store=store,
            out_dir=out_dir,
            adapter_label=f"demo:{version}",
            run_id=run_id,
        )
        status = "failed" if fail_v2 and version == "v2" else "done"
        return result.model_copy(
            update={"conversation_status": dict.fromkeys(result.conversation_status, status)}
        )

    monkeypatch.setattr(cli, "_demo_clients", clients)
    monkeypatch.setattr(cli, "_demo_run", fake_run)
    return package_dir / cli.CASSETTE_NAME


def test_demo_record_writes_cassette(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cassette = _fake_recording(monkeypatch, tmp_path, fail_v2=False)
    result = runner.invoke(app, ["demo", "--record"])
    assert result.exit_code == 0, result.output
    assert f"wrote 2 entries to {cassette}" in result.output
    meta = json.loads(cassette.read_text(encoding="utf-8"))["meta"]
    assert meta["simulator"] == cli.DEMO_SIMULATOR
    assert meta["judge"] == cli.DEMO_JUDGE
    assert (tmp_path / ".convoprobe" / "demo-record.db").is_file()


def test_demo_record_failure_keeps_old_cassette(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cassette = _fake_recording(monkeypatch, tmp_path, fail_v2=True)
    result = runner.invoke(app, ["demo", "--record"])
    assert result.exit_code == 1
    assert "cooperative-r0: no simulator model configured" in result.output
    assert "cassette not written: not every conversation finished" in result.output
    assert not cassette.exists()


def test_invalid_cassette_json(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    result = runner.invoke(app, run_args(tmp_path, demo_scenario_path(), "--replay", str(bad)))
    assert result.exit_code == 2
    assert f"error: cassette {bad} is not valid JSON" in result.output


def test_null_cache_is_inert() -> None:
    null = cli._NullCache()
    assert null.get("k") is None
    null.put("k", "s", "text", {})
    null.mark_used("r", "k")


def test_demo_record_without_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    marker = tmp_path / ".convoprobe" / "demo-record.db"
    marker.parent.mkdir()
    marker.write_text("keep", encoding="utf-8")
    result = runner.invoke(app, ["demo", "--record"])
    assert result.exit_code == 2
    assert "GROQ_API_KEY is not set" in result.output
    assert marker.read_text(encoding="utf-8") == "keep"
