"""The key M3 test: the same agent behind three adapters gives identical results.

The toy agent runs as a plain function (CallableAdapter), as a LangGraph graph
with a human approval interrupt (LangGraphAdapter), and behind a local HTTP
server (HTTPAdapter). If the adapters are faithful, every check result, reply
text, and tool call matches. Then the LangGraph agent replays the recorded demo
cassette and must reproduce the recorded summary with zero live calls.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from importlib.resources import as_file, files
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("langgraph")

from examples.http_agent.server import make_server
from examples.toy_support_agent import graph as toy_graph

from convoprobe import cli
from convoprobe.adapters import CallableAdapter, HTTPAdapter, HTTPAdapterConfig
from convoprobe.config import load_settings
from convoprobe.datamodels import RunResult, Scenario
from convoprobe.demo import CASSETTE_NAME
from convoprobe.demo.toy_agent import toy_agent_v1, toy_agent_v2
from convoprobe.models.cache import JsonCassette
from convoprobe.runner import run_scenario
from convoprobe.scenario import load_scenario
from convoprobe.simulator.llm import LLMSimulator
from convoprobe.store import RunStore


@pytest.fixture
def scenario() -> Scenario:
    with as_file(files("convoprobe.demo") / "scenario.yaml") as path:
        return load_scenario(path)


@pytest.fixture
def http_urls() -> Iterator[dict[str, str]]:
    servers = {version: make_server("127.0.0.1", 0, version) for version in ("v1", "v2")}
    threads = [threading.Thread(target=s.serve_forever, daemon=True) for s in servers.values()]
    for thread in threads:
        thread.start()
    yield {
        version: f"http://127.0.0.1:{server.server_address[1]}/chat"
        for version, server in servers.items()
    }
    for server in servers.values():
        server.shutdown()
        server.server_close()
    for thread in threads:
        thread.join(timeout=5)


def http_adapter(url: str) -> HTTPAdapter:
    return HTTPAdapter(
        HTTPAdapterConfig(
            url=url,
            request_template={"session": "{{conversation_id}}", "message": "{{message}}"},
            text_path="reply",
            tool_calls_path="tool_calls",
        )
    )


async def run(
    scenario: Scenario, adapter: Any, tmp_path: Path, name: str, **clients: Any
) -> RunResult:
    with RunStore(tmp_path / f"{name}.db") as store:
        return await run_scenario(
            scenario,
            adapter,
            store=store,
            out_dir=tmp_path / "out",
            adapter_label=name,
            run_id=name,
            **clients,
        )


def fingerprint(result: RunResult) -> dict[str, Any]:
    """Everything that must match across adapters. Timings and metadata excluded."""
    return {
        "status": result.conversation_status,
        "checks": [(c.rubric_id, c.persona_id, c.status, c.detail) for c in result.checks],
        "turns": {
            conv.id: [(m.role, m.text, [call.name for call in m.tool_calls]) for m in conv.messages]
            for conv in result.conversations
        },
    }


@pytest.mark.parametrize("version", ["v1", "v2"])
async def test_three_adapters_identical(
    version: str, scenario: Scenario, http_urls: dict[str, str], tmp_path: Path
) -> None:
    function = toy_agent_v1 if version == "v1" else toy_agent_v2
    graph_adapter = toy_graph.adapter_v1 if version == "v1" else toy_graph.adapter_v2
    plain = await run(scenario, CallableAdapter(function), tmp_path, "plain")
    graph = await run(scenario, graph_adapter, tmp_path, "graph")
    http = await run(scenario, http_adapter(http_urls[version]), tmp_path, "http")

    assert fingerprint(graph) == fingerprint(plain)
    assert fingerprint(http) == fingerprint(plain)
    assert plain.checks, "the comparison must not be vacuous"
    refund_turns = [
        m.metadata
        for conv in graph.conversations
        for m in conv.messages
        if any(c.name == "issue_refund" for c in m.tool_calls)
    ]
    assert refund_turns and all(meta["resumes"] == 1 for meta in refund_turns)
    assert all("approve" in meta["node_path"] for meta in refund_turns)


async def test_langgraph_reproduces_recorded_demo(scenario: Scenario, tmp_path: Path) -> None:
    with as_file(files("convoprobe.demo") / CASSETTE_NAME) as path:
        cassette = JsonCassette(path)
    settings = load_settings()

    def clients() -> dict[str, Any]:
        sim, judge = cli._demo_clients(settings, cassette, replay_only=True, run_id=None)
        return {"llm_simulator": LLMSimulator(sim), "judge_client": judge, "persona_client": sim}

    plain = await run(scenario, CallableAdapter(toy_agent_v2), tmp_path, "plain", **clients())
    graph_clients = clients()
    graph = await run(scenario, toy_graph.adapter_v2, tmp_path, "graph", **graph_clients)

    assert set(graph.conversation_status.values()) == {"done"}
    assert graph.summary == plain.summary
    assert fingerprint(graph) == fingerprint(plain)
    usage = {u.spec: u for u in graph.llm_usage}
    assert all(u.calls == 0 for u in usage.values()), "replay must make no live calls"
    assert usage[cli.DEMO_JUDGE].cache_hits == 7
