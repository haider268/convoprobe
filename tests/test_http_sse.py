"""HTTPAdapter SSE mode, history mapping, header templates, and the SSE example server.

Parser tests use httpx.MockTransport. The equivalence and rate-limit tests run the
example server on a loopback port that the test starts itself.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from importlib.resources import as_file, files
from pathlib import Path
from typing import Any

import httpx
import pytest
import yaml
from examples.sse_agent.server import make_server, sentences

from convoprobe.adapters import CallableAdapter, load_adapter
from convoprobe.adapters.http_adapter import HTTPAdapter, HTTPAdapterConfig, parse_sse
from convoprobe.datamodels import RunResult, Scenario, ToolCallRecord
from convoprobe.demo.toy_agent import toy_agent_v1, toy_agent_v2
from convoprobe.errors import AdapterLoadError, ConvoprobeError, TransientError
from convoprobe.runner import run_scenario
from convoprobe.scenario import load_scenario
from convoprobe.store import RunStore

URL = "http://agent.example/chat"
EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "sse_agent" / "adapter.yaml"


def sse_config(**overrides: Any) -> HTTPAdapterConfig:
    data: dict[str, Any] = {
        "url": URL,
        "request_template": {"message": "{{message}}"},
        "response_mode": "sse",
        "sse_tool_event": "tool",
        "sse_tool_result_fields": ["ok", "path"],
        "sse_transient_events": ["limited"],
        "sse_note_events": ["degraded"],
    }
    data.update(overrides)
    return HTTPAdapterConfig(**data)


def stream(*lines: str) -> str:
    return "\n".join(lines) + "\n"


def event(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload)}\n"


async def no_sleep(_seconds: float) -> None:
    return None


class Recorder:
    """MockTransport handler: replays bodies as text/event-stream, keeps requests."""

    def __init__(self, *bodies: str, status: int = 200) -> None:
        self.bodies = list(bodies)
        self.status = status
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return httpx.Response(
            self.status,
            text=self.bodies.pop(0),
            headers={"content-type": "text/event-stream"},
        )

    def sent(self) -> list[Any]:
        return [json.loads(r.content) for r in self.requests]


async def started(rec: Recorder, cfg: HTTPAdapterConfig, cid: str = "c1") -> HTTPAdapter:
    adapter = HTTPAdapter(cfg, transport=httpx.MockTransport(rec), sleep=no_sleep)
    await adapter.start(cid)
    return adapter


# Parser


def test_text_from_last_done_event_and_tools() -> None:
    body = stream(
        ": a comment line",
        "event: message",
        event({"type": "tool", "name": "lookup_order", "args": {"id": "1"}, "ok": True}),
        "",
        event({"type": "chunk", "text": "partial "}),
        event({"type": "done", "text": "first"}),
        event({"type": "tool", "name": "save", "path": "/x", "other": 1}),
        "",
        "data:" + json.dumps({"type": "done", "text": "final"}),
    )
    turn = parse_sse(body, sse_config())
    assert turn.text == "final"
    assert turn.tool_calls == [
        ToolCallRecord(name="lookup_order", args={"id": "1"}, result={"ok": True}),
        ToolCallRecord(name="save", args={}, result={"path": "/x"}),
    ]
    assert turn.metadata == {}


def test_tool_result_none_without_result_fields_and_no_tool_capture() -> None:
    body = stream(
        event({"type": "tool", "name": "t", "ok": 1}), event({"type": "done", "text": "x"})
    )
    assert parse_sse(body, sse_config(sse_tool_result_fields=[])).tool_calls == [
        ToolCallRecord(name="t", args={}, result=None)
    ]
    assert parse_sse(body, sse_config(sse_tool_event=None)).tool_calls == []


def test_custom_field_names() -> None:
    cfg = sse_config(
        sse_type_field="kind",
        sse_text_event="final",
        sse_text_field="content",
        sse_tool_name_field="tool",
        sse_tool_args_field="input",
        sse_tool_event="call",
    )
    body = stream(
        event({"kind": "call", "tool": "t", "input": {"a": 1}}),
        event({"kind": "final", "content": "ok"}),
    )
    turn = parse_sse(body, cfg)
    assert turn.text == "ok" and turn.tool_calls[0].name == "t"
    assert turn.tool_calls[0].args == {"a": 1}


@pytest.mark.parametrize(
    ("body", "message"),
    [
        (stream(event({"type": "chunk", "text": "x"})), "agent stream ended without a done event"),
        (stream("", ": only comments"), "agent stream ended without a done event"),
        (
            stream(event({"type": "done", "text": "x"}), "data: {not json"),
            "line 3 is not valid JSON",
        ),
        (stream("data: [1, 2]"), "line 1 is not a JSON object"),
        (stream(event({"type": "done", "text": 5})), "event field 'text' must be a string"),
        (stream(event({"type": "tool", "args": {}})), "tool event needs a string 'name'"),
    ],
)
def test_parser_errors(body: str, message: str) -> None:
    with pytest.raises(ConvoprobeError, match=message):
        parse_sse(body, sse_config())


def test_transient_event_raises_after_stream() -> None:
    body = stream(event({"type": "limited"}), event({"type": "done", "text": "later"}))
    with pytest.raises(TransientError, match=r"^agent reported limited$"):
        parse_sse(body, sse_config())
    # A limited stream with no done event is still transient, not a stream error.
    with pytest.raises(TransientError):
        parse_sse(stream(event({"type": "limited"})), sse_config())


def test_note_events_in_metadata() -> None:
    body = stream(
        event({"type": "degraded"}),
        event({"type": "chunk", "text": "a"}),
        event({"type": "degraded"}),
        event({"type": "done", "text": "a"}),
    )
    assert parse_sse(body, sse_config()).metadata == {"events": ["degraded", "degraded"]}


async def test_send_in_sse_mode_and_transient_not_retried_in_turn() -> None:
    rec = Recorder(
        stream(event({"type": "done", "text": "hello"})),
        stream(event({"type": "limited"}), event({"type": "done", "text": "x"})),
    )
    adapter = await started(rec, sse_config())
    assert (await adapter.send("c1", "hi")).text == "hello"
    with pytest.raises(TransientError):
        await adapter.send("c1", "again")
    assert len(rec.requests) == 2  # the limited turn was sent once, not retried


async def test_http_status_handling_unchanged_in_sse_mode() -> None:
    rec = Recorder("busy", stream(event({"type": "done", "text": "ok"})))
    statuses = iter([503, 200])

    def handler(request: httpx.Request) -> httpx.Response:
        response = rec(request)
        return httpx.Response(next(statuses), text=response.text)

    adapter = HTTPAdapter(sse_config(), transport=httpx.MockTransport(handler), sleep=no_sleep)
    await adapter.start("c1")
    assert (await adapter.send("c1", "hi")).text == "ok"
    bad = HTTPAdapter(
        sse_config(), transport=httpx.MockTransport(Recorder("nope", status=400)), sleep=no_sleep
    )
    await bad.start("c1")
    with pytest.raises(ConvoprobeError, match="agent endpoint returned 400"):
        await bad.send("c1", "hi")


# History mapping


def done(text: str) -> str:
    return stream(event({"type": "done", "text": text}))


async def test_history_roles_and_fields_renamed() -> None:
    cfg = sse_config(
        request_template={"history": "{{history}}", "text": "{{message}}"},
        history_role_field="speaker",
        history_text_field="content",
        history_roles={"user": "human", "agent": "model"},
    )
    rec = Recorder(done("a1"), done("a2"))
    adapter = await started(rec, cfg)
    await adapter.send("c1", "u1")
    await adapter.send("c1", "u2")
    assert rec.sent()[1]["history"] == [
        {"speaker": "human", "content": "u1"},
        {"speaker": "model", "content": "a1"},
        {"speaker": "human", "content": "u2"},
    ]


async def test_history_includes_current_false() -> None:
    cfg = sse_config(
        request_template={"history": "{{history}}", "text": "{{message}}"},
        history_includes_current=False,
    )
    rec = Recorder(done("a1"), done("a2"))
    adapter = await started(rec, cfg)
    await adapter.send("c1", "u1")
    await adapter.send("c1", "u2")
    first, second = rec.sent()
    assert first == {"history": [], "text": "u1"}
    assert second == {
        "history": [{"role": "user", "text": "u1"}, {"role": "agent", "text": "a1"}],
        "text": "u2",
    }


async def test_json_mode_history_default_shape_unchanged() -> None:
    cfg = HTTPAdapterConfig(url=URL, request_template={"h": "{{history}}"}, text_path="reply")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"reply": "r"})

    adapter = HTTPAdapter(cfg, transport=httpx.MockTransport(handler), sleep=no_sleep)
    await adapter.start("c1")
    await adapter.send("c1", "u1")
    assert adapter.history_value([("user", "u1"), ("agent", "r")]) == [
        {"role": "user", "text": "u1"},
        {"role": "agent", "text": "r"},
    ]


# Header templates


async def test_header_conversation_id_per_request(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_TOKEN", "tok-123")
    cfg = sse_config(
        headers={
            "x-session": "convoprobe-{{conversation_id}}",
            "authorization": "Bearer ${AGENT_TOKEN}",
        }
    )
    rec = Recorder(done("a"), done("b"))
    adapter = HTTPAdapter(cfg, transport=httpx.MockTransport(rec), sleep=no_sleep)
    await adapter.start("conv-a")
    await adapter.start("conv-b")
    await adapter.send("conv-a", "hi")
    await adapter.send("conv-b", "hi")
    assert [r.headers["x-session"] for r in rec.requests] == [
        "convoprobe-conv-a",
        "convoprobe-conv-b",
    ]
    assert {r.headers["authorization"] for r in rec.requests} == {"Bearer tok-123"}
    # ${VAR} is resolved once at construction.
    monkeypatch.setenv("AGENT_TOKEN", "changed")
    assert adapter.request_headers("x")["authorization"] == "Bearer tok-123"


# text_path validator


def test_text_path_required_only_in_json_mode() -> None:
    with pytest.raises(ValueError, match="text_path is required when response_mode is json"):
        HTTPAdapterConfig(url=URL, request_template={})
    assert HTTPAdapterConfig(url=URL, request_template={}, response_mode="sse").text_path == ""


def test_loader_reports_missing_text_path(tmp_path: Path) -> None:
    path = tmp_path / "a.yaml"
    path.write_text("type: http\nurl: http://agent.example\nrequest_template: {}\n", "utf-8")
    with pytest.raises(AdapterLoadError) as info:
        load_adapter(f"http:{path}")
    assert str(info.value) == (
        f"adapter config {path}: text_path is required when response_mode is json"
    )


# Example server: equivalence and rate limits


def test_sentences_join_back() -> None:
    text = "Refund of $89.00 issued. Anything else? Thanks!"
    assert sentences(text) == ["Refund of $89.00 issued. ", "Anything else? ", "Thanks!"]
    assert "".join(sentences(text)) == text


@contextmanager
def serving(version: str, limit: int = 1000) -> Iterator[str]:
    server = make_server("127.0.0.1", 0, version, limit)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/chat"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def example_adapter(url: str, **overrides: Any) -> HTTPAdapter:
    data = yaml.safe_load(EXAMPLE.read_text("utf-8"))
    data.update(url=url, **overrides)
    return HTTPAdapter(HTTPAdapterConfig(**data))


@pytest.fixture
def scenario() -> Scenario:
    with as_file(files("convoprobe.demo") / "scenario.yaml") as path:
        return load_scenario(path)


async def run(scenario: Scenario, adapter: Any, store: RunStore, out: Path, name: str) -> RunResult:
    return await run_scenario(
        scenario, adapter, store=store, out_dir=out, adapter_label=name, run_id=name
    )


def fingerprint(result: RunResult) -> dict[str, Any]:
    """What must match across adapters, as in test_adapter_equivalence.py."""
    return {
        "status": result.conversation_status,
        "checks": [(c.rubric_id, c.persona_id, c.status, c.detail) for c in result.checks],
        "turns": {
            conv.id: [(m.role, m.text, [call.name for call in m.tool_calls]) for m in conv.messages]
            for conv in result.conversations
        },
    }


@pytest.mark.parametrize("version", ["v1", "v2"])
async def test_sse_example_matches_callable(
    version: str, scenario: Scenario, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SSE_TEST_TOKEN", "never-in-logs-7f3a")
    function = toy_agent_v1 if version == "v1" else toy_agent_v2
    with RunStore(tmp_path / "state.db") as store, serving(version) as url:
        plain = await run(scenario, CallableAdapter(function), store, tmp_path / "out", "plain")
        headers = {"x-session": "hdr-{{conversation_id}}", "authorization": "${SSE_TEST_TOKEN}"}
        sse = await run(
            scenario, example_adapter(url, headers=headers), store, tmp_path / "out", "sse"
        )
    assert fingerprint(sse) == fingerprint(plain)
    assert plain.checks, "the comparison must not be vacuous"
    refunds = [c for conv in sse.conversations for m in conv.messages for c in m.tool_calls]
    assert refunds and all(c.result == {"ok": True} for c in refunds)
    log = (tmp_path / "out" / "sse" / "log.jsonl").read_text("utf-8")
    assert "never-in-logs-7f3a" not in log
    assert "hdr-" not in log


async def test_rate_limit_is_transient_then_resumes(scenario: Scenario, tmp_path: Path) -> None:
    out = tmp_path / "out"
    with RunStore(tmp_path / "state.db") as store:
        with serving("v1", limit=1) as url:
            first = await run(scenario, example_adapter(url), store, out, "limited")
        statuses = first.conversation_status
        assert "failed_transient" in statuses.values()
        transient = [cid for cid, s in statuses.items() if s == "failed_transient"]
        for cid in transient:
            conv = next(c for c in first.conversations if c.id == cid)
            assert conv.error == "agent reported limited"
        with serving("v1") as url:
            second = await run(scenario, example_adapter(url), store, out, "limited")
    assert "failed_transient" not in second.conversation_status.values()
    assert all(second.conversation_status[cid] == "done" for cid in transient)
