"""HTTPAdapter over httpx.MockTransport. No sockets."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from convoprobe.adapters.http_adapter import (
    HTTPAdapter,
    HTTPAdapterConfig,
    dotted_get,
    fill_template,
)
from convoprobe.errors import ConvoprobeError, TransientError

URL = "http://agent.example/chat"


def config(**overrides: Any) -> HTTPAdapterConfig:
    data: dict[str, Any] = {
        "url": URL,
        "request_template": {"session": "{{conversation_id}}", "message": "{{message}}"},
        "text_path": "reply",
        "tool_calls_path": "tool_calls",
    }
    data.update(overrides)
    return HTTPAdapterConfig(**data)


class Recorder:
    """A MockTransport handler that replays scripted responses and keeps requests."""

    def __init__(self, *responses: httpx.Response) -> None:
        self.responses = list(responses)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self.responses.pop(0)

    def bodies(self) -> list[Any]:
        return [json.loads(r.content) for r in self.requests]


def ok(reply: str = "hi", tool_calls: Any = None) -> httpx.Response:
    return httpx.Response(200, json={"reply": reply, "tool_calls": tool_calls or []})


async def no_sleep(_seconds: float) -> None:
    return None


async def adapter_with(rec: Recorder, cfg: HTTPAdapterConfig | None = None) -> HTTPAdapter:
    adapter = HTTPAdapter(cfg or config(), transport=httpx.MockTransport(rec), sleep=no_sleep)
    await adapter.start("c1")
    return adapter


def test_fill_template_whole_values_only() -> None:
    template = {
        "message": "{{message}}",
        "note": "prefix {{message}}",
        "nested": [{"id": "{{conversation_id}}"}, 3, None],
        "history": "{{history}}",
    }
    values = {"{{message}}": "hi", "{{conversation_id}}": "c1", "{{history}}": [{"a": 1}]}
    assert fill_template(template, values) == {
        "message": "hi",
        "note": "prefix {{message}}",
        "nested": [{"id": "c1"}, 3, None],
        "history": [{"a": 1}],
    }
    assert template["history"] == "{{history}}"


def test_dotted_get() -> None:
    data = {"a": {"b": [{"c": "deep"}, {"c": "second"}]}}
    assert dotted_get(data, "a.b.0.c") == "deep"
    assert dotted_get(data, "a.b.1.c") == "second"
    for path in ("a.x", "a.b.5.c", "a.b.c", "a.b.0.c.d"):
        with pytest.raises(ConvoprobeError, match=f"agent response has no field '{path}'"):
            dotted_get(data, path)


async def test_request_body_and_history() -> None:
    rec = Recorder(ok("first"), ok("second"))
    cfg = config(request_template={"s": "{{conversation_id}}", "h": "{{history}}", "keep": "x"})
    adapter = await adapter_with(rec, cfg)
    await adapter.send("c1", "hello")
    await adapter.send("c1", "again")
    assert rec.bodies() == [
        {"s": "c1", "h": [{"role": "user", "text": "hello"}], "keep": "x"},
        {
            "s": "c1",
            "h": [
                {"role": "user", "text": "hello"},
                {"role": "agent", "text": "first"},
                {"role": "user", "text": "again"},
            ],
            "keep": "x",
        },
    ]


async def test_tool_calls_parsed() -> None:
    calls = [
        {"name": "lookup_order", "args": {"order_id": "1042"}, "result": {"found": True}},
        {"name": "ping"},
    ]
    adapter = await adapter_with(Recorder(ok("done", calls)))
    turn = await adapter.send("c1", "hi")
    assert turn.text == "done"
    assert [(c.name, c.args, c.result) for c in turn.tool_calls] == [
        ("lookup_order", {"order_id": "1042"}, {"found": True}),
        ("ping", {}, None),
    ]


async def test_nested_text_path_and_no_tool_path() -> None:
    rec = Recorder(httpx.Response(200, json={"data": {"choices": [{"text": "nested"}]}}))
    cfg = config(text_path="data.choices.0.text", tool_calls_path=None)
    turn = await (await adapter_with(rec, cfg)).send("c1", "hi")
    assert (turn.text, turn.tool_calls) == ("nested", [])


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"tool_calls": []}, "agent response has no field 'reply'"),
        ({"reply": 5, "tool_calls": []}, "agent response field 'reply' must be a string"),
        ({"reply": "x", "tool_calls": "no"}, "must be a list of objects with a name"),
        ({"reply": "x", "tool_calls": [{"args": {}}]}, "must be a list of objects with a name"),
        ({"reply": "x", "tool_calls": [{"name": "a", "args": [1]}]}, "list of objects"),
    ],
)
async def test_bad_response_shapes(payload: Any, message: str) -> None:
    adapter = await adapter_with(Recorder(httpx.Response(200, json=payload)))
    with pytest.raises(ConvoprobeError, match=message):
        await adapter.send("c1", "hi")


async def test_503_retried_then_success() -> None:
    rec = Recorder(httpx.Response(503, text="busy"), httpx.Response(429), ok("finally"))
    turn = await (await adapter_with(rec)).send("c1", "hi")
    assert turn.text == "finally"
    assert len(rec.requests) == 3


async def test_503_always_is_transient() -> None:
    rec = Recorder(*[httpx.Response(503, text="busy")] * 4)
    with pytest.raises(TransientError, match=f"http:{URL}: gave up after 4 attempts"):
        await (await adapter_with(rec)).send("c1", "hi")


async def test_400_is_an_error_not_retried() -> None:
    rec = Recorder(httpx.Response(400, text="bad request body"))
    with pytest.raises(ConvoprobeError) as info:
        await (await adapter_with(rec)).send("c1", "hi")
    assert str(info.value) == "agent endpoint returned 400: bad request body"
    assert not isinstance(info.value, TransientError)
    assert len(rec.requests) == 1


async def test_non_json_body() -> None:
    rec = Recorder(httpx.Response(200, text="<html>oops</html>"))
    with pytest.raises(ConvoprobeError, match="not JSON: <html>oops</html>"):
        await (await adapter_with(rec)).send("c1", "hi")


async def test_header_env_substitution(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_TOKEN", "test-token")
    rec = Recorder(ok())
    cfg = config(headers={"Authorization": "Bearer ${AGENT_TOKEN}", "X-Plain": "plain"})
    await (await adapter_with(rec, cfg)).send("c1", "hi")
    assert rec.requests[0].headers["Authorization"] == "Bearer test-token"
    assert rec.requests[0].headers["X-Plain"] == "plain"


def test_header_missing_env_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MISSING_AGENT_TOKEN", raising=False)
    cfg = config(headers={"Authorization": "Bearer ${MISSING_AGENT_TOKEN}"})
    with pytest.raises(ConvoprobeError) as info:
        HTTPAdapter(cfg)
    assert str(info.value) == "HTTP adapter header needs environment variable MISSING_AGENT_TOKEN"


async def test_start_end_and_send_before_start() -> None:
    adapter = await adapter_with(Recorder(ok()))
    await adapter.end("c1")
    with pytest.raises(ConvoprobeError, match="conversation c1 not started"):
        await adapter.send("c1", "hi")


def test_config_rejects_unknown_fields() -> None:
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        config(surprise=True)
