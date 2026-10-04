"""Adapter for an agent behind an HTTP endpoint.

A YAML config names the URL, a request template, and where the reply is: dotted
paths into a JSON response, or event fields in a server-sent event (SSE) stream.
Retries 408, 429, and 5xx with the same backoff as model calls.
"""

from __future__ import annotations

import asyncio
import copy
import json
import os
import re
from collections.abc import Awaitable, Callable
from typing import Any, Literal

import httpx
from pydantic import BaseModel, ConfigDict, model_validator

from convoprobe.datamodels import AgentTurn, ToolCallRecord
from convoprobe.errors import ConvoprobeError, TransientError
from convoprobe.models.retry import TRANSIENT_STATUS, RetryPolicy, call_with_retry

# 408, 429, 500, 502, 503, 504 are retried: the same set as model calls.
_ENV_REF = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
PLACEHOLDERS = ("{{message}}", "{{conversation_id}}", "{{history}}")


class HTTPAdapterConfig(BaseModel):
    """How to call the agent endpoint and where to find its reply."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["http"] = "http"
    url: str
    # Values may use ${ENV_VAR} (resolved once) and {{conversation_id}} (per request).
    headers: dict[str, str] = {}
    request_template: dict[str, Any]
    text_path: str = ""  # required in json mode
    tool_calls_path: str | None = None
    timeout_s: float = 30.0

    response_mode: Literal["json", "sse"] = "json"
    # SSE: each event is a "data: <json>" line; events are separated by blank
    # lines; non-data lines are ignored.
    sse_type_field: str = "type"
    sse_text_event: str = "done"  # the event whose text field holds the full reply
    sse_text_field: str = "text"
    sse_tool_event: str | None = None  # None: tool calls are not captured
    sse_tool_name_field: str = "name"
    sse_tool_args_field: str = "args"
    sse_tool_result_fields: list[str] = []  # copied into ToolCallRecord.result
    sse_transient_events: list[str] = []  # "rate limited, retry later": TransientError
    sse_note_events: list[str] = []  # recorded in AgentTurn.metadata["events"]

    # Shape of the "{{history}}" value.
    history_role_field: str = "role"
    history_text_field: str = "text"
    history_roles: dict[str, str] = {"user": "user", "agent": "agent"}
    history_includes_current: bool = True  # False: the new message goes only in {{message}}

    @model_validator(mode="after")
    def _text_path_for_json(self) -> HTTPAdapterConfig:
        if self.response_mode == "json" and not self.text_path:
            raise ValueError("text_path is required when response_mode is json")
        return self


class _RetryableStatusError(Exception):
    """A status the retry policy should retry. Carries .status_code."""

    def __init__(self, status_code: int, body: str) -> None:
        super().__init__(f"agent endpoint returned {status_code}: {body[:200]}")
        self.status_code = status_code


def _expand_headers(headers: dict[str, str]) -> dict[str, str]:
    """Replace ${NAME} with the environment value. Values are never printed."""

    def value_of(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in os.environ:
            raise ConvoprobeError(f"HTTP adapter header needs environment variable {name}")
        return os.environ[name]

    return {key: _ENV_REF.sub(value_of, value) for key, value in headers.items()}


def fill_template(template: Any, values: dict[str, Any]) -> Any:
    """Deep copy of template with whole-value placeholders replaced."""
    if isinstance(template, str):
        return copy.deepcopy(values[template]) if template in values else template
    if isinstance(template, dict):
        return {key: fill_template(value, values) for key, value in template.items()}
    if isinstance(template, list):
        return [fill_template(item, values) for item in template]
    return copy.deepcopy(template)


def dotted_get(data: Any, path: str) -> Any:
    """Walk "a.b.0.c" through dict keys and list indices."""
    current = data
    for part in path.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
        elif isinstance(current, list) and part.isdigit() and int(part) < len(current):
            current = current[int(part)]
        else:
            raise ConvoprobeError(f"agent response has no field {path!r}")
    return current


def parse_sse(body: str, config: HTTPAdapterConfig) -> AgentTurn:
    """Read a complete SSE body into an AgentTurn.

    The reply is the text of the last text event. A transient event raises
    TransientError once the whole stream is read, so the runner can resume later.
    """
    events: list[dict[str, Any]] = []
    for number, line in enumerate(body.splitlines(), start=1):
        if not line.startswith("data:"):
            continue  # blank separators, comments, and event:/id:/retry: lines
        payload = line[len("data:") :].strip()
        try:
            event = json.loads(payload)
        except ValueError as exc:
            raise ConvoprobeError(f"agent stream line {number} is not valid JSON") from exc
        if not isinstance(event, dict):
            raise ConvoprobeError(f"agent stream line {number} is not a JSON object")
        events.append(event)

    text: str | None = None
    tool_calls: list[ToolCallRecord] = []
    transient: str | None = None
    notes: list[str] = []
    for event in events:
        kind = event.get(config.sse_type_field)
        if kind == config.sse_text_event:
            value = event.get(config.sse_text_field)
            if not isinstance(value, str):
                raise ConvoprobeError(
                    f"agent {config.sse_text_event} event field"
                    f" {config.sse_text_field!r} must be a string"
                )
            text = value
        elif config.sse_tool_event is not None and kind == config.sse_tool_event:
            tool_calls.append(_sse_tool_call(event, config))
        if kind in config.sse_transient_events and transient is None:
            transient = str(kind)
        if kind in config.sse_note_events:
            notes.append(str(kind))
    if transient is not None:
        raise TransientError(f"agent reported {transient}")
    if text is None:
        raise ConvoprobeError(f"agent stream ended without a {config.sse_text_event} event")
    return AgentTurn(text=text, tool_calls=tool_calls, metadata={"events": notes} if notes else {})


def _sse_tool_call(event: dict[str, Any], config: HTTPAdapterConfig) -> ToolCallRecord:
    name = event.get(config.sse_tool_name_field)
    args = event.get(config.sse_tool_args_field) or {}
    if not isinstance(name, str) or not isinstance(args, dict):
        raise ConvoprobeError(
            f"agent {config.sse_tool_event} event needs a string"
            f" {config.sse_tool_name_field!r} and an object {config.sse_tool_args_field!r}"
        )
    result: dict[str, Any] | None = None
    if config.sse_tool_result_fields:
        result = {f: event[f] for f in config.sse_tool_result_fields if f in event}
    return ToolCallRecord(name=name, args=args, result=result)


def _tool_calls(raw: Any, path: str) -> list[ToolCallRecord]:
    problem = ConvoprobeError(
        f"agent response field {path!r} must be a list of objects with a name"
    )
    if not isinstance(raw, list):
        raise problem
    records = []
    for item in raw:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str):
            raise problem
        args = item.get("args") or {}
        if not isinstance(args, dict):
            raise problem
        records.append(ToolCallRecord(name=item["name"], args=args, result=item.get("result")))
    return records


class HTTPAdapter:
    """POSTs each user message to an agent endpoint and reads the reply."""

    def __init__(
        self,
        config: HTTPAdapterConfig,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        retry: RetryPolicy = RetryPolicy(),  # noqa: B008
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self.config = config
        self._headers = _expand_headers(config.headers)
        self._transport = transport
        self._retry = retry
        self._sleep = sleep
        # (convoprobe role, text) per conversation; mapped to the agent's shape per request.
        self._histories: dict[str, list[tuple[str, str]]] = {}

    async def start(self, conversation_id: str) -> None:
        self._histories[conversation_id] = []

    async def end(self, conversation_id: str) -> None:
        self._histories.pop(conversation_id, None)

    def history_value(self, history: list[tuple[str, str]]) -> list[dict[str, str]]:
        """The "{{history}}" value in the agent's own field and role names."""
        config = self.config
        entries = history if config.history_includes_current else history[:-1]
        return [
            {
                config.history_role_field: config.history_roles.get(role, role),
                config.history_text_field: text,
            }
            for role, text in entries
        ]

    def request_headers(self, conversation_id: str) -> dict[str, str]:
        """Headers with {{conversation_id}} filled in. Values are never logged."""
        return {
            key: value.replace("{{conversation_id}}", conversation_id)
            for key, value in self._headers.items()
        }

    async def send(self, conversation_id: str, user_message: str) -> AgentTurn:
        history = self._histories.get(conversation_id)
        if history is None:
            raise ConvoprobeError(f"conversation {conversation_id} not started")
        history.append(("user", user_message))
        body = fill_template(
            self.config.request_template,
            {
                "{{message}}": user_message,
                "{{conversation_id}}": conversation_id,
                "{{history}}": self.history_value(history),
            },
        )
        headers = self.request_headers(conversation_id)
        data = await call_with_retry(
            lambda: self._post(body, headers),
            policy=self._retry,
            label=f"http:{self.config.url}",
            sleep=self._sleep,
        )
        if self.config.response_mode == "sse":
            turn = parse_sse(data, self.config)
            history.append(("agent", turn.text))
            return turn
        text = dotted_get(data, self.config.text_path)
        if not isinstance(text, str):
            raise ConvoprobeError(
                f"agent response field {self.config.text_path!r} must be a string"
            )
        tool_calls: list[ToolCallRecord] = []
        if self.config.tool_calls_path is not None:
            raw = dotted_get(data, self.config.tool_calls_path)
            tool_calls = _tool_calls(raw, self.config.tool_calls_path)
        history.append(("agent", text))
        return AgentTurn(text=text, tool_calls=tool_calls)

    async def _post(self, body: Any, headers: dict[str, str]) -> Any:
        """The parsed JSON body, or in sse mode the full stream as text."""
        async with httpx.AsyncClient(
            timeout=self.config.timeout_s, transport=self._transport
        ) as client:
            response = await client.post(self.config.url, json=body, headers=headers)
        status = response.status_code
        if status in TRANSIENT_STATUS:
            raise _RetryableStatusError(status, response.text)
        if not 200 <= status < 300:
            raise ConvoprobeError(f"agent endpoint returned {status}: {response.text[:200]}")
        if self.config.response_mode == "sse":
            return response.text
        try:
            return response.json()
        except ValueError as exc:
            raise ConvoprobeError(
                f"agent endpoint returned a body that is not JSON: {response.text[:200]}"
            ) from exc
