"""Adapter for an agent behind an HTTP endpoint.

A YAML config names the URL, a request template, and dotted paths to the reply
text and tool calls in the JSON response. Retries 408, 429, and 5xx with the
same backoff as model calls.
"""

from __future__ import annotations

import asyncio
import copy
import os
import re
from collections.abc import Awaitable, Callable
from typing import Any, Literal

import httpx
from pydantic import BaseModel, ConfigDict

from convoprobe.datamodels import AgentTurn, ToolCallRecord
from convoprobe.errors import ConvoprobeError
from convoprobe.models.retry import TRANSIENT_STATUS, RetryPolicy, call_with_retry

# 408, 429, 500, 502, 503, 504 are retried: the same set as model calls.
_ENV_REF = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
PLACEHOLDERS = ("{{message}}", "{{conversation_id}}", "{{history}}")


class HTTPAdapterConfig(BaseModel):
    """How to call the agent endpoint and where to find its reply."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["http"] = "http"
    url: str
    headers: dict[str, str] = {}
    request_template: dict[str, Any]
    text_path: str
    tool_calls_path: str | None = None
    timeout_s: float = 30.0


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
        self._histories: dict[str, list[dict[str, str]]] = {}

    async def start(self, conversation_id: str) -> None:
        self._histories[conversation_id] = []

    async def end(self, conversation_id: str) -> None:
        self._histories.pop(conversation_id, None)

    async def send(self, conversation_id: str, user_message: str) -> AgentTurn:
        history = self._histories.get(conversation_id)
        if history is None:
            raise ConvoprobeError(f"conversation {conversation_id} not started")
        history.append({"role": "user", "text": user_message})
        body = fill_template(
            self.config.request_template,
            {
                "{{message}}": user_message,
                "{{conversation_id}}": conversation_id,
                "{{history}}": list(history),
            },
        )
        data = await call_with_retry(
            lambda: self._post(body),
            policy=self._retry,
            label=f"http:{self.config.url}",
            sleep=self._sleep,
        )
        text = dotted_get(data, self.config.text_path)
        if not isinstance(text, str):
            raise ConvoprobeError(
                f"agent response field {self.config.text_path!r} must be a string"
            )
        tool_calls: list[ToolCallRecord] = []
        if self.config.tool_calls_path is not None:
            raw = dotted_get(data, self.config.tool_calls_path)
            tool_calls = _tool_calls(raw, self.config.tool_calls_path)
        history.append({"role": "agent", "text": text})
        return AgentTurn(text=text, tool_calls=tool_calls)

    async def _post(self, body: Any) -> Any:
        async with httpx.AsyncClient(
            timeout=self.config.timeout_s, transport=self._transport
        ) as client:
            response = await client.post(self.config.url, json=body, headers=self._headers)
        status = response.status_code
        if status in TRANSIENT_STATUS:
            raise _RetryableStatusError(status, response.text)
        if not 200 <= status < 300:
            raise ConvoprobeError(f"agent endpoint returned {status}: {response.text[:200]}")
        try:
            return response.json()
        except ValueError as exc:
            raise ConvoprobeError(
                f"agent endpoint returned a body that is not JSON: {response.text[:200]}"
            ) from exc
