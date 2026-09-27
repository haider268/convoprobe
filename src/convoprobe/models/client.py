"""LLMClient: one model spec with cache, replay, rate limits, and retries."""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Awaitable, Callable, Sequence
from typing import Any, TypeVar

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from pydantic import BaseModel, ValidationError

from convoprobe.datamodels import LLMUsage
from convoprobe.errors import ConvoprobeError
from convoprobe.models.cache import LLMCache
from convoprobe.models.ratelimit import ProviderLimiter, estimate_tokens
from convoprobe.models.retry import RetryPolicy, call_with_retry, is_transient

T = TypeVar("T", bound=BaseModel)

# Provider error text meaning the model could not fit the schema. Not retryable.
_PROVIDER_SCHEMA_FAIL = ("tool_use_failed", "json_validate_failed", "failed_generation")


class StructuredOutputError(ConvoprobeError):
    """The model's output did not match the requested schema."""


class ReplayMissError(ConvoprobeError):
    """Replay mode and no recorded response for this request."""


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and part.get("type") == "text":
                parts.append(str(part.get("text", "")))
        return "".join(parts)
    return str(content)


class LLMClient:
    """Calls one model. Cache first; on a miss, rate limit, retry, then store.

    Requests are keyed by spec, kind, temperature, schema, messages, and salt.
    The salt lets repeats of the same conversation get independent samples
    instead of one cached answer. Callers pass for example conversation id + turn.
    """

    def __init__(
        self,
        spec: str,
        *,
        cache: LLMCache,
        chat_model: BaseChatModel | None,
        temperature: float,
        limiter: ProviderLimiter | None = None,
        retry: RetryPolicy = RetryPolicy(),  # noqa: B008
        replay_only: bool = False,
        run_id: str | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        structured_method: str | None = None,
    ) -> None:
        if chat_model is None and not replay_only:
            raise ConvoprobeError(f"{spec}: a chat model is required unless replay_only is set")
        self.spec = spec
        self.cache = cache
        self.chat_model = chat_model
        self.temperature = temperature
        self.limiter = limiter
        self.retry = retry
        self.replay_only = replay_only
        self.run_id = run_id
        self._sleep = sleep
        self.structured_method = structured_method
        self.usage = LLMUsage(spec=spec)

    def cache_key(
        self,
        kind: str,
        messages: Sequence[BaseMessage],
        schema: type[BaseModel] | None,
        salt: str,
    ) -> str:
        payload = {
            "v": 2,
            "spec": self.spec,
            "method": self.structured_method,
            "kind": kind,
            "temperature": self.temperature,
            "schema": schema.model_json_schema() if schema else None,
            "messages": [{"role": m.type, "content": m.content} for m in messages],
            "salt": salt,
        }
        raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    async def text(self, messages: Sequence[BaseMessage], *, salt: str = "") -> str:
        """Plain text reply."""
        response = await self._call("text", messages, None, salt)
        return str(response["text"])

    async def structured(
        self, messages: Sequence[BaseMessage], schema: type[T], *, salt: str = ""
    ) -> T:
        """Reply parsed into schema. Bad output raises StructuredOutputError."""
        response = await self._call("structured", messages, schema, salt)
        return schema.model_validate(response["data"])

    async def _call(
        self,
        kind: str,
        messages: Sequence[BaseMessage],
        schema: type[BaseModel] | None,
        salt: str,
    ) -> dict[str, Any]:
        key = self.cache_key(kind, messages, schema, salt)
        if self.run_id:
            self.cache.mark_used(self.run_id, key)
        cached = self.cache.get(key)
        if cached is not None:
            self.usage.cache_hits += 1
            return cached
        if self.replay_only or self.chat_model is None:
            raise ReplayMissError(
                f"no recorded response for {self.spec} ({kind}); record it with a live model first"
            )
        chat_model = self.chat_model

        async def attempt() -> tuple[dict[str, Any], AIMessage | None]:
            if self.limiter is None:
                return await self._invoke(chat_model, kind, messages, schema)
            async with self.limiter.slot(estimate_tokens(messages)):
                return await self._invoke(chat_model, kind, messages, schema)

        def count_retry() -> None:
            self.usage.retries += 1

        response, message = await call_with_retry(
            attempt, policy=self.retry, label=self.spec, on_retry=count_retry, sleep=self._sleep
        )
        self.usage.calls += 1
        usage = getattr(message, "usage_metadata", None)
        if usage:
            self.usage.input_tokens += int(usage.get("input_tokens", 0))
            self.usage.output_tokens += int(usage.get("output_tokens", 0))
        self.cache.put(key, self.spec, kind, response)
        return response

    async def _invoke(
        self,
        chat_model: BaseChatModel,
        kind: str,
        messages: Sequence[BaseMessage],
        schema: type[BaseModel] | None,
    ) -> tuple[dict[str, Any], AIMessage | None]:
        if schema is None:
            message = await chat_model.ainvoke(list(messages))
            ai = message if isinstance(message, AIMessage) else None
            return {"text": _content_text(message.content)}, ai
        kwargs: dict[str, Any] = {"include_raw": True}
        if self.structured_method is not None:
            kwargs["method"] = self.structured_method
        try:
            out = await chat_model.with_structured_output(schema, **kwargs).ainvoke(list(messages))
        except Exception as exc:
            text = str(exc)
            if not is_transient(exc) and any(marker in text for marker in _PROVIDER_SCHEMA_FAIL):
                raise StructuredOutputError(
                    f"{self.spec} could not produce {schema.__name__}: {text[:200]}"
                ) from exc
            raise
        if not isinstance(out, dict):
            out = {"raw": None, "parsed": out, "parsing_error": None}
        raw = out.get("raw")
        mismatch = StructuredOutputError(
            f"{self.spec} returned output that does not match {schema.__name__}:"
            f" {_content_text(getattr(raw, 'content', ''))[:200]}"
        )
        if out.get("parsing_error") is not None or out.get("parsed") is None:
            raise mismatch
        try:
            parsed = schema.model_validate(out["parsed"])
        except ValidationError as exc:
            raise mismatch from exc
        raw_ai = raw if isinstance(raw, AIMessage) else None
        return {"data": parsed.model_dump(mode="json")}, raw_ai
