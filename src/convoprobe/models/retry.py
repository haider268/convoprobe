"""Retry with exponential backoff and jitter for transient provider errors."""

from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TypeVar

from convoprobe.errors import ConvoprobeError, TransientError

T = TypeVar("T")

TRANSIENT_STATUS = {408, 429, 500, 502, 503, 504}

# Matched by class name so provider packages never need importing here.
TRANSIENT_NAMES = {
    "RateLimitError",
    "APIConnectionError",
    "APITimeoutError",
    "InternalServerError",
    "ServerError",
    "ConnectError",
    "ConnectTimeout",
    "ReadTimeout",
    "WriteTimeout",
    "PoolTimeout",
    "TimeoutException",
    "RemoteProtocolError",
    "RequestError",
}

VPN_MESSAGE = "Groq returned 403. This usually means a VPN is on. Turn the VPN off and rerun."


def status_of(exc: BaseException) -> int | None:
    """HTTP-like status from .status_code, then .code. None if neither is an int."""
    for attr in ("status_code", "code"):
        value = getattr(exc, attr, None)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return None


def _chain(exc: BaseException) -> list[BaseException]:
    """The exception plus everything reachable via __cause__ and __context__."""
    seen: list[BaseException] = []
    stack = [exc]
    while stack:
        current = stack.pop(0)
        if any(current is s for s in seen):
            continue
        seen.append(current)
        for linked in (current.__cause__, current.__context__):
            if linked is not None:
                stack.append(linked)
    return seen


def is_too_large(exc: BaseException) -> bool:
    """A 429 that means one request exceeds the per-minute token limit. Waiting cannot fix it."""
    return any(status_of(e) == 429 and "Request too large" in str(e) for e in _chain(exc))


def is_transient(exc: BaseException) -> bool:
    """True for rate limits, 5xx, timeouts, and connection failures."""
    if is_too_large(exc):
        return False
    for e in _chain(exc):
        if status_of(e) in TRANSIENT_STATUS:
            return True
        if any(cls.__name__ in TRANSIENT_NAMES for cls in type(e).__mro__):
            return True
        if isinstance(e, TimeoutError | ConnectionError):
            return True
    return False


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 4
    base_delay_s: float = 1.0
    max_delay_s: float = 30.0


async def call_with_retry(
    fn: Callable[[], Awaitable[T]],
    *,
    policy: RetryPolicy,
    label: str,
    on_retry: Callable[[], None] | None = None,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    rand: Callable[[], float] = random.random,
) -> T:
    """Call fn, retrying transient errors. Gives up with TransientError."""
    for attempt in range(policy.max_attempts):
        try:
            return await fn()
        except Exception as exc:
            if status_of(exc) == 403 and label.startswith("groq:"):
                raise ConvoprobeError(VPN_MESSAGE) from exc
            if is_too_large(exc):
                raise ConvoprobeError(
                    f"{label}: request is larger than the provider's per-minute token limit."
                    " Use a model with a higher limit or a shorter scenario."
                ) from exc
            if not is_transient(exc):
                raise
            if attempt == policy.max_attempts - 1:
                raise TransientError(
                    f"{label}: gave up after {policy.max_attempts} attempts:"
                    f" {type(exc).__name__}: {exc}"
                ) from exc
            delay = min(policy.max_delay_s, policy.base_delay_s * 2**attempt)
            await sleep(delay / 2 + rand() * delay / 2)
            if on_retry is not None:
                on_retry()
    raise ConvoprobeError(f"{label}: max_attempts must be at least 1")
