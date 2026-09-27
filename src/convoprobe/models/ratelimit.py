"""Token buckets and a per-provider limiter (requests/min, tokens/min, concurrency)."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from contextlib import asynccontextmanager

from langchain_core.messages import BaseMessage

Clock = Callable[[], float]
Sleep = Callable[[float], Awaitable[None]]


class TokenBucket:
    """Refills at rate_per_minute up to capacity. Starts full."""

    def __init__(
        self,
        rate_per_minute: float,
        capacity: float,
        clock: Clock = time.monotonic,
        sleep: Sleep = asyncio.sleep,
    ) -> None:
        self.rate_per_minute = rate_per_minute
        self.capacity = capacity
        self.tokens = capacity
        self._clock = clock
        self._sleep = sleep
        self._last = clock()
        self._lock = asyncio.Lock()

    def _refill(self) -> None:
        now = self._clock()
        elapsed = now - self._last
        self._last = now
        self.tokens = min(self.capacity, self.tokens + elapsed * self.rate_per_minute / 60)

    async def acquire(self, amount: float = 1.0) -> None:
        """Wait until amount tokens are available, then take them.

        Amounts above capacity are clamped so a large request cannot wait forever.
        """
        amount = min(amount, self.capacity)
        async with self._lock:
            while True:
                self._refill()
                if self.tokens >= amount:
                    self.tokens -= amount
                    return
                await self._sleep((amount - self.tokens) * 60 / self.rate_per_minute)


class ProviderLimiter:
    """One per provider. Buckets set to None are skipped."""

    def __init__(
        self,
        rpm: float | None,
        tpm: float | None,
        concurrency: int,
        clock: Clock = time.monotonic,
        sleep: Sleep = asyncio.sleep,
    ) -> None:
        self.requests = TokenBucket(rpm, rpm, clock, sleep) if rpm is not None else None
        self.tokens = TokenBucket(tpm, tpm, clock, sleep) if tpm is not None else None
        self._semaphore = asyncio.Semaphore(concurrency)

    @asynccontextmanager
    async def slot(self, estimated_tokens: int) -> AsyncIterator[None]:
        """Hold a concurrency slot after paying the request and token buckets."""
        async with self._semaphore:
            if self.requests is not None:
                await self.requests.acquire(1)
            if self.tokens is not None:
                await self.tokens.acquire(estimated_tokens)
            yield


def estimate_tokens(messages: Sequence[BaseMessage]) -> int:
    """Rough input size (4 characters per token) plus room for the reply."""
    return sum(len(str(m.content)) for m in messages) // 4 + 500
