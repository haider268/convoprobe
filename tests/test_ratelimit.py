"""Token buckets and ProviderLimiter with a fake clock."""

from __future__ import annotations

import asyncio

import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from convoprobe.models.ratelimit import ProviderLimiter, TokenBucket, estimate_tokens


class FakeTime:
    """clock() reads the fake time; sleep() advances it and records the wait."""

    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def clock(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds
        await asyncio.sleep(0)


async def test_bucket_starts_full_then_waits() -> None:
    t = FakeTime()
    bucket = TokenBucket(60, 60, clock=t.clock, sleep=t.sleep)
    for _ in range(60):
        await bucket.acquire()
    assert t.sleeps == []
    await bucket.acquire()
    assert t.sleeps == [pytest.approx(1.0)]


async def test_bucket_refills_over_time() -> None:
    t = FakeTime()
    bucket = TokenBucket(60, 10, clock=t.clock, sleep=t.sleep)
    await bucket.acquire(10)
    t.now += 5.0
    await bucket.acquire(5)
    assert t.sleeps == []


async def test_bucket_amount_clamped_to_capacity() -> None:
    t = FakeTime()
    bucket = TokenBucket(6000, 6000, clock=t.clock, sleep=t.sleep)
    await bucket.acquire(50_000)  # more than capacity: takes the whole bucket, no wait
    assert t.sleeps == []
    await bucket.acquire(50_000)
    assert t.sleeps == [pytest.approx(60.0)]


async def test_limiter_concurrency_one_never_overlaps() -> None:
    t = FakeTime()
    limiter = ProviderLimiter(None, None, 1, clock=t.clock, sleep=t.sleep)
    active = 0
    peak = 0

    async def work() -> None:
        nonlocal active, peak
        async with limiter.slot(100):
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            active -= 1

    await asyncio.gather(work(), work(), work())
    assert peak == 1


async def test_limiter_none_buckets_skipped() -> None:
    limiter = ProviderLimiter(None, None, 2)
    assert limiter.requests is None
    assert limiter.tokens is None
    async with limiter.slot(10**9):
        pass


async def test_limiter_uses_both_buckets() -> None:
    t = FakeTime()
    limiter = ProviderLimiter(60, 1000, 4, clock=t.clock, sleep=t.sleep)
    async with limiter.slot(1000):
        pass
    assert limiter.requests is not None and limiter.tokens is not None
    assert limiter.requests.tokens == 59
    assert limiter.tokens.tokens == 0
    async with limiter.slot(500):
        pass
    assert t.sleeps == [pytest.approx(30.0)]


def test_estimate_tokens() -> None:
    messages = [SystemMessage(content="x" * 400), HumanMessage(content="y" * 40)]
    assert estimate_tokens(messages) == 110 + 500
    assert estimate_tokens([]) == 500
