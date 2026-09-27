"""is_transient and call_with_retry with injected sleep and randomness."""

from __future__ import annotations

import pytest

from convoprobe.errors import ConvoprobeError, TransientError
from convoprobe.models.retry import (
    VPN_MESSAGE,
    RetryPolicy,
    call_with_retry,
    is_transient,
    status_of,
)


class StatusError(Exception):
    def __init__(self, status_code: int) -> None:
        super().__init__(f"status {status_code}")
        self.status_code = status_code


class CodeError(Exception):
    def __init__(self, code: int) -> None:
        super().__init__(f"code {code}")
        self.code = code


class RateLimitError(Exception):
    """Named like the provider class, matched by name."""


class ProviderRateLimit(RateLimitError):
    """Subclass: matched through the MRO."""


def caused_by(outer: Exception, inner: Exception) -> Exception:
    try:
        raise outer from inner
    except Exception as exc:
        return exc


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (StatusError(429), True),
        (StatusError(503), True),
        (CodeError(429), True),
        (StatusError(400), False),
        (RateLimitError("slow down"), True),
        (ProviderRateLimit("slow down"), True),
        (TimeoutError("timed out"), True),
        (ConnectionError("refused"), True),
        (ValueError("bad"), False),
        (caused_by(RuntimeError("wrapper"), StatusError(502)), True),
        (caused_by(RuntimeError("wrapper"), ValueError("bad")), False),
    ],
)
def test_is_transient(exc: Exception, expected: bool) -> None:
    assert is_transient(exc) is expected


def test_is_transient_via_context_and_cycle() -> None:
    inner = TimeoutError("slow")
    outer = RuntimeError("wrapper")
    outer.__context__ = inner
    inner.__context__ = outer  # a cycle must not loop forever
    assert is_transient(outer) is True


def test_status_of() -> None:
    assert status_of(StatusError(503)) == 503
    assert status_of(CodeError(429)) == 429
    assert status_of(ValueError("x")) is None
    flag = Exception()
    flag.code = True  # type: ignore[attr-defined]
    assert status_of(flag) is None


class Recorder:
    def __init__(self) -> None:
        self.sleeps: list[float] = []
        self.retries = 0

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)

    def on_retry(self) -> None:
        self.retries += 1


def failing(errors: list[Exception], result: str = "ok"):  # type: ignore[no-untyped-def]
    calls = {"n": 0}

    async def fn() -> str:
        calls["n"] += 1
        if errors:
            raise errors.pop(0)
        return result

    return fn, calls


async def test_retries_then_succeeds_with_exact_sleeps() -> None:
    rec = Recorder()
    fn, calls = failing([StatusError(503), StatusError(503), StatusError(503)])
    result = await call_with_retry(
        fn,
        policy=RetryPolicy(),
        label="ollama:x",
        on_retry=rec.on_retry,
        sleep=rec.sleep,
        rand=lambda: 0.5,
    )
    assert result == "ok"
    assert rec.sleeps == [0.75, 1.5, 3.0]
    assert rec.retries == 3
    assert calls["n"] == 4


async def test_gives_up_after_max_attempts() -> None:
    rec = Recorder()
    fn, calls = failing([StatusError(503) for _ in range(10)])
    with pytest.raises(TransientError) as info:
        await call_with_retry(
            fn, policy=RetryPolicy(), label="ollama:x", sleep=rec.sleep, rand=lambda: 0.5
        )
    assert calls["n"] == 4
    assert len(rec.sleeps) == 3
    assert isinstance(info.value.__cause__, StatusError)
    assert str(info.value) == "ollama:x: gave up after 4 attempts: StatusError: status 503"


async def test_non_transient_raised_immediately() -> None:
    rec = Recorder()
    fn, calls = failing([ValueError("bad input")])
    with pytest.raises(ValueError, match="bad input"):
        await call_with_retry(fn, policy=RetryPolicy(), label="ollama:x", sleep=rec.sleep)
    assert rec.sleeps == []
    assert calls["n"] == 1


async def test_groq_403_is_vpn_message() -> None:
    rec = Recorder()
    fn, _ = failing([StatusError(403)])
    with pytest.raises(ConvoprobeError) as info:
        await call_with_retry(fn, policy=RetryPolicy(), label="groq:x", sleep=rec.sleep)
    assert str(info.value) == VPN_MESSAGE
    assert isinstance(info.value.__cause__, StatusError)
    assert rec.sleeps == []


async def test_other_403_re_raised_unchanged() -> None:
    rec = Recorder()
    original = StatusError(403)
    fn, _ = failing([original])
    with pytest.raises(StatusError) as info:
        await call_with_retry(fn, policy=RetryPolicy(), label="ollama:x", sleep=rec.sleep)
    assert info.value is original


async def test_max_delay_cap() -> None:
    rec = Recorder()
    fn, _ = failing([StatusError(429) for _ in range(5)])
    policy = RetryPolicy(max_attempts=6, base_delay_s=1.0, max_delay_s=4.0)
    await call_with_retry(fn, policy=policy, label="groq:x", sleep=rec.sleep, rand=lambda: 1.0)
    assert rec.sleeps == [1.0, 2.0, 4.0, 4.0, 4.0]


async def test_zero_attempts_is_an_error() -> None:
    fn, calls = failing([])
    with pytest.raises(ConvoprobeError, match="max_attempts must be at least 1"):
        await call_with_retry(fn, policy=RetryPolicy(max_attempts=0), label="x")
    assert calls["n"] == 0
