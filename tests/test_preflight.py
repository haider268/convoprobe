"""check_ollama against httpx.MockTransport. No sockets."""

from __future__ import annotations

import httpx
import pytest

from convoprobe.errors import ConvoprobeError
from convoprobe.models.preflight import check_ollama

BASE = "http://ollama.example:11434"
UNREACHABLE = f"Ollama is not reachable at {BASE}. Start Ollama, or rerun with --replay <cassette>."


def tags(*names: str) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/tags"
        return httpx.Response(200, json={"models": [{"name": n} for n in names]})

    return httpx.MockTransport(handler)


async def test_ok() -> None:
    await check_ollama(
        BASE, ["qwen2.5:3b", "llama3.2:3b"], transport=tags("qwen2.5:3b", "llama3.2:3b")
    )


async def test_one_missing() -> None:
    with pytest.raises(ConvoprobeError) as info:
        await check_ollama(BASE, ["qwen2.5:3b", "llama3.2:3b"], transport=tags("qwen2.5:3b"))
    assert (
        str(info.value) == 'Ollama model "llama3.2:3b" is not pulled. Run: ollama pull llama3.2:3b'
    )


async def test_two_missing_in_one_message() -> None:
    with pytest.raises(ConvoprobeError) as info:
        await check_ollama(BASE, ["qwen2.5:3b", "llama3.2:3b"], transport=tags("other:1b"))
    assert str(info.value).splitlines() == [
        'Ollama model "qwen2.5:3b" is not pulled. Run: ollama pull qwen2.5:3b',
        'Ollama model "llama3.2:3b" is not pulled. Run: ollama pull llama3.2:3b',
    ]


async def test_connection_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    with pytest.raises(ConvoprobeError) as info:
        await check_ollama(BASE, ["qwen2.5:3b"], transport=httpx.MockTransport(handler))
    assert str(info.value) == UNREACHABLE


async def test_server_error() -> None:
    transport = httpx.MockTransport(lambda request: httpx.Response(500, text="boom"))
    with pytest.raises(ConvoprobeError) as info:
        await check_ollama(BASE, ["qwen2.5:3b"], transport=transport)
    assert str(info.value) == UNREACHABLE


async def test_malformed_body() -> None:
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json={"nope": []}))
    with pytest.raises(ConvoprobeError) as info:
        await check_ollama(BASE, [], transport=transport)
    assert str(info.value) == UNREACHABLE
