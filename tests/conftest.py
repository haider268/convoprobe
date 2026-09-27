"""Shared test setup. Every offline test is isolated from the real .env, its keys,
and the network."""

from __future__ import annotations

import asyncio
import socket
from pathlib import Path
from typing import Any

import pytest

OLLAMA_PORT = 11434
LOOPBACK = {"127.0.0.1", "::1", "localhost"}


class NetworkBlockedError(OSError):
    """An offline test tried to reach the network."""


@pytest.fixture(autouse=True)
def _no_real_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point settings at a missing env file and drop provider keys from the environment.

    Live tests opt back in by setting CONVOPROBE_ENV_FILE to the repo's .env.
    """
    monkeypatch.setenv("CONVOPROBE_ENV_FILE", str(tmp_path / "no.env"))
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)


@pytest.fixture(autouse=True)
def _no_network(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """Block DNS and outbound connects in every test not marked live.

    Loopback connects stay allowed because asyncio on Windows builds its event
    loop's self-pipe from a loopback socket pair. The Ollama port is blocked.
    """
    if request.node.get_closest_marker("live"):
        return
    real_connect = socket.socket.connect

    def guarded_connect(self: socket.socket, address: Any) -> None:
        if isinstance(address, tuple) and address[0] in LOOPBACK and address[1] != OLLAMA_PORT:
            real_connect(self, address)
            return
        raise NetworkBlockedError(f"network blocked in offline tests: {address!r}")

    def blocked_getaddrinfo(*args: Any, **kwargs: Any) -> Any:
        raise NetworkBlockedError(f"DNS blocked in offline tests: {args[:2]!r}")

    async def blocked_loop_connect(self: Any, *args: Any, **kwargs: Any) -> Any:
        raise NetworkBlockedError(f"asyncio connect blocked in offline tests: {args[1:3]!r}")

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    monkeypatch.setattr(socket, "getaddrinfo", blocked_getaddrinfo)
    # On Windows the Proactor loop connects with ConnectEx, bypassing socket.connect.
    monkeypatch.setattr(asyncio.BaseEventLoop, "create_connection", blocked_loop_connect)
    monkeypatch.setattr(asyncio.BaseEventLoop, "sock_connect", blocked_loop_connect)
