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

    Loopback stays allowed: asyncio on Windows builds its event loop's self-pipe
    from a loopback socket pair, and some tests start their own local server.
    The Ollama port is blocked even on loopback.
    """
    if request.node.get_closest_marker("live"):
        return
    real_connect = socket.socket.connect
    real_getaddrinfo = socket.getaddrinfo
    real_create_connection = asyncio.BaseEventLoop.create_connection
    real_sock_connect = asyncio.BaseEventLoop.sock_connect

    def allowed(host: Any, port: Any) -> bool:
        return host in LOOPBACK and port != OLLAMA_PORT

    def guarded_connect(self: socket.socket, address: Any) -> None:
        if isinstance(address, tuple) and allowed(address[0], address[1]):
            real_connect(self, address)
            return
        raise NetworkBlockedError(f"network blocked in offline tests: {address!r}")

    def guarded_getaddrinfo(host: Any, port: Any, *args: Any, **kwargs: Any) -> Any:
        if allowed(host, port):
            return real_getaddrinfo(host, port, *args, **kwargs)
        raise NetworkBlockedError(f"DNS blocked in offline tests: {(host, port)!r}")

    async def guarded_create_connection(
        self: Any, protocol_factory: Any, host: Any = None, port: Any = None, **kwargs: Any
    ) -> Any:
        if allowed(host, port):
            return await real_create_connection(self, protocol_factory, host, port, **kwargs)
        raise NetworkBlockedError(f"asyncio connect blocked in offline tests: {(host, port)!r}")

    async def guarded_sock_connect(self: Any, sock: Any, address: Any) -> Any:
        if isinstance(address, tuple) and allowed(address[0], address[1]):
            return await real_sock_connect(self, sock, address)
        raise NetworkBlockedError(f"asyncio connect blocked in offline tests: {address!r}")

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    monkeypatch.setattr(socket, "getaddrinfo", guarded_getaddrinfo)
    # On Windows the Proactor loop connects with ConnectEx, bypassing socket.connect.
    monkeypatch.setattr(asyncio.BaseEventLoop, "create_connection", guarded_create_connection)
    monkeypatch.setattr(asyncio.BaseEventLoop, "sock_connect", guarded_sock_connect)
