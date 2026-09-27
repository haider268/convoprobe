"""The example HTTP server, on a loopback port chosen by the OS."""

from __future__ import annotations

import http.client
import json
import threading
from collections.abc import Iterator
from http.server import ThreadingHTTPServer
from typing import Any

import pytest
from examples.http_agent.server import make_server


def serve(version: str) -> tuple[ThreadingHTTPServer, threading.Thread]:
    server = make_server("127.0.0.1", 0, version)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


@pytest.fixture
def server() -> Iterator[ThreadingHTTPServer]:
    running, thread = serve("v1")
    yield running
    running.shutdown()
    running.server_close()
    thread.join(timeout=5)


def request(
    server: ThreadingHTTPServer, method: str, path: str, body: Any = None
) -> tuple[int, Any]:
    host, port = server.server_address[:2]
    conn = http.client.HTTPConnection(str(host), int(port), timeout=5)
    try:
        payload = None if body is None else json.dumps(body)
        conn.request(method, path, body=payload, headers={"Content-Type": "application/json"})
        response = conn.getresponse()
        return response.status, json.loads(response.read() or b"null")
    finally:
        conn.close()


def test_chat_and_per_session_history(server: ThreadingHTTPServer) -> None:
    status, first = request(server, "POST", "/chat", {"session": "a", "message": "refund 1042"})
    assert status == 200
    assert first["reply"].startswith("I looked up order 1042")
    assert [c["name"] for c in first["tool_calls"]] == ["lookup_order", "issue_refund"]
    assert first["tool_calls"][0]["result"]["found"] is True

    _, again = request(server, "POST", "/chat", {"session": "a", "message": "thanks"})
    assert again["reply"].startswith("Your refund is already in progress.")
    _, other = request(server, "POST", "/chat", {"session": "b", "message": "thanks"})
    assert other["reply"].startswith("Hi, I can help")


def test_v2_server() -> None:
    running, thread = serve("v2")
    try:
        _, reply = request(running, "POST", "/chat", {"session": "a", "message": "refund 1042"})
        assert reply["reply"] == "Done, I have issued your refund."
    finally:
        running.shutdown()
        running.server_close()
        thread.join(timeout=5)


def test_404_and_bad_request(server: ThreadingHTTPServer) -> None:
    assert request(server, "POST", "/other", {"session": "a", "message": "x"})[0] == 404
    assert request(server, "GET", "/chat")[0] == 404
    assert request(server, "POST", "/chat", {"session": "a"})[0] == 400
