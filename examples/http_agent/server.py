"""A tiny local HTTP server that wraps the toy support agent. Stdlib only.

POST /chat with {"session": str, "message": str} returns
{"reply": str, "tool_calls": [{"name", "args", "result"}]}. History is kept in
memory per session. For local testing only: it binds to 127.0.0.1 by default.

    python examples/http_agent/server.py --port 8765 --version v1
"""

from __future__ import annotations

import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from convoprobe.datamodels import Message
from convoprobe.demo.toy_agent import toy_agent_v1, toy_agent_v2


def make_server(host: str, port: int, version: str) -> ThreadingHTTPServer:
    """A server for toy agent v1 or v2. Port 0 picks a free port."""
    agent = toy_agent_v1 if version == "v1" else toy_agent_v2
    sessions: dict[str, list[Message]] = {}
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def _send_json(self, status: int, payload: Any) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:
            if self.path != "/chat":
                self._send_json(404, {"error": "not found"})
                return
            length = int(self.headers.get("Content-Length") or 0)
            try:
                request = json.loads(self.rfile.read(length) or b"{}")
                session, message = str(request["session"]), str(request["message"])
            except (ValueError, KeyError, TypeError):
                self._send_json(400, {"error": "expected JSON with session and message"})
                return
            with lock:
                history = sessions.setdefault(session, [])
                history.append(Message(role="user", text=message))
                turn = agent(list(history))
                history.append(Message(role="agent", text=turn.text, tool_calls=turn.tool_calls))
            self._send_json(
                200,
                {
                    "reply": turn.text,
                    "tool_calls": [call.model_dump(mode="json") for call in turn.tool_calls],
                },
            )

        def do_GET(self) -> None:
            self._send_json(404, {"error": "not found"})

        def log_message(self, format: str, *args: Any) -> None:
            return None

    return ThreadingHTTPServer((host, port), Handler)


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the toy support agent over HTTP.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--version", choices=["v1", "v2"], default="v1")
    args = parser.parse_args()
    server = make_server(args.host, args.port, args.version)
    host, port = server.server_address[:2]
    print(f"toy agent {args.version} serving on http://{host}:{port}/chat (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
