"""A tiny local agent that streams its reply as server-sent events (SSE). Stdlib only.

POST /chat with {"text": str, "history": [{"role": "user" | "model", "text": str}],
"session": str}. The client sends its own history; "model" is the agent's role
name. The server rebuilds the conversation and runs the toy support agent, then
streams:

    data: {"type": "tool", "name": ..., "args": ..., "ok": true}   one per tool call
    data: {"type": "chunk", "text": ...}                            one per sentence
    data: {"type": "done", "text": <full reply>}

After more than --limit requests in one session it streams a "limited" event and
a done event instead of calling the agent. For local testing only: it binds to
127.0.0.1 by default.

    python examples/sse_agent/server.py --port 8766 --version v1
"""

from __future__ import annotations

import argparse
import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from convoprobe.datamodels import Message, ToolCallRecord
from convoprobe.demo.toy_agent import toy_agent_v1, toy_agent_v2

LIMITED_TEXT = "Too many requests in this session. Try again later."
# A sentence ends at . ! or ? followed by whitespace, so "$89.00" stays whole.
_SENTENCE = re.compile(r".+?(?:[.!?](?=\s)|$)\s*", re.DOTALL)


def sentences(text: str) -> list[str]:
    """Split a reply into sentence chunks that join back to the full text."""
    return _SENTENCE.findall(text) or [text]


def make_server(host: str, port: int, version: str, limit: int = 1000) -> ThreadingHTTPServer:
    """A server for toy agent v1 or v2. Port 0 picks a free port."""
    agent = toy_agent_v1 if version == "v1" else toy_agent_v2
    # Text-only history loses tool calls, but the toy agent's rules read earlier
    # tool calls, so the server keeps each session's calls per agent reply.
    session_tools: dict[str, list[list[ToolCallRecord]]] = {}
    requests: dict[str, int] = {}
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def _send_json(self, status: int, payload: Any) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _event(self, payload: dict[str, Any]) -> None:
            self.wfile.write(f"data: {json.dumps(payload)}\n\n".encode())
            self.wfile.flush()

        def do_POST(self) -> None:
            if self.path != "/chat":
                self._send_json(404, {"error": "not found"})
                return
            length = int(self.headers.get("Content-Length") or 0)
            try:
                request = json.loads(self.rfile.read(length) or b"{}")
                session, text = str(request["session"]), str(request["text"])
                history = [(str(h["role"]), str(h["text"])) for h in request.get("history", [])]
            except (ValueError, KeyError, TypeError):
                self._send_json(400, {"error": "expected JSON with session, text, history"})
                return
            with lock:
                requests[session] = requests.get(session, 0) + 1
                limited = requests[session] > limit
                if not limited:
                    tools = session_tools.setdefault(session, [])
                    messages: list[Message] = []
                    agent_index = 0
                    for role, message_text in history:
                        if role == "model":
                            calls = tools[agent_index] if agent_index < len(tools) else []
                            messages.append(
                                Message(role="agent", text=message_text, tool_calls=calls)
                            )
                            agent_index += 1
                        else:
                            messages.append(Message(role="user", text=message_text))
                    messages.append(Message(role="user", text=text))
                    turn = agent(messages)
                    del tools[agent_index:]
                    tools.append(list(turn.tool_calls))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            if limited:
                self._event({"type": "limited"})
                self._event({"type": "done", "text": LIMITED_TEXT})
                return
            for call in turn.tool_calls:
                self._event({"type": "tool", "name": call.name, "args": call.args, "ok": True})
            for chunk in sentences(turn.text):
                self._event({"type": "chunk", "text": chunk})
            self._event({"type": "done", "text": turn.text})

        def log_message(self, format: str, *args: Any) -> None:
            return  # keep test output quiet

    return ThreadingHTTPServer((host, port), Handler)


def main() -> None:
    parser = argparse.ArgumentParser(description="Toy support agent streaming over SSE.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--version", choices=["v1", "v2"], default="v1")
    parser.add_argument("--limit", type=int, default=1000, help="Requests per session.")
    args = parser.parse_args()
    server = make_server(args.host, args.port, args.version, args.limit)
    print(f"toy agent {args.version} streaming on http://{args.host}:{args.port}/chat")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
