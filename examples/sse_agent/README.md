# Toy support agent over SSE

A local stdlib server that wraps the demo's toy support agent and streams each
reply as server-sent events (SSE). It shows how to test an agent that streams
its answer and takes the conversation history in its own format.

## The endpoint

`POST /chat` with:

```json
{"session": "...", "text": "new user message",
 "history": [{"role": "user", "text": "..."}, {"role": "model", "text": "..."}]}
```

The history holds only earlier turns, and the agent's role is called `model`.
The reply is a stream of `data: <json>` lines:

```
data: {"type": "tool", "name": "lookup_order", "args": {"order_id": "1042"}, "ok": true}

data: {"type": "chunk", "text": "Order 1042 (Trail Jacket) is delivered. "}

data: {"type": "done", "text": "<the full reply>"}
```

After more than `--limit` requests in one session (default 1000), the server
streams `{"type": "limited"}` and a done event instead of answering. It binds to
`127.0.0.1` and is for local testing only.

## The adapter config

`adapter.yaml` maps all of that:

- `response_mode: sse` reads the stream. The reply is the `text` of the last
  `done` event.
- `sse_tool_event: tool` turns each tool event into a tool call, and
  `sse_tool_result_fields: [ok]` keeps its `ok` field as the result.
- `sse_transient_events: [limited]` treats `limited` as a rate limit: the
  conversation ends as `failed_transient` and runs again on a rerun with the
  same run id.
- `history_roles: {user: user, agent: model}` and
  `history_includes_current: false` send the history in the agent's format.
- `headers` may use `{{conversation_id}}` (filled per request) and `${NAME}`
  (an environment variable). Header values are never printed.

## Run it

Start the server in one terminal, from the repo root:

```
uv run python examples/sse_agent/server.py --port 8766 --version v1
```

Run convoprobe in another:

```
uv run convoprobe run src/convoprobe/demo/scenario.yaml \
    --adapter http:examples/sse_agent/adapter.yaml --no-llm \
    --out results --db .convoprobe/sse.db
```

Stop the server with Ctrl+C. Start it with `--version v2` to see the
deliberate refund bug caught.
