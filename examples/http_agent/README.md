# Toy support agent over HTTP

A local stdlib HTTP server that wraps the demo's toy support agent, and the
adapter config that points convoprobe at it. It shows how to test an agent
that only exposes an HTTP endpoint.

## The endpoint

`POST /chat` with `{"session": "...", "message": "..."}` returns
`{"reply": "...", "tool_calls": [{"name", "args", "result"}]}`. The server keeps
each session's history in memory. Any other path returns 404. It binds to
`127.0.0.1` and is for local testing only.

## The adapter config

`adapter.yaml`:

```yaml
type: http
url: http://127.0.0.1:8765/chat
request_template:
  session: "{{conversation_id}}"
  message: "{{message}}"
text_path: reply
tool_calls_path: tool_calls
```

- `request_template` is sent as JSON. String values that are exactly
  `{{message}}`, `{{conversation_id}}` or `{{history}}` are replaced.
- `text_path` and `tool_calls_path` are dotted paths into the JSON response,
  for example `data.choices.0.text`.
- `headers` may use `${NAME}` to read an environment variable, for example
  `Authorization: "Bearer ${AGENT_TOKEN}"`. Values are never printed.
- 408, 429 and 5xx responses are retried with backoff.

## Run it

Start the server in one terminal, from the repo root:

```
uv run python examples/http_agent/server.py --port 8765 --version v1
```

Run convoprobe in another:

```
uv run convoprobe run src/convoprobe/demo/scenario.yaml \
    --adapter http:examples/http_agent/adapter.yaml --no-llm \
    --out results --db .convoprobe/m3.db
```

Stop the server with Ctrl+C. Start it with `--version v2` to see the
deliberate refund bug caught.
