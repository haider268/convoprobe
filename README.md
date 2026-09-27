# convoprobe

Stress-test conversational agents with simulated users.

Status: early scaffold. Nothing here is usable yet.

convoprobe will run multi-turn conversations between a user-simulator LLM and
an agent, score each conversation with deterministic checks and an LLM judge,
write reports, and compare against a saved baseline so it can gate CI.

This project is a Python library. It has no Docker image.

## Development

```
uv sync
uv run pytest
uv run ruff check
uv run mypy src
```

## License

MIT. See [LICENSE](LICENSE).
