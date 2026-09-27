# CLAUDE.md for convoprobe

convoprobe stress-tests conversational agents with simulated users. Scenario
YAML plus an agent adapter in; scored conversations, reports, and a CI exit
code out. Status: scaffold (step 0). This file grows with each milestone.

## Hard rules

- Git identity is local to this repo: `haider268` /
  `haider268@users.noreply.github.com`. The global identity is a company
  account and must never appear in history. Check with
  `git log --format="%an <%ae>" | sort -u` before any checkpoint.
- Never `git remote add`, never `git push`, never upload to PyPI or TestPyPI.
  Tags are local only. Pushes happen from the review lane on Haider's word.
- Never download anything over about 200 MB from the terminal (Ollama
  installer, `ollama pull`, torch, CUDA, Docker images). Stop and ask.
- No system installs. Everything goes through uv into `.venv`.
- Keys only in a git-ignored `.env`. Never print, log, or commit a key.
- Fake data only: fixture people are obviously fake, domains use `.example`.
- Rules in code, language in the LLM. Anything decidable in Python is Python.

## How to run

uv lives at `C:\Users\maria\.local\bin\uv.exe` if it is not on PATH.

```
uv sync              # core + dev group, no extras
uv run pytest        # offline tests with coverage
uv run ruff check
uv run ruff format --check
uv run mypy src
```

## Layout

- `src/convoprobe/datamodels.py`: Pydantic data models. Not `models.py`,
  because `models/` is the chat model factory.
- `src/convoprobe/models/`: `provider:model` string to chat model, replay model.
- `src/convoprobe/adapters/`: agent adapters (callable now, LangGraph and HTTP in M3).
- `src/convoprobe/simulator/`, `scoring/`, `report/`: simulator, checks, reports.
- `src/convoprobe/demo/`: offline demo agent, scenario, cassette. Package data.
- `store.py` SQLite state and cache, `runner.py` orchestration,
  `scenario.py` YAML loading, `config.py` settings, `logging.py` JSONL logs,
  `cli.py` typer app.

## Contracts

- Core must import and test without any extra installed. Provider packages
  are imported lazily with an "install convoprobe[<extra>]" error.
- Test suite is offline: no sockets. HTTP tests use `httpx.MockTransport`.
