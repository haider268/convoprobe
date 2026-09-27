# CLAUDE.md for convoprobe

convoprobe stress-tests conversational agents with simulated users. Scenario
YAML plus an agent adapter in; scored conversations, reports, and a CI exit
code out. Status: milestone 1 (core loop, scripted personas, offline demo).

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
- Write files with the file tool, not shell heredocs (a heredoc once
  corrupted a regex backslash).

## How to run

uv lives at `C:\Users\maria\.local\bin\uv.exe` if it is not on PATH.

```
uv sync                   # core + dev group, no extras
uv run convoprobe demo    # offline v1 vs v2 regression demo, zero keys
uv run convoprobe run src/convoprobe/demo/scenario.yaml \
    --adapter convoprobe.demo.toy_agent:toy_agent_v1
uv run pytest             # offline tests with coverage
uv run ruff check
uv run ruff format --check
uv run mypy src
```

`run` options: `--out results`, `--run-id` (reuse to resume), `--db
.convoprobe/convoprobe.db`, `--concurrency 4`. Exit 2 on scenario, adapter,
or store errors.

## Layout

- `datamodels.py`: all Pydantic models. Not `models.py`, because `models/` is
  the chat model factory (milestone 2).
- `scenario.py`: `load_scenario` / `parse_scenario`, errors as
  `<path>: <dotted.loc>: <msg>`.
- `errors.py`: `ConvoprobeError`, `TransientError` (retried on next run),
  `AdapterLoadError`.
- `adapters/`: `AgentAdapter` protocol (`start`, `send`, `end`, all async),
  `CallableAdapter` for `(history) -> str | AgentTurn` functions, sync or
  async, and `load_adapter("module:attr")`. Classes are rejected, not built.
- `simulator/simulator.py`: `UserSimulator` protocol, `ScriptedSimulator`.
- `scoring/deterministic.py`: the 8 non-LLM checks, dispatched via `_CHECKS`.
- `summary.py`: per-rubric counts and pass rates.
- `store.py`: `RunStore`, SQLite.
- `runner.py`: `run_scenario`.
- `jsonlog.py`: `JsonlLogger`, one JSON object per line with `run_id`.
- `report/json_report.py`: `write_results`, `print_run_report`.
- `demo/`: `toy_agent.py` (v1 correct, v2 refunds without lookup) and
  `scenario.yaml`. Package data, ships in the wheel.
- `cli.py`: `version`, `run`, `demo`.
- No module may shadow a stdlib name (hence `jsonlog.py`, `json_report.py`).

## Store schema

- `runs(run_id, scenario_name, scenario_hash, adapter, started_at, finished_at)`.
  Resuming a run id with a different scenario hash is an error.
- `conversations(run_id, conversation_id, persona_id, repeat_index, status,
  attempts, data_json, error, updated_at)`. Status is one of `pending`,
  `done`, `failed_transient`, `failed`, `skipped`.
- `check_results(run_id, conversation_id, rubric_id, status, detail)`,
  replaced per conversation on each save.
- All timestamps UTC ISO 8601.

## Runner flow

1. Create or resume the run. Conversation ids are `<persona>-r<repeat>`.
2. `generate: N` entries are noted, not run. LLM personas without a
   simulator are saved as `skipped` with a note.
3. Only `pending` and `failed_transient` conversations run, under a semaphore.
4. Loop up to `max_turns`: simulator message (None ends it), `adapter.send`,
   record latency and tool calls. `adapter.end` always runs.
5. `TransientError` saves `failed_transient` with the partial transcript.
   Any other exception saves `failed` with `agent_error` and still scores.
6. Results are rebuilt from the store, so resumed runs include earlier work.
   Writes `<out>/<run_id>/results.json` and `<out>/<run_id>/log.jsonl`.

## How to extend

- New check type: add a model to `datamodels.py`, add it to `RubricItem`
  and the tag set in `scenario.py`, write a function in
  `scoring/deterministic.py`, register it in `_CHECKS`, test pass and fail
  with exact detail strings.
- New adapter: implement the three async `AgentAdapter` methods. Raise
  `TransientError` for retryable failures.
- New provider: comes with the model factory in milestone 2.

## Contracts

- Core must import and test without any extra installed. Provider packages
  are imported lazily with an "install convoprobe[<extra>]" error.
- Test suite is offline: no sockets. HTTP tests use `httpx.MockTransport`.
- The demo needs nothing outside the package: no model, no key, no network.
