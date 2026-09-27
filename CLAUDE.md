# CLAUDE.md for convoprobe

convoprobe stress-tests conversational agents with simulated users. Scenario
YAML plus an agent adapter in; scored conversations, reports, and a CI exit
code out. Status: milestone 2 (LLM simulator, judge, personas, replay).

## Hard rules

- Git identity is local: `haider268` / `haider268@users.noreply.github.com`.
  The global identity is a company account and must never appear in history.
  Check `git log --format="%an <%ae>" | sort -u` before any checkpoint.
- Never `git remote add`, `git push`, or upload to PyPI/TestPyPI. Local tags only.
- Never download anything over about 200 MB from the terminal (Ollama
  installer, `ollama pull`, torch, CUDA, Docker images). Stop and ask.
- No system installs. Everything goes through uv into `.venv`.
- Keys only in the git-ignored `.env`. Never print, log, read out, or commit one.
- Fake data only: fixture people are obviously fake, domains use `.example`.
- Rules in code, language in the LLM. Anything decidable in Python is Python.
- Write files with the file tool, not shell heredocs (one corrupted a regex).
- Live runs: VPN OFF for Groq (it answers 403 to VPN and datacenter IPs).

## How to run

uv lives at `C:\Users\maria\.local\bin\uv.exe` if it is not on PATH.

```
uv sync --all-extras                 # as CI does
uv run convoprobe demo               # replays the recorded cassette, no keys
uv run convoprobe run SCENARIO --adapter module:attr [--no-llm | --hosted |
    --replay CASSETTE | --simulator P:M --judge P:M] [--repeats N]
uv run convoprobe personas generate SCENARIO -n 4 [--hosted | --replay C]
uv run convoprobe cassette export --run-id ID [--run-id ID] --out FILE
uv run convoprobe demo --record      # live, needs GROQ_API_KEY
uv run pytest                        # offline; live tests excluded
uv run pytest -m live -q --no-cov    # live: needs Ollama (and .env for Groq)
uv run ruff check && uv run ruff format --check && uv run mypy src
```

`run` modes: default is live with `CONVOPROBE_SIMULATOR` / `CONVOPROBE_JUDGE`
(Ollama `qwen2.5:3b` / `llama3.2:3b`, preflighted with `check_ollama`).
`--hosted` fills unset models with `HOSTED_SIMULATOR` (`groq:openai/gpt-oss-20b`)
and `HOSTED_JUDGE` (`groq:openai/gpt-oss-120b`). The judge is the strongest
available model, measured 4/4 valid outputs on 28 Sep 2026; qwen3.8-27b was 1/4.
For real agents it should also be a different family from the agent under test.
`--replay` reads a cassette, needs no key or provider package. `--no-llm` is the
M1 behaviour and cannot be combined with model options. Simulator runs at
temperature 0.7, judge at 0.0. The LLM cache shares the `--db` file.
Exit 2 on scenario, adapter, model, key, Ollama, or store errors.

## Layout

- `datamodels.py`: all Pydantic models (`models/` is the LLM package).
- `config.py`: `Settings`, `load_settings()` (reads `CONVOPROBE_ENV_FILE`,
  default `.env`), `HOSTED_*`. Blank env values count as unset.
- `scenario.py`, `errors.py`, `adapters/`, `store.py`, `runner.py`,
  `summary.py`, `jsonlog.py`, `report/json_report.py`, `cli.py`.
- `scoring/deterministic.py` (8 checks) and `scoring/judge.py`.
- `simulator/`: `simulator.py` (protocol, scripted), `llm.py`, `personas.py`.
- `demo/`: toy agent, `scenario.yaml`, `cassette.json`, `DEMO_*` constants.
  All package data. No module may shadow a stdlib name.

## LLM layer (models/)

- `factory.py`: `parse_spec` splits on the first colon; `make_chat_model`
  imports the provider lazily, `max_retries=0`.
- `client.py`: `LLMClient.text()` / `.structured(schema)`. Cache key = spec,
  method, kind, temperature, schema, messages, and a **salt** (conversation id
  + turn) so repeats get independent samples. Bad structured output, or a
  provider `tool_use_failed` / `json_validate_failed`, is `StructuredOutputError`
  and never cached. `json_schema` method for Ollama and Groq.
- `cache.py`: `SqliteLLMCache` records which run used which key;
  `export_cassette(run_ids, path, meta)`; `JsonCassette` is read-only, `.meta`.
- `retry.py`: 408/429/5xx, timeouts, connection errors retry with jittered
  backoff. Groq 403 = VPN. 429 "Request too large" is not retried.
- `ratelimit.py`, `registry.py` (one limiter per provider), `preflight.py`.

## Simulator, judge, personas

- `LLMSimulator` flips roles: its own past lines are AIMessages, the agent's
  are HumanMessages, so a small model does not play the agent. It returns a
  `SimTurn` (`goal_status`, `message`); achieved/impossible stops only after
  the agent replied once. One retry on bad output; a second failure ends the
  conversation as `simulator_error` (status `failed`, no checks).
- Judge: `YesNoVerdict` / `ScaleVerdict`, reasoning first. Transcript lines
  are `turn N USER|AGENT|AGENT TOOL CALL: ...`; a user message and the reply
  after it share N. Evidence turns outside the transcript make an attempt
  invalid; two invalid attempts give status `error`, never a pass.
  `quote_found` checks each quote against its turn. No docstrings on schema
  classes: pydantic sends them to the model.
- Personas: `generate: N` uses the simulator client, ids `gen_<slug>`, saved
  per run so a resume reuses them.

## Replay and the demo cassette

- `demo --record` runs v1 and v2 live on Groq (run ids `demo-v1`, `demo-v2`,
  db `.convoprobe/demo-record.db`) and writes `demo/cassette.json` only if
  every conversation is `done`. Otherwise it lists failures and exits 1.
- `demo` replays that cassette with `DEMO_SIMULATOR` / `DEMO_JUDGE`, which are
  fixed constants, not `HOSTED_*`: the cache keys include the spec.
- Anything that changes a prompt, schema, salt, or the demo scenario changes
  cache keys. Re-record the cassette after such a change.

## Runner flow

1. Create or resume the run. Conversation ids are `<persona>-r<repeat>`.
2. Expand `generate:` entries (or reuse saved ones). LLM personas without a
   simulator are `skipped`, and run on a later resume that has one.
3. Only `pending`, `failed_transient`, and rerunnable `skipped` conversations run.
4. Deterministic checks, then judge checks, saved together. Judge
   `TransientError` = `failed_transient`, judge replay miss = `failed`.
5. Results come from the store, with repeat stdev, per-persona rates,
   `models`, and `llm_usage`.

## Store schema

`runs`, `conversations` (status `pending|done|failed_transient|failed|skipped`),
`check_results` (+ `extra_json` for judge fields, added by migration),
`generated_personas`, `llm_cache`, `llm_cache_usage`. Timestamps UTC ISO 8601.

## How to extend

- New check: model in `datamodels.py`, add to `RubricItem` and the tag set in
  `scenario.py`, function in `scoring/deterministic.py`, register in `_CHECKS`.
- New adapter: the three async `AgentAdapter` methods; raise `TransientError`
  for retryable failures.
- New provider: `PROVIDERS` and `_CLASSES` in `models/factory.py`, an extra in
  `pyproject.toml`, limiter settings in `config.py` and `LLMRegistry.limiter`,
  its structured method in `registry.STRUCTURED_METHOD`.

## Contracts

- Core imports without extras; a subprocess test checks no provider loads.
- Offline tests: `conftest.py` blocks DNS and non-loopback connects, blocks the
  Ollama port, and hides `.env`. HTTP tests use `httpx.MockTransport`.
- The demo needs nothing outside the package: no key, no network.
