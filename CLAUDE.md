# CLAUDE.md for convoprobe

convoprobe stress-tests conversational agents with simulated users. Scenario
YAML plus an agent adapter in; scored conversations, reports, and a CI exit
code out. Status: milestone 4 (HTML report, baselines, compare).

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
- User-facing text (CLI, HTML): plain, short, no em dashes.

## How to run

uv lives at `C:\Users\maria\.local\bin\uv.exe` if it is not on PATH.

```
uv sync --all-extras                 # as CI does
uv run convoprobe demo [--out DIR]   # replays the recorded cassette, no keys
uv run convoprobe run SCENARIO --adapter module:attr [--no-llm | --hosted |
    --replay CASSETTE | --simulator P:M --judge P:M] [--repeats N] [--html]
uv run convoprobe report RESULTS [--html] [--out PATH]
uv run convoprobe baseline save RESULTS [--path .convoprobe/baseline.json]
uv run convoprobe compare RESULTS [--baseline PATH] [--threshold 0..1] [--json]
uv run convoprobe personas generate SCENARIO -n 4 [--hosted | --replay C]
uv run convoprobe cassette export --run-id ID [--run-id ID] --out FILE
uv run convoprobe demo --record      # live, needs GROQ_API_KEY
uv run pytest                        # offline; live tests excluded
uv run pytest -m live -q --no-cov    # live: needs Ollama (and .env for Groq)
uv run ruff check && uv run ruff format --check && uv run mypy src
```

`run` modes: default is live Ollama (`qwen2.5:3b` / `llama3.2:3b`, preflighted;
a live Ollama judge prints a note, a 3B judge gave wrong verdicts in review).
`--hosted` uses `groq:openai/gpt-oss-20b` / `groq:openai/gpt-oss-120b` (judge
4/4 valid on 28 Sep 2026, qwen3.8-27b 1/4; for real agents pick a judge family
different from the agent). `--replay` needs no key or provider. `--no-llm`
cannot be combined with model options. Simulator 0.7, judge 0.0 temperature.
RESULTS is a run folder or its `results.json`.

## Reports, baselines, compare (the CI gate)

- Terminal report: rubric table, pass rate by persona (numbered columns plus a
  legend so it fits 80 columns), models, usage, counts, notes.
- `report/html.py` + `report/templates/report.html.j2`: one self-contained
  file, inline CSS, light and dark, no script or external URL, autoescaped.
  Failed conversations first and open; evidence links jump to
  `<conversation id>-turn-<N>`. Checked at 375 px: only tables scroll.
- `baseline.py`: `baseline save` freezes rubric, persona, and overall rates
  plus the scenario's `pass_rate_drop`. `compare` gates a later run:
  - drop greater than the threshold (default `pass_rate_drop`) = regression;
    exactly the threshold is OK;
  - a rubric item with a baseline rate but no results now = missing, fails;
  - overall drop greater than the threshold fails too;
  - rise greater than the threshold = improved; items only in the new run =
    new; neither gates. A changed scenario is noted, never gating alone.
- Exit codes: 0 no regression, 1 regression, 2 usage or config error.
- `.convoprobe/` is git-ignored. A baseline meant to be committed is saved
  outside it with `--path` (for example `ci/baseline.json`).

## Adapters

`--adapter` takes `module.path:attr` (the cwd is put on `sys.path`, so
`examples.toy_support_agent.graph:adapter_v1` works from the repo root) or
`http:<config.yaml>`. A compiled LangGraph graph is wrapped automatically.
- `CallableAdapter`: `(history) -> str | AgentTurn`, sync or async.
- `LangGraphAdapter(graph, resume=...)`: needs a checkpointer; a fresh thread
  per conversation start. Interrupts resume with `resume` (value or callable
  given the payload); `None` makes an interrupt an error; `max_resumes` caps a
  turn. Node path, interrupts, resumes go to message `metadata`. langgraph is
  imported only inside methods.
- `HTTPAdapter`: YAML (`url`, `request_template` with whole-value `{{message}}`
  / `{{conversation_id}}` / `{{history}}`, dotted `text_path` and
  `tool_calls_path`, `headers` with `${ENV}`). 408/429/500/502/503/504 retry.
- `tests/test_adapter_equivalence.py` runs the demo scenario through all three
  shapes (plain function, LangGraph graph, HTTP server in `examples/`) and
  requires identical results, then replays the cassette through the graph.
  It proves an adapter is faithful; keep it green.
- The toy agent reads only whole 4-digit order ids, kept on purpose as a demo finding.

## Layout

- `datamodels.py` (all Pydantic models), `config.py` (`Settings`,
  `load_settings()` reads `CONVOPROBE_ENV_FILE`, default `.env`), `scenario.py`,
  `errors.py`, `store.py`, `runner.py`, `summary.py`, `baseline.py`,
  `jsonlog.py`, `cli.py`, `report/` (`json_report.py`, `html.py`, `load.py`).
- `scoring/deterministic.py` (8 checks) and `scoring/judge.py`.
- `simulator/`: `simulator.py` (protocol, scripted), `llm.py`, `personas.py`.
- `demo/`: toy agent (`decide()` plus reply texts, shared by every agent shape),
  `scenario.yaml`, `cassette.json`, `DEMO_*`. All package data.
- No module may shadow a stdlib or dependency name.

## LLM layer (models/)

- `factory.py`: `parse_spec` splits on the first colon; providers load lazily.
- `client.py`: cache key = spec, method, kind, temperature, schema, messages,
  and a **salt** (conversation id + turn) so repeats sample independently. Bad
  structured output (or `tool_use_failed` / `json_validate_failed`) raises
  `StructuredOutputError` and is never cached. `json_schema` for Ollama and Groq.
- `cache.py`: `SqliteLLMCache` records which run used which key;
  `export_cassette(run_ids, path, meta)`; `JsonCassette` is read-only.
- `retry.py`: 408/429/5xx, timeouts, connection errors retry with jittered
  backoff. Groq 403 = VPN. 429 "Request too large" is not retried.
- `ratelimit.py`, `registry.py` (one limiter per Groq/Gemini model, since their
  limits are per model; one shared Ollama limiter), `preflight.py`.
  `GROQ_MAX_TOKENS` sets `max_tokens`; qwen3.8-27b needs it below 1000.

## Simulator, judge, personas

- `LLMSimulator` flips roles (its own lines are AIMessages) so a small model
  does not play the agent. `SimTurn` stops on achieved/impossible only after an
  agent reply. One retry on bad output, then `simulator_error` (no checks).
- Judge: reasoning first; transcript lines `turn N USER|AGENT|AGENT TOOL CALL`.
  Evidence turns outside the transcript are invalid; two invalid attempts give
  `error`, never a pass. `quote_found` checks each quote. No docstrings on
  schema classes: pydantic sends them to the model.
- Personas: `generate: N` uses the simulator client, ids `gen_<slug>`, saved
  per run so a resume reuses them.

## Demo cassette, runner, store

- `demo --record` (live Groq) writes `demo/cassette.json` only if every
  conversation is `done`. `demo` replays it with fixed `DEMO_*` specs. Changing
  a prompt, schema, salt, or the demo scenario changes cache keys: re-record.
- Resume by run id (`<persona>-r<repeat>` ids); `pending`, `failed_transient`,
  and `skipped` LLM personas run. Judge `TransientError` = `failed_transient`,
  replay miss = `failed`. SQLite tables include `check_results.extra_json`
  (migrated), `generated_personas`, `llm_cache`, `llm_cache_usage`.

## How to extend

- New check: model in `datamodels.py`, add to `RubricItem` and the tag set in
  `scenario.py`, function in `scoring/deterministic.py`, register in `_CHECKS`.
- New adapter: the three async `AgentAdapter` methods; raise `TransientError`
  for retryable failures, and add it to the equivalence test.
- New provider: `models/factory.py` (`PROVIDERS`, `_CLASSES`), an extra, limits
  in `config.py` and `LLMRegistry.limiter`, `registry.STRUCTURED_METHOD`.

## Contracts

- Core imports without extras (no provider, no langgraph); subprocess tests check.
- Offline tests: `conftest.py` hides `.env` and blocks all but loopback, and the
  Ollama port. The demo needs nothing outside the package: no key, no network.
