# CLAUDE.md for convoprobe

convoprobe stress-tests conversational agents with simulated users. Scenario
YAML plus an agent adapter in; scored conversations, reports, and a CI exit
code out. Status: 0.1.0 (README, example CI gate in `ci/`, judge calibration).

## Hard rules

- Git identity is local: `haider268` / `haider268@users.noreply.github.com`
  (the global one is a company account). Check `git log --format="%an <%ae>"`.
- Never push, add a remote, or upload to PyPI. No downloads over about 200 MB.
- No system installs; everything goes through uv. Rules in code, language in the LLM.
- Keys only in the git-ignored `.env`. Never print, log, read out, or commit one.
- Fake data only (`.example` domains). Write files with the file tool, not heredocs.
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
uv run convoprobe calibrate prepare --from RESULTS [--from ...] --out DIR [--max 30]
uv run convoprobe calibrate score --set DIR --scenario S [--scenario S] --judge P:M
    --out FILE [--replay CASSETTE | --db PATH]
uv run convoprobe calibrate table FILE [FILE ...] [--out F] [--title T] [--note N]
    [--append] [--no-table] [--footer]
uv run convoprobe personas generate SCENARIO -n 4 [--hosted | --replay C]
uv run convoprobe cassette export --run-id ID [--run-id ID] --out FILE
uv run convoprobe demo --record      # live, needs GROQ_API_KEY
uv run pytest                        # offline; live tests excluded
uv run pytest -m live -q --no-cov    # live: needs Ollama (and .env for Groq)
uv run ruff check && uv run ruff format --check && uv run mypy src
```

`run` modes: default live Ollama (`qwen2.5:3b` / `llama3.2:3b`, preflighted; a
live Ollama judge prints a reliability note). `--hosted`: Groq `gpt-oss-20b` /
`gpt-oss-120b` (for real agents pick a judge family different from the agent).
Simulator 0.7, judge 0.0. RESULTS is a run folder or its `results.json`.

## Reports, baselines, compare (the CI gate)

- Terminal report: rubric table, persona matrix (numbered columns, 80 wide).
- `report/html.py` + `templates/report.html.j2`: one file, inline CSS, light
  and dark, no script or external URL, autoescaped. Failed conversations first
  and open; evidence links jump to `<conversation id>-turn-<N>`.
- `baseline.py`: `baseline save` freezes rubric, persona, and overall rates
  plus the scenario's `pass_rate_drop`. `compare` gates a later run:
  a drop greater than the threshold (default `pass_rate_drop`; exactly the
  threshold is OK), a missing rubric item, or an overall drop fails. Improved
  and new items never gate; a changed scenario is only noted.
- Exit codes: 0 no regression, 1 regression, 2 usage or config error.
- `.convoprobe/` is git-ignored: save a baseline meant to be committed with
  `--path` outside it (for example `ci/baseline.json`).

## Clinic example and calibration

- `examples/clinic_agent/` (fictional clinic, Groq, lazy client): `agent_strict`
  gives no advice or outcome estimates; `agent_loose` shares typical outcomes to
  produce failures. Say "candidacy call" only (a rubric regex checks the other
  word). Contact details are reserved fakes (`.example`, 555-0100 to 0199).
- Flow (`calibration.py`): **prepare** writes a blind set (refuses to overwrite
  labels or to write real-looking emails or phones); **labels** are filled;
  **score** runs one judge over every label (salt `calib:<item>:<rubric>`, errors
  recorded) into a `CalibrationReport` (agreement, Cohen's kappa on pass/fail
  decisions, exact / within 1 for scales); **table** writes `RESULTS.md`.
- Blind rule: the set never shows a check result, verdict, or score. Wording:
  labels were drafted by Claude without seeing judge output, then reviewed and
  approved by a human (never plain "human labels"). Never edit label values.
- Reproduce offline: `calibrate score ... --replay calibration/cassette.json`
  (commands in `RESULTS.md`). Live runs use `--db`, run id `calib-<slug>`, then
  `cassette export`. qwen3.8-27b needs `GROQ_MAX_TOKENS=800`; gpt-oss must not.
- Finding: judges counted "many patients see relief" as medical advice (criterion
  bleed). Fixed in the question wording, not the labels; both tables are kept.
- Only hosted judges are calibrated; the local 3B judge is too slow on CPU.
- Replay fallback (judge, simulator, personas): invalid output is never cached,
  so on a replay miss of the first attempt, try the `:retry1` salt the live run
  used; a second miss raises the first. Live calls never do this.

## Adapters

`--adapter` takes `module.path:attr` (the cwd is on `sys.path`, so
`examples.toy_support_agent.graph:adapter_v1` works from the repo root) or
`http:<config.yaml>`. A compiled LangGraph graph is wrapped automatically.
- `CallableAdapter`: `(history) -> str | AgentTurn`, sync or async.
- `LangGraphAdapter(graph, resume=...)`: checkpointer, fresh thread per
  conversation, interrupts resumed with `resume` up to `max_resumes`.
- `HTTPAdapter`: YAML `url`, `request_template` (`{{message}}` and friends),
  dotted `text_path`, `headers` with `${ENV}`; 408/429/5xx retry.
- `tests/test_adapter_equivalence.py` runs the demo through all three shapes and
  requires identical results. Keep it green. The toy agent reads only whole
  4-digit order ids, kept on purpose as a demo finding.

## Layout

- `datamodels.py` (all models), `config.py` (`load_settings()` reads
  `CONVOPROBE_ENV_FILE`, default `.env`), `runner.py`, `calibration.py`, `cli.py`,
  `report/`, `scoring/`, `simulator/`, `demo/` (toy agent, scenario, cassette).
  No module may shadow a stdlib or dependency name.

## LLM layer (models/)

- `factory.py`: `parse_spec` splits on the first colon; providers load lazily.
- `client.py`: cache key = spec, method, kind, temperature, schema, messages,
  and a **salt** (conversation id + turn) so repeats sample independently. Bad
  structured output (or `tool_use_failed` / `json_validate_failed`) raises
  `StructuredOutputError` and is never cached. `json_schema` for Ollama and Groq.
- `cache.py`: `SqliteLLMCache` (tracks run use, `export_cassette`), `JsonCassette`.
- `retry.py`: 408/429/5xx, timeouts, connection errors retry with backoff. Groq
  403 = VPN. 429 "Request too large" is not retried (message redacts org ids).
- `registry.py`: one limiter per Groq/Gemini model, one shared Ollama limiter.

## Simulator, judge, demo, runner

- `LLMSimulator` flips roles (its lines are AIMessages); `SimTurn` stops only
  after an agent reply. One retry on bad output or an empty first message, then
  `simulator_error`; a zero-message conversation is never `done`.
- Judge: reasoning first; lines `turn N USER|AGENT|AGENT TOOL CALL`. Bad
  evidence twice gives `error`, never a pass; `quote_found` checks each quote.
  No docstrings on schema classes: pydantic sends them to the model.
- `generate: N` personas (`gen_<slug>`) are saved per run; a resume reuses them.
- `demo --record` (live Groq) writes `demo/cassette.json` only if every
  conversation is `done`. A changed prompt, schema, salt, or demo scenario
  changes cache keys: re-record.
- Resume by run id; `pending`, `failed_transient`, and `skipped` LLM personas
  run. Judge `TransientError` = `failed_transient`, replay miss = `failed`.

## How to extend

- New check: model + `RubricItem` + tag set (`scenario.py`), then `_CHECKS`.
- New adapter: three async `AgentAdapter` methods, `TransientError`, equivalence test.
- New provider: `models/factory.py` (`PROVIDERS`, `_CLASSES`), an extra, limits
  in `config.py` and `LLMRegistry.limiter`, `registry.STRUCTURED_METHOD`.
- Release: version in two places, `pyproject.toml` and `src/convoprobe/__init__.py`
  (tests check it; then `uv lock --offline`), plus a `CHANGELOG.md` entry.

## Contracts

- Core imports without extras (no provider, no langgraph); subprocess tests check.
- Old `results.json` files stay loadable: new `RunResult` fields get defaults.
- Offline tests: `conftest.py` hides `.env` and blocks all but loopback, and the
  Ollama port. The demo needs nothing outside the package: no key, no network.
