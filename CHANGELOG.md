# Changelog

## 0.1.0 (2026-10-04)

First release.

- Scenario files in YAML: agent description, personas, rubric, thresholds, repeats.
- Personas: scripted, LLM-played, and generated from the agent description.
- LLM user simulator with structured turns and a role-flipped prompt.
- Adapters for a Python callable, a compiled LangGraph graph (interrupts resumed, node paths
  recorded), and an HTTP endpoint configured in YAML.
- Deterministic checks: tool called, tool not called, tool order, tool args, regex present,
  regex absent, max turns, max latency.
- LLM judge with yes/no or 1 to 5 answers, reasoning before the verdict, evidence turns, and a
  check that each quoted piece of evidence is in the transcript.
- Model providers: Ollama, Groq, and Gemini as optional extras, with per-model rate limits,
  retries with backoff, and an LLM cache.
- Record and replay cassettes for runs with no keys and no network.
- Resumable runs in SQLite: transient failures rerun on the same run id.
- Terminal report, `results.json`, and a self-contained HTML report.
- `baseline save` and `compare` with exit codes 0, 1, 2 for CI, plus an example GitHub Actions
  workflow and a committed baseline in `ci/`.
- `convoprobe demo`: a toy shop agent, a correct v1 and a v2 with a bug, replayed from a
  recorded cassette.
- Judge calibration: `calibrate prepare`, `calibrate score`, and `calibrate table`, with a
  labelled set, results for three hosted judges, and their cassette in `calibration/`.
- A fictional clinic example agent in `examples/clinic_agent/`.
