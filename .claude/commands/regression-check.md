Check an agent for regressions against the saved baseline. Agent: $ARGUMENTS

Run everything from the convoprobe folder. If `uv` is not on PATH, use
`C:\Users\maria\.local\bin\uv.exe`.

1. Work out the adapter spec from $ARGUMENTS: `module.path:attr` for a Python
   agent or LangGraph graph, or `http:<config.yaml>` for an HTTP agent. If a
   scenario path is given, use it; otherwise use
   `src/convoprobe/demo/scenario.yaml`.
2. Pick the model mode. Use `--no-llm` when the scenario has only scripted
   personas and no `llm_judge` items, or when no cassette fits. Use
   `--replay <cassette>` when a cassette recorded for this scenario exists,
   for example `src/convoprobe/demo/cassette.json` for the demo scenario with
   `--simulator groq:openai/gpt-oss-20b --judge groq:openai/gpt-oss-120b`.
   Never start a live Groq or Ollama run without asking first.
3. Run it with a fixed run id:
   `uv run convoprobe run <scenario> --adapter <spec> <mode> --out results --run-id regression-check --html`
4. Check `.convoprobe/baseline.json` exists. If it does not, stop and say so:
   a baseline is saved from a known-good run with
   `uv run convoprobe baseline save results/<run_id>`.
5. Compare: `uv run convoprobe compare results/regression-check`. Report the
   exit code (0 no regression, 1 regression, 2 error).
6. Explain each regression line in plain words: which rubric item dropped, by
   how many points, against which threshold, and which personas dropped. Point
   at the conversations to read in `results/regression-check/report.html`
   (failed conversations are listed first and open).
7. If the output says the scenario changed since the baseline, say that the
   comparison may not be like for like.

Do not change any files except the run output under `results/`.
