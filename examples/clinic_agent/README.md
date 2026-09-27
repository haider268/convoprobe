# Clinic agent (fictional)

A front-desk assistant for Meridian Regenerative Clinic. The clinic is
fictional and exists only for testing: the name, address (100 Example Street),
services, and hours are made up. Nothing here is medical information.

The agent answers questions about the clinic and books a free 20-minute
candidacy call with a specialist. It calls Groq (`openai/gpt-oss-20b`,
temperature 0.7), so it needs `GROQ_API_KEY` in `.env` and the VPN off.
Replies are cached in `.convoprobe/clinic-agent-cache.db`.

## Two variants

Both share the same clinic facts. Only the instructions differ.

- `agent_strict`: never gives medical advice, a diagnosis, a treatment
  recommendation, or any promise or estimate of results. It says the
  specialist covers that on the candidacy call.
- `agent_loose`: told to be warm and to share typical outcomes to build
  confidence. It exists to produce borderline and failing conversations, so the
  judge calibration set has real failures to label, not only clean passes.

## Run it with convoprobe

From the repo root:

```
uv run convoprobe run examples/scenarios/clinic_intake.yaml --adapter examples.clinic_agent.agent:agent_strict --hosted --out results --db .convoprobe/calib.db --run-id calib-strict --html
uv run convoprobe run examples/scenarios/clinic_intake.yaml --adapter examples.clinic_agent.agent:agent_loose --hosted --out results --db .convoprobe/calib.db --run-id calib-loose --html
```

`--hosted` uses Groq for the simulator (`openai/gpt-oss-20b`) and the judge
(`openai/gpt-oss-120b`), so a run makes three kinds of Groq calls: agent,
simulator, and judge. Rerun the same command with the same run id to resume
conversations that ended on a transient error.

To build a blind labelling set from the results:

```
uv run convoprobe calibrate prepare --from results/calib-strict --from results/calib-loose --out calibration --max 30
```
