# Judge calibration results

## Current rubric

| judge | decisions agree | kappa | no_medical_advice kappa | no_outcome_promise kappa | stayed_on_task exact / within 1 | honest_about_refund agreement | errors |
|---|---|---|---|---|---|---|---|
| groq:openai/gpt-oss-120b | 94% | 0.86 | 0.64 | 1.00 | 44% / 81% | 100% | 0 of 62 |
| groq:openai/gpt-oss-20b | 94% | 0.86 | -0.07 | 1.00 | 44% / 81% | 100% | 0 of 62 |
| groq:qwen/qwen3.8-27b | 95% | 0.90 | 0.64 | 1.00 | 44% / 94% | 100% | 0 of 62 |

## Before the wording fix

| judge | decisions agree | kappa | no_medical_advice kappa | no_outcome_promise kappa | stayed_on_task exact / within 1 | honest_about_refund agreement | errors |
|---|---|---|---|---|---|---|---|
| groq:openai/gpt-oss-120b | 87% | 0.74 | 0.20 | 1.00 | 44% / 81% | 100% | 0 of 62 |
| groq:openai/gpt-oss-20b | 89% | 0.77 | 0.20 | 1.00 | 44% / 88% | 100% | 0 of 62 |
| groq:qwen/qwen3.8-27b | 85% | 0.71 | 0.12 | 1.00 | 44% / 94% | 100% | 0 of 62 |

The first wording of no_medical_advice let judges count outcome statements as medical advice. It is reproducible from commit 344c891.

## Local judge

A local 3B judge (ollama:llama3.2:3b) was not scored. On the development laptop (CPU only, 7.4 GB RAM) it produced 17 of 62 answers in 2.5 hours, and in an earlier smoke test it returned a wrong verdict. Use a hosted judge or the recorded cassette for results you rely on.

---

n = 62 labels (48 clinic, 14 toy). Decisions are pass or fail per label; kappa is Cohen's kappa on those decisions.

Labels: drafted by Claude (Opus 5.5) without seeing judge verdicts; reviewed and approved by the repository owner (haider268) on 2026-10-04.

Date: 2026-10-04.

Reproduce offline from the recorded cassette (no keys, no network):

```
uv run convoprobe calibrate score --set calibration --scenario examples/scenarios/clinic_intake.yaml --scenario src/convoprobe/demo/scenario.yaml --judge groq:openai/gpt-oss-120b --replay calibration/cassette.json --out replay-groq-openai-gpt-oss-120b.json
uv run convoprobe calibrate score --set calibration --scenario examples/scenarios/clinic_intake.yaml --scenario src/convoprobe/demo/scenario.yaml --judge groq:openai/gpt-oss-20b --replay calibration/cassette.json --out replay-groq-openai-gpt-oss-20b.json
uv run convoprobe calibrate score --set calibration --scenario examples/scenarios/clinic_intake.yaml --scenario src/convoprobe/demo/scenario.yaml --judge groq:qwen/qwen3.8-27b --replay calibration/cassette.json --out replay-groq-qwen-qwen3-8-27b.json
```

Kappa corrects agreement for chance: 0 means no better than chance, 1 means perfect agreement. The labels were drafted by Claude without seeing any judge output, then reviewed and approved by a human. honest_about_refund labels come from recorded tool calls (objective ground truth). Small sample: treat differences of a few points as noise.
