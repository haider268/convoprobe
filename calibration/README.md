# Calibration set

Each item is one conversation between a simulated user and an agent. Your
labels are the ground truth the LLM judge is measured against.

## How to label

1. Open `transcripts/<item>.md` (for example `transcripts/c01.md`).
2. Read only the transcript and the agent description above it.
3. In `labels.yaml`, find the entries for that item and fill in `label`:
   - yes or no questions: `label: yes` or `label: no`
   - 1 to 5 questions: `label: 1` to `label: 5` (5 is excellent)
4. If you are unsure, still pick the closest answer and say why in `note`.

The judge's verdicts are deliberately not shown anywhere in this folder, so
they cannot bias your labels. Do not open the run results while labelling.

Labels must come from a person, or be drafted by a model and then reviewed by
a person. Record which one in the labeled_by field. A model's labels would
measure the judge against another model, not against human judgement.

`conversations/<item>.json` holds the full conversation data for tooling.
