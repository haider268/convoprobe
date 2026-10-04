"""Calibration: build a blind labelling set from finished runs.

A person labels each conversation against the judge questions without seeing
the judge's verdicts, so the labels cannot be biased by them. Nothing written
here carries a check result, verdict, score, reasoning, or pass/fail status.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from convoprobe.datamodels import Conversation, LLMJudgeItem, RunResult
from convoprobe.errors import ConvoprobeError
from convoprobe.report.load import load_results
from convoprobe.scoring.judge import format_transcript

EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@([A-Za-z0-9-]+\.)+[A-Za-z]{2,}")
PHONE = re.compile(r"\+?\(?\d[\d\s().-]{6,}\d")
# Domains and numbers reserved for documentation and fiction, so they reach no one.
SAFE_DOMAINS = ("example.com", "example.org", "example.net")
SAFE_SUFFIXES = (".example", ".test", ".invalid")
SAFE_PHONE = re.compile(r"55501\d\d")

LABELS_FILE = "labels.yaml"
OVERWRITE_REFUSED = "labels.yaml already has labels; refusing to overwrite"
GENERATED = re.compile(r"^c\d+\.(json|md)$")

README = """# Calibration set

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

Labels must come from a person, not a model. A model's labels would measure
the judge against another model, not against human judgement.

`conversations/<item>.json` holds the full conversation data for tooling.
"""


@dataclass
class PreparedSet:
    items: int = 0
    labels: int = 0
    per_source: list[tuple[str, int]] = field(default_factory=list)


def judge_items_for(result: RunResult, persona_id: str) -> list[LLMJudgeItem]:
    """The run's llm_judge rubric items that apply to a persona."""
    return [
        item
        for item in result.rubric
        if isinstance(item, LLMJudgeItem)
        and (item.applies_to is None or persona_id in item.applies_to)
    ]


def _how_to_answer(item: LLMJudgeItem) -> str:
    return "yes or no" if item.pass_if is not None else "1 to 5"


def _fence(text: str) -> str:
    """A code fence longer than any backtick run inside the text."""
    longest = max((len(run) for run in re.findall(r"`+", text)), default=0)
    return "`" * max(3, longest + 1)


def render_transcript(
    item_id: str, conversation: Conversation, agent_description: str, items: list[LLMJudgeItem]
) -> str:
    """The Markdown page a labeller reads: description, transcript, questions."""
    transcript = format_transcript(conversation)
    fence = _fence(transcript)
    lines = [
        f"# {item_id}",
        "",
        "## What the agent is supposed to do",
        "",
        agent_description.strip(),
        "",
        "## Transcript",
        "",
        f"{fence}text",
        transcript,
        fence,
        "",
        "## Questions",
        "",
    ]
    for number, item in enumerate(items, start=1):
        lines.append(f"{number}. {item.id}: {item.question} (answer {_how_to_answer(item)})")
    return "\n".join(lines) + "\n"


def _safe_email(address: str) -> bool:
    domain = address.rsplit("@", 1)[1].lower()
    return domain in SAFE_DOMAINS or domain.endswith(SAFE_SUFFIXES)


def contact_details(text: str) -> list[tuple[str, str]]:
    """(kind, matched text) for every email or phone number not in a reserved range."""
    found: list[tuple[str, str]] = []
    for match in EMAIL.finditer(text):
        if not _safe_email(match.group()):
            found.append(("email", match.group()))
    for match in PHONE.finditer(text):
        digits = re.sub(r"\D", "", match.group())
        if len(digits) >= 7 and not SAFE_PHONE.search(digits):
            found.append(("phone", match.group()))
    return found


def _conversation_texts(conversation: Conversation) -> list[str]:
    texts: list[str] = []
    for message in conversation.messages:
        texts.append(message.text)
        for call in message.tool_calls:
            texts.append(json.dumps(call.args, ensure_ascii=False, default=str))
            texts.append(json.dumps(call.result, ensure_ascii=False, default=str))
    return texts


def _check_contact_details(picked: list[tuple[str, Conversation]]) -> None:
    """Refuse the whole set if any conversation holds a real-looking email or phone."""
    problems = [
        f"{item_id} {kind}: {value}"
        for item_id, conversation in picked
        for text in _conversation_texts(conversation)
        for kind, value in contact_details(text)
    ]
    if problems:
        raise ConvoprobeError(
            "real-looking contact details found; nothing written:\n  " + "\n  ".join(problems)
        )


def _has_labels(path: Path) -> bool:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConvoprobeError(f"{path} is not valid YAML; refusing to overwrite") from exc
    entries = data.get("items") if isinstance(data, dict) else None
    if not isinstance(entries, list):
        return False
    return any(isinstance(e, dict) and e.get("label") is not None for e in entries)


def prepare(sources: list[Path], out_dir: Path, max_items: int = 30) -> PreparedSet:
    """Write a blind labelling set for up to max_items conversations."""
    out_dir = Path(out_dir)
    labels_path = out_dir / LABELS_FILE
    if labels_path.is_file() and _has_labels(labels_path):
        raise ConvoprobeError(OVERWRITE_REFUSED)

    picked: list[tuple[RunResult, Conversation, list[LLMJudgeItem]]] = []
    summary = PreparedSet()
    for source in sources:
        result = load_results(source)
        count = 0
        for conversation in sorted(result.conversations, key=lambda c: c.id):
            if len(picked) >= max_items:
                break
            if result.conversation_status.get(conversation.id) != "done":
                continue
            items = judge_items_for(result, conversation.persona_id)
            if items:
                picked.append((result, conversation, items))
                count += 1
        summary.per_source.append((str(source), count))

    _check_contact_details(
        [(f"c{index:02d}", conversation) for index, (_r, conversation, _i) in enumerate(picked, 1)]
    )

    conversations_dir = out_dir / "conversations"
    transcripts_dir = out_dir / "transcripts"
    for folder in (conversations_dir, transcripts_dir):
        folder.mkdir(parents=True, exist_ok=True)
        for old in folder.iterdir():
            if old.is_file() and GENERATED.match(old.name):
                old.unlink()

    entries: list[dict[str, Any]] = []
    for index, (result, conversation, items) in enumerate(picked, start=1):
        item_id = f"c{index:02d}"
        record = conversation.model_dump(mode="json")
        record.update(
            {
                "source_run": result.run_id,
                "adapter": result.adapter,
                "agent_description": result.agent_description,
            }
        )
        (conversations_dir / f"{item_id}.json").write_text(
            json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
        )
        (transcripts_dir / f"{item_id}.md").write_text(
            render_transcript(item_id, conversation, result.agent_description, items),
            encoding="utf-8",
            newline="\n",
        )
        for item in items:
            entries.append(
                {
                    "item": item_id,
                    "rubric_id": item.id,
                    "question": item.question,
                    "answer_type": "yes_no" if item.pass_if is not None else "scale",
                    "label": None,
                    "note": "",
                }
            )

    labels_path.write_text(
        yaml.safe_dump({"format": 1, "items": entries}, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
        newline="\n",
    )
    (out_dir / "README.md").write_text(README, encoding="utf-8", newline="\n")
    summary.items = len(picked)
    summary.labels = len(entries)
    return summary
