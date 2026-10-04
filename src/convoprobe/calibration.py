"""Calibration: build a blind labelling set, then measure a judge against it.

prepare: labels are written without seeing the judge's verdicts, so they cannot be
biased by them. Nothing prepare writes carries a check result, verdict, score,
reasoning, or pass/fail status.

score: run one judge model over every labelled item and compare its answers with
the reviewed labels (percent agreement, Cohen's kappa, confusion counts).
"""

from __future__ import annotations

import asyncio
import json
import re
from collections import Counter
from collections.abc import Hashable, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel

from convoprobe.datamodels import Conversation, LLMJudgeItem, RunResult, Scenario
from convoprobe.errors import ConvoprobeError
from convoprobe.models.client import LLMClient
from convoprobe.report.load import load_results
from convoprobe.scenario import load_scenario
from convoprobe.scoring.judge import format_transcript, judge_item

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

Labels must come from a person, or be drafted by a model and then reviewed by
a person. Record which one in the labeled_by field. A model's labels would
measure the judge against another model, not against human judgement.

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


# Scoring: one judge against the reviewed labels.

AnswerType = Literal["yes_no", "scale"]
Decision = Literal["pass", "fail"]

KAPPA_EXPLANATION = (
    "Kappa corrects agreement for chance: 0 means no better than chance, 1 means perfect"
    " agreement. The labels were drafted by Claude without seeing any judge output, then"
    " reviewed and approved by a human. honest_about_refund labels come from recorded tool"
    " calls (objective ground truth). Small sample: treat differences of a few points as noise."
)


@dataclass(frozen=True)
class LabelEntry:
    item: str
    rubric_id: str
    answer_type: AnswerType
    label: str | int


def _parse_label(entry: Any) -> LabelEntry:
    if not isinstance(entry, dict):
        raise ConvoprobeError(f"labels.yaml: every item must be a mapping (got {entry!r})")
    item, rubric_id = str(entry.get("item")), str(entry.get("rubric_id"))
    where = f"labels.yaml: item {item} rubric {rubric_id}"
    answer_type, label = entry.get("answer_type"), entry.get("label")
    if label is None:
        raise ConvoprobeError(f"{where}: label is missing")
    if answer_type == "yes_no":
        if isinstance(label, bool):
            return LabelEntry(item, rubric_id, "yes_no", "yes" if label else "no")
        if isinstance(label, str) and label.strip().lower() in ("yes", "no"):
            return LabelEntry(item, rubric_id, "yes_no", label.strip().lower())
        raise ConvoprobeError(f"{where}: label must be yes or no (got {label!r})")
    if answer_type == "scale":
        if isinstance(label, int) and not isinstance(label, bool) and 1 <= label <= 5:
            return LabelEntry(item, rubric_id, "scale", label)
        raise ConvoprobeError(f"{where}: label must be a whole number from 1 to 5 (got {label!r})")
    raise ConvoprobeError(f"{where}: answer_type must be yes_no or scale (got {answer_type!r})")


def load_labels(set_dir: Path) -> tuple[str, list[LabelEntry]]:
    """(labeled_by, entries) from labels.yaml. Every label must be filled and valid."""
    path = Path(set_dir) / LABELS_FILE
    if not path.is_file():
        raise ConvoprobeError(f"no {LABELS_FILE} in {set_dir}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConvoprobeError(f"{path} is not valid YAML: {exc}") from exc
    if not isinstance(data, dict) or not isinstance(data.get("items"), list):
        raise ConvoprobeError(f"{path}: expected a mapping with an items list")
    labeled_by = data.get("labeled_by")
    return (labeled_by if isinstance(labeled_by, str) else ""), [
        _parse_label(entry) for entry in data["items"]
    ]


@dataclass(frozen=True)
class JudgeDefinition:
    item: LLMJudgeItem
    source: str


def judge_definitions(scenario_paths: Sequence[Path]) -> dict[str, list[JudgeDefinition]]:
    """Every llm_judge item in the scenarios, by rubric id.

    The source is the first word of the scenario name ("clinic" for clinic-intake,
    "toy" for toy-support), used to group results.
    """
    found: dict[str, list[JudgeDefinition]] = {}
    for path in scenario_paths:
        scenario = load_scenario(path)
        source = scenario.name.split("-")[0]
        for item in scenario.rubric:
            if isinstance(item, LLMJudgeItem):
                found.setdefault(item.id, []).append(JudgeDefinition(item, source))
    return found


def resolve_definition(
    definitions: dict[str, list[JudgeDefinition]], entry: LabelEntry
) -> JudgeDefinition:
    """The one judge item a label refers to; its answer type must match the label's."""
    candidates = definitions.get(entry.rubric_id, [])
    if not candidates:
        raise ConvoprobeError(
            f"item {entry.item}: rubric {entry.rubric_id} is not an llm_judge item"
            " in any --scenario file"
        )
    first = candidates[0]
    if any(c.item.model_dump() != first.item.model_dump() for c in candidates[1:]):
        raise ConvoprobeError(
            f"rubric {entry.rubric_id} is defined differently in two --scenario files"
        )
    expected: AnswerType = "yes_no" if first.item.pass_if is not None else "scale"
    if entry.answer_type != expected:
        raise ConvoprobeError(
            f"item {entry.item}: rubric {entry.rubric_id} is {expected},"
            f" but the label is {entry.answer_type}"
        )
    return first


def percent_agreement(a: Sequence[Hashable], b: Sequence[Hashable]) -> float | None:
    """Share of positions where a and b agree. None for empty input."""
    if len(a) != len(b):
        raise ValueError("both raters must rate the same number of items")
    if not a:
        return None
    return sum(x == y for x, y in zip(a, b, strict=True)) / len(a)


def cohens_kappa(a: Sequence[Hashable], b: Sequence[Hashable]) -> float | None:
    """Cohen's kappa for two raters over categories.

    None when there are no items or when chance agreement is 1 (both raters always
    give the same single category), where kappa is undefined.
    """
    if len(a) != len(b):
        raise ValueError("both raters must rate the same number of items")
    n = len(a)
    if n == 0:
        return None
    observed = Fraction(sum(x == y for x, y in zip(a, b, strict=True)), n)
    count_a, count_b = Counter(a), Counter(b)
    expected = sum((Fraction(count_a[k] * count_b[k], n * n) for k in count_a), Fraction(0))
    if expected == 1:
        return None
    return float((observed - expected) / (1 - expected))


def confusion(a: Sequence[Hashable], b: Sequence[Hashable]) -> dict[tuple[Any, Any], int]:
    """Counts keyed by (label, judge)."""
    if len(a) != len(b):
        raise ValueError("both raters must rate the same number of items")
    return dict(Counter(zip(a, b, strict=True)))


class ConfusionRow(BaseModel):
    label: str | int
    judge: str | int
    count: int


class RubricCalibration(BaseModel):
    rubric_id: str
    answer_type: AnswerType
    n: int
    n_errors: int
    agreement: float | None = None
    exact: float | None = None
    within1: float | None = None
    decision_agreement: float | None
    decision_kappa: float | None
    confusion: list[ConfusionRow]


class SourceCalibration(BaseModel):
    n: int
    decision_agreement: float | None
    decision_kappa: float | None


class ItemCalibration(BaseModel):
    item: str
    rubric_id: str
    source: str
    label: str | int
    judge_answer: str | int | None
    agree: bool | None
    decision_agree: bool | None
    label_decision: Decision
    judge_decision: Decision | None
    error: str | None
    reasoning: str
    evidence_turns: list[int]


class CalibrationReport(BaseModel):
    judge: str
    labeled_by: str
    n_labels: int
    n_errors: int
    overall_decision_agreement: float | None
    overall_decision_kappa: float | None
    per_rubric: list[RubricCalibration]
    per_source: dict[str, SourceCalibration]
    items: list[ItemCalibration]
    # How the report was made, for the reproduce commands in the results table.
    set_dir: str = ""
    scenarios: list[str] = []


def decision(item: LLMJudgeItem, value: str | int) -> Decision:
    """Pass or fail for an answer: yes/no against pass_if, a score against min_pass."""
    if item.pass_if is not None:
        return "pass" if value == item.pass_if else "fail"
    assert isinstance(value, int)
    return "pass" if value >= (item.min_pass or 1) else "fail"


def _load_item(set_dir: Path, item: str) -> tuple[Conversation, str]:
    path = Path(set_dir) / "conversations" / f"{item}.json"
    if not path.is_file():
        raise ConvoprobeError(f"item {item}: no conversation file at {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    description = str(data.pop("agent_description", ""))
    data.pop("source_run", None)
    data.pop("adapter", None)
    return Conversation.model_validate(data), description


def _reasoning(detail: str) -> str:
    """The judge's reasoning from a check detail like "verdict no (pass if no): ..."."""
    return detail.split(": ", 1)[1][:REASONING_CHARS] if ": " in detail else ""


REASONING_CHARS = 300


async def _judge_one(
    set_dir: Path, entry: LabelEntry, definition: JudgeDefinition, client: LLMClient
) -> ItemCalibration:
    item = definition.item
    row: dict[str, Any] = {
        "item": entry.item,
        "rubric_id": entry.rubric_id,
        "source": definition.source,
        "label": entry.label,
        "judge_answer": None,
        "agree": None,
        "decision_agree": None,
        "label_decision": decision(item, entry.label),
        "judge_decision": None,
        "error": None,
        "reasoning": "",
        "evidence_turns": [],
    }
    conversation, description = _load_item(set_dir, entry.item)
    context = Scenario.model_construct(agent_description=description)
    try:
        check = await judge_item(
            item, conversation, context, client, salt=f"calib:{entry.item}:{entry.rubric_id}"
        )
    except Exception as exc:  # recorded, never fatal: transient, replay miss, provider errors
        row["error"] = f"{type(exc).__name__}: {exc}"[:REASONING_CHARS]
        return ItemCalibration(**row)
    if check.status == "error":
        row["error"] = check.detail[:REASONING_CHARS]
        return ItemCalibration(**row)
    answer: str | int = check.verdict if check.verdict is not None else int(check.score or 0)
    judge_decision = decision(item, answer)
    row.update(
        judge_answer=answer,
        agree=answer == entry.label,
        decision_agree=judge_decision == row["label_decision"],
        judge_decision=judge_decision,
        reasoning=_reasoning(check.detail),
        evidence_turns=[e.turn for e in check.evidence],
    )
    return ItemCalibration(**row)


def _decision_metrics(rows: list[ItemCalibration]) -> tuple[float | None, float | None]:
    ok = [r for r in rows if r.error is None]
    labels = [r.label_decision for r in ok]
    judged = [r.judge_decision for r in ok]
    return percent_agreement(labels, judged), cohens_kappa(labels, judged)


def _rubric_calibration(
    rubric_id: str, answer_type: AnswerType, rows: list[ItemCalibration]
) -> RubricCalibration:
    ok = [r for r in rows if r.error is None]
    labels = [r.label for r in ok]
    answers = [r.judge_answer for r in ok]
    decision_agreement, decision_kappa = _decision_metrics(rows)
    counts = confusion(labels, answers)
    result = RubricCalibration(
        rubric_id=rubric_id,
        answer_type=answer_type,
        n=len(rows),
        n_errors=len(rows) - len(ok),
        decision_agreement=decision_agreement,
        decision_kappa=decision_kappa,
        confusion=[
            ConfusionRow(label=label, judge=judge, count=count)
            for (label, judge), count in sorted(counts.items(), key=lambda kv: str(kv[0]))
        ],
    )
    if answer_type == "yes_no":
        result.agreement = percent_agreement(labels, answers)
    else:
        result.exact = percent_agreement(labels, answers)
        result.within1 = (
            sum(abs(int(a or 0) - int(b)) <= 1 for a, b in zip(answers, labels, strict=True))
            / len(ok)
            if ok
            else None
        )
    return result


def build_report(
    judge: str, labeled_by: str, rows: list[ItemCalibration], answer_types: dict[str, AnswerType]
) -> CalibrationReport:
    """Aggregate per-label rows into per-rubric, per-source, and overall numbers."""
    by_rubric: dict[str, list[ItemCalibration]] = {}
    by_source: dict[str, list[ItemCalibration]] = {}
    for row in rows:
        by_rubric.setdefault(row.rubric_id, []).append(row)
        by_source.setdefault(row.source, []).append(row)
    overall_agreement, overall_kappa = _decision_metrics(rows)
    per_source = {}
    for source, source_rows in by_source.items():
        agreement, kappa = _decision_metrics(source_rows)
        per_source[source] = SourceCalibration(
            n=len(source_rows), decision_agreement=agreement, decision_kappa=kappa
        )
    return CalibrationReport(
        judge=judge,
        labeled_by=labeled_by,
        n_labels=len(rows),
        n_errors=sum(r.error is not None for r in rows),
        overall_decision_agreement=overall_agreement,
        overall_decision_kappa=overall_kappa,
        per_rubric=[
            _rubric_calibration(rubric_id, answer_types[rubric_id], rubric_rows)
            for rubric_id, rubric_rows in by_rubric.items()
        ],
        per_source=per_source,
        items=rows,
    )


async def score_judge(
    set_dir: Path, scenario_paths: Sequence[Path], judge: str, client: LLMClient
) -> CalibrationReport:
    """Judge every labelled item and compare. Judge errors are recorded, never fatal."""
    labeled_by, labels = load_labels(set_dir)
    definitions = judge_definitions(scenario_paths)
    resolved = [(entry, resolve_definition(definitions, entry)) for entry in labels]
    for entry, _definition in resolved:
        _load_item(set_dir, entry.item)  # fail fast on a missing conversation file
    rows = await asyncio.gather(
        *(_judge_one(set_dir, entry, definition, client) for entry, definition in resolved)
    )
    report = build_report(
        judge, labeled_by, list(rows), {e.rubric_id: e.answer_type for e in labels}
    )
    report.set_dir = Path(set_dir).as_posix()
    report.scenarios = [Path(p).as_posix() for p in scenario_paths]
    return report


def judge_slug(spec: str) -> str:
    """A file-safe model spec: groq:openai/gpt-oss-120b -> groq-openai-gpt-oss-120b."""
    return re.sub(r"[^a-z0-9]", "-", spec)


# Results table across judges.

TABLE_COLUMNS = (
    "judge",
    "decisions agree",
    "kappa",
    "no_medical_advice kappa",
    "no_outcome_promise kappa",
    "stayed_on_task exact / within 1",
    "honest_about_refund agreement",
    "errors",
)
TABLE_HEADER = "| " + " | ".join(TABLE_COLUMNS) + " |"


def format_share(value: float | None) -> str:
    return "-" if value is None else f"{round(value * 100)}%"


def format_kappa(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}"


def load_report(path: Path) -> CalibrationReport:
    try:
        return CalibrationReport.model_validate_json(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConvoprobeError(f"no calibration report at {path}") from exc
    except ValueError as exc:
        raise ConvoprobeError(f"{path}: not a calibration report: {exc}") from exc


def _table_row(report: CalibrationReport) -> str:
    rubric = {r.rubric_id: r for r in report.per_rubric}

    def kappa(rubric_id: str) -> str:
        return format_kappa(rubric[rubric_id].decision_kappa) if rubric_id in rubric else "-"

    task = rubric.get("stayed_on_task")
    honest = rubric.get("honest_about_refund")
    cells = [
        report.judge,
        format_share(report.overall_decision_agreement),
        format_kappa(report.overall_decision_kappa),
        kappa("no_medical_advice"),
        kappa("no_outcome_promise"),
        f"{format_share(task.exact)} / {format_share(task.within1)}" if task else "-",
        format_share(honest.agreement) if honest else "-",
        f"{report.n_errors} of {report.n_labels}",
    ]
    return "| " + " | ".join(cells) + " |"


def render_table(
    reports: Sequence[CalibrationReport], date: str, cassette: str = "calibration/cassette.json"
) -> str:
    """The Markdown results table, sample notes, reproduce commands, and explanation."""
    if not reports:
        raise ConvoprobeError("no calibration reports given")
    first = reports[0]
    sources = ", ".join(f"{source.n} {name}" for name, source in first.per_source.items())
    lines = [
        "# Judge calibration results",
        "",
        TABLE_HEADER,
        "|" + "|".join("---" for _ in TABLE_COLUMNS) + "|",
        *(_table_row(report) for report in reports),
        "",
        f"n = {first.n_labels} labels ({sources}). Decisions are pass or fail per label;"
        " kappa is Cohen's kappa on those decisions.",
        "",
        f"Labels: {first.labeled_by}.",
        "",
        f"Date: {date}.",
        "",
        "Reproduce offline from the recorded cassette (no keys, no network):",
        "",
        "```",
    ]
    for report in reports:
        scenarios = " ".join(f"--scenario {s}" for s in report.scenarios)
        lines.append(
            f"uv run convoprobe calibrate score --set {report.set_dir} {scenarios}"
            f" --judge {report.judge} --replay {cassette}"
            f" --out replay-{judge_slug(report.judge)}.json"
        )
    lines += ["```", "", KAPPA_EXPLANATION, ""]
    return "\n".join(lines)
