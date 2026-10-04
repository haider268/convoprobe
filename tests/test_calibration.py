"""calibrate prepare: selection, order, files, blindness, and overwrite refusal."""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import TypeAdapter
from typer.testing import CliRunner

from convoprobe.calibration import (
    OVERWRITE_REFUSED,
    README,
    contact_details,
    prepare,
    render_transcript,
)
from convoprobe.cli import app
from convoprobe.datamodels import (
    CheckResult,
    Conversation,
    Evidence,
    Message,
    RubricItem,
    RunResult,
    ToolCallRecord,
)
from convoprobe.errors import ConvoprobeError
from convoprobe.report.json_report import write_results
from convoprobe.report.load import load_results
from convoprobe.scoring.judge import format_transcript

runner = CliRunner()

YES_NO = {"id": "no_advice", "type": "llm_judge", "question": "Did it advise?", "pass_if": "no"}
SCALE_P1 = {
    "id": "on_task",
    "type": "llm_judge",
    "question": "How well did it do?",
    "scale": "1-5",
    "min_pass": 4,
    "applies_to": ["p1"],
}
REGEX = {"id": "no_word", "type": "regex_absent", "pattern": "bad"}
ONLY_X = {**YES_NO, "id": "x_only", "applies_to": ["x"]}

# Strings that exist only in check results. None may leak into the labelling set.
JUDGE_DETAILS = ["JUDGE-DETAIL-SECRET", "judge reasoning text", "REGEX-DETAIL-SECRET"]
STATUS_IN_CHECK = re.compile(
    r"""["']?(status|verdict|score|pass_rate)["']?\s*[:=]\s*["']?(pass|fail|error|skip|yes|no|\d)""",
    re.IGNORECASE,
)


def rubric(*items: dict[str, Any]) -> list[Any]:
    return TypeAdapter(list[RubricItem]).validate_python(list(items))


def conversation(cid: str, persona: str) -> Conversation:
    return Conversation(
        id=cid,
        persona_id=persona,
        ended_reason="max_turns",
        messages=[
            Message(role="user", text=f"hello from {cid}"),
            Message(
                role="agent",
                text="Sure, ```code``` here.",
                tool_calls=[ToolCallRecord(name="book", args={"day": "mon"}, result="ok")],
            ),
            Message(role="user", text="thanks"),
            Message(role="agent", text="bye"),
        ],
    )


def judge_check(cid: str, persona: str, rubric_id: str) -> CheckResult:
    return CheckResult(
        rubric_id=rubric_id,
        persona_id=persona,
        conversation_id=cid,
        status="fail",
        detail="JUDGE-DETAIL-SECRET judge reasoning text",
        source="judge",
        verdict="yes",
        evidence=[Evidence(turn=1, quote="hello", quote_found=True)],
    )


def write_run(
    root: Path,
    run_id: str,
    items: list[Any],
    convs: list[tuple[str, str, str]],
) -> Path:
    """convs: (conversation id, persona id, status)."""
    conversations = [conversation(cid, persona) for cid, persona, _ in convs]
    checks = [judge_check(cid, persona, "no_advice") for cid, persona, _ in convs]
    checks += [
        CheckResult(
            rubric_id="no_word",
            persona_id=persona,
            conversation_id=cid,
            status="pass",
            detail="REGEX-DETAIL-SECRET",
        )
        for cid, persona, _ in convs
    ]
    result = RunResult(
        run_id=run_id,
        scenario_name="s",
        adapter=f"adapter:{run_id}",
        started_at=datetime(2026, 9, 28, tzinfo=UTC),
        finished_at=None,
        conversation_status={cid: status for cid, _, status in convs},  # type: ignore[misc]
        conversations=conversations,
        checks=checks,
        summary=[],
        overall_pass_rate=0.5,
        notes=[],
        mode="live",
        rubric=items,
        agent_description=f"Agent for {run_id}.",
    )
    write_results(result, root)
    return root / run_id


@pytest.fixture
def sources(tmp_path: Path) -> list[Path]:
    runs = tmp_path / "runs"
    a = write_run(
        runs,
        "run-a",
        rubric(YES_NO, SCALE_P1, REGEX),
        [
            ("p2-r1", "p2", "done"),
            ("p1-r0", "p1", "done"),
            ("p3-r0", "p3", "failed"),
            ("p2-r0", "p2", "done"),
            ("p4-r0", "p4", "failed_transient"),
        ],
    )
    b = write_run(runs, "run-b", rubric(REGEX), [("p1-r0", "p1", "done")])
    c = write_run(
        runs, "run-c", rubric(ONLY_X, REGEX), [("y-r0", "y", "done"), ("x-r0", "x", "done")]
    )
    return [a, b, c]


def load_labels(out: Path) -> dict[str, Any]:
    data: dict[str, Any] = yaml.safe_load((out / "labels.yaml").read_text("utf-8"))
    return data


def test_selection_and_order(tmp_path: Path, sources: list[Path]) -> None:
    out = tmp_path / "calib"
    prepared = prepare(sources, out)
    assert prepared.items == 4
    assert prepared.labels == 5
    assert prepared.per_source == [(str(sources[0]), 3), (str(sources[1]), 0), (str(sources[2]), 1)]
    picked = [
        (json.loads((out / "conversations" / f"c0{n}.json").read_text("utf-8")))
        for n in range(1, 5)
    ]
    assert [(p["source_run"], p["id"]) for p in picked] == [
        ("run-a", "p1-r0"),
        ("run-a", "p2-r0"),
        ("run-a", "p2-r1"),
        ("run-c", "x-r0"),
    ]
    labels = load_labels(out)
    assert labels["format"] == 1
    assert [(e["item"], e["rubric_id"], e["answer_type"]) for e in labels["items"]] == [
        ("c01", "no_advice", "yes_no"),
        ("c01", "on_task", "scale"),
        ("c02", "no_advice", "yes_no"),
        ("c03", "no_advice", "yes_no"),
        ("c04", "x_only", "yes_no"),
    ]
    assert labels["items"][1] == {
        "item": "c01",
        "rubric_id": "on_task",
        "question": "How well did it do?",
        "answer_type": "scale",
        "label": None,
        "note": "",
    }
    text = (out / "labels.yaml").read_text("utf-8")
    assert text.startswith("format: 1\nitems:\n- item: c01\n  rubric_id: no_advice\n")


def test_max_stops_early(tmp_path: Path, sources: list[Path]) -> None:
    out = tmp_path / "calib"
    prepared = prepare(sources, out, max_items=2)
    assert prepared.items == 2
    assert prepared.per_source == [(str(sources[0]), 2), (str(sources[1]), 0), (str(sources[2]), 0)]
    assert sorted(p.name for p in (out / "transcripts").iterdir()) == ["c01.md", "c02.md"]
    assert {e["item"] for e in load_labels(out)["items"]} == {"c01", "c02"}


def test_file_contents(tmp_path: Path, sources: list[Path]) -> None:
    out = tmp_path / "calib"
    prepare(sources, out)
    record = json.loads((out / "conversations" / "c01.json").read_text("utf-8"))
    expected = conversation("p1-r0", "p1").model_dump(mode="json")
    assert {k: record[k] for k in expected} == expected
    assert record["source_run"] == "run-a"
    assert record["adapter"] == "adapter:run-a"
    assert record["agent_description"] == "Agent for run-a."
    assert set(record) == set(expected) | {"source_run", "adapter", "agent_description"}
    assert (out / "README.md").read_text("utf-8") == README
    assert "deliberately not shown" in README
    assert (
        "Labels must come from a person, or be drafted by a model and then reviewed by\n"
        "a person. Record which one in the labeled_by field." in README
    )
    assert "not a model" not in README


def test_transcripts_match_format_transcript(tmp_path: Path, sources: list[Path]) -> None:
    out = tmp_path / "calib"
    prepare(sources, out)
    page = (out / "transcripts" / "c01.md").read_text("utf-8")
    transcript = format_transcript(conversation("p1-r0", "p1"))
    # The agent text contains ``` so the fence grows to four backticks.
    assert f"````text\n{transcript}\n````\n" in page
    assert page.startswith("# c01\n\n## What the agent is supposed to do\n\nAgent for run-a.\n")
    assert page.endswith(
        "## Questions\n\n"
        "1. no_advice: Did it advise? (answer yes or no)\n"
        "2. on_task: How well did it do? (answer 1 to 5)\n"
    )
    plain = Conversation(id="z", persona_id="z", messages=[Message(role="user", text="hi")])
    items = rubric(YES_NO)
    assert "```text\nturn 1 USER: hi\n```\n" in render_transcript("c09", plain, "d", items)


def test_blind_nothing_from_checks_leaks(tmp_path: Path, sources: list[Path]) -> None:
    out = tmp_path / "calib"
    prepare(sources, out)
    written = [p for p in out.rglob("*") if p.is_file()]
    assert len(written) == 4 + 4 + 2
    for path in written:
        text = path.read_text("utf-8")
        for secret in JUDGE_DETAILS:
            assert secret not in text, (path, secret)
        assert not STATUS_IN_CHECK.search(text), (path, STATUS_IN_CHECK.search(text))
        assert "quote_found" not in text
        if path.name == "README.md":
            continue  # fixed text that says verdicts are hidden; holds no run data
        lowered = text.lower()
        for word in ("verdict", "reasoning", "passed", "failed", "pass_rate"):
            assert word not in lowered, (path, word)


def test_refuses_to_overwrite_labels(tmp_path: Path, sources: list[Path]) -> None:
    out = tmp_path / "calib"
    prepare(sources, out)
    prepare(sources, out)  # all labels still null: overwriting is fine
    labels_path = out / "labels.yaml"
    labels_path.write_text(
        labels_path.read_text("utf-8").replace("label: null", "label: 'no'", 1), encoding="utf-8"
    )
    before = labels_path.read_text("utf-8")
    with pytest.raises(ConvoprobeError, match=OVERWRITE_REFUSED):
        prepare(sources, out)
    result = runner.invoke(
        app, ["calibrate", "prepare", "--from", str(sources[0]), "--out", str(out)]
    )
    assert result.exit_code == 2
    assert result.output.strip() == f"error: {OVERWRITE_REFUSED}"
    assert labels_path.read_text("utf-8") == before


def test_refuses_to_overwrite_reviewed_labels_file(tmp_path: Path, sources: list[Path]) -> None:
    reviewed = Path(__file__).resolve().parents[1] / "calibration" / "labels.yaml"
    data = yaml.safe_load(reviewed.read_text("utf-8"))
    assert isinstance(data["labeled_by"], str)
    out = tmp_path / "calib"
    out.mkdir()
    (out / "labels.yaml").write_bytes(reviewed.read_bytes())
    with pytest.raises(ConvoprobeError, match=OVERWRITE_REFUSED):
        prepare(sources, out)
    assert (out / "labels.yaml").read_bytes() == reviewed.read_bytes()


def test_refuses_unreadable_labels(tmp_path: Path, sources: list[Path]) -> None:
    out = tmp_path / "calib"
    out.mkdir()
    (out / "labels.yaml").write_text("items: [unclosed", encoding="utf-8")
    with pytest.raises(ConvoprobeError, match="not valid YAML; refusing to overwrite"):
        prepare(sources, out)
    (out / "labels.yaml").write_text("just text", encoding="utf-8")
    assert prepare(sources, out).items == 4


def test_rerun_removes_stale_items(tmp_path: Path, sources: list[Path]) -> None:
    out = tmp_path / "calib"
    prepare(sources, out)
    (out / "transcripts" / "keep.txt").write_text("mine", encoding="utf-8")
    prepare(sources, out, max_items=1)
    assert sorted(p.name for p in (out / "transcripts").iterdir()) == ["c01.md", "keep.txt"]
    assert [p.name for p in (out / "conversations").iterdir()] == ["c01.json"]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("mail pat@example.com please", []),
        ("write to bookings@meridian.example", []),
        ("a@b.test", []),
        ("x@example.org and y@EXAMPLE.net", []),
        ("someone@email.com", [("email", "someone@email.com")]),
        ("x@meridianclinic.com", [("email", "x@meridianclinic.com")]),
        ("call 555-0123", []),
        ("call 555-0100", []),
        ("call (555) 987-6543", [("phone", "(555) 987-6543")]),
        ("call +1 212 555 7788", [("phone", "+1 212 555 7788")]),
        ("order 1042 and 987654, refund 89.0", []),
        ('{"amount": 89.0, "order_id": "1042"}', []),
    ],
)
def test_contact_details(text: str, expected: list[tuple[str, str]]) -> None:
    assert contact_details(text) == expected


def contact_run(root: Path, text_by_cid: dict[str, str], tool_result: object = "ok") -> Path:
    run_dir = write_run(root, "contacts", rubric(YES_NO), [(c, "p", "done") for c in text_by_cid])
    result = load_results(run_dir)
    conversations = []
    for conv in result.conversations:
        messages = [
            Message(role="user", text=text_by_cid[conv.id]),
            Message(
                role="agent",
                text="Booked.",
                tool_calls=[ToolCallRecord(name="book", args={"day": "mon"}, result=tool_result)],
            ),
        ]
        conversations.append(conv.model_copy(update={"messages": messages}))
    write_results(result.model_copy(update={"conversations": conversations}), root)
    return run_dir


def test_refusal_lists_items_and_writes_nothing(tmp_path: Path) -> None:
    source = contact_run(
        tmp_path / "runs",
        {
            "p-r0": "I am pat@example.com, 555-0123",
            "p-r1": "Reach me at someone@email.com or (555) 987-6543",
            "p-r2": "Order 1042 for 89.0",
        },
        tool_result={"contact": "x@meridianclinic.com"},
    )
    out = tmp_path / "calib"
    with pytest.raises(ConvoprobeError) as info:
        prepare([source], out)
    assert str(info.value) == (
        "real-looking contact details found; nothing written:\n"
        "  c01 email: x@meridianclinic.com\n"
        "  c02 email: someone@email.com\n"
        "  c02 phone: (555) 987-6543\n"
        "  c02 email: x@meridianclinic.com\n"
        "  c03 email: x@meridianclinic.com"
    )
    assert not out.exists()
    result = runner.invoke(app, ["calibrate", "prepare", "--from", str(source), "--out", str(out)])
    assert result.exit_code == 2
    assert result.output.startswith("error: real-looking contact details found; nothing written")
    assert not out.exists()


def test_tool_args_are_scanned(tmp_path: Path) -> None:
    source = contact_run(tmp_path / "runs", {"p-r0": "hi"}, tool_result={"phone": "212 555 7788"})
    with pytest.raises(ConvoprobeError, match="c01 phone: 212 555 7788"):
        prepare([source], tmp_path / "calib")


def test_reserved_details_pass(tmp_path: Path) -> None:
    source = contact_run(
        tmp_path / "runs",
        {"p-r0": "pat@example.com, 555-0123"},
        tool_result={"email": "bookings@meridian.example", "phone": "555-0100"},
    )
    assert prepare([source], tmp_path / "calib").items == 1


def test_cli_prepare(tmp_path: Path, sources: list[Path]) -> None:
    out = tmp_path / "calib"
    args = ["calibrate", "prepare", "--out", str(out), "--max", "30"]
    for source in sources:
        args += ["--from", str(source)]
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    assert result.output.splitlines() == [
        f"{sources[0]}: 3 item(s)",
        f"{sources[1]}: 0 item(s)",
        f"{sources[2]}: 1 item(s)",
        f"prepared 4 item(s), 5 label(s) to fill: {out}",
    ]
    missing = runner.invoke(
        app, ["calibrate", "prepare", "--from", str(tmp_path / "none"), "--out", str(out)]
    )
    assert missing.exit_code == 2
    assert "error: no results.json at" in missing.output
