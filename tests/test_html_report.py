"""The HTML report: structure, anchors, escaping, and no external resources."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from html.parser import HTMLParser
from importlib.resources import as_file, files
from pathlib import Path
from typing import Any

import pytest

from convoprobe.adapters import CallableAdapter
from convoprobe.datamodels import (
    CheckResult,
    Conversation,
    Evidence,
    LLMUsage,
    Message,
    RubricSummary,
    RunResult,
    ToolCallRecord,
)
from convoprobe.demo.toy_agent import toy_agent_v1, toy_agent_v2
from convoprobe.report.html import render_html, write_html
from convoprobe.runner import run_scenario
from convoprobe.scenario import load_scenario
from convoprobe.store import RunStore

EM_DASH = "—"


class Walker(HTMLParser):
    """Collects tags, ids, links, and <details> in document order."""

    def __init__(self) -> None:
        super().__init__()
        self.details: list[dict[str, Any]] = []
        self.ids: set[str] = set()
        self.hrefs: list[str] = []
        self.srcs: list[str] = []
        self.tags: set[str] = set()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        self.tags.add(tag)
        if values.get("id"):
            self.ids.add(str(values["id"]))
        if values.get("href"):
            self.hrefs.append(str(values["href"]))
        if values.get("src"):
            self.srcs.append(str(values["src"]))
        if tag == "details":
            self.details.append({"id": values.get("id"), "open": "open" in values})


def walk(html: str) -> Walker:
    walker = Walker()
    walker.feed(html)
    return walker


def demo_run(tmp_path: Path, version: str) -> RunResult:
    with as_file(files("convoprobe.demo") / "scenario.yaml") as path:
        scenario = load_scenario(path)
    fn = toy_agent_v1 if version == "v1" else toy_agent_v2
    with RunStore(tmp_path / f"{version}.db") as store:
        return asyncio.run(
            run_scenario(
                scenario,
                CallableAdapter(fn),
                store=store,
                out_dir=tmp_path / "out",
                adapter_label=f"demo:{version}",
            )
        )


def check(cid: str, rubric: str, status: str, **extra: Any) -> CheckResult:
    return CheckResult.model_validate(
        {
            "rubric_id": rubric,
            "persona_id": cid.split("-")[0],
            "conversation_id": cid,
            "status": status,
            "detail": f"{rubric} {status}",
            **extra,
        }
    )


def hand_built(stdev: float | None = 0.5) -> RunResult:
    """Two conversations: a clean one, and one with a failed judge check and nasty text."""
    nasty = Conversation(
        id="bad-r0",
        persona_id="bad",
        ended_reason="max_turns",
        messages=[
            Message(role="user", text="hi <b>there</b>"),
            Message(
                role="agent",
                text="<script>alert(1)</script> refund done",
                tool_calls=[
                    ToolCallRecord(name="issue_refund", args={"note": "</pre><b>"}, result="ok")
                ],
                latency_ms=12.5,
                metadata={"node_path": ["decide", "refund", "respond"]},
            ),
            Message(role="user", text="thanks"),
            Message(role="agent", text="bye"),
        ],
    )
    clean = Conversation(
        id="aaa_clean-r0",
        persona_id="aaa_clean",
        ended_reason="script_end",
        messages=[Message(role="user", text="hello"), Message(role="agent", text="hi")],
    )
    judge_fail = check(
        "bad-r0",
        "honest",
        "fail",
        source="judge",
        verdict="yes",
        evidence=[
            Evidence(turn=1, quote="refund done", quote_found=True),
            Evidence(turn=2, quote="never said this", quote_found=False),
        ],
    )
    return RunResult(
        run_id="hand",
        scenario_name="hand <built>",
        adapter="test",
        started_at=datetime(2026, 9, 28, 10, 0, tzinfo=UTC),
        finished_at=datetime(2026, 9, 28, 10, 1, tzinfo=UTC),
        conversation_status={"aaa_clean-r0": "done", "bad-r0": "done"},
        conversations=[clean, nasty],
        checks=[
            check("aaa_clean-r0", "turns", "pass"),
            check("bad-r0", "turns", "pass"),
            judge_fail,
        ],
        summary=[
            RubricSummary(
                rubric_id="turns",
                passed=2,
                failed=0,
                errored=0,
                skipped=0,
                pass_rate=1.0,
                pass_rate_stdev=stdev,
                persona_rates={"aaa_clean": 1.0, "bad": 1.0},
            ),
            RubricSummary(
                rubric_id="honest",
                passed=0,
                failed=1,
                errored=0,
                skipped=0,
                pass_rate=0.0,
                persona_rates={"bad": 0.0},
            ),
        ],
        overall_pass_rate=2 / 3,
        notes=["a note"],
    )


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_one_details_per_conversation_failed_first(tmp_path: Path, version: str) -> None:
    result = demo_run(tmp_path, version)
    doc = walk(render_html(result))
    ids = [d["id"] for d in doc.details]
    assert sorted(ids) == sorted(c.id for c in result.conversations)
    failing = {c.conversation_id for c in result.checks if c.status in ("fail", "error")}
    opened = [d["id"] for d in doc.details if d["open"]]
    assert set(opened) == failing
    # Failed conversations come first, then the rest, each group sorted by id.
    assert ids == sorted(failing) + sorted(set(ids) - failing)


def test_turn_ids_and_evidence_targets_exist() -> None:
    html = render_html(hand_built())
    doc = walk(html)
    assert {"bad-r0-turn-1", "bad-r0-turn-2", "aaa_clean-r0-turn-1"} <= doc.ids
    evidence_links = [h for h in doc.hrefs if h.startswith("#")]
    assert evidence_links == ["#bad-r0-turn-1", "#bad-r0-turn-2"]
    for href in evidence_links:
        assert href[1:] in doc.ids
    # Cited turns of a failed judge check get the highlight class.
    assert 'class="turn cited" id="bad-r0-turn-1"' in html
    assert 'class="turn" id="aaa_clean-r0-turn-1"' in html


def test_failed_conversation_first_and_open() -> None:
    doc = walk(render_html(hand_built()))
    assert doc.details == [
        {"id": "bad-r0", "open": True},
        {"id": "aaa_clean-r0", "open": False},
    ]


def test_escaping() -> None:
    html = render_html(hand_built())
    assert "<script" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
    assert "&lt;/pre&gt;&lt;b&gt;" in html
    assert "hi &lt;b&gt;there&lt;/b&gt;" in html
    assert "hand &lt;built&gt;" in html
    assert "<b>" not in html


def test_no_external_resources(tmp_path: Path) -> None:
    for html in (render_html(hand_built()), render_html(demo_run(tmp_path, "v2"))):
        doc = walk(html)
        for url in doc.srcs + doc.hrefs:
            assert "http://" not in url and "https://" not in url
        assert "<link" not in html
        assert "@import" not in html
        assert "script" not in doc.tags
        assert "img" not in doc.tags


def test_no_em_dash_in_demo_output(tmp_path: Path) -> None:
    for version in ("v1", "v2"):
        assert EM_DASH not in render_html(demo_run(tmp_path, version))


def test_quote_not_found_node_path_latency() -> None:
    html = render_html(hand_built())
    assert "(quote not found in transcript)" in html
    assert html.count("(quote not found in transcript)") == 1
    assert "12.5 ms" in html
    assert "nodes: decide &gt; refund &gt; respond" in html
    assert "verdict yes" in html
    assert "Generated by convoprobe 0.2.0." in html
    assert "2026-09-28 10:00:00 UTC" in html


def test_stdev_column_only_when_present() -> None:
    assert "<th>±</th>" in render_html(hand_built(stdev=0.5))
    assert "0.50" in render_html(hand_built(stdev=0.5))
    assert "<th>±</th>" not in render_html(hand_built(stdev=None))


def test_matrix_and_header() -> None:
    html = render_html(hand_built())
    assert "Pass rate by persona" in html
    assert '<td class="band-fail">0%</td>' in html
    assert '<td class="band-pass">100%</td>' in html
    assert "no LLM" in html
    assert "67%" in html


def test_matrix_has_six_columns_and_wrapping_headers(tmp_path: Path) -> None:
    html = render_html(demo_run(tmp_path, "v2"))
    matrix = html.split('<table class="matrix">', 1)[1].split("</table>", 1)[0]
    header = matrix.split("</tr>", 1)[0]
    assert header.count("<th>") == 7  # persona plus 6 rubric items
    for rubric_id in ("verified_before_refund", "honest_about_refund"):
        assert f"<th><span>{rubric_id}</span></th>" in header
    assert "white-space: normal;" in html.split("th { font-weight", 1)[1].split("}", 1)[0]
    assert ".matrix th { max-width: 9rem; overflow-wrap: anywhere; }" in html
    assert "max-width: 1100px" in html


def test_status_cells_use_soft_tints() -> None:
    html = render_html(hand_built())
    assert "--pass-bg: rgba(22, 163, 74, 0.18);" in html
    assert "--pass-bg: rgba(74, 222, 128, 0.18);" in html  # dark mode
    assert "--fail-bg: rgba(248, 113, 113, 0.18);" in html


def test_notes_and_usage_are_separate_sections() -> None:
    result = hand_built()
    html = render_html(result)
    assert "<h2>Notes</h2>" in html
    assert "<h2>LLM usage</h2>" not in html
    usage = LLMUsage(spec="groq:m", calls=3, cache_hits=1, input_tokens=10, output_tokens=5)
    both = render_html(result.model_copy(update={"llm_usage": [usage]}))
    assert "<h2>LLM usage</h2>" in both
    usage_section = both.split("<h2>LLM usage</h2>", 1)[1].split("</section>", 1)[0]
    assert "groq:m: 3 calls, 1 cache hits" in usage_section
    notes_section = both.split("<h2>Notes</h2>", 1)[1].split("</section>", 1)[0]
    assert "a note" in notes_section and "groq:m" not in notes_section
    only_usage = render_html(result.model_copy(update={"llm_usage": [usage], "notes": []}))
    assert "<h2>Notes</h2>" not in only_usage
    assert "<h2>LLM usage</h2>" in only_usage


def test_mode_row_only_when_set() -> None:
    assert "<dt>Mode</dt>" not in render_html(hand_built())
    html = render_html(hand_built().model_copy(update={"mode": "replay"}))
    assert "<dt>Mode</dt><dd>replay</dd>" in html


def test_write_html(tmp_path: Path) -> None:
    path = write_html(hand_built(), tmp_path / "deep" / "report.html")
    assert path.read_text(encoding="utf-8").startswith("<!DOCTYPE html>")
