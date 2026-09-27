"""Self-contained HTML report: one file, inline CSS, no scripts, no external URLs.

Python builds a plain view model; the Jinja template only renders it. Autoescape
is on and nothing from a run (agent, user, judge, or tool text) is marked safe.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from jinja2 import Environment, PackageLoader

from convoprobe import __version__
from convoprobe.datamodels import CheckResult, Conversation, Message, RunResult
from convoprobe.report.json_report import format_rate, persona_matrix, rate_band

_env = Environment(
    loader=PackageLoader("convoprobe", "report/templates"),
    autoescape=True,
    trim_blocks=True,
    lstrip_blocks=True,
)

BAD = ("fail", "error")


@dataclass
class ToolView:
    name: str
    args: str
    result: str


@dataclass
class AgentView:
    text: str
    tools: list[ToolView]
    latency_ms: float | None
    node_path: list[str]


@dataclass
class TurnView:
    anchor: str
    number: int
    user: str | None
    agents: list[AgentView] = field(default_factory=list)
    cited: bool = False


@dataclass
class EvidenceView:
    turn: int
    quote: str
    found: bool | None
    href: str


@dataclass
class CheckView:
    rubric_id: str
    status: str
    source: str
    detail: str
    verdict: str | None
    score: int | None
    evidence: list[EvidenceView]


@dataclass
class ConversationView:
    id: str
    persona_id: str
    status: str
    ended_reason: str
    failed: int
    open: bool
    checks: list[CheckView]
    turns: list[TurnView]
    error: str | None


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str, ensure_ascii=False)


def _utc(value: datetime | None) -> str:
    if value is None:
        return "-"
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")


def turn_anchor(conversation_id: str, number: int) -> str:
    return f"{conversation_id}-turn-{number}"


def _agent_view(message: Message) -> AgentView:
    node_path = message.metadata.get("node_path") or []
    return AgentView(
        text=message.text,
        tools=[ToolView(c.name, _json(c.args), _json(c.result)) for c in message.tool_calls],
        latency_ms=message.latency_ms,
        node_path=[str(n) for n in node_path] if isinstance(node_path, list) else [],
    )


def _turns(conversation: Conversation, cited: set[int]) -> list[TurnView]:
    """Group messages into exchanges numbered like the judge transcript."""
    turns: list[TurnView] = []
    for message in conversation.messages:
        if message.role == "user":
            number = len(turns) + 1
            turns.append(TurnView(turn_anchor(conversation.id, number), number, message.text))
            continue
        if not turns:
            turns.append(TurnView(turn_anchor(conversation.id, 1), 1, None))
        turns[-1].agents.append(_agent_view(message))
    for turn in turns:
        turn.cited = turn.number in cited
    return turns


def _check_view(check: CheckResult) -> CheckView:
    return CheckView(
        rubric_id=check.rubric_id,
        status=check.status,
        source=check.source,
        detail=check.detail,
        verdict=check.verdict,
        score=check.score,
        evidence=[
            EvidenceView(
                turn=e.turn,
                quote=e.quote,
                found=e.quote_found,
                href="#" + turn_anchor(check.conversation_id, e.turn),
            )
            for e in check.evidence
        ],
    )


def _conversation_views(result: RunResult) -> list[ConversationView]:
    views = []
    for conversation in result.conversations:
        checks = [c for c in result.checks if c.conversation_id == conversation.id]
        bad = [c for c in checks if c.status in BAD]
        cited = {
            e.turn for c in checks if c.source == "judge" and c.status in BAD for e in c.evidence
        }
        ordered = bad + [c for c in checks if c.status not in BAD]
        status = result.conversation_status.get(conversation.id, "done")
        views.append(
            ConversationView(
                id=conversation.id,
                persona_id=conversation.persona_id,
                status=status,
                ended_reason=conversation.ended_reason or "-",
                failed=len(bad),
                open=bool(bad),
                checks=[_check_view(c) for c in ordered],
                turns=_turns(conversation, cited),
                error=conversation.error,
            )
        )
    problem = {"failed", "failed_transient"}
    views.sort(key=lambda v: (not (v.failed or v.status in problem), v.id))
    return views


def _source(result: RunResult, rubric_id: str) -> str:
    judged = any(c.rubric_id == rubric_id and c.source == "judge" for c in result.checks)
    return "judge" if judged else "deterministic"


def build_view(result: RunResult) -> dict[str, Any]:
    """Everything the template needs, already formatted."""
    personas, rubric_ids, rates = persona_matrix(result)
    show_stdev = any(row.pass_rate_stdev is not None for row in result.summary)
    models = (
        f"simulator {result.models.get('simulator', 'none')},"
        f" judge {result.models.get('judge', 'none')}"
        if result.models
        else "no LLM"
    )
    return {
        "result": result,
        "models": models,
        "started": _utc(result.started_at),
        "finished": _utc(result.finished_at),
        "overall": format_rate(result.overall_pass_rate),
        "overall_band": rate_band(result.overall_pass_rate) or "none",
        "show_stdev": show_stdev,
        "rubric_rows": [
            {
                "id": row.rubric_id,
                "source": _source(result, row.rubric_id),
                "passed": row.passed,
                "failed": row.failed,
                "errored": row.errored,
                "skipped": row.skipped,
                "rate": format_rate(row.pass_rate),
                "band": rate_band(row.pass_rate) or "none",
                "stdev": "-" if row.pass_rate_stdev is None else f"{row.pass_rate_stdev:.2f}",
            }
            for row in result.summary
        ],
        "rubric_ids": rubric_ids,
        "matrix": [
            {
                "persona": persona_id,
                "cells": [
                    {
                        "text": format_rate(rates[persona_id].get(rubric_id)),
                        "band": rate_band(rates[persona_id].get(rubric_id)) or "none",
                    }
                    for rubric_id in rubric_ids
                ],
            }
            for persona_id in personas
        ],
        "usage": [
            f"{u.spec}: {u.calls} calls, {u.cache_hits} cache hits, {u.retries} retries,"
            f" {u.input_tokens} in / {u.output_tokens} out tokens"
            for u in result.llm_usage
        ],
        "conversations": _conversation_views(result),
        "version": __version__,
    }


def render_html(result: RunResult) -> str:
    """The report as one HTML string."""
    return _env.get_template("report.html.j2").render(**build_view(result))


def write_html(result: RunResult, path: Path) -> Path:
    """Write the report and return its path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_html(result), encoding="utf-8", newline="\n")
    return path
