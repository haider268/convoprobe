"""Run orchestration: conversations, concurrency, resume, checks, results."""

from __future__ import annotations

import asyncio
import secrets
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from convoprobe.adapters.base import AgentAdapter
from convoprobe.datamodels import (
    Conversation,
    ConversationStatus,
    GeneratePersonas,
    LLMPersona,
    Message,
    RunResult,
    Scenario,
    ScriptedPersona,
)
from convoprobe.errors import TransientError
from convoprobe.jsonlog import JsonlLogger
from convoprobe.report.json_report import write_results
from convoprobe.scoring.deterministic import run_deterministic_checks
from convoprobe.simulator.simulator import ScriptedSimulator, UserSimulator
from convoprobe.store import RunStore
from convoprobe.summary import summarize

NO_SIMULATOR = "no simulator model configured"
RUNNABLE = {"pending", "failed_transient"}


def new_run_id() -> str:
    """UTC timestamp plus a short random suffix."""
    return datetime.now(UTC).strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(2)


async def run_scenario(
    scenario: Scenario,
    adapter: AgentAdapter,
    *,
    store: RunStore,
    out_dir: Path,
    adapter_label: str,
    run_id: str | None = None,
    concurrency: int = 4,
    llm_simulator: UserSimulator | None = None,
) -> RunResult:
    """Run every persona x repeat, resume unfinished work, score, write results."""
    run_id = run_id or new_run_id()
    out_dir = Path(out_dir)
    resumed = store.create_or_resume_run(run_id, scenario, adapter_label)
    notes: list[str] = []
    scripted = ScriptedSimulator()
    semaphore = asyncio.Semaphore(concurrency)

    with JsonlLogger(out_dir / run_id / "log.jsonl", run_id) as log:
        log.log("run_started", resumed=resumed, scenario=scenario.name, adapter=adapter_label)

        for entry in scenario.personas:
            if isinstance(entry, GeneratePersonas):
                notes.append(
                    f"generate: {entry.generate} personas not expanded"
                    " (needs a simulator model, added later)"
                )

        work: list[tuple[LLMPersona | ScriptedPersona, int, str]] = []
        for persona in scenario.concrete_personas():
            if isinstance(persona, LLMPersona) and llm_simulator is None:
                notes.append(f"persona {persona.id} skipped: {NO_SIMULATOR}")
            for repeat_index in range(scenario.repeats):
                cid = f"{persona.id}-r{repeat_index}"
                store.ensure_conversation(run_id, cid, persona.id, repeat_index)
                work.append((persona, repeat_index, cid))

        async def run_one(
            persona: LLMPersona | ScriptedPersona, repeat_index: int, cid: str
        ) -> None:
            status = store.conversation_status(run_id, cid)
            if status not in RUNNABLE:
                return
            if isinstance(persona, ScriptedPersona):
                simulator: UserSimulator = scripted
            elif llm_simulator is not None:
                simulator = llm_simulator
            else:
                skipped = Conversation(
                    id=cid, persona_id=persona.id, repeat_index=repeat_index, error=NO_SIMULATOR
                )
                store.save_conversation(run_id, skipped, "skipped", NO_SIMULATOR)
                log.log("conversation_skipped", conversation_id=cid, reason=NO_SIMULATOR)
                return
            async with semaphore:
                await _converse(
                    scenario, adapter, simulator, persona, repeat_index, cid, run_id, store, log
                )

        await asyncio.gather(*(run_one(p, r, c) for p, r, c in work))

        store.finish_run(run_id)
        log.log("run_finished")

    started_at, finished_at = store.run_times(run_id)
    conversations = [c for c, _status, _error in store.load_conversations(run_id) if c]
    checks = store.load_checks(run_id)
    summary, overall = summarize(scenario.rubric, checks)
    result = RunResult(
        run_id=run_id,
        scenario_name=scenario.name,
        adapter=adapter_label,
        started_at=datetime.fromisoformat(started_at),
        finished_at=datetime.fromisoformat(finished_at) if finished_at else None,
        conversation_status=cast(
            dict[str, ConversationStatus], store.conversation_statuses(run_id)
        ),
        conversations=conversations,
        checks=checks,
        summary=summary,
        overall_pass_rate=overall,
        notes=notes,
    )
    write_results(result, out_dir)
    return result


async def _converse(
    scenario: Scenario,
    adapter: AgentAdapter,
    simulator: UserSimulator,
    persona: LLMPersona | ScriptedPersona,
    repeat_index: int,
    cid: str,
    run_id: str,
    store: RunStore,
    log: JsonlLogger,
) -> None:
    """One conversation: talk, save, score. Transient errors leave it retryable."""
    conversation = Conversation(id=cid, persona_id=persona.id, repeat_index=repeat_index)
    log.log("conversation_started", conversation_id=cid, persona_id=persona.id)
    try:
        await adapter.start(cid)
        try:
            for _ in range(scenario.max_turns):
                text = await simulator.next_message(scenario, persona, conversation.messages)
                if text is None:
                    stop = isinstance(persona, ScriptedPersona)
                    conversation.ended_reason = "script_end" if stop else "simulator_stop"
                    break
                conversation.messages.append(Message(role="user", text=text))
                started = time.perf_counter()
                turn = await adapter.send(cid, text)
                latency_ms = round((time.perf_counter() - started) * 1000, 1)
                conversation.messages.append(
                    Message(
                        role="agent",
                        text=turn.text,
                        tool_calls=turn.tool_calls,
                        latency_ms=latency_ms,
                    )
                )
                log.log(
                    "turn",
                    conversation_id=cid,
                    turn=len(conversation.agent_messages()),
                    latency_ms=latency_ms,
                    tool_calls=[call.name for call in turn.tool_calls],
                )
            else:
                conversation.ended_reason = "max_turns"
        finally:
            await adapter.end(cid)
    except TransientError as exc:
        conversation.error = str(exc)
        store.save_conversation(run_id, conversation, "failed_transient", str(exc))
        log.log("conversation_failed", conversation_id=cid, error=str(exc), transient=True)
        return
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        conversation.ended_reason = "agent_error"
        conversation.error = error
        store.save_conversation(run_id, conversation, "failed", error)
        store.save_checks(run_id, cid, run_deterministic_checks(scenario.rubric, conversation))
        log.log("conversation_failed", conversation_id=cid, error=error, transient=False)
        return
    store.save_conversation(run_id, conversation, "done")
    store.save_checks(run_id, cid, run_deterministic_checks(scenario.rubric, conversation))
    log.log("conversation_finished", conversation_id=cid, ended_reason=conversation.ended_reason)
