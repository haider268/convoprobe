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
    CheckResult,
    Conversation,
    ConversationStatus,
    GeneratePersonas,
    LLMJudgeItem,
    LLMPersona,
    LLMUsage,
    Message,
    RunResult,
    Scenario,
    ScriptedPersona,
)
from convoprobe.errors import TransientError
from convoprobe.jsonlog import JsonlLogger
from convoprobe.models.client import LLMClient, ReplayMissError, StructuredOutputError
from convoprobe.report.json_report import write_results
from convoprobe.scoring.deterministic import run_deterministic_checks
from convoprobe.scoring.judge import run_judge_checks
from convoprobe.simulator.personas import generate_personas
from convoprobe.simulator.simulator import ScriptedSimulator, UserSimulator
from convoprobe.store import RunStore
from convoprobe.summary import summarize

NO_SIMULATOR = "no simulator model configured"
RUNNABLE = {"pending", "failed_transient"}

Persona = LLMPersona | ScriptedPersona


class _SimulatorFailure(Exception):
    """The simulator, not the agent, failed (bad output or replay miss)."""


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
    judge_client: LLMClient | None = None,
    persona_client: LLMClient | None = None,
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

        if judge_client is None and any(isinstance(i, LLMJudgeItem) for i in scenario.rubric):
            notes.append("llm_judge items skipped: no judge model configured")

        personas: list[Persona] = list(scenario.concrete_personas())
        taken = {p.id for p in personas}
        for index, entry in enumerate(scenario.personas):
            if not isinstance(entry, GeneratePersonas):
                continue
            generated = store.load_generated_personas(run_id, index)
            if generated is None and persona_client is None:
                notes.append(f"generate: {entry.generate} personas not expanded: {NO_SIMULATOR}")
                continue
            if generated is None:
                assert persona_client is not None
                salt = f"personas:{store.scenario_hash(scenario)}:{index}"
                generated, generation_notes = await generate_personas(
                    scenario, entry.generate, persona_client, salt=salt, taken=taken
                )
                notes.extend(generation_notes)
                store.save_generated_personas(run_id, index, generated)
                log.log(
                    "personas_generated",
                    entry_index=index,
                    persona_ids=[p.id for p in generated],
                )
            taken.update(p.id for p in generated)
            personas.extend(generated)

        work: list[tuple[Persona, int, str]] = []
        for persona in personas:
            if isinstance(persona, LLMPersona) and llm_simulator is None:
                notes.append(f"persona {persona.id} skipped: {NO_SIMULATOR}")
            for repeat_index in range(scenario.repeats):
                cid = f"{persona.id}-r{repeat_index}"
                store.ensure_conversation(run_id, cid, persona.id, repeat_index)
                work.append((persona, repeat_index, cid))

        async def run_one(persona: Persona, repeat_index: int, cid: str) -> None:
            status = store.conversation_status(run_id, cid)
            rerun_skipped = (
                status == "skipped"
                and isinstance(persona, LLMPersona)
                and llm_simulator is not None
            )
            if status not in RUNNABLE and not rerun_skipped:
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
                    scenario,
                    adapter,
                    simulator,
                    persona,
                    repeat_index,
                    cid,
                    run_id,
                    store,
                    log,
                    judge_client,
                )

        await asyncio.gather(*(run_one(p, r, c) for p, r, c in work))

        store.finish_run(run_id)
        log.log("run_finished")

    started_at, finished_at = store.run_times(run_id)
    conversations = [c for c, _status, _error in store.load_conversations(run_id) if c]
    checks = store.load_checks(run_id)
    summary, overall = summarize(scenario.rubric, checks, conversations)
    models, usage = _models_and_usage(llm_simulator, judge_client, persona_client)
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
        models=models,
        llm_usage=usage,
    )
    write_results(result, out_dir)
    return result


def _models_and_usage(
    llm_simulator: UserSimulator | None,
    judge_client: LLMClient | None,
    persona_client: LLMClient | None,
) -> tuple[dict[str, str], list[LLMUsage]]:
    """Model specs and usage of the distinct LLM clients. Empty when none were given."""
    sim_client = getattr(llm_simulator, "client", None)
    sim_client = sim_client if isinstance(sim_client, LLMClient) else None
    clients: list[LLMClient] = []
    for client in (sim_client, judge_client, persona_client):
        if client is not None and all(client is not c for c in clients):
            clients.append(client)
    if llm_simulator is None and not clients:
        return {}, []
    models = {
        "simulator": sim_client.spec if sim_client else "none",
        "judge": judge_client.spec if judge_client else "none",
    }
    return models, [c.usage for c in clients]


async def _converse(
    scenario: Scenario,
    adapter: AgentAdapter,
    simulator: UserSimulator,
    persona: Persona,
    repeat_index: int,
    cid: str,
    run_id: str,
    store: RunStore,
    log: JsonlLogger,
    judge_client: LLMClient | None,
) -> None:
    """One conversation: talk, score, save. Transient errors leave it retryable."""
    conversation = Conversation(id=cid, persona_id=persona.id, repeat_index=repeat_index)
    log.log("conversation_started", conversation_id=cid, persona_id=persona.id)
    status: ConversationStatus = "done"
    error: str | None = None
    try:
        await adapter.start(cid)
        try:
            for _ in range(scenario.max_turns):
                try:
                    text = await simulator.next_message(
                        scenario, persona, conversation.messages, conversation_id=cid
                    )
                except (StructuredOutputError, ReplayMissError) as exc:
                    raise _SimulatorFailure(str(exc)) from exc
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
        _fail(store, log, run_id, conversation, "failed_transient", str(exc))
        return
    except _SimulatorFailure as exc:
        conversation.ended_reason = "simulator_error"
        _fail(store, log, run_id, conversation, "failed", f"simulator: {exc}")
        return
    except Exception as exc:
        status = "failed"
        error = f"{type(exc).__name__}: {exc}"
        conversation.ended_reason = "agent_error"
        conversation.error = error

    checks: list[CheckResult] = run_deterministic_checks(scenario.rubric, conversation)
    try:
        checks += await run_judge_checks(scenario.rubric, conversation, scenario, judge_client)
    except TransientError as exc:
        _fail(store, log, run_id, conversation, "failed_transient", f"judge: {exc}")
        return
    except ReplayMissError as exc:
        _fail(store, log, run_id, conversation, "failed", f"replay: {exc}")
        return

    store.save_conversation(run_id, conversation, status, error)
    store.save_checks(run_id, cid, checks)
    if status == "done":
        log.log(
            "conversation_finished", conversation_id=cid, ended_reason=conversation.ended_reason
        )
    else:
        log.log("conversation_failed", conversation_id=cid, error=error, transient=False)


def _fail(
    store: RunStore,
    log: JsonlLogger,
    run_id: str,
    conversation: Conversation,
    status: ConversationStatus,
    error: str,
) -> None:
    """Save a failed conversation with no checks."""
    conversation.error = error
    store.save_conversation(run_id, conversation, status, error)
    store.save_checks(run_id, conversation.id, [])
    log.log(
        "conversation_failed",
        conversation_id=conversation.id,
        error=error,
        transient=status == "failed_transient",
    )
