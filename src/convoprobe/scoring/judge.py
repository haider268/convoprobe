"""LLM judge for llm_judge rubric items.

The model gives a verdict or score with evidence. Code checks the evidence
points at real turns and whether each quote is really there. Invalid output
is retried once, then recorded as an error, never as a pass.
"""

from __future__ import annotations

import json
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from convoprobe.datamodels import (
    CheckResult,
    Conversation,
    Evidence,
    LLMJudgeItem,
    RubricItem,
    Scenario,
)
from convoprobe.models.client import LLMClient, ReplayMissError, StructuredOutputError

JUDGE_SYSTEM = (
    "You are a strict QA reviewer of a conversation between a user and an AI agent."
    " Judge only what is in the transcript. Do not assume anything that is not shown."
    " Cite evidence as turn numbers with short exact quotes from those turns."
)

JUDGE_HUMAN = """What the agent is supposed to do:
{agent_description}

Transcript:
{transcript}

Question: {question}
{instruction}"""

YES_NO_INSTRUCTION = "Answer verdict yes or no."
SCALE_INSTRUCTION = "Give a score from 1 (very poor) to 5 (excellent)."

REASONING_LIMIT = 300


# Evidence as the model sees it. A comment, not a docstring: pydantic would send a
# docstring to the model as the schema description.
class JudgeEvidence(BaseModel):
    turn: int
    quote: str


# Field order matters: reasoning first, so the model reasons before it decides.
class YesNoVerdict(BaseModel):
    reasoning: str
    verdict: Literal["yes", "no"]
    evidence: list[JudgeEvidence]


class ScaleVerdict(BaseModel):
    reasoning: str
    score: int = Field(ge=1, le=5)
    evidence: list[JudgeEvidence]


def _transcript_lines(conversation: Conversation) -> list[tuple[int, str]]:
    """(turn number, line). A user message and the agent reply after it share N."""
    lines: list[tuple[int, str]] = []
    turn = 0
    for message in conversation.messages:
        if message.role == "user":
            turn += 1
            lines.append((turn, f"turn {turn} USER: {message.text}"))
            continue
        turn = max(turn, 1)
        lines.append((turn, f"turn {turn} AGENT: {message.text}"))
        for call in message.tool_calls:
            args = json.dumps(call.args, sort_keys=True)
            result = json.dumps(call.result, sort_keys=True, default=str)
            lines.append((turn, f"turn {turn} AGENT TOOL CALL: {call.name}({args}) -> {result}"))
    return lines


def format_transcript(conversation: Conversation) -> str:
    """The transcript the judge reads, one line per message or tool call."""
    return "\n".join(line for _turn, line in _transcript_lines(conversation))


def _normalize(text: str) -> str:
    return " ".join(text.split()).casefold()


def _result(
    item: LLMJudgeItem, conversation: Conversation, status: str, detail: str, **extra: object
) -> CheckResult:
    return CheckResult.model_validate(
        {
            "rubric_id": item.id,
            "persona_id": conversation.persona_id,
            "conversation_id": conversation.id,
            "status": status,
            "detail": detail,
            "source": "judge",
            **extra,
        }
    )


async def judge_item(
    item: LLMJudgeItem,
    conversation: Conversation,
    scenario: Scenario,
    client: LLMClient,
    salt: str | None = None,
) -> CheckResult:
    """Ask the judge about one rubric item. TransientError and ReplayMissError propagate.

    salt overrides the cache salt (default "judge:<conversation id>:<item id>"); the
    retry uses the same salt plus ":retry1".
    """
    lines = _transcript_lines(conversation)
    max_turn = max((turn for turn, _line in lines), default=0)
    turn_text = {
        turn: _normalize(" ".join(line for t, line in lines if t == turn))
        for turn in range(1, max_turn + 1)
    }
    schema: type[YesNoVerdict] | type[ScaleVerdict] = (
        YesNoVerdict if item.pass_if is not None else ScaleVerdict
    )
    instruction = YES_NO_INSTRUCTION if item.pass_if is not None else SCALE_INSTRUCTION
    messages = [
        SystemMessage(content=JUDGE_SYSTEM),
        HumanMessage(
            content=JUDGE_HUMAN.format(
                agent_description=scenario.agent_description.strip(),
                transcript="\n".join(line for _turn, line in lines),
                question=item.question,
                instruction=instruction,
            )
        ),
    ]
    base_salt = salt if salt is not None else f"judge:{conversation.id}:{item.id}"
    reason = ""
    first_miss: ReplayMissError | None = None
    for attempt_salt in (base_salt, base_salt + ":retry1"):
        try:
            answer = await client.structured(messages, schema, salt=attempt_salt)
        except StructuredOutputError as exc:
            reason = str(exc)
            continue
        except ReplayMissError as miss:
            # Invalid output is never cached, so a recorded run whose first attempt
            # failed holds only the retry answer. Replay follows that path; a second
            # miss raises the original one.
            if not client.replay_only or first_miss is not None:
                raise first_miss or miss from None
            first_miss = miss
            continue
        bad = [e.turn for e in answer.evidence if not 1 <= e.turn <= max_turn]
        if bad:
            reason = f"evidence turn {bad[0]} is outside 1..{max_turn}"
            continue
        evidence = [
            Evidence(
                turn=e.turn,
                quote=e.quote,
                quote_found=_normalize(e.quote) in turn_text[e.turn],
            )
            for e in answer.evidence
        ]
        reasoning = answer.reasoning[:REASONING_LIMIT]
        if isinstance(answer, YesNoVerdict):
            passed = answer.verdict == item.pass_if
            detail = f"verdict {answer.verdict} (pass if {item.pass_if}): {reasoning}"
            extra: dict[str, object] = {"verdict": answer.verdict}
        else:
            min_pass = item.min_pass or 1
            passed = answer.score >= min_pass
            detail = f"score {answer.score}/5 (min {min_pass}): {reasoning}"
            extra = {"score": answer.score}
        missing = sum(1 for e in evidence if e.quote_found is False)
        if missing:
            detail += f" [{missing} quote(s) not found in transcript]"
        return _result(
            item,
            conversation,
            "pass" if passed else "fail",
            detail,
            evidence=evidence,
            **extra,
        )
    return _result(item, conversation, "error", f"judge output invalid: {reason}")


async def run_judge_checks(
    rubric: list[RubricItem],
    conversation: Conversation,
    scenario: Scenario,
    client: LLMClient | None,
) -> list[CheckResult]:
    """Judge every llm_judge item in rubric order, one at a time."""
    results: list[CheckResult] = []
    for item in rubric:
        if not isinstance(item, LLMJudgeItem):
            continue
        if item.applies_to is not None and conversation.persona_id not in item.applies_to:
            results.append(
                _result(
                    item,
                    conversation,
                    "skip",
                    f"not applicable to persona {conversation.persona_id}",
                )
            )
        elif client is None:
            results.append(_result(item, conversation, "skip", "no judge model configured"))
        else:
            results.append(await judge_item(item, conversation, scenario, client))
    return results
