"""Pydantic data models for scenarios, turns, conversations, and results.

Named datamodels.py, not models.py, because models/ holds the chat model factory.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Annotated, Any, Literal, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    Discriminator,
    Field,
    Tag,
    field_validator,
    model_validator,
)

Id = Annotated[str, Field(pattern=r"^[a-z0-9_]+$")]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# Conversation data


class ToolCallRecord(_Strict):
    """One tool call made by the agent."""

    name: str
    args: dict[str, Any] = {}
    result: Any = None


class AgentTurn(_Strict):
    """What an adapter returns for one agent reply."""

    text: str
    tool_calls: list[ToolCallRecord] = []
    metadata: dict[str, Any] = {}


class Message(_Strict):
    """One message in a conversation, from the user or the agent."""

    role: Literal["user", "agent"]
    text: str
    tool_calls: list[ToolCallRecord] = []
    latency_ms: float | None = None
    metadata: dict[str, Any] = {}


class Conversation(_Strict):
    """A full conversation between one persona and the agent."""

    id: str
    persona_id: str
    repeat_index: int = 0
    messages: list[Message] = []
    ended_reason: (
        Literal["max_turns", "simulator_stop", "script_end", "agent_error", "simulator_error"]
        | None
    ) = None
    error: str | None = None

    def agent_messages(self) -> list[Message]:
        """Agent messages in order."""
        return [m for m in self.messages if m.role == "agent"]

    def tool_calls_in_order(self) -> list[tuple[int, ToolCallRecord]]:
        """Every tool call in order, paired with its 0-based agent message index."""
        return [
            (index, call)
            for index, message in enumerate(self.agent_messages())
            for call in message.tool_calls
        ]


class Evidence(_Strict):
    """A judge citation. quote_found is set by convoprobe, not the model."""

    turn: int = Field(ge=1)
    quote: str
    quote_found: bool | None = None


class CheckResult(_Strict):
    """The outcome of one rubric item on one conversation."""

    rubric_id: str
    persona_id: str
    conversation_id: str
    status: Literal["pass", "fail", "skip", "error"]
    detail: str
    source: Literal["deterministic", "judge"] = "deterministic"
    verdict: Literal["yes", "no"] | None = None
    score: int | None = None
    evidence: list[Evidence] = []


# Personas


class LLMPersona(_Strict):
    """A persona played by the simulator LLM."""

    id: Id
    description: str = Field(min_length=1)
    goal: str = Field(min_length=1)


class ScriptedPersona(_Strict):
    """A persona that replays fixed user turns. No LLM involved."""

    id: Id
    script: list[str] = Field(min_length=1)
    description: str = ""


class GeneratePersonas(_Strict):
    """A request to generate N personas from the scenario."""

    generate: int = Field(ge=1, le=50)


def _persona_tag(value: Any) -> str:
    if isinstance(value, GeneratePersonas):
        return "generate"
    if isinstance(value, ScriptedPersona):
        return "scripted"
    if isinstance(value, LLMPersona):
        return "llm"
    if isinstance(value, dict):
        if "generate" in value:
            return "generate"
        if "script" in value:
            return "scripted"
    return "llm"


PersonaEntry = Annotated[
    Union[  # noqa: UP007
        Annotated[LLMPersona, Tag("llm")],
        Annotated[ScriptedPersona, Tag("scripted")],
        Annotated[GeneratePersonas, Tag("generate")],
    ],
    Discriminator(_persona_tag),
]


# Rubric items


class _RubricBase(_Strict):
    id: Id
    applies_to: list[str] | None = None


class ToolCalledItem(_RubricBase):
    type: Literal["tool_called"]
    tool: str


class ToolNotCalledItem(_RubricBase):
    type: Literal["tool_not_called"]
    tool: str


class ToolOrderItem(_RubricBase):
    type: Literal["tool_order"]
    before: str
    after: str

    @model_validator(mode="after")
    def _distinct(self) -> ToolOrderItem:
        if self.before == self.after:
            raise ValueError(f"before and after must differ, both are {self.before!r}")
        return self


class _RegexBase(_RubricBase):
    pattern: str
    speaker: Literal["agent", "user"] = "agent"
    ignore_case: bool = True

    @field_validator("pattern")
    @classmethod
    def _compiles(cls, value: str) -> str:
        try:
            re.compile(value)
        except re.error as exc:
            raise ValueError(f"invalid regex {value!r}: {exc}") from exc
        return value


class RegexPresentItem(_RegexBase):
    type: Literal["regex_present"]


class RegexAbsentItem(_RegexBase):
    type: Literal["regex_absent"]


class MaxTurnsItem(_RubricBase):
    type: Literal["max_turns"]
    value: int = Field(ge=1)


class MaxLatencyItem(_RubricBase):
    type: Literal["max_latency_ms"]
    value: float = Field(gt=0)


class ToolArgsItem(_RubricBase):
    type: Literal["tool_args"]
    tool: str
    required: list[str] = Field(min_length=1)


class LLMJudgeItem(_RubricBase):
    """Parsed now, scored in a later step."""

    type: Literal["llm_judge"]
    question: str = Field(min_length=1)
    pass_if: Literal["yes", "no"] | None = None
    scale: Literal["1-5"] | None = None
    min_pass: int | None = Field(default=None, ge=1, le=5)

    @model_validator(mode="after")
    def _mode(self) -> LLMJudgeItem:
        if (self.pass_if is None) == (self.scale is None):
            raise ValueError("set exactly one of pass_if or scale")
        if self.scale is not None and self.min_pass is None:
            raise ValueError("min_pass is required when scale is set")
        if self.pass_if is not None and self.min_pass is not None:
            raise ValueError("min_pass is only allowed with scale, not with pass_if")
        return self


RubricItem = Annotated[
    Union[  # noqa: UP007
        ToolCalledItem,
        ToolNotCalledItem,
        ToolOrderItem,
        RegexPresentItem,
        RegexAbsentItem,
        MaxTurnsItem,
        MaxLatencyItem,
        ToolArgsItem,
        LLMJudgeItem,
    ],
    Field(discriminator="type"),
]


# Scenario


class Thresholds(_Strict):
    pass_rate_drop: float = Field(default=0.05, ge=0, le=1)


class Scenario(_Strict):
    """A scenario file: agent description, personas, rubric, thresholds."""

    name: str = Field(min_length=1)
    agent_description: str = Field(min_length=1)
    max_turns: int = Field(default=10, ge=1, le=50)
    repeats: int = Field(default=1, ge=1, le=20)
    personas: list[PersonaEntry] = Field(min_length=1)
    rubric: list[RubricItem] = Field(min_length=1)
    thresholds: Thresholds = Thresholds()

    @model_validator(mode="after")
    def _cross_checks(self) -> Scenario:
        persona_ids: set[str] = set()
        for persona in self.concrete_personas():
            if persona.id in persona_ids:
                raise ValueError(f"duplicate persona id {persona.id!r}")
            persona_ids.add(persona.id)
        rubric_ids: set[str] = set()
        for item in self.rubric:
            if item.id in rubric_ids:
                raise ValueError(f"duplicate rubric id {item.id!r}")
            rubric_ids.add(item.id)
            for name in item.applies_to or []:
                if name not in persona_ids:
                    raise ValueError(
                        f"rubric item {item.id!r}: applies_to names unknown persona {name!r}"
                    )
        return self

    def concrete_personas(self) -> list[LLMPersona | ScriptedPersona]:
        """Personas with an id. Generate requests are left out."""
        return [p for p in self.personas if isinstance(p, LLMPersona | ScriptedPersona)]


# Results

ConversationStatus = Literal["pending", "done", "failed_transient", "failed", "skipped"]


class RubricSummary(_Strict):
    """Counts for one rubric item across all conversations in a run."""

    rubric_id: str
    passed: int
    failed: int
    errored: int
    skipped: int
    pass_rate: float | None
    pass_rate_stdev: float | None = None
    persona_rates: dict[str, float | None] = {}


class LLMUsage(_Strict):
    """Call counts and token totals for one model spec."""

    spec: str
    calls: int = 0
    cache_hits: int = 0
    retries: int = 0
    input_tokens: int = 0
    output_tokens: int = 0


class RunResult(_Strict):
    """Everything a run produced. Written to results.json."""

    run_id: str
    scenario_name: str
    adapter: str
    started_at: datetime
    finished_at: datetime | None
    conversation_status: dict[str, ConversationStatus]
    conversations: list[Conversation]
    checks: list[CheckResult]
    summary: list[RubricSummary]
    overall_pass_rate: float | None
    notes: list[str]
    models: dict[str, str] = {}
    llm_usage: list[LLMUsage] = []
    # Defaults keep results.json files from older versions loadable.
    scenario_hash: str = ""
    pass_rate_drop: float = 0.05
    mode: str = ""
    rubric: list[RubricItem] = []
