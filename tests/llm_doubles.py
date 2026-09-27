"""A scripted chat model for LLMClient tests. Real BaseChatModel, no network."""

from __future__ import annotations

import json
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda
from pydantic import Field

USAGE = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}


class FakeChatModel(BaseChatModel):
    """Pops the next scripted output per call.

    Text calls: a str becomes the reply, an Exception is raised.
    Structured calls: a dict is the parsed result, "BAD" is a parsing error,
    an Exception is raised.
    """

    outputs: list[Any] = Field(default_factory=list)
    calls: int = 0

    @property
    def _llm_type(self) -> str:
        return "fake"

    def _next(self) -> Any:
        self.calls += 1
        output = self.outputs.pop(0)
        if isinstance(output, Exception):
            raise output
        return output

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        text = self._next()
        message = AIMessage(content=text, usage_metadata=USAGE)  # type: ignore[arg-type]
        return ChatResult(generations=[ChatGeneration(message=message)])

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        return self._generate(messages, stop, **kwargs)

    def with_structured_output(  # type: ignore[override]
        self, schema: Any, *, include_raw: bool = False, **kwargs: Any
    ) -> RunnableLambda[Any, dict[str, Any]]:
        def run(_messages: Any) -> dict[str, Any]:
            output = self._next()
            if output == "BAD":
                return {
                    "raw": AIMessage(content="not json"),
                    "parsed": None,
                    "parsing_error": ValueError("bad"),
                }
            raw = AIMessage(content=json.dumps(output), usage_metadata=USAGE)  # type: ignore[arg-type]
            return {"raw": raw, "parsed": output, "parsing_error": None}

        return RunnableLambda(run)
