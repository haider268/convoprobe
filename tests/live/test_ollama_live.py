"""Live Ollama smoke test. Excluded by default; run with: uv run pytest -m live"""

from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from convoprobe.config import load_settings
from convoprobe.models import LLMRegistry, SqliteLLMCache, check_ollama

pytestmark = pytest.mark.live


class Greeting(BaseModel):
    text: str


REPO_ENV = Path(__file__).resolve().parents[2] / ".env"


async def test_ollama_preflight_and_structured_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Live tests opt back in to the real .env. Keys are never printed.
    monkeypatch.setenv("CONVOPROBE_ENV_FILE", str(REPO_ENV))
    settings = load_settings()
    await check_ollama(settings.ollama_base_url, ["qwen2.5:3b", "llama3.2:3b"])
    with SqliteLLMCache(tmp_path / "cache.db") as cache:
        registry = LLMRegistry(settings, cache)
        client = registry.client("ollama:qwen2.5:3b", temperature=0)
        messages = [
            SystemMessage(content="Reply with JSON only."),
            HumanMessage(content="Give a one-sentence greeting for a shop at shop.example."),
        ]
        greeting = await client.structured(messages, Greeting)
        assert greeting.text.strip()
        again = await client.structured(messages, Greeting)
        assert again == greeting
        assert client.usage.calls == 1
        assert client.usage.cache_hits == 1
        print(f"\nlive greeting: {greeting.text!r}")
        print(f"usage: {client.usage.model_dump()}")
