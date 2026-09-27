"""Runtime settings read from the environment and an optional .env file."""

from __future__ import annotations

import os

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# Hosted defaults, chosen from review-lane measurements on 28 Sep 2026 (gpt-oss-120b 4/4
# valid judge outputs, qwen3.8-27b 1/4). The judge should be the strongest available model
# and, for real agents, a different model family from the agent under test.
HOSTED_SIMULATOR = "groq:openai/gpt-oss-20b"
HOSTED_JUDGE = "groq:openai/gpt-oss-120b"


class Settings(BaseSettings):
    """Provider keys, endpoints, models, and rate limits. Keys are never logged."""

    # env_ignore_empty: a blank "GROQ_API_KEY=" line counts as not set.
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", env_ignore_empty=True
    )

    groq_api_key: SecretStr | None = None
    google_api_key: SecretStr | None = None
    ollama_base_url: str = "http://localhost:11434"

    convoprobe_simulator: str = "ollama:qwen2.5:3b"
    convoprobe_judge: str = "ollama:llama3.2:3b"

    # Local CPU inference: one call at a time, calls can take 5 to 15 s on a laptop.
    ollama_concurrency: int = 1
    ollama_timeout_s: float = 180.0

    llm_timeout_s: float = 60.0

    # Groq free tier as last verified 28 Sep 2026 from response headers: about 30
    # requests/min, 8,000 tokens/min, 1,000 requests/day per model.
    groq_rpm: float = 30.0
    groq_tpm: float | None = 8000.0
    groq_concurrency: int = 2

    # Conservative default. Verify the current Gemini free tier before relying on it.
    gemini_rpm: float = 10.0
    gemini_tpm: float | None = None
    gemini_concurrency: int = 2


def load_settings() -> Settings:
    """Settings from the env file named by CONVOPROBE_ENV_FILE (default .env)."""
    env_file = os.environ.get("CONVOPROBE_ENV_FILE", ".env")
    return Settings(_env_file=env_file)  # type: ignore[call-arg]
