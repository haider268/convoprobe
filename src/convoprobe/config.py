"""Runtime settings read from the environment and an optional .env file."""

from __future__ import annotations

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


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

    # Groq free tier as last verified Sep 2026: about 30 requests/min, 6,000 tokens/min.
    groq_rpm: float = 30.0
    groq_tpm: float | None = 6000.0
    groq_concurrency: int = 2

    # Conservative default. Verify the current Gemini free tier before relying on it.
    gemini_rpm: float = 10.0
    gemini_tpm: float | None = None
    gemini_concurrency: int = 2
