"""Runtime settings read from the environment and an optional .env file."""

from __future__ import annotations

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Provider keys and endpoints. Keys stay secret and are never logged."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    groq_api_key: SecretStr | None = None
    google_api_key: SecretStr | None = None
    ollama_base_url: str = "http://localhost:11434"
