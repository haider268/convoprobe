"""Scaffold smoke tests: the package imports and the CLI answers."""

import pytest
from typer.testing import CliRunner

import convoprobe
from convoprobe.cli import app
from convoprobe.config import Settings


def test_version_string() -> None:
    assert convoprobe.__version__ == "0.0.1"


def test_cli_version() -> None:
    result = CliRunner().invoke(app, ["version"])
    assert result.exit_code == 0
    assert result.stdout.strip() == "0.0.1"


def test_settings_defaults_without_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    assert settings.groq_api_key is None
    assert settings.ollama_base_url == "http://localhost:11434"
