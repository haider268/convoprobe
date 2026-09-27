"""Offline tests can never see a real key, even when the repo has a .env."""

from __future__ import annotations

import os
import socket
from pathlib import Path

import pytest

from convoprobe.config import load_settings
from convoprobe.errors import ConvoprobeError
from convoprobe.models.preflight import check_ollama
from tests.conftest import NetworkBlockedError

REPO_ENV = Path(__file__).resolve().parents[1] / ".env"


def test_load_settings_has_no_keys_in_tests() -> None:
    settings = load_settings()
    assert settings.groq_api_key is None
    assert settings.google_api_key is None


def test_env_file_points_away_from_repo() -> None:
    env_file = Path(os.environ["CONVOPROBE_ENV_FILE"])
    assert env_file.resolve() != REPO_ENV
    assert not env_file.exists()


def test_load_settings_reads_named_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = tmp_path / "fake.env"
    fake.write_text("GROQ_API_KEY=test-key\nGROQ_TPM=1234\n", encoding="utf-8")
    monkeypatch.setenv("CONVOPROBE_ENV_FILE", str(fake))
    settings = load_settings()
    assert settings.groq_api_key is not None
    assert settings.groq_api_key.get_secret_value() == "test-key"
    assert settings.groq_tpm == 1234.0


async def test_ollama_port_is_blocked() -> None:
    with pytest.raises(ConvoprobeError, match="Ollama is not reachable"):
        await check_ollama("http://127.0.0.1:11434", ["qwen2.5:3b"])


def test_dns_and_remote_connect_blocked() -> None:
    with pytest.raises(NetworkBlockedError):
        socket.getaddrinfo("api.groq.com", 443)
    with socket.socket() as sock, pytest.raises(NetworkBlockedError):
        sock.connect(("203.0.113.10", 443))


def test_default_groq_tpm() -> None:
    assert load_settings().groq_tpm == 8000.0
