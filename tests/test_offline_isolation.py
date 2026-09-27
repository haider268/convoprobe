"""Offline tests can never see a real key, even when the repo has a .env."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from convoprobe.config import load_settings

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


def test_default_groq_tpm() -> None:
    assert load_settings().groq_tpm == 8000.0
