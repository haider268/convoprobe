"""Shared test setup. Every test is isolated from the real .env and its keys."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _no_real_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point settings at a missing env file and drop provider keys from the environment.

    Live tests opt back in by setting CONVOPROBE_ENV_FILE to the repo's .env.
    """
    monkeypatch.setenv("CONVOPROBE_ENV_FILE", str(tmp_path / "no.env"))
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
