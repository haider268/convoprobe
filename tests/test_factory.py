"""parse_spec and make_chat_model. Constructors make no network calls."""

from __future__ import annotations

import subprocess
import sys
from types import ModuleType

import pytest
from pydantic import SecretStr

from convoprobe.config import Settings
from convoprobe.errors import ConvoprobeError
from convoprobe.models import factory
from convoprobe.models.factory import make_chat_model, parse_spec


def settings(**overrides: object) -> Settings:
    return Settings(_env_file=None, **overrides)  # type: ignore[call-arg, arg-type]


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        ("ollama:qwen2.5:3b", ("ollama", "qwen2.5:3b")),
        ("groq:llama-3.1-8b-instant", ("groq", "llama-3.1-8b-instant")),
        ("gemini:some-model", ("gemini", "some-model")),
    ],
)
def test_parse_spec_valid(spec: str, expected: tuple[str, str]) -> None:
    assert parse_spec(spec) == expected


@pytest.mark.parametrize("spec", ["", "qwen", "openai:x", "ollama:"])
def test_parse_spec_invalid(spec: str) -> None:
    with pytest.raises(ConvoprobeError) as info:
        parse_spec(spec)
    assert str(info.value) == (
        f'model spec must look like provider:model, for example ollama:qwen2.5:3b (got "{spec}")'
    )


def test_ollama_model() -> None:
    pytest.importorskip("langchain_ollama")
    s = settings(ollama_base_url="http://ollama.example:11434", ollama_timeout_s=99.0)
    chat = make_chat_model("ollama:qwen2.5:3b", s, temperature=0.3)
    assert type(chat).__name__ == "ChatOllama"
    assert chat.model == "qwen2.5:3b"  # type: ignore[attr-defined]
    assert chat.base_url == "http://ollama.example:11434"  # type: ignore[attr-defined]
    assert chat.num_ctx == 4096  # type: ignore[attr-defined]
    assert chat.temperature == 0.3  # type: ignore[attr-defined]
    assert chat.client_kwargs == {"timeout": 99.0}  # type: ignore[attr-defined]


def test_groq_model() -> None:
    pytest.importorskip("langchain_groq")
    s = settings(groq_api_key=SecretStr("test-key"), llm_timeout_s=12.0)
    chat = make_chat_model("groq:llama-3.1-8b-instant", s, temperature=0.7)
    assert type(chat).__name__ == "ChatGroq"
    assert chat.model_name == "llama-3.1-8b-instant"  # type: ignore[attr-defined]
    assert chat.max_retries == 0  # type: ignore[attr-defined]
    assert chat.request_timeout == 12.0  # type: ignore[attr-defined]
    assert chat.groq_api_key.get_secret_value() == "test-key"  # type: ignore[attr-defined]


def test_groq_max_tokens_only_when_set() -> None:
    pytest.importorskip("langchain_groq")
    unset = make_chat_model(
        "groq:openai/gpt-oss-20b", settings(groq_api_key=SecretStr("k")), temperature=0.7
    )
    assert unset.max_tokens is None  # type: ignore[attr-defined]
    s = settings(groq_api_key=SecretStr("k"), groq_max_tokens=800)
    chat = make_chat_model("groq:qwen/qwen3.8-27b", s, temperature=0.7)
    assert chat.max_tokens == 800  # type: ignore[attr-defined]


def test_groq_max_tokens_passed_to_constructor(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[dict[str, object]] = []

    def fake_class(provider: str) -> object:
        return lambda **kwargs: seen.append(kwargs)

    monkeypatch.setattr(factory, "_chat_class", fake_class)
    make_chat_model("groq:m", settings(groq_api_key=SecretStr("k")), temperature=0.0)
    make_chat_model(
        "groq:m", settings(groq_api_key=SecretStr("k"), groq_max_tokens=200), temperature=0.0
    )
    assert "max_tokens" not in seen[0]
    assert seen[1]["max_tokens"] == 200


def test_groq_max_tokens_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_MAX_TOKENS", "800")
    assert settings().groq_max_tokens == 800
    monkeypatch.setenv("GROQ_MAX_TOKENS", "")
    assert settings().groq_max_tokens is None


def test_gemini_model() -> None:
    pytest.importorskip("langchain_google_genai")
    s = settings(google_api_key=SecretStr("test-key"))
    chat = make_chat_model("gemini:some-model", s, temperature=0.0)
    assert type(chat).__name__ == "ChatGoogleGenerativeAI"
    assert chat.model.endswith("some-model")  # type: ignore[attr-defined]
    assert chat.max_retries == 0  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    ("spec", "var"), [("groq:llama-3.1-8b-instant", "GROQ_API_KEY"), ("gemini:m", "GOOGLE_API_KEY")]
)
def test_missing_key(spec: str, var: str) -> None:
    with pytest.raises(ConvoprobeError) as info:
        make_chat_model(spec, settings(), temperature=0.0)
    assert str(info.value) == f"{var} is not set. Put it in .env (never in code or chat)."


def test_blank_env_key_counts_as_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "")
    assert settings().groq_api_key is None


@pytest.mark.parametrize("provider", ["ollama", "groq", "gemini"])
def test_missing_extra(provider: str, monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(name: str, package: str | None = None) -> ModuleType:
        raise ImportError(f"No module named {name}")

    monkeypatch.setattr(factory.importlib, "import_module", refuse)
    s = settings(groq_api_key=SecretStr("k"), google_api_key=SecretStr("k"))
    with pytest.raises(ConvoprobeError) as info:
        make_chat_model(f"{provider}:m", s, temperature=0.0)
    assert str(info.value) == (
        f'provider "{provider}" needs the extra: pip install "convoprobe[{provider}]"'
    )


def test_cli_and_models_do_not_import_providers() -> None:
    code = (
        "import sys, convoprobe.cli, convoprobe.models; "
        "bad=[m for m in sys.modules if m.startswith("
        "('langchain_ollama','langchain_groq','langchain_google_genai'))]; "
        "assert not bad, bad"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
