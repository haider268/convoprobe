"""Map "provider:model" strings to LangChain chat models.

Provider packages are imported lazily, so core convoprobe works without any extra.
"""

from __future__ import annotations

import importlib
from typing import Any

from langchain_core.language_models import BaseChatModel

from convoprobe.config import Settings
from convoprobe.errors import ConvoprobeError

PROVIDERS = ("ollama", "groq", "gemini")

_CLASSES = {
    "ollama": ("langchain_ollama", "ChatOllama"),
    "groq": ("langchain_groq", "ChatGroq"),
    "gemini": ("langchain_google_genai", "ChatGoogleGenerativeAI"),
}


def parse_spec(spec: str) -> tuple[str, str]:
    """Split "provider:model" on the first colon. Model names may contain colons."""
    provider, sep, model = spec.partition(":")
    if not sep or provider not in PROVIDERS or not model:
        raise ConvoprobeError(
            "model spec must look like provider:model, for example ollama:qwen2.5:3b"
            f' (got "{spec}")'
        )
    return provider, model


def _chat_class(provider: str) -> Any:
    module_name, class_name = _CLASSES[provider]
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise ConvoprobeError(
            f'provider "{provider}" needs the extra: pip install "convoprobe[{provider}]"'
        ) from exc
    return getattr(module, class_name)


def make_chat_model(spec: str, settings: Settings, temperature: float) -> BaseChatModel:
    """Build a chat model. max_retries=0 because convoprobe does its own retries."""
    provider, model = parse_spec(spec)
    if provider == "ollama":
        chat: BaseChatModel = _chat_class("ollama")(
            model=model,
            base_url=settings.ollama_base_url,
            temperature=temperature,
            num_ctx=4096,
            client_kwargs={"timeout": settings.ollama_timeout_s},
        )
        return chat
    if provider == "groq":
        if settings.groq_api_key is None:
            raise ConvoprobeError(
                "GROQ_API_KEY is not set. Put it in .env (never in code or chat)."
            )
        chat = _chat_class("groq")(
            model=model,
            api_key=settings.groq_api_key.get_secret_value(),
            temperature=temperature,
            max_retries=0,
            timeout=settings.llm_timeout_s,
        )
        return chat
    if settings.google_api_key is None:
        raise ConvoprobeError("GOOGLE_API_KEY is not set. Put it in .env (never in code or chat).")
    chat = _chat_class("gemini")(
        model=model,
        google_api_key=settings.google_api_key.get_secret_value(),
        temperature=temperature,
        max_retries=0,
        timeout=settings.llm_timeout_s,
    )
    return chat
