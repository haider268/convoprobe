"""LLMRegistry: builds clients and shares rate limiters (per model, or per provider for Ollama)."""

from __future__ import annotations

from convoprobe.config import Settings
from convoprobe.datamodels import LLMUsage
from convoprobe.models import factory
from convoprobe.models.cache import LLMCache
from convoprobe.models.client import LLMClient
from convoprobe.models.ratelimit import ProviderLimiter

# json_schema works on Ollama and on every current Groq chat model. Groq's default
# (function calling) has failed with "tool_use_failed". Gemini keeps its default.
STRUCTURED_METHOD: dict[str, str | None] = {"ollama": "json_schema", "groq": "json_schema"}


class LLMRegistry:
    """Creates LLMClients. In replay mode no provider package is imported."""

    def __init__(
        self,
        settings: Settings,
        cache: LLMCache,
        *,
        replay_only: bool = False,
        run_id: str | None = None,
    ) -> None:
        self.settings = settings
        self.cache = cache
        self.replay_only = replay_only
        self.run_id = run_id
        self._limiters: dict[str, ProviderLimiter] = {}
        self._clients: list[LLMClient] = []

    def limiter(self, spec: str) -> ProviderLimiter:
        """The shared limiter for a model spec, created on first use.

        Groq and Gemini limits are per model, so each spec gets its own limiter
        (the provider's settings apply to each). Ollama shares one local CPU, so
        every Ollama model shares one limiter.
        """
        provider, _model = factory.parse_spec(spec)
        key = provider if provider == "ollama" else spec
        if key not in self._limiters:
            s = self.settings
            if provider == "ollama":
                limiter = ProviderLimiter(None, None, s.ollama_concurrency)
            elif provider == "groq":
                limiter = ProviderLimiter(s.groq_rpm, s.groq_tpm, s.groq_concurrency)
            else:
                limiter = ProviderLimiter(s.gemini_rpm, s.gemini_tpm, s.gemini_concurrency)
            self._limiters[key] = limiter
        return self._limiters[key]

    def client(self, spec: str, temperature: float) -> LLMClient:
        provider, _model = factory.parse_spec(spec)
        chat_model = (
            None if self.replay_only else factory.make_chat_model(spec, self.settings, temperature)
        )
        client = LLMClient(
            spec,
            cache=self.cache,
            chat_model=chat_model,
            temperature=temperature,
            limiter=self.limiter(spec),
            replay_only=self.replay_only,
            run_id=self.run_id,
            structured_method=STRUCTURED_METHOD.get(provider),
        )
        self._clients.append(client)
        return client

    def usage(self) -> list[LLMUsage]:
        """Usage for every client created, in creation order."""
        return [c.usage for c in self._clients]
