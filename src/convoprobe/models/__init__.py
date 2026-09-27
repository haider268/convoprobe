"""Chat model factory, LLM client with cache and replay, retries, rate limits.

Importing this package never imports a provider package.
"""

from convoprobe.models.cache import JsonCassette, SqliteLLMCache
from convoprobe.models.client import LLMClient, ReplayMissError, StructuredOutputError
from convoprobe.models.factory import make_chat_model, parse_spec
from convoprobe.models.preflight import check_ollama
from convoprobe.models.registry import LLMRegistry
from convoprobe.models.retry import RetryPolicy

__all__ = [
    "JsonCassette",
    "LLMClient",
    "LLMRegistry",
    "ReplayMissError",
    "RetryPolicy",
    "SqliteLLMCache",
    "StructuredOutputError",
    "check_ollama",
    "make_chat_model",
    "parse_spec",
]
