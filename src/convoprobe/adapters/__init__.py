"""Adapters that let convoprobe talk to an agent under test."""

from __future__ import annotations

import importlib
import inspect

from convoprobe.adapters.base import AgentAdapter
from convoprobe.adapters.callable import CallableAdapter
from convoprobe.errors import AdapterLoadError

__all__ = ["AgentAdapter", "CallableAdapter", "load_adapter"]


def load_adapter(spec: str) -> AgentAdapter:
    """Load "module.path:attr" as an adapter.

    An AgentAdapter instance is used as is. Any other callable is wrapped in
    CallableAdapter. Classes are rejected, not instantiated.
    """
    module_name, sep, attr = spec.partition(":")
    if not sep or not module_name or not attr or ":" in attr:
        raise AdapterLoadError(f"adapter {spec!r}: expected the form module.path:attr")
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        raise AdapterLoadError(
            f"adapter {spec!r}: cannot import {module_name}: {type(exc).__name__}: {exc}"
        ) from exc
    if not hasattr(module, attr):
        raise AdapterLoadError(f"adapter {spec!r}: module {module_name} has no attribute {attr}")
    obj = getattr(module, attr)
    if inspect.isclass(obj):
        raise AdapterLoadError(f"adapter {spec!r}: is a class, pass an instance or function")
    if isinstance(obj, AgentAdapter):
        return obj
    if callable(obj):
        return CallableAdapter(obj)
    raise AdapterLoadError(f"adapter {spec!r}: is neither an AgentAdapter nor callable")
