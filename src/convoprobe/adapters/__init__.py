"""Adapters that let convoprobe talk to an agent under test."""

from __future__ import annotations

import builtins
import importlib
import inspect
import os
import sys
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from convoprobe.adapters.base import AgentAdapter
from convoprobe.adapters.callable import CallableAdapter
from convoprobe.adapters.http_adapter import HTTPAdapter, HTTPAdapterConfig
from convoprobe.adapters.langgraph_adapter import LangGraphAdapter
from convoprobe.errors import AdapterLoadError, ConvoprobeError

__all__ = [
    "AgentAdapter",
    "CallableAdapter",
    "HTTPAdapter",
    "HTTPAdapterConfig",
    "LangGraphAdapter",
    "load_adapter",
]

HTTP_PREFIX = "http:"


def _load_http(path_text: str) -> HTTPAdapter:
    path = Path(path_text)
    if not path.is_file():
        raise AdapterLoadError(f"adapter config {path}: file not found")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise AdapterLoadError(f"adapter config {path}: YAML error: {exc}") from exc
    if not isinstance(data, dict):
        raise AdapterLoadError(f"adapter config {path}: expected a mapping at the top level")
    try:
        config = HTTPAdapterConfig(**data)
    except ValidationError as exc:
        lines = [
            f"adapter config {path}: {'.'.join(str(p) for p in error['loc'])}: {error['msg']}"
            for error in exc.errors()
        ]
        raise AdapterLoadError("\n".join(lines)) from None
    try:
        return HTTPAdapter(config)
    except ConvoprobeError as exc:
        raise AdapterLoadError(f"adapter config {path}: {exc}") from exc


def _is_langgraph(obj: Any) -> bool:
    module = type(obj).__module__ or ""
    return module.startswith("langgraph") and all(
        hasattr(obj, name) for name in ("aget_state", "astream")
    )


def load_adapter(spec: str) -> AgentAdapter:
    """Load an adapter from "module.path:attr" or "http:<config.yaml>".

    A compiled LangGraph graph is wrapped in LangGraphAdapter (no resume value,
    so a graph that interrupts fails with a clear message). An AgentAdapter
    instance is used as is. Any other callable is wrapped in CallableAdapter.
    Classes are rejected, not instantiated.
    """
    if spec.startswith(HTTP_PREFIX):
        return _load_http(spec[len(HTTP_PREFIX) :])
    module_name, sep, attr = spec.partition(":")
    if not sep or not module_name or not attr or ":" in attr:
        raise AdapterLoadError(f"adapter {spec!r}: expected the form module.path:attr")
    # So --adapter examples.pkg.module:attr works from the project folder.
    cwd = os.getcwd()
    if cwd not in sys.path:
        sys.path.insert(0, cwd)
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        raise AdapterLoadError(
            f"adapter {spec!r}: cannot import {module_name}: {type(exc).__name__}: {exc}"
        ) from exc
    if not hasattr(module, attr):
        raise AdapterLoadError(f"adapter {spec!r}: module {module_name} has no attribute {attr}")
    obj = getattr(module, attr)
    if _is_langgraph(obj):
        try:
            return LangGraphAdapter(obj)
        except ConvoprobeError as exc:
            raise AdapterLoadError(f"adapter {spec!r}: {exc}") from exc
    if inspect.isclass(obj):
        raise AdapterLoadError(f"adapter {spec!r}: is a class, pass an instance or function")
    if isinstance(obj, AgentAdapter):
        return obj
    # The adapters.callable submodule shadows the builtin name in this package.
    if builtins.callable(obj):
        return CallableAdapter(obj)
    raise AdapterLoadError(f"adapter {spec!r}: is neither an AgentAdapter nor callable")
