"""LLM response cache (SQLite) and read-only replay cassettes (JSON).

A cassette is an export of the cache entries a set of runs used. Both share
one entry shape: {"spec", "kind", "response"} keyed by the request hash.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Any, Protocol

from convoprobe.errors import ConvoprobeError

CASSETTE_FORMAT = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS llm_cache (
    key TEXT PRIMARY KEY,
    spec TEXT NOT NULL,
    kind TEXT NOT NULL,
    response_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS llm_cache_usage (
    run_id TEXT NOT NULL,
    key TEXT NOT NULL,
    PRIMARY KEY(run_id, key)
);
"""


class LLMCache(Protocol):
    """Where LLM responses are looked up and stored."""

    def get(self, key: str) -> dict[str, Any] | None: ...

    def put(self, key: str, spec: str, kind: str, response: dict[str, Any]) -> None: ...

    def mark_used(self, run_id: str, key: str) -> None: ...


class SqliteLLMCache:
    """Read-write cache in its own SQLite file."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path)
        with self._conn:
            self._conn.executescript(_SCHEMA)

    def __enter__(self) -> SqliteLLMCache:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        self._conn.close()

    def get(self, key: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT response_json FROM llm_cache WHERE key = ?", (key,)
        ).fetchone()
        if row is None:
            return None
        response: dict[str, Any] = json.loads(row[0])
        return response

    def put(self, key: str, spec: str, kind: str, response: dict[str, Any]) -> None:
        with self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO llm_cache (key, spec, kind, response_json, created_at)"
                " VALUES (?, ?, ?, ?, ?)",
                (key, spec, kind, json.dumps(response), datetime.now(UTC).isoformat()),
            )

    def mark_used(self, run_id: str, key: str) -> None:
        with self._conn:
            self._conn.execute(
                "INSERT OR IGNORE INTO llm_cache_usage (run_id, key) VALUES (?, ?)",
                (run_id, key),
            )

    def export_cassette(self, run_ids: list[str], out_path: Path) -> int:
        """Write every cached entry used by these runs to a cassette. Returns the count."""
        marks = ",".join("?" for _ in run_ids)
        rows = (
            self._conn.execute(
                "SELECT DISTINCT c.key, c.spec, c.kind, c.response_json FROM llm_cache c"
                f" JOIN llm_cache_usage u ON u.key = c.key WHERE u.run_id IN ({marks})",
                run_ids,
            ).fetchall()
            if run_ids
            else []
        )
        entries = {
            key: {"spec": spec, "kind": kind, "response": json.loads(response)}
            for key, spec, kind, response in sorted(rows)
        }
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"format": CASSETTE_FORMAT, "entries": entries}
        text = json.dumps(payload, indent=1, sort_keys=True, ensure_ascii=False) + "\n"
        out_path.write_text(text, encoding="utf-8")
        return len(entries)


class JsonCassette:
    """Read-only cache backed by a cassette file. Used for replay."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        data = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or data.get("format") != CASSETTE_FORMAT:
            found = data.get("format") if isinstance(data, dict) else None
            raise ConvoprobeError(
                f"unsupported cassette format in {self.path}: {found!r}"
                f" (expected {CASSETTE_FORMAT})"
            )
        self._entries: dict[str, dict[str, Any]] = data.get("entries", {})

    def get(self, key: str) -> dict[str, Any] | None:
        entry = self._entries.get(key)
        return None if entry is None else entry["response"]

    def put(self, key: str, spec: str, kind: str, response: dict[str, Any]) -> None:
        raise ConvoprobeError(f"cassette {self.path} is read-only")

    def mark_used(self, run_id: str, key: str) -> None:
        return None
