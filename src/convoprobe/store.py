"""SQLite store for runs, conversations, and check results.

Rerunning a run id resumes it: done conversations are kept, transient
failures are retried.
"""

from __future__ import annotations

import hashlib
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType

from convoprobe.datamodels import CheckResult, Conversation, Scenario
from convoprobe.errors import ConvoprobeError

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    scenario_name TEXT NOT NULL,
    scenario_hash TEXT NOT NULL,
    adapter TEXT NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT
);
CREATE TABLE IF NOT EXISTS conversations (
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    conversation_id TEXT NOT NULL,
    persona_id TEXT NOT NULL,
    repeat_index INTEGER NOT NULL,
    status TEXT NOT NULL
        CHECK(status IN ('pending','done','failed_transient','failed','skipped')),
    attempts INTEGER NOT NULL DEFAULT 0,
    data_json TEXT,
    error TEXT,
    updated_at TEXT NOT NULL,
    PRIMARY KEY(run_id, conversation_id)
);
CREATE TABLE IF NOT EXISTS check_results (
    run_id TEXT NOT NULL,
    conversation_id TEXT NOT NULL,
    rubric_id TEXT NOT NULL,
    status TEXT NOT NULL,
    detail TEXT NOT NULL,
    PRIMARY KEY(run_id, conversation_id, rubric_id)
);
"""


def utc_now() -> str:
    """Current time as a UTC ISO 8601 string."""
    return datetime.now(UTC).isoformat()


class RunStore:
    """Runs, conversations, and check results in one SQLite file."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path)
        self._conn.execute("PRAGMA foreign_keys=ON")
        with self._conn:
            self._conn.executescript(_SCHEMA)

    def __enter__(self) -> RunStore:
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

    @staticmethod
    def scenario_hash(scenario: Scenario) -> str:
        return hashlib.sha256(scenario.model_dump_json().encode("utf-8")).hexdigest()

    def create_or_resume_run(self, run_id: str, scenario: Scenario, adapter_label: str) -> bool:
        """Create the run, or resume it. Returns True when resuming."""
        digest = self.scenario_hash(scenario)
        row = self._conn.execute(
            "SELECT scenario_hash FROM runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        if row is not None:
            if row[0] != digest:
                raise ConvoprobeError(f"run {run_id} was created from a different scenario")
            return True
        with self._conn:
            self._conn.execute(
                "INSERT INTO runs (run_id, scenario_name, scenario_hash, adapter, started_at)"
                " VALUES (?, ?, ?, ?, ?)",
                (run_id, scenario.name, digest, adapter_label, utc_now()),
            )
        return False

    def run_times(self, run_id: str) -> tuple[str, str | None]:
        """(started_at, finished_at) for a run."""
        row = self._conn.execute(
            "SELECT started_at, finished_at FROM runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        if row is None:
            raise ConvoprobeError(f"run {run_id} not found")
        return row[0], row[1]

    def ensure_conversation(
        self, run_id: str, conversation_id: str, persona_id: str, repeat_index: int
    ) -> None:
        with self._conn:
            self._conn.execute(
                "INSERT OR IGNORE INTO conversations"
                " (run_id, conversation_id, persona_id, repeat_index, status, updated_at)"
                " VALUES (?, ?, ?, ?, 'pending', ?)",
                (run_id, conversation_id, persona_id, repeat_index, utc_now()),
            )

    def conversation_status(self, run_id: str, conversation_id: str) -> str:
        row = self._conn.execute(
            "SELECT status FROM conversations WHERE run_id = ? AND conversation_id = ?",
            (run_id, conversation_id),
        ).fetchone()
        if row is None:
            raise ConvoprobeError(f"conversation {conversation_id} not found in run {run_id}")
        return str(row[0])

    def conversation_attempts(self, run_id: str, conversation_id: str) -> int:
        row = self._conn.execute(
            "SELECT attempts FROM conversations WHERE run_id = ? AND conversation_id = ?",
            (run_id, conversation_id),
        ).fetchone()
        if row is None:
            raise ConvoprobeError(f"conversation {conversation_id} not found in run {run_id}")
        return int(row[0])

    def save_conversation(
        self, run_id: str, conversation: Conversation, status: str, error: str | None = None
    ) -> None:
        """Store the conversation and its status. Each save counts as one attempt."""
        with self._conn:
            self._conn.execute(
                "INSERT INTO conversations (run_id, conversation_id, persona_id, repeat_index,"
                " status, attempts, data_json, error, updated_at)"
                " VALUES (?, ?, ?, ?, ?, 1, ?, ?, ?)"
                " ON CONFLICT(run_id, conversation_id) DO UPDATE SET"
                " status = excluded.status, attempts = conversations.attempts + 1,"
                " data_json = excluded.data_json, error = excluded.error,"
                " updated_at = excluded.updated_at",
                (
                    run_id,
                    conversation.id,
                    conversation.persona_id,
                    conversation.repeat_index,
                    status,
                    conversation.model_dump_json(),
                    error,
                    utc_now(),
                ),
            )

    def save_checks(self, run_id: str, conversation_id: str, results: list[CheckResult]) -> None:
        """Replace all check results for one conversation."""
        with self._conn:
            self._conn.execute(
                "DELETE FROM check_results WHERE run_id = ? AND conversation_id = ?",
                (run_id, conversation_id),
            )
            self._conn.executemany(
                "INSERT INTO check_results (run_id, conversation_id, rubric_id, status, detail)"
                " VALUES (?, ?, ?, ?, ?)",
                [(run_id, conversation_id, r.rubric_id, r.status, r.detail) for r in results],
            )

    def load_conversations(self, run_id: str) -> list[tuple[Conversation | None, str, str | None]]:
        """(conversation or None if never saved, status, error), by conversation id."""
        rows = self._conn.execute(
            "SELECT data_json, status, error FROM conversations WHERE run_id = ?"
            " ORDER BY conversation_id",
            (run_id,),
        ).fetchall()
        return [
            (Conversation.model_validate_json(data) if data else None, status, error)
            for data, status, error in rows
        ]

    def load_checks(self, run_id: str) -> list[CheckResult]:
        rows = self._conn.execute(
            "SELECT c.conversation_id, c.rubric_id, c.status, c.detail, v.persona_id"
            " FROM check_results c JOIN conversations v"
            " ON v.run_id = c.run_id AND v.conversation_id = c.conversation_id"
            " WHERE c.run_id = ? ORDER BY c.conversation_id, c.rubric_id",
            (run_id,),
        ).fetchall()
        return [
            CheckResult(
                conversation_id=cid, rubric_id=rid, status=status, detail=detail, persona_id=pid
            )
            for cid, rid, status, detail, pid in rows
        ]

    def finish_run(self, run_id: str) -> None:
        with self._conn:
            self._conn.execute(
                "UPDATE runs SET finished_at = ? WHERE run_id = ?", (utc_now(), run_id)
            )
