"""Structured JSONL logging. Every line carries the run id."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Any


class JsonlLogger:
    """Appends one JSON object per line to a log file."""

    def __init__(self, path: Path, run_id: str) -> None:
        self.path = Path(path)
        self.run_id = run_id
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("a", encoding="utf-8")

    def log(self, event: str, **fields: Any) -> None:
        record = {
            "ts": datetime.now(UTC).isoformat(),
            "event": event,
            "run_id": self.run_id,
            **fields,
        }
        self._file.write(json.dumps(record, default=str) + "\n")
        self._file.flush()

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> JsonlLogger:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
