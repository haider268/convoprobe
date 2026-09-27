"""SqliteLLMCache and JsonCassette."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from convoprobe.errors import ConvoprobeError
from convoprobe.models.cache import JsonCassette, SqliteLLMCache


def test_put_get_round_trip(tmp_path: Path) -> None:
    with SqliteLLMCache(tmp_path / "a" / "cache.db") as cache:
        assert cache.get("k") is None
        cache.put("k", "ollama:m", "text", {"text": "hi"})
        assert cache.get("k") == {"text": "hi"}


def test_insert_or_replace(tmp_path: Path) -> None:
    with SqliteLLMCache(tmp_path / "cache.db") as cache:
        cache.put("k", "ollama:m", "text", {"text": "old"})
        cache.put("k", "ollama:m", "text", {"text": "new"})
        assert cache.get("k") == {"text": "new"}


def test_export_cassette(tmp_path: Path) -> None:
    out = tmp_path / "out" / "cassette.json"
    with SqliteLLMCache(tmp_path / "cache.db") as cache:
        cache.put("b-key", "ollama:m", "text", {"text": "b"})
        cache.put("a-key", "ollama:j", "structured", {"data": {"verdict": "yes"}})
        cache.put("unused", "ollama:m", "text", {"text": "x"})
        cache.put("other-run", "ollama:m", "text", {"text": "y"})
        cache.mark_used("run-1", "b-key")
        cache.mark_used("run-1", "b-key")
        cache.mark_used("run-2", "a-key")
        cache.mark_used("run-3", "other-run")
        count = cache.export_cassette(["run-1", "run-2"], out)
        assert cache.export_cassette([], tmp_path / "empty.json") == 0
    assert count == 2
    text = out.read_text(encoding="utf-8")
    assert text.endswith("}\n")
    data = json.loads(text)
    assert data["format"] == 1
    assert list(data["entries"]) == ["a-key", "b-key"]
    assert data["entries"]["b-key"] == {
        "spec": "ollama:m",
        "kind": "text",
        "response": {"text": "b"},
    }


def test_cassette_replays_export(tmp_path: Path) -> None:
    path = tmp_path / "cassette.json"
    with SqliteLLMCache(tmp_path / "cache.db") as cache:
        cache.put("k", "ollama:m", "text", {"text": "hi"})
        cache.mark_used("run-1", "k")
        cache.export_cassette(["run-1"], path)
    cassette = JsonCassette(path)
    assert cassette.get("k") == {"text": "hi"}
    assert cassette.get("missing") is None
    cassette.mark_used("run-1", "k")
    with pytest.raises(ConvoprobeError, match="is read-only"):
        cassette.put("k", "ollama:m", "text", {"text": "new"})
    assert cassette.meta == {}
    assert "meta" not in json.loads(path.read_text(encoding="utf-8"))


def test_cassette_meta_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "cassette.json"
    meta = {"recorded_at": "2026-09-28", "simulator": "groq:a", "judge": "groq:b"}
    with SqliteLLMCache(tmp_path / "cache.db") as cache:
        cache.put("k", "groq:a", "text", {"text": "hi"})
        cache.mark_used("run-1", "k")
        assert cache.export_cassette(["run-1"], path, meta=meta) == 1
    assert json.loads(path.read_text(encoding="utf-8"))["meta"] == meta
    cassette = JsonCassette(path)
    assert cassette.meta == meta
    assert cassette.get("k") == {"text": "hi"}
    assert b"\r\n" not in path.read_bytes()


@pytest.mark.parametrize("payload", [{"format": 2, "entries": {}}, {"entries": {}}, [1, 2]])
def test_cassette_bad_format(tmp_path: Path, payload: object) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ConvoprobeError, match="unsupported cassette format"):
        JsonCassette(path)
