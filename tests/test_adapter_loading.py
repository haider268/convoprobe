"""load_adapter: http configs, compiled graphs, the examples package, lazy langgraph."""

from __future__ import annotations

import subprocess
import sys
import types
from pathlib import Path

import pytest

from convoprobe.adapters import HTTPAdapter, LangGraphAdapter, load_adapter
from convoprobe.errors import AdapterLoadError

REPO = Path(__file__).resolve().parents[1]

HTTP_YAML = """\
type: http
url: http://127.0.0.1:8765/chat
request_template:
  session: "{{conversation_id}}"
  message: "{{message}}"
text_path: reply
tool_calls_path: tool_calls
"""


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "adapter.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_http_config_loads(tmp_path: Path) -> None:
    adapter = load_adapter(f"http:{write(tmp_path, HTTP_YAML)}")
    assert isinstance(adapter, HTTPAdapter)
    assert adapter.config.text_path == "reply"


def test_repo_http_example_loads(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(REPO)
    assert isinstance(load_adapter("http:examples/http_agent/adapter.yaml"), HTTPAdapter)


@pytest.mark.parametrize(
    ("text", "message"),
    [
        (HTTP_YAML.replace("text_path: reply\n", ""), "text_path: Field required"),
        (HTTP_YAML + "retries: 3\n", "retries: Extra inputs are not permitted"),
        (HTTP_YAML.replace("type: http", "type: grpc"), "type: Input should be 'http'"),
        ("- just\n- a list\n", "expected a mapping at the top level"),
        ("url: [unclosed\n", "YAML error"),
    ],
)
def test_bad_http_config(tmp_path: Path, text: str, message: str) -> None:
    path = write(tmp_path, text)
    with pytest.raises(AdapterLoadError) as info:
        load_adapter(f"http:{path}")
    assert f"adapter config {path}" in str(info.value)
    assert message in str(info.value)


def test_http_config_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "nope.yaml"
    with pytest.raises(AdapterLoadError, match="file not found"):
        load_adapter(f"http:{missing}")


def test_http_config_missing_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NOT_SET_TOKEN", raising=False)
    path = write(tmp_path, HTTP_YAML + "headers:\n  Authorization: Bearer ${NOT_SET_TOKEN}\n")
    with pytest.raises(AdapterLoadError, match="needs environment variable NOT_SET_TOKEN"):
        load_adapter(f"http:{path}")


def test_compiled_graph_is_wrapped(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("langgraph")
    from examples.toy_support_agent.graph import build_graph

    module = types.ModuleType("fake_graph_module")
    module.graph = build_graph("v1")  # type: ignore[attr-defined]
    module.no_saver = build_graph("v1").builder.compile()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "fake_graph_module", module)
    adapter = load_adapter("fake_graph_module:graph")
    assert isinstance(adapter, LangGraphAdapter)
    assert adapter.resume is None
    with pytest.raises(AdapterLoadError, match="needs a graph compiled with a checkpointer"):
        load_adapter("fake_graph_module:no_saver")


def test_examples_adapter_loads_from_repo_root(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("langgraph")
    monkeypatch.chdir(REPO)
    monkeypatch.setattr(sys, "path", [p for p in sys.path if Path(p or ".").resolve() != REPO])
    adapter = load_adapter("examples.toy_support_agent.graph:adapter_v1")
    assert isinstance(adapter, LangGraphAdapter)
    assert adapter.resume == "approve"
    assert sys.path[0] == str(REPO)


def test_existing_errors_unchanged() -> None:
    with pytest.raises(AdapterLoadError, match=r"expected the form module\.path:attr"):
        load_adapter("no_colon")
    with pytest.raises(AdapterLoadError, match="is a class, pass an instance or function"):
        load_adapter("convoprobe.adapters.callable:CallableAdapter")


def test_adapters_package_does_not_import_langgraph() -> None:
    code = (
        "import sys, convoprobe.adapters, convoprobe.cli; "
        "bad = [m for m in sys.modules if m.startswith('langgraph')]; "
        "assert not bad, bad"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
