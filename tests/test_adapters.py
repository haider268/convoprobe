"""CallableAdapter and load_adapter."""

from __future__ import annotations

import pytest

from convoprobe.adapters import AgentAdapter, CallableAdapter, load_adapter
from convoprobe.datamodels import AgentTurn, Message, ToolCallRecord
from convoprobe.errors import AdapterLoadError, ConvoprobeError, TransientError


def echo(history: list[Message]) -> str:
    return f"echo {history[-1].text} ({len(history)})"


async def async_echo(history: list[Message]) -> str:
    return f"async {history[-1].text}"


# Module attributes used by load_adapter tests.
ADAPTER_INSTANCE = CallableAdapter(echo)
NOT_CALLABLE = 42


async def test_sync_fn_returning_str() -> None:
    adapter = CallableAdapter(echo)
    await adapter.start("c1")
    turn = await adapter.send("c1", "hi")
    assert turn == AgentTurn(text="echo hi (1)")


async def test_async_fn() -> None:
    adapter = CallableAdapter(async_echo)
    await adapter.start("c1")
    assert (await adapter.send("c1", "hi")).text == "async hi"


async def test_sync_fn_returning_awaitable() -> None:
    adapter = CallableAdapter(lambda history: async_echo(history))
    await adapter.start("c1")
    assert (await adapter.send("c1", "hi")).text == "async hi"


async def test_fn_returning_agent_turn_and_history_records_it() -> None:
    seen: list[list[Message]] = []

    def agent(history: list[Message]) -> AgentTurn:
        seen.append(history)
        return AgentTurn(text="ok", tool_calls=[ToolCallRecord(name="lookup_order")])

    adapter = CallableAdapter(agent)
    await adapter.start("c1")
    turn = await adapter.send("c1", "first")
    assert turn.tool_calls[0].name == "lookup_order"
    await adapter.send("c1", "second")
    last = seen[-1]
    assert [(m.role, m.text) for m in last] == [
        ("user", "first"),
        ("agent", "ok"),
        ("user", "second"),
    ]
    assert last[1].tool_calls[0].name == "lookup_order"


async def test_fn_gets_a_copy() -> None:
    def mutating(history: list[Message]) -> str:
        history.clear()
        return "ok"

    adapter = CallableAdapter(mutating)
    await adapter.start("c1")
    await adapter.send("c1", "a")
    turn = await adapter.send("c1", "b")
    assert turn.text == "ok"
    assert len(adapter._histories["c1"]) == 4


async def test_histories_are_separate() -> None:
    adapter = CallableAdapter(echo)
    await adapter.start("c1")
    await adapter.start("c2")
    await adapter.send("c1", "a")
    await adapter.send("c1", "b")
    assert (await adapter.send("c2", "x")).text == "echo x (1)"
    assert (await adapter.send("c1", "c")).text == "echo c (5)"


async def test_end_clears_history() -> None:
    adapter = CallableAdapter(echo)
    await adapter.start("c1")
    await adapter.send("c1", "a")
    await adapter.end("c1")
    with pytest.raises(ConvoprobeError, match="conversation c1 not started"):
        await adapter.send("c1", "b")


async def test_send_before_start() -> None:
    with pytest.raises(ConvoprobeError, match="conversation nope not started"):
        await CallableAdapter(echo).send("nope", "hi")


async def test_exceptions_propagate_unchanged() -> None:
    def flaky(history: list[Message]) -> str:
        raise TransientError("rate limited")

    adapter = CallableAdapter(flaky)
    await adapter.start("c1")
    with pytest.raises(TransientError, match="rate limited"):
        await adapter.send("c1", "hi")


def test_callable_adapter_is_agent_adapter() -> None:
    assert isinstance(CallableAdapter(echo), AgentAdapter)


# load_adapter


async def test_load_adapter_function() -> None:
    adapter = load_adapter("convoprobe.demo.toy_agent:toy_agent_v1")
    assert isinstance(adapter, CallableAdapter)
    await adapter.start("c1")
    assert "order status and refunds" in (await adapter.send("c1", "hello")).text


def test_load_adapter_instance_returned_as_is() -> None:
    assert load_adapter("tests.test_adapters:ADAPTER_INSTANCE") is ADAPTER_INSTANCE


@pytest.mark.parametrize(
    ("spec", "reason"),
    [
        ("no_colon", "expected the form module.path:attr"),
        ("a:b:c", "expected the form module.path:attr"),
        (":attr", "expected the form module.path:attr"),
        ("module_only:", "expected the form module.path:attr"),
        ("convoprobe_no_such_module_xyz:fn", "cannot import convoprobe_no_such_module_xyz"),
        ("convoprobe.demo.toy_agent:missing", "has no attribute missing"),
        ("convoprobe.adapters.callable:CallableAdapter", "is a class"),
        ("tests.test_adapters:NOT_CALLABLE", "is neither an AgentAdapter nor callable"),
    ],
)
def test_load_adapter_errors(spec: str, reason: str) -> None:
    with pytest.raises(AdapterLoadError) as info:
        load_adapter(spec)
    message = str(info.value)
    assert repr(spec) in message
    assert reason in message


def test_load_adapter_class_message() -> None:
    with pytest.raises(AdapterLoadError, match="is a class, pass an instance or function"):
        load_adapter("convoprobe.adapters.callable:CallableAdapter")
