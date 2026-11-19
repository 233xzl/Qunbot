"""Offline checks for Pydantic AI function-tool execution and per-request accounting."""

import asyncio
from dataclasses import replace

from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.usage import RequestUsage
import pytest

from group_agent.models import BoundCall, BudgetExceeded, Ledger, LiveModel, ModelCallError, Usage
from test_live_replay import GEMINI


def test_image_reservation_counts_visual_allowance_not_encoded_payload():
    from pydantic_ai import BinaryContent
    from pydantic_ai.messages import ModelRequest, UserPromptPart
    from group_agent.models import _input_token_bound

    small = [ModelRequest(parts=[UserPromptPart([BinaryContent(b"x", media_type="image/png")])])]
    large = [
        ModelRequest(
            parts=[UserPromptPart([BinaryContent(b"x" * 1_000_000, media_type="image/png")])]
        )
    ]
    assert _input_token_bound(small) == _input_token_bound(large)
    assert _input_token_bound(large) >= 20_000
    assert (
        _input_token_bound([*large, ModelRequest(parts=[UserPromptPart("证据" * 1000)])])
        > _input_token_bound(large) + 6000
    )


async def web_search(query: str) -> str:
    """Search for a public fact."""
    return "found:" + query


def tool_model(fn, ledger=None):
    return LiveModel(GEMINI, lambda: FunctionModel(fn), ledger or Ledger(1.0))


def test_multiround_async_tools_and_usage():
    seen = []
    ledger = Ledger(1.0)

    def fn(messages, info):
        assert info.function_tools[0].name == "web_search"
        seen.append(messages)
        if len(seen) <= 2:
            return ModelResponse(
                parts=[ToolCallPart("web_search", {"query": "test"}, tool_call_id=str(len(seen)))],
                usage=RequestUsage(input_tokens=100, output_tokens=20),
            )
        assert any(
            isinstance(p, ToolReturnPart) and p.content == "found:test"
            for m in messages
            for p in m.parts
        )
        return ModelResponse(
            parts=[TextPart("final")], usage=RequestUsage(input_tokens=100, output_tokens=20)
        )

    text, record = asyncio.run(
        tool_model(fn, ledger).generate(
            "a",
            "i",
            ["p"],
            5,
            tools=[web_search],
        )
    )
    assert text == "final" and record.responses == 3
    assert record.usage == Usage(input=300, output=60, thinking=0, requests=3)
    assert len(ledger.records) == 3 and ledger.pending == 0
    assert ledger.spent == pytest.approx(record.cost_cny)


def test_budget_refuses_later_request_before_provider():
    ledger = Ledger(1.0)
    calls = []

    def fn(messages, info):
        calls.append(1)
        # Leave too little for the second request; no pending first reservation survives.
        ledger.budget_cny = 0.00001
        return ModelResponse(
            parts=[ToolCallPart("web_search", {"query": "test"})],
            usage=RequestUsage(input_tokens=10, output_tokens=10),
        )

    with pytest.raises(BudgetExceeded) as err:
        asyncio.run(tool_model(fn, ledger).generate("a", "i", ["p"], 5, tools=[web_search]))
    assert err.value.record.status == "budget" and calls == [1]
    assert ledger.pending == 0 and ledger.spent == err.value.record.cost_cny > 0


@pytest.mark.parametrize("limits, expected", [({"max_requests": 2}, 2), ({"max_tool_calls": 1}, 2)])
def test_loop_limits_stop_repeating_model(limits, expected):
    calls, tools = [], []

    async def work(query: str) -> str:
        """Return a result."""
        tools.append(query)
        return "done"

    def fn(messages, info):
        calls.append(1)
        return ModelResponse(
            parts=[ToolCallPart("work", {"query": "q"})],
            usage=RequestUsage(input_tokens=10, output_tokens=10),
        )

    model = tool_model(fn)
    with pytest.raises(ModelCallError) as err:
        asyncio.run(model.generate("a", "i", ["p"], 5, tools=[work], **limits))
    assert err.value.record.error == "UsageLimitExceeded"
    assert len(calls) == expected and len(tools) <= limits.get("max_tool_calls", 8)
    assert model.ledger.pending == 0


def test_timeout_after_tool_response_preserves_all_fees():
    calls = []

    async def fn(messages, info):
        calls.append(1)
        if len(calls) == 1:
            return ModelResponse(
                parts=[ToolCallPart("web_search", {"query": "test"})],
                usage=RequestUsage(input_tokens=100, output_tokens=20),
            )
        await asyncio.sleep(2)

    model = tool_model(fn)
    with pytest.raises(ModelCallError) as err:
        asyncio.run(model.generate("a", "i", ["p"], 0.05, tools=[web_search]))
    assert err.value.kind == "timeout" and len(calls) == 2
    records = model.ledger.records
    assert len(records) == 2 and records[1].cost_cny > 0
    assert err.value.record.cost_cny == pytest.approx(sum(r.cost_cny for r in records))
    assert model.ledger.pending == 0


def test_cancel_settles_inflight_request():
    async def exercise():
        entered = asyncio.Event()

        async def fn(messages, info):
            entered.set()
            await asyncio.sleep(2)

        model = tool_model(fn)
        task = asyncio.create_task(model.generate("a", "i", ["p"], 5, tools=[web_search]))
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert model.ledger.pending == 0 and model.ledger.spent > 0

    asyncio.run(exercise())


def test_timeout_in_tool_only_charges_completed_request():
    async def slow(query: str) -> str:
        """Wait for a result."""
        await asyncio.sleep(2)
        return "done"

    def fn(messages, info):
        return ModelResponse(
            parts=[ToolCallPart("slow", {"query": "q"})],
            usage=RequestUsage(input_tokens=100, output_tokens=20),
        )

    model = tool_model(fn)
    with pytest.raises(ModelCallError) as err:
        asyncio.run(model.generate("a", "i", ["p"], 0.05, tools=[slow]))
    assert err.value.kind == "timeout"
    assert len(model.ledger.records) == 1 and model.ledger.records[0].status == "ok"
    assert model.ledger.pending == 0


def test_protocol_capabilities_follow_protocol_not_name():
    model = tool_model(lambda *_: None)
    assert model.supports_native_output
    for protocol in ("anthropic", "openai-compatible"):
        spec = replace(GEMINI, protocol=protocol)
        other = LiveModel(spec, lambda: None, Ledger())
        assert not other.google and not other.supports_native_output
        assert "google_thinking_config" not in other.settings()


def test_bound_call_passes_tools_and_reports_aggregate():
    calls = []

    def fn(messages, info):
        calls.append(1)
        parts = (
            [ToolCallPart("web_search", {"query": "q"})] if len(calls) == 1 else [TextPart("final")]
        )
        return ModelResponse(parts=parts, usage=RequestUsage(input_tokens=10, output_tokens=10))

    model = tool_model(fn)
    sink = []
    bound = BoundCall(
        model,
        "a",
        "i",
        ["p"],
        lambda: 5,
        sink,
        tools=[web_search],
        max_requests=3,
        max_tool_calls=2,
    )
    assert asyncio.run(bound.run(None)) == "final"
    assert len(sink) == 1 and sink[0].responses == 2


def test_fallback_keeps_failed_loop_costs():
    from group_agent.contracts import FallbackModel

    ledger, sink = Ledger(1.0), []
    calls = []

    def fail_after_search(messages, info):
        calls.append(1)
        if len(calls) == 1:
            return ModelResponse(
                parts=[ToolCallPart("web_search", {"query": "q"})],
                usage=RequestUsage(input_tokens=100, output_tokens=20),
            )
        raise ConnectionError("private provider detail")

    def good(messages, info):
        return ModelResponse(
            parts=[TextPart("fallback")], usage=RequestUsage(input_tokens=100, output_tokens=20)
        )

    first = tool_model(fail_after_search, ledger)
    second = tool_model(good, ledger)
    runners = [
        BoundCall(m, "a", "i", ["p"], lambda: 5, sink, tools=[web_search]) for m in (first, second)
    ]
    assert asyncio.run(FallbackModel(*runners).run(None)) == "fallback"
    assert [r.status for r in sink] == ["error", "ok"]
    assert len(ledger.records) == 3 and ledger.pending == 0
    assert ledger.spent == pytest.approx(sum(r.cost_cny for r in sink))
    assert "private" not in str([r.as_dict() for r in ledger.records])


def test_fallback_parts_factory_reads_evidence_after_first_attempt():
    from group_agent.contracts import FallbackModel

    materials, seen = [], []

    async def search(query: str) -> str:
        """Find a fact."""
        materials.append("evidence")
        return "evidence"

    calls = []

    def first_fn(messages, info):
        calls.append(1)
        if len(calls) == 1:
            return ModelResponse(parts=[ToolCallPart("search", {"query": "q"})])
        raise ConnectionError("unavailable")

    def second_fn(messages, info):
        seen.append(str(messages))
        return ModelResponse(parts=[TextPart("final")])

    runners = [
        BoundCall(
            tool_model(fn),
            "a",
            "i",
            ["base"],
            lambda: 5,
            [],
            tools=[search],
            parts_factory=lambda: ["base", *materials],
        )
        for fn in (first_fn, second_fn)
    ]
    assert asyncio.run(FallbackModel(*runners).run(None)) == "final"
    assert "evidence" in seen[0]


def test_batch_over_tool_limit_executes_no_callbacks():
    executed = []

    async def work(query: str) -> str:
        """Find information."""
        executed.append(query)
        return "done"

    def fn(messages, info):
        return ModelResponse(
            parts=[
                ToolCallPart("work", {"query": "a"}, tool_call_id="a"),
                ToolCallPart("work", {"query": "b"}, tool_call_id="b"),
            ]
        )

    with pytest.raises(ModelCallError) as err:
        asyncio.run(tool_model(fn).generate("a", "i", ["p"], 5, tools=[work], max_tool_calls=1))
    assert err.value.record.error == "UsageLimitExceeded" and not executed


def test_structured_retry_reserves_again_before_provider():
    from group_agent.live_agent import RouteDecision
    from test_live_replay import DEEPSEEK

    ledger, calls = Ledger(1.0), []

    def fn(messages, info):
        calls.append(1)
        ledger.budget_cny = 0.00001
        return ModelResponse(
            parts=[ToolCallPart("submit", {"route": "bad"})],
            usage=RequestUsage(input_tokens=10, output_tokens=10),
        )

    model = LiveModel(DEEPSEEK, lambda: FunctionModel(fn), ledger)
    with pytest.raises(BudgetExceeded) as err:
        asyncio.run(model.structured("a", "i", ["p"], RouteDecision, "tool", 5))
    assert calls == [1] and ledger.pending == 0
    assert err.value.record.status == "budget"
    assert err.value.record.cost_cny == ledger.spent > 0
    assert len(ledger.records) == 1


def test_bound_call_budget_stops_fallback_and_keeps_record():
    from group_agent.contracts import FallbackModel

    ledger, calls, sink = Ledger(1e-12), [], []

    def fn(messages, info):
        calls.append(1)
        return ModelResponse(parts=[TextPart("final")])

    model = tool_model(fn, ledger)
    first = BoundCall(model, "a", "i", ["p"], lambda: 5, sink, tools=[web_search])
    second = BoundCall(model, "a", "i", ["p"], lambda: 5, sink)
    chain = FallbackModel(first, second)
    with pytest.raises(BudgetExceeded):
        asyncio.run(chain.run(None))
    assert not calls and len(chain.attempts) == 1
    assert len(sink) == 1 and sink[0].status == "budget"
    assert ledger.pending == 0


@pytest.mark.parametrize("tools", [(), (web_search,)])
def test_bound_cancel_keeps_aggregate_record(tools):
    async def exercise():
        entered = asyncio.Event()

        async def fn(messages, info):
            entered.set()
            await asyncio.sleep(2)

        model, sink = tool_model(fn), []
        bound = BoundCall(model, "a", "i", ["p"], lambda: 5, sink, tools=tools)
        task = asyncio.create_task(bound.run(None))
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError) as err:
            await task
        assert len(sink) == 1 and sink[0] is err.value.record
        assert sink[0].status == "cancelled" and sink[0].cost_cny > 0
        assert sink[0].cost_cny == model.ledger.spent and model.ledger.pending == 0

    asyncio.run(exercise())
