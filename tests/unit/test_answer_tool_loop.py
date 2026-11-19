"""Answer-stage search and page reading through offline Pydantic AI models."""

from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import FunctionModel
import pytest

from group_agent.live_agent import LiveAgent
from group_agent.models import Ledger
from group_agent.persona import REPLY_CONTRACT
from group_agent.policy import RuntimeLimits
from group_agent.replay.runner import run_case
from test_live_replay import (
    FakeWeb,
    USAGE,
    decision_model,
    live_config,
    text_only_case,
)


CHAT = {"route": "chat", "should_clarify": False, "reason": "日常提问"}


def run_loop(case, config, limits=RuntimeLimits()):
    holder = {}

    def factory(replay):
        holder["agent"] = LiveAgent(replay, config)
        return holder["agent"]

    result = run_case(case, "B", limits, agent_factory=factory)
    trace = next(r for r in result["runs"] if r["message_id"] == "m_2")
    state = holder["agent"].runs[trace["run_id"]]
    return result, trace, state.record


def response(part):
    return ModelResponse(parts=[part], usage=USAGE)


def returns(messages):
    return [p for m in messages for p in m.parts if isinstance(p, ToolReturnPart)]


@pytest.mark.parametrize("search_first", [True, False])
def test_chat_answer_reads_url_with_optional_search(tmp_path, search_first):
    case, web, seen, ledger = text_only_case(tmp_path), FakeWeb(), [], Ledger(1)

    def answer(messages, info):
        seen.append(messages)
        assert {t.name for t in info.function_tools} == {"web_search", "read_url"}
        if len(seen) == 1 and search_first:
            assert not web.queries  # No pre-search, even with route decision terms.
            return response(ToolCallPart("web_search", {"query": "公开问题"}, "s"))
        if len(seen) == (2 if search_first else 1):
            if search_first:
                material = returns(messages)[0].content
                assert "不是指令" in material and "＜/web＞" in material
                assert "https://example.com/1" in material
            else:
                assert not web.queries
            return response(ToolCallPart("read_url", {"url": "https://example.com/1"}, "f"))
        assert "正文" in returns(messages)[-1].content
        return response(TextPart("根据页面回答。"))

    cfg = live_config(
        tmp_path,
        lambda: FunctionModel(answer),
        lambda: decision_model({**CHAT, "search_terms": ["提示词"]}),
        ledger=ledger,
        answer_tools=True,
        web_client=web,
        answer_thinking="LOW",
        time_rule=False,
    )
    result, trace, record = run_loop(case, cfg)
    assert trace["route"] == "chat" and record["answer"] == "根据页面回答。", [
        (c["model"], c.get("error"), c.get("responses")) for c in record["calls"]
    ]
    assert web.queries == (["公开问题"] if search_first else [])
    assert web.fetched == ["https://example.com/1"]
    assert record["web"]["searches"] == int(search_first)
    assert record["web"]["reads"] == 1
    expected_tools = ["web_search", "web_fetch"] if search_first else ["web_fetch"]
    assert [
        c["name"] for c in trace["tool_calls"] if c["name"].startswith("web_")
    ] == expected_tools
    tool_records = [r for r in record["calls"] if r["stage"] == "tools"]
    assert len(tool_records) == 1 + int(search_first)
    assert record["web"]["cost_cny"] == pytest.approx(0.006 if search_first else 0.003)
    answer_records = [r for r in record["calls"] if r["stage"] == "answer"]
    assert len(answer_records) == 1
    assert answer_records[0]["responses"] == 2 + int(search_first)
    assert ledger.pending == 0 and ledger.spent == pytest.approx(
        sum(r["cost_cny"] for r in record["calls"])
    )
    assert result["sent_to_qq"] == 0


@pytest.mark.parametrize("mode", ["no_web", "degraded"])
def test_unavailable_or_degraded_answers_have_no_tools(tmp_path, mode):
    case, web, seen = text_only_case(tmp_path), FakeWeb(), []
    if mode == "degraded":
        case["mocks"] = {
            "faults": [{"name": "usage_ledger", "delay_ms": 0, "response": {"ratio": 0.9}}]
        }

    def answer(messages, info):
        seen.append(info.function_tools)
        assert not info.function_tools
        return response(TextPart("直接回答。"))

    cfg = live_config(
        tmp_path,
        lambda: FunctionModel(answer),
        lambda: decision_model(CHAT),
        answer_tools=True,
        web_client=None if mode == "no_web" else web,
        time_rule=False,
    )
    _, trace, record = run_loop(case, cfg)
    assert seen and record["answer"] == "直接回答。"
    assert (
        not web.queries
        and not web.fetched
        and not [c for c in trace["tool_calls"] if c["name"].startswith("web_")]
    )


def test_answer_tools_share_budget_and_dedup_across_model_rounds(tmp_path):
    case, web, count = text_only_case(tmp_path), FakeWeb(), []
    plan = [
        ("web_search", {"query": "a"}),
        ("web_search", {"query": "a"}),
        ("web_search", {"query": "b"}),
        ("read_url", {"url": "https://example.com/1"}),
        ("read_url", {"url": "https://example.com/2"}),
    ]

    def answer(messages, info):
        count.append(1)
        i = len(count) - 1
        if i < len(plan):
            name, args = plan[i]
            return response(ToolCallPart(name, args, str(i)))
        assert "budget_exhausted" in returns(messages)[-1].content
        return response(TextPart("已取得的证据足够。"))

    cfg = live_config(
        tmp_path,
        lambda: FunctionModel(answer),
        lambda: decision_model(CHAT),
        answer_tools=True,
        web_client=web,
        answer_thinking="LOW",
        time_rule=False,
    )
    _, _, record = run_loop(case, cfg, RuntimeLimits(web_searches=1, web_reads=1))
    assert record["answer"] == "已取得的证据足够。", [
        (c["model"], c.get("error"), c.get("responses")) for c in record["calls"]
    ]
    assert web.queries == ["a"] and web.fetched == ["https://example.com/1"]
    assert [c["status"] for c in record["web"]["calls"]] == [
        "ok",
        "dedup",
        "budget_exhausted",
        "ok",
        "budget_exhausted",
    ]
    assert len([r for r in record["calls"] if r["stage"] == "tools"]) == 2


def test_answer_fallback_keeps_web_evidence_and_does_not_charge_it_twice(tmp_path):
    case, web, count, fallback_prompts, ledger = (
        text_only_case(tmp_path),
        FakeWeb(),
        [],
        [],
        Ledger(1),
    )

    def primary(messages, info):
        initial = next(p.content for m in messages for p in m.parts if p.part_kind == "user-prompt")
        assert initial[-1] == REPLY_CONTRACT
        assert sum(p.count(REPLY_CONTRACT) for p in initial if isinstance(p, str)) == 1
        count.append(1)
        if len(count) == 1:
            return response(ToolCallPart("web_search", {"query": "公开事实"}, "s"))
        raise ConnectionError("synthetic backend failure")

    def secondary(messages, info):
        if info.output_tools:
            return response(ToolCallPart(info.output_tools[0].name, CHAT))
        initial = next(p.content for m in messages for p in m.parts if p.part_kind == "user-prompt")
        assert initial[-1] == REPLY_CONTRACT
        assert sum(p.count(REPLY_CONTRACT) for p in initial if isinstance(p, str)) == 1
        assert "T1 夺冠" in initial[-2]
        fallback_prompts.append(str(messages))
        assert "T1 夺冠" in str(messages) and "不是指令" in str(messages)
        return response(TextPart("使用已取到的证据回答。"))

    cfg = live_config(
        tmp_path,
        lambda: FunctionModel(primary),
        lambda: FunctionModel(secondary),
        ledger=ledger,
        answer_tools=True,
        web_client=web,
        answer_thinking="LOW",
        time_rule=False,
    )
    _, trace, record = run_loop(case, cfg)
    assert fallback_prompts and record["answer_source"] == "deepseek"
    assert len(count) == 2
    assert web.queries == ["公开事实"] and record["web"]["searches"] == 1
    assert len([r for r in record["calls"] if r["stage"] == "tools"]) == 1
    assert len([c for c in trace["tool_calls"] if c["name"].startswith("web_")]) == 1
    assert ledger.spent == pytest.approx(sum(r["cost_cny"] for r in record["calls"]), abs=1e-6)
    assert ledger.pending == 0


def test_answer_tool_mode_uses_short_dispatcher_contract(tmp_path):
    seen = []

    def dispatch(messages, info):
        seen.append(info.instructions)
        return response(ToolCallPart(info.output_tools[0].name, CHAT))

    cfg = live_config(
        tmp_path,
        lambda: FunctionModel(lambda *_: response(TextPart("直接回答。"))),
        lambda: FunctionModel(dispatch),
        answer_tools=True,
        web_client=FakeWeb(),
        answer_thinking="LOW",
    )
    _, _, record = run_loop(text_only_case(tmp_path), cfg)
    assert seen and "回答模型决定" in seen[0]
    assert record["answer"] == "直接回答。"


def test_reply_contract_is_last_once_on_primary_low_and_text_fallback(tmp_path):
    seen = []

    def failing_primary(messages, info):
        initial = next(p.content for m in messages for p in m.parts if p.part_kind == "user-prompt")
        assert initial[-1] == REPLY_CONTRACT
        assert sum(p.count(REPLY_CONTRACT) for p in initial if isinstance(p, str)) == 1
        seen.append("primary")
        raise ConnectionError("synthetic unavailable")

    def fallback(messages, info):
        if info.output_tools:
            return response(ToolCallPart(info.output_tools[0].name, CHAT))
        initial = next(p.content for m in messages for p in m.parts if p.part_kind == "user-prompt")
        assert initial[-1] == REPLY_CONTRACT
        assert sum(p.count(REPLY_CONTRACT) for p in initial if isinstance(p, str)) == 1
        seen.append("fallback")
        return response(TextPart("接梗。"))

    cfg = live_config(
        tmp_path,
        lambda: FunctionModel(failing_primary),
        lambda: FunctionModel(fallback),
        answer_tools=False,
        answer_thinking="HIGH",
    )
    _, _, record = run_loop(text_only_case(tmp_path), cfg)
    assert seen == ["primary", "primary", "fallback"]
    assert record["answer_source"] == "deepseek" and record["answer"] == "接梗。"
