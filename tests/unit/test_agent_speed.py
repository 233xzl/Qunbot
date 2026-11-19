"""Offline speed policies retain special capabilities and run budgets."""

from pydantic_ai.messages import TextPart, ToolCallPart
from pydantic_ai.models.function import FunctionModel
import pytest

from group_agent.config import ConfigError, Settings, load_settings
from group_agent.live_agent import LiveAgent, choose_answer_thinking
from group_agent.policy import RuntimeLimits
from group_agent.replay.runner import run_case
from test_answer_tool_loop import CHAT, response, run_loop
from test_live_replay import FakeWeb, decision_model, live_config, make_case, text_only_case


def request_text(case, text):
    anchor = next(e for e in case["events"] if e["message_id"] == "m_2")
    anchor["message"] = [
        {"type": "at", "data": {"qq": "u_bot"}},
        {"type": "text", "data": {"text": text}},
    ]


def forbidden_decider():
    raise AssertionError("ordinary lightweight request invoked the decision model")


def test_lightweight_uses_answer_tools_and_local_candidates_without_decider(tmp_path):
    case, web, prompts, count = text_only_case(tmp_path), FakeWeb(), [], []
    request_text(case, "红色警报这个梗是什么意思")

    def answer(messages, info):
        count.append(1)
        prompts.append(str(messages))
        assert {t.name for t in info.function_tools} == {"web_search", "read_url"}
        if len(count) < 3:
            return response(
                ToolCallPart("web_search", {"query": "客观背景" + str(len(count))}, str(len(count)))
            )
        return response(TextPart("接梗回答。"))

    cfg = live_config(
        tmp_path,
        lambda: FunctionModel(answer),
        forbidden_decider,
        decision_mode="lightweight",
        answer_tools=True,
        answer_thinking="complex_only",
        web_client=web,
    )
    result, trace, record = run_loop(case, cfg, RuntimeLimits(web_searches=1))
    assert record["answer"] == "接梗回答。" and record["answer_thinking"] == "LOW"
    assert record["decision_mode"] == record["decision_source"] == "lightweight"
    assert record["decision_attempts"] == [] and record["decision"]["knowledge_ids"] == []
    assert "C0001" in record["knowledge_candidates"]
    assert "未确认匹配" in prompts[0] and "红色警报" in prompts[0]
    assert "路线判断：chat" not in prompts[0] and "直达回答" not in prompts[0]
    assert "只查答案必需" not in prompts[0]  # static evidence policy owns the search rules
    assert not [c for c in record["calls"] if c["stage"] == "decision"]
    assert (
        web.queries == ["客观背景1"] and record["web"]["calls"][-1]["status"] == "budget_exhausted"
    )
    assert result["sent_to_qq"] == 0


def test_picture_meme_stays_low_without_decider(tmp_path):
    case = make_case(tmp_path)
    # Keep the ordinary image fixture free of the separate admin-capability guard.
    case["events"][1]["message"][1] = {"type": "text", "data": {"text": "图片里的笑点"}}
    cfg = live_config(
        tmp_path,
        lambda: FunctionModel(lambda *_: response(TextPart("图片笑点。"))),
        forbidden_decider,
        decision_mode="lightweight",
        answer_tools=True,
        answer_thinking="complex_only",
        web_client=FakeWeb(),
    )
    _, trace, record = run_loop(case, cfg)
    assert record["decision_source"] == "lightweight"
    assert record["answer_thinking"] == "LOW" and record["answer"] == "图片笑点。"
    assert trace["route"] == "chat"


def test_complex_decision_remains_high_and_model_mode_restores_classifier(tmp_path):
    cfg = live_config(
        tmp_path,
        lambda: FunctionModel(lambda *_: response(TextPart("分析。"))),
        lambda: decision_model({**CHAT, "complex": True}),
        decision_mode="model",
        answer_tools=True,
        answer_thinking="complex_only",
        web_client=FakeWeb(),
    )
    _, _, record = run_loop(text_only_case(tmp_path), cfg)
    assert record["decision_source"] == "model" and record["answer_thinking"] == "HIGH"
    assert len([c for c in record["calls"] if c["stage"] == "decision"]) == 1


@pytest.mark.parametrize(
    "text", ["本群有多少人？", "管理员有哪些？", "我什么时候入群的？", "群主是谁？"]
)
def test_lightweight_preserves_group_info_judgement(tmp_path, text):
    case, queries = text_only_case(tmp_path), []
    request_text(case, text)

    class GroupTool:
        async def call(self, args):
            queries.append(args["query"])
            return {"status": "ok", "material": "本群资料"}

    cfg = live_config(
        tmp_path,
        lambda: FunctionModel(lambda *_: response(TextPart("群资料。"))),
        lambda: decision_model({**CHAT, "needs_group_info": True}),
        decision_mode="lightweight",
        answer_tools=True,
        group_info=lambda group, user: GroupTool(),
    )
    _, _, record = run_loop(case, cfg)
    assert record["decision_source"] == "model" and record["decision_guard"] == "group_info"
    assert queries and record["answer"] == "群资料。"


def test_lightweight_sparse_request_retains_clarification(tmp_path):
    case = text_only_case(tmp_path)
    request_text(case, "在吗")
    cfg = live_config(
        tmp_path,
        forbidden_decider,
        lambda: decision_model(
            {
                "route": "clarify",
                "should_clarify": True,
                "clarify_question": "想聊什么？",
                "reason": "没有对象",
            }
        ),
        decision_mode="lightweight",
        answer_tools=True,
    )
    _, trace, record = run_loop(case, cfg)
    assert trace["route"] == "clarify" and record["decision_guard"] == "sparse_request"
    assert record["answer"] == "想聊什么？"


@pytest.mark.parametrize("owner,expected", [(None, "unauthorized"), ("u_2", "committed")])
def test_lightweight_keeps_code_teaching_permissions(tmp_path, owner, expected):
    case = text_only_case(tmp_path)
    request_text(case, "记住 本群：「开黑」指周五组队")
    case["environment"]["owner_id"] = owner
    cfg = live_config(
        tmp_path,
        forbidden_decider,
        forbidden_decider,
        decision_mode="lightweight",
        answer_tools=True,
    )
    result = run_case(case, "B", agent_factory=lambda replay: LiveAgent(replay, cfg))
    trace = next(r for r in result["runs"] if r["message_id"] == "m_2")
    assert trace["route"] == "teaching" and trace["teaching"]["outcome"] == expected
    assert not cfg.ledger.records


def test_lightweight_degraded_has_no_web_tools(tmp_path):
    case, web = text_only_case(tmp_path), FakeWeb()
    case["mocks"] = {
        "faults": [{"name": "usage_ledger", "delay_ms": 0, "response": {"ratio": 0.9}}]
    }

    def answer(messages, info):
        assert not info.function_tools
        return response(TextPart("直接回答。"))

    cfg = live_config(
        tmp_path,
        lambda: FunctionModel(answer),
        forbidden_decider,
        decision_mode="lightweight",
        answer_tools=True,
        web_client=web,
    )
    _, _, record = run_loop(case, cfg)
    assert record["decision_source"] == "lightweight" and not web.queries


def test_speed_settings_defaults_and_legacy_options(tmp_path):
    settings = Settings()
    assert settings.decision_mode == "lightweight" and settings.answer_thinking == "complex_only"
    path = tmp_path / "settings.toml"
    path.write_text('[agent]\ndecision_mode = "model"\nanswer_thinking = "auto"', encoding="utf-8")
    loaded = load_settings(path)
    assert loaded.decision_mode == "model" and loaded.answer_thinking == "auto"
    for field in ("decision_mode", "answer_thinking"):
        path.write_text("[agent]\n" + field + ' = "invalid"', encoding="utf-8")
        with pytest.raises(ConfigError, match="agent." + field):
            load_settings(path)
    assert (
        choose_answer_thinking("auto", "LOW", images=True, web_material=False, complex_=False)
        == "HIGH"
    )
    assert (
        choose_answer_thinking(
            "complex_only", "LOW", images=True, web_material=True, complex_=False
        )
        == "LOW"
    )
    assert (
        choose_answer_thinking(
            "complex_only", "LOW", images=False, web_material=False, complex_=True
        )
        == "HIGH"
    )


def test_lightweight_requires_answer_tool_loop_and_direct_configs_keep_legacy(tmp_path):
    cfg = live_config(
        tmp_path,
        lambda: FunctionModel(lambda *_: response(TextPart("直接回答。"))),
        lambda: decision_model(CHAT),
    )
    assert cfg.decision_mode == "model" and cfg.answer_thinking == "auto"
    cfg.decision_mode = "lightweight"
    assert not cfg.answer_tools
    _, _, record = run_loop(text_only_case(tmp_path), cfg)
    assert record["decision_mode"] == record["decision_source"] == "model"
    assert record["decision_attempts"]


def test_lightweight_decision_diagnostic_still_uses_classifier(tmp_path):
    cfg = live_config(
        tmp_path,
        forbidden_decider,
        lambda: decision_model(CHAT),
        decision_mode="lightweight",
        answer_tools=True,
        decision_only=True,
    )
    _, _, record = run_loop(text_only_case(tmp_path), cfg)
    assert record["decision_guard"] == "decision_diagnostic"
    assert record["decision_source"] == "model"
    assert record["answer_source"] == "decision_only"


def test_lightweight_short_reply_inherits_quoted_group_info_guard(tmp_path):
    case = text_only_case(tmp_path)
    case["events"][1]["message"] = [{"type": "text", "data": {"text": "本群有几个机器人？"}}]
    request_text(case, "帮看看")
    anchor = next(e for e in case["events"] if e["message_id"] == "m_2")
    anchor["message"].insert(0, {"type": "reply", "data": {"id": "m_1"}})
    cfg = live_config(
        tmp_path,
        lambda: FunctionModel(lambda *_: response(TextPart("本群资料。"))),
        lambda: decision_model({**CHAT, "needs_group_info": True}),
        decision_mode="lightweight",
        answer_tools=True,
    )
    _, _, record = run_loop(case, cfg)
    assert record["decision_source"] == "model"
    assert record["decision_guard"] == "group_info"
    assert record["decision"]["needs_group_info"]


def test_lightweight_unrelated_group_info_history_does_not_trigger_guard(tmp_path):
    case = text_only_case(tmp_path)
    case["events"][1]["message"] = [{"type": "text", "data": {"text": "本群有几个机器人？"}}]
    request_text(case, "帮忙看看这个笑话")
    cfg = live_config(
        tmp_path,
        lambda: FunctionModel(lambda *_: response(TextPart("接个笑话。"))),
        forbidden_decider,
        decision_mode="lightweight",
        answer_tools=True,
    )
    _, _, record = run_loop(case, cfg)
    assert record["decision_source"] == "lightweight"
    assert "decision_guard" not in record
