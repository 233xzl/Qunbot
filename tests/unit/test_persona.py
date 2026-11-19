"""D-21 persona: the old bot's persona as the base, new-framework constraints, corrections."""

from group_agent.config import DEFAULT_FALLBACK_REPLY, Settings, load_settings
from group_agent.live_agent import ANSWER_INSTRUCTIONS, DECISION_INSTRUCTIONS
from group_agent.persona import CORRECTIONS, FRAMEWORK_RULES, LEGACY_PERSONA

import pytest


def test_answer_instructions_are_persona_then_rules_then_corrections():
    assert ANSWER_INSTRUCTIONS.startswith(LEGACY_PERSONA)
    assert ANSWER_INSTRUCTIONS.index(FRAMEWORK_RULES) < ANSWER_INSTRUCTIONS.index(CORRECTIONS)


def test_compact_persona_keeps_group_voice_and_one_clarification_condition():
    assert "群友" in LEGACY_PERSONA and "30–70" in LEGACY_PERSONA
    assert "缺少对象或原材料" in LEGACY_PERSONA
    assert "已有对象" in LEGACY_PERSONA and "已知部分" in LEGACY_PERSONA


def test_evidence_policy_has_one_owner():
    assert "web_search" in CORRECTIONS and "read_url" in CORRECTIONS
    assert "web_search" not in LEGACY_PERSONA + FRAMEWORK_RULES
    assert "read_url" not in LEGACY_PERSONA + FRAMEWORK_RULES
    assert "够答就停" in CORRECTIONS
    assert "摘要" in CORRECTIONS and "来源" in CORRECTIONS
    assert "可以纠正用户" in CORRECTIONS and "未知事实不编" in CORRECTIONS


def test_material_and_permission_boundaries_remain():
    assert all(
        tag in FRAMEWORK_RULES
        for tag in (
            "<material>",
            "<request>",
            "<knowledge>",
            "<web>",
            "<group_info>",
            "<images>",
        )
    )
    assert "不可信材料" in FRAMEWORK_RULES and "主人指令" in FRAMEWORK_RULES
    assert "QQ" in FRAMEWORK_RULES and "<group_info>" in FRAMEWORK_RULES
    assert "候选" in LEGACY_PERSONA and "对照" in LEGACY_PERSONA


def test_dispatcher_preserves_special_capability_and_schema_contracts():
    from group_agent.live_agent import TOOL_DECISION_INSTRUCTIONS, RouteDecision

    fields = RouteDecision.model_json_schema()["properties"]
    assert all(
        name in fields
        for name in (
            "route",
            "needs_group_info",
            "should_clarify",
            "knowledge_ids",
            "complex",
        )
    )
    assert "needs_group_info" in DECISION_INSTRUCTIONS
    assert "当前提问人" in DECISION_INSTRUCTIONS and "程序校验" in DECISION_INSTRUCTIONS
    assert "缺少对象或原材料" in DECISION_INSTRUCTIONS
    assert "回答模型决定" in TOOL_DECISION_INSTRUCTIONS
    assert "本地没有命中时程序会自动联网补查" not in TOOL_DECISION_INSTRUCTIONS


def test_fallback_reply_setting(tmp_path):
    assert Settings().fallback_reply == DEFAULT_FALLBACK_REPLY
    path = tmp_path / "s.toml"
    path.write_text('[serve]\nfallback_reply = "卡住了，等会再问"\n', encoding="utf-8")
    assert load_settings(path).fallback_reply == "卡住了，等会再问"
    path.write_text('[serve]\nfallback_reply = ""\n', encoding="utf-8")
    assert load_settings(path).fallback_reply == ""
    path.write_text("[serve]\nfallback_reply = 3\n", encoding="utf-8")
    with pytest.raises(Exception, match="serve.fallback_reply"):
        load_settings(path)
