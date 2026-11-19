"""The answer prompt names the configured models (owner 2026-11-14)."""

from dataclasses import replace

from group_agent.live_agent import LiveAgent
from group_agent.persona import ANSWER_INSTRUCTIONS
from group_agent.web import WebBackends
from test_live_replay import DEEPSEEK, GEMINI, live_config, make_case, run_live, text_model


class Search:
    provider_name = "gemini_search"
    model = "gemini-3.8-flash"
    use_url_context = False


class Fetch:
    provider_name = "deepseek_web"


def note_for(tmp_path, **kw):
    config = live_config(tmp_path, lambda: text_model(), lambda: text_model(), **kw)
    return LiveAgent(None, config).model_note()


def test_note_names_roles_and_web_backends(tmp_path):
    gemini = replace(GEMINI, model="gemini-4-hydrogen", channel="vertex")
    fallback = replace(DEEPSEEK, model="deepseek-flash", channel="official:api.deepseek.com")
    config = live_config(
        tmp_path,
        lambda: text_model(),
        lambda: text_model(),
        web_client=WebBackends(search=Search(), fetch=Fetch()),
    )
    config = replace(config, gemini=gemini, deepseek=fallback)
    note = LiveAgent(None, config).model_note()
    assert "回答 gemini-4-hydrogen（Google Vertex AI）" in note
    assert "deepseek-flash（官方 API） 接替" in note
    assert "搜索 gemini-3.8-flash 内置 Google 搜索" in note
    assert "网页读取 DeepSeek 官方联网工具" in note


def test_note_without_web_and_in_prompt(tmp_path):
    (tmp_path / "a").mkdir()
    assert "搜索 不可用" in note_for(tmp_path / "a")
    assert "本次模型" in ANSWER_INSTRUCTIONS
    seen = []

    def capture():
        from pydantic_ai.messages import ModelResponse, TextPart
        from pydantic_ai.models.function import FunctionModel

        def fn(messages, info):
            seen.append(str(messages))
            return ModelResponse(parts=[TextPart("好")])

        return FunctionModel(fn)

    config = live_config(tmp_path, capture, lambda: text_model(), decider="deepseek")
    run_live(make_case(tmp_path), config)
    assert any("本次模型：回答" in s for s in seen)
