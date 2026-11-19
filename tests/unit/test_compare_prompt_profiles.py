"""The comparison driver freezes instructions and uses real fallback control flow."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys

from pydantic_ai.messages import TextPart
from pydantic_ai.models.function import FunctionModel
import pytest

from group_agent import live_agent
from group_agent.models import ModelBreaker
from test_answer_tool_loop import response, run_loop
from test_live_replay import FakeWeb, live_config, make_case


def driver():
    path = Path(__file__).resolve().parents[2] / "scripts/compare_prompt_profiles.py"
    spec = importlib.util.spec_from_file_location("compare_prompt_profiles_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def snapshot(tmp_path):
    (tmp_path / "before-live_agent.py").write_text(
        Path(live_agent.__file__).read_text(encoding="utf-8"), encoding="utf-8"
    )
    (tmp_path / "before-prompts.json").write_text(
        json.dumps(
            {"answer_instructions": "frozen answer", "decision_instructions": "frozen decision"}
        ),
        encoding="utf-8",
    )


def test_before_loads_whole_module_with_relative_imports_and_frozen_prompts(tmp_path, monkeypatch):
    mod = driver()
    snapshot(tmp_path)
    agent, metadata = mod.load_profile("before", tmp_path)
    frozen = sys.modules[agent.__module__]
    assert agent is not live_agent.LiveAgent
    assert agent.__module__.startswith("group_agent._prompt_profile_before_")
    assert frozen.ANSWER_INSTRUCTIONS == "frozen answer"
    assert frozen.DECISION_INSTRUCTIONS == "frozen decision"
    assert hasattr(agent, "_answer") and hasattr(agent, "run_tools")
    monkeypatch.setattr(live_agent, "ANSWER_INSTRUCTIONS", "changed after loading")
    assert frozen.ANSWER_INSTRUCTIONS == "frozen answer"
    assert metadata["answer_prompt_sha256"] == mod._sha("frozen answer")
    assert len(metadata["live_agent_sha256"]) == 64


def test_before_requires_complete_files(tmp_path):
    mod = driver()
    with pytest.raises(SystemExit, match="complete frozen"):
        mod.load_profile("before", tmp_path)
    snapshot(tmp_path)
    (tmp_path / "before-prompts.json").write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit, match="both nonempty"):
        mod.load_profile("before", tmp_path)


def test_forced_breaker_only_blocks_primary_and_labels_injection():
    mod = driver()
    breaker = mod.ForcedPrimaryBreaker("primary", ModelBreaker())
    assert not breaker.available("primary", count_skip=True)
    assert breaker.acquire("primary") is None
    assert breaker.state("primary") == "open_injected"
    breaker.record("primary", None, "ok")
    assert not breaker.available("primary")  # Success cannot accidentally reset injection.
    assert breaker.acquire("fallback") == "closed"
    assert breaker.summary()["primary"]["skipped"] == 2
    assert breaker.summary()["primary"]["fault_injection"] == "controlled_primary_unavailable"


def test_forced_fallback_uses_existing_image_and_tool_branch_without_primary(tmp_path):
    mod = driver()
    seen, web = [], FakeWeb()

    def primary():
        pytest.fail("forced primary must never build or call a real primary model")

    def fallback(messages, info):
        seen.append((messages, info))
        assert {t.name for t in info.function_tools} == {"web_search", "read_url"}
        assert any(
            getattr(item, "media_type", None) == "image/jpeg"
            for item in messages[0].parts[0].content
        )
        return response(TextPart("fallback answer"))

    cfg = live_config(
        tmp_path,
        primary,
        lambda: FunctionModel(fallback),
        answer_tools=True,
        decision_mode="lightweight",
        web_client=web,
    )
    mod.inject_config(cfg)
    result, trace, record = run_loop(make_case(tmp_path), cfg)
    assert seen and record["answer_source"] == "deepseek"
    assert record["answer_fallback_used"]
    assert "answer_gemini_skipped" in trace["degraded"]
    assert record["breaker"]["answer"] == "open_injected"
    assert cfg.ledger.spent > 0 and cfg.ledger.pending == 0
    assert result["sent_to_qq"] == 0


def test_cli_passes_replay_arguments_and_tags_summary_and_pairs(tmp_path, monkeypatch):
    mod = driver()
    data = tmp_path / "data"
    out = data / "comparison"
    monkeypatch.setattr(mod, "DATA_ROOT", data)
    monkeypatch.setattr(
        mod,
        "load_profile",
        lambda *args: ("chosen-agent", {"profile": "after", "prompt_sha256": "hash"}),
    )
    original_config = SimpleNamespace(
        gemini=SimpleNamespace(name="p"), deepseek=SimpleNamespace(name="f"), model_breaker=None
    )
    replay = SimpleNamespace(LiveAgent="old-agent", build_live_config=lambda: original_config)
    forwarded = []

    def main(args):
        forwarded.extend(args)
        assert replay.LiveAgent == "chosen-agent"
        assert isinstance(replay.build_live_config().model_breaker, mod.ForcedPrimaryBreaker)
        out.mkdir(parents=True)
        (out / "summary.json").write_text('{"sent_to_qq": 0}', encoding="utf-8")
        (out / "pairs.jsonl").write_text('{"answer_source":"deepseek"}\n', encoding="utf-8")
        return 0

    replay.main = main
    monkeypatch.setattr(mod, "_load_replay", lambda: replay)
    assert (
        mod.main(
            [
                "--profile",
                "after",
                "--force-fallback",
                "--budget-cny",
                "0.2",
                "--ids",
                "synthetic",
                "--out",
                str(out),
            ]
        )
        == 0
    )
    assert forwarded[:4] == ["--budget-cny", "0.2", "--ids", "synthetic"]
    assert replay.LiveAgent == "old-agent"
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    pair = json.loads((out / "pairs.jsonl").read_text(encoding="utf-8"))
    assert summary["prompt_profile"]["forced_fallback"]
    assert "controlled" in pair["prompt_profile"]["label"]
    assert pair["answer_source"] == "deepseek"


@pytest.mark.parametrize(
    "args",
    [
        [],
        ["--budget-cny", "nan"],
        ["--budget-cny", "-1"],
        ["--budget-cny", "0.1", "--out", "docs/out"],
    ],
)
def test_cli_requires_budget_and_private_output(args):
    with pytest.raises(SystemExit):
        driver().main(["--profile", "after", *args])
