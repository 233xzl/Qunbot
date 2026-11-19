"""Role aliases select protocol behavior independently of historical model names."""

from dataclasses import replace

import pytest

from group_agent.config import Settings
from group_agent.serve import build_live_config, dry_model_builder
from group_agent.models import Ledger
from test_live_replay import DEEPSEEK, GEMINI, make_case, run_live


@pytest.mark.parametrize(
    "answer_protocol,decision_protocol",
    [
        ("anthropic", "openai-compatible"),
        ("openai-compatible", "anthropic"),
        ("google-genai", "openai-compatible"),
        ("openai-compatible", "google-genai"),
    ],
)
def test_generation_roles_are_independent_of_model_aliases(
    tmp_path,
    answer_protocol,
    decision_protocol,
):
    answer = replace(
        GEMINI, name="answer_alias", protocol=answer_protocol, thinking_level=None, api_version=None
    )
    decision = replace(
        GEMINI,
        name="route_alias",
        protocol=decision_protocol,
        thinking_level=None,
        api_version=None,
    )
    fallback = replace(DEEPSEEK, name="backup_alias")
    settings = Settings(
        decision_mode="model",
        models={m.name: m for m in (answer, decision, fallback)},
        model_roles={"answer": answer.name, "decision": decision.name, "fallback": fallback.name},
    )
    cfg = build_live_config(
        settings,
        ledger=Ledger(1.0),
        media_root=tmp_path,
        dry_models=True,
        web=False,
        group_info=None,
        knowledge_path=None,
    )
    seen = []

    def tracked_builder(spec, creds):
        seen.append((spec.name, spec.protocol))
        return dry_model_builder(spec, creds)

    cfg = replace(cfg, model_builder=tracked_builder)
    result, anchor, record = run_live(make_case(tmp_path), cfg)
    assert record["answer_source"] == answer.name
    assert record["decision_attempts"] == [
        decision.name + (":native:ok" if decision_protocol == "google-genai" else ":tool:ok")
    ]
    assert (answer.name, answer_protocol) in seen
    assert (decision.name, decision_protocol) in seen
    assert not any(name == fallback.name for name, _ in seen)
    assert cfg.ledger.pending == 0 and cfg.ledger.spent == 0
