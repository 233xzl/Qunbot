"""replay_shadow summary: answer/decision sources and degradation counts (persona v2)."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import replay_shadow  # noqa: E402


def line(source, decision_source, degraded, route="chat"):
    return {
        "new": "答",
        "old": "旧",
        "shadow_new": "影",
        "state": "succeeded",
        "error_kind": None,
        "answer_source": source,
        "answer_fallback_used": bool(degraded),
        "decision_fallback": "rule" if decision_source == "rule" else None,
        "decision_source": decision_source,
        "degraded": degraded,
        "route": route,
        "shadow_route": "chat",
        "latency_s": 1.0,
    }


def test_summary_counts_sources_and_degradations():
    lines = [
        line("gemini", "gemini", []),
        line(
            "deepseek",
            "deepseek",
            [
                "decision_gemini_skipped",
                "decision_fallback:deepseek",
                "answer_gemini_skipped",
                "answer_fallback:deepseek",
            ],
        ),
        line("deepseek:no_images", "rule", ["decision_rule_fallback", "answer_fallback:x"]),
        line("decision_clarify_question", "gemini", [], route="clarify"),
    ]
    s = replay_shadow.summarize(lines, 0.1, {"name": "t"})
    assert s["answer_models"] == {"gemini": 1, "deepseek": 2, "decision_clarify_question": 1}
    assert s["decision_sources"] == {"gemini": 2, "deepseek": 1, "rule": 1}
    assert s["degraded_runs"] == 2
    assert s["degraded"]["answer_gemini_skipped"] == 1
    assert s["degraded"]["decision_rule_fallback"] == 1
    assert s["routes"] == {"chat": 3, "clarify": 1}
