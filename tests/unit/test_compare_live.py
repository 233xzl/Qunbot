import importlib.util
from pathlib import Path
import sys

sys.path.insert(0, str(Path("scripts/replay").resolve()))
SPEC = importlib.util.spec_from_file_location(
    "compare_live", Path("scripts/replay/compare_live.py")
)
compare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(compare)


def row(context, answer, route="chat"):
    return {
        "case_id": "c1",
        "status": "completed",
        "anchor_state": "sent",
        "anchor_trace": {"context_ids": context, "context_reasons": {"m1": "reply_to/quote_chain"}},
        "score": {
            "checks": {
                "route": {"expected": "chat", "actual": route, "pass": route == "chat"},
                "required_context": {"expected": ["m1"], "missing": [] if context else ["m1"]},
                "forbidden_context": {"used": []},
            }
        },
        "live": {
            "answer": answer,
            "anchor_text": "<b>问</b>",
            "material": {"messages": len(context), "images_attached": 0, "chars": 10},
            "calls": [{"usage": {"input": 100, "output": 20}, "cost_cny": 0.001}],
            "latency_total_s": 3.0,
        },
    }


def test_row_stats_and_context_of():
    stats = compare.row_stats(row(["m1"], "a"))
    assert stats["required_hit"] == 1 and stats["context_messages"] == 1
    assert stats["input_tokens"] == 100 and stats["cost_cny"] == 0.001
    assert compare.context_of(None) == () and compare.context_of({"anchor_trace": None}) == ()


def test_review_html_escapes_and_marks_context_differences():
    case = {"annotations": {"status": "reviewed", "expected_route": "chat", "answer_points": []}}
    runs = {"B": {"c1": row([], "<script>x</script>")}, "C": {"c1": row(["m1"], "ok")}}
    page = compare.review_html(runs, {"c1": case}, ["c1"])
    assert "<script>x" not in page and "&lt;script&gt;" in page
    assert "&lt;b&gt;问" in page
    assert "各策略上下文不同" in page and "reply_to/quote_chain×1" in page
    same = compare.review_html({"B": runs["C"], "C": runs["C"]}, {"c1": case}, ["c1"])
    assert "上下文相同" in same


def test_parse_runs_requires_name_and_dir():
    assert compare.parse_runs(["A=data/x"]) == {"A": Path("data/x")}
    try:
        compare.parse_runs(["nodir"])
    except SystemExit:
        pass
    else:
        raise AssertionError("expected SystemExit")
