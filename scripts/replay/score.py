"""Score context strategies and the legacy pipeline (baseline A) on annotated replay cases (T1.10).

Prints aggregate metrics only (no chat text, no identifiers beyond case ids on request).

Metrics (requirements section 9):
- context: required-context recall (micro and macro), irrelevant share = forbidden prior
  messages / selected, unmarked share, selected count, exact-cover rate;
- legacy behaviour (from legacy_result fields): route agreement on a coarse scale
  (teaching / web / clarify / direct), effective-clarify rate, false-clarify rate,
  latency p50/p95, tokens and cost per request.
Isolation metrics are not derivable from legacy fields; the deterministic runner reports
them for the new pipeline (scripts/replay/run_cases.py).

Usage:
    python scripts/replay/score.py <batch_dir> [--annotations <sidecar>] [--reviewed <file>]
        [--json] [--per-case]
"""

import argparse
import json
from pathlib import Path
import re
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from group_agent.context_strategies import STRATEGIES, explain_c, strategy_c  # noqa: E402
from validate_cases import annotated_case  # noqa: E402

# CNY per 1M tokens: (input, cached input, output); keep in sync with config prices.
PRICES = {
    "gemini-3.8-flash": (0.225, 0.0225, 1.125),
    "deepseek-v4.1-flash": (0.45, 0.045, 2.25),
}
# The legacy reply asks the user for material or meaning instead of answering.
CLARIFY_PATTERN = re.compile(
    r"(重新发|再发|发一下|发我|带上|补充|截图|原图|看不到|看不清|没看到|是指|指的是|哪个|哪一|什么意思|具体说说|说清楚)"
)
COARSE = {"chat": "direct", "local_knowledge": "direct"}


def load_cases(batch: Path, sidecar_path: Path | None, reviewed_path: Path | None) -> list[dict]:
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8")) if sidecar_path else None
    reviewed = json.loads(reviewed_path.read_text(encoding="utf-8")) if reviewed_path else None
    cases = []
    for path in sorted(batch.glob("*.json")):
        case = json.loads(path.read_text(encoding="utf-8"))
        if "case_id" not in case or "events" not in case:
            continue
        if sidecar is not None and case["case_id"] not in sidecar.get("cases", {}):
            continue
        cases.append(annotated_case(case, sidecar, reviewed))
    return cases


def prior_ids(case: dict) -> list[str]:
    anchor = case["anchor"]["message_id"]
    ids = []
    for event in case["events"]:
        mid = str(event.get("message_id"))
        if mid == anchor:
            break
        ids.append(mid)
    return ids


def score_context(cases: list[dict], strategies: dict | None = None) -> tuple[dict, dict]:
    per_strategy, per_case = {}, {}
    for name, strategy in (strategies or STRATEGIES).items():
        hit = need = selected_total = forbidden_total = unmarked_total = exact = 0
        macro = []
        for case in cases:
            ann = case["annotations"]
            if ann.get("status") not in ("draft", "reviewed") or not ann.get("should_reply"):
                continue
            prior = set(prior_ids(case))
            required = set(ann.get("required_context_ids") or [])
            forbidden = set(ann.get("forbidden_context_ids") or []) & prior
            chosen = set(strategy(case["events"], case["anchor"]["message_id"]))
            got = len(chosen & required)
            hit, need = hit + got, need + len(required)
            if required:
                macro.append(got / len(required))
            selected_total += len(chosen)
            forbidden_total += len(chosen & forbidden)
            unmarked_total += len(chosen - required - forbidden)
            exact += int(required <= chosen)
            per_case.setdefault(case["case_id"], {})[name] = {
                "required": len(required),
                "hit": got,
                "selected": len(chosen),
                "forbidden": len(chosen & forbidden),
            }
        scored = sum(1 for c in cases if c["case_id"] in per_case)
        per_strategy[name] = {
            "cases": scored,
            "required_recall_micro": round(hit / need, 3) if need else None,
            "required_recall_macro": round(statistics.mean(macro), 3) if macro else None,
            "all_required_covered_rate": round(exact / scored, 3) if scored else None,
            "irrelevant_share": round(forbidden_total / selected_total, 3)
            if selected_total
            else None,
            "unmarked_share": round(unmarked_total / selected_total, 3) if selected_total else None,
            "selected_per_request": round(selected_total / scored, 1) if scored else None,
        }
    return per_strategy, per_case


C_RULES = (
    "continues",
    "refers_to",
    "author_burst",
    "same_topic",
    "asker_filter",
    "recent_chat",
    "quote_context",
    "urges",
)


def _scored(cases: list[dict]) -> list[dict]:
    return [
        c
        for c in cases
        if c["annotations"].get("status") in ("draft", "reviewed")
        and c["annotations"].get("should_reply")
    ]


def relation_contribution(cases: list[dict]) -> dict:
    """Strategy C picks grouped by relation/method: how many were selected, and how many
    of those are annotated required, forbidden or unmarked."""
    out: dict[str, dict[str, int]] = {}
    for case in _scored(cases):
        ann = case["annotations"]
        required = set(ann.get("required_context_ids") or [])
        forbidden = set(ann.get("forbidden_context_ids") or [])
        for pick in explain_c(case["events"], case["anchor"]["message_id"]):
            key = f"{pick.relation}/{pick.method}"
            row = out.setdefault(key, {"selected": 0, "required": 0, "forbidden": 0, "unmarked": 0})
            row["selected"] += 1
            if pick.message_id in required:
                row["required"] += 1
            elif pick.message_id in forbidden:
                row["forbidden"] += 1
            else:
                row["unmarked"] += 1
    return dict(sorted(out.items()))


def rule_ablation(cases: list[dict]) -> dict:
    """Strategy C with one implicit rule switched off at a time (and all off)."""
    variants = {"C": strategy_c}
    for rule in C_RULES:
        rest = frozenset(set(C_RULES) - {rule})
        variants[f"C-{rule}"] = lambda e, a, rest=rest: strategy_c(e, a, rules=rest)
    variants["C-none"] = lambda e, a: strategy_c(e, a, rules=frozenset())
    metrics, _ = score_context(cases, variants)
    keep = ("required_recall_micro", "all_required_covered_rate", "irrelevant_share")
    return {
        name: {k: m[k] for k in keep} | {"selected": m["selected_per_request"]}
        for name, m in metrics.items()
    }


def split_cases(cases: list[dict], tune_dir: Path | None) -> dict[str, list[dict]]:
    """Overall, owner-reviewed, and (with `tune_dir`) the tuning split = cases also present
    in `tune_dir` (C's rules were written while looking only at those) vs held-out."""
    splits = {
        "overall": cases,
        "reviewed": [c for c in cases if c["annotations"].get("status") == "reviewed"],
    }
    if tune_dir is not None:
        tune = {p.stem for p in tune_dir.glob("*.json")}
        splits["tune"] = [c for c in cases if c["case_id"] in tune]
        splits["held_out"] = [c for c in cases if c["case_id"] not in tune]
    return splits


def legacy_route(case: dict) -> str | None:
    legacy = case.get("legacy_result") or {}
    diag = legacy.get("diagnostics")
    reply_id = (legacy.get("bot_reply") or {}).get("message_id")
    if diag is None and reply_id is None:
        return None
    if diag and diag.get("teaching"):
        return "teaching"
    if diag and (
        diag.get("search_count")
        or diag.get("fetch_count")
        or diag.get("web_calls")
        or diag.get("url_fetch_calls")
        or diag.get("account_lookup")
    ):
        return "web"
    text = ""
    for event in case["events"]:
        if str(event.get("message_id")) == str(reply_id):
            text = "".join(
                str((s.get("data") or {}).get("text", ""))
                for s in event.get("message") or []
                if isinstance(s, dict) and s.get("type") == "text"
            )
            break
    if text and len(text) <= 120 and ("?" in text or "？" in text) and CLARIFY_PATTERN.search(text):
        return "clarify"
    return "direct"


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))
    return ordered[index]


def score_legacy(cases: list[dict]) -> dict:
    agree = compared = 0
    clarify_expected = clarify_done = false_clarify = no_clarify_expected = 0
    confusion: dict[str, dict[str, int]] = {}
    latencies, costs, tokens_in, tokens_out = [], [], [], []
    for case in cases:
        ann = case["annotations"]
        if ann.get("status") not in ("draft", "reviewed") or not ann.get("should_reply"):
            continue
        legacy = case.get("legacy_result") or {}
        observed = legacy_route(case)
        expected = ann.get("expected_route")
        if observed is not None and expected:
            want = COARSE.get(expected, expected)
            compared += 1
            agree += int(want == observed)
            confusion.setdefault(want, {}).setdefault(observed, 0)
            confusion[want][observed] += 1
            if ann.get("should_clarify"):
                clarify_expected += 1
                clarify_done += int(observed == "clarify")
            else:
                no_clarify_expected += 1
                false_clarify += int(observed == "clarify")
        diag = legacy.get("diagnostics") or {}
        latency = legacy.get("latency_ms") or diag.get("latency_ms")
        if isinstance(latency, (int, float)):
            latencies.append(latency / 1000)
        calls = legacy.get("provider_calls") or {}
        if calls.get("call_count"):
            models = calls.get("models") or []
            price = PRICES.get(models[0] if len(models) == 1 else "gemini-3.8-flash")
            other, cached, out = (
                calls.get("token_input_other") or 0,
                calls.get("token_input_cached") or 0,
                calls.get("token_output") or 0,
            )
            tokens_in.append(other + cached)
            tokens_out.append(out)
            costs.append((other * price[0] + cached * price[1] + out * price[2]) / 1e6)
    return {
        "route_compared": compared,
        "route_agreement_coarse": round(agree / compared, 3) if compared else None,
        "route_confusion_expected_to_observed": confusion,
        "effective_clarify_rate": round(clarify_done / clarify_expected, 3)
        if clarify_expected
        else None,
        "clarify_expected": clarify_expected,
        "false_clarify_rate": round(false_clarify / no_clarify_expected, 3)
        if no_clarify_expected
        else None,
        "latency_seconds": {
            "n": len(latencies),
            "p50": round(percentile(latencies, 0.5), 1) if latencies else None,
            "p95": round(percentile(latencies, 0.95), 1) if latencies else None,
        },
        "tokens_per_request": {
            "n": len(tokens_in),
            "input_mean": round(statistics.mean(tokens_in)) if tokens_in else None,
            "output_mean": round(statistics.mean(tokens_out)) if tokens_out else None,
        },
        "cost_cny_per_request": {
            "mean": round(statistics.mean(costs), 5) if costs else None,
            "p95": round(percentile(costs, 0.95), 5) if costs else None,
        },
        "isolation_errors": "not measurable from legacy fields; new-pipeline isolation"
        " counters come from scripts/replay/run_cases.py (deterministic runner, T1.9)",
        "notes": [
            "legacy route is inferred: teaching/web from diagnostics flags, clarify by reply-text"
            " heuristic, everything else 'direct' (chat and local_knowledge are merged because"
            " legacy always runs local retrieval)",
            "cost uses relay prices; calls with several models are priced as Gemini",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch", type=Path)
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--reviewed", type=Path)
    parser.add_argument("--per-case", action="store_true", help="include per-case counts")
    parser.add_argument(
        "--tune-dir",
        type=Path,
        help="cases also in this directory form the tuning split; the rest are held out",
    )
    parser.add_argument(
        "--relations", action="store_true", help="strategy C per-relation counts and ablation"
    )
    args = parser.parse_args()
    cases = load_cases(args.batch, args.annotations, args.reviewed)
    context, per_case = score_context(cases)
    result = {
        "batch": args.batch.name,
        "cases": len(cases),
        "context_strategies": context,
        "legacy_baseline_A": score_legacy(cases),
    }
    splits = split_cases(cases, args.tune_dir)
    if len(splits) > 1:
        result["context_by_split"] = {
            name: {"cases": len(subset), **score_context(subset)[0]}
            for name, subset in splits.items()
            if name != "overall"
        }
    if args.relations:
        result["c_relations"] = {name: relation_contribution(s) for name, s in splits.items()}
        result["c_ablation"] = {name: rule_ablation(s) for name, s in splits.items()}
    if args.per_case:
        result["per_case"] = per_case
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
