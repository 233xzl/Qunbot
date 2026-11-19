"""Aggregate scorecard for live-model replay runs (T1.9 live tier).

Input rows are per-case results written by `scripts/replay/run_live.py`:
`{case_id, reviewed, status, score (scoring.score_case), live (LiveAgent record),
anchor_state, cost_cny, ...}`. Output is aggregate only (no chat text).

Route scales: fine = the 5 annotation routes; coarse = BASELINE.md scale where chat and
local_knowledge merge into `direct` (legacy always ran local retrieval).
`knowledge_ids` is not scored: batch-02 annotations do not label knowledge ids.

Route exclusions (`tests/replay/route_exclusions.json`): cases whose label depends on
material that the replay cannot provide (e.g. a quoted forward removed at extraction) are
left out of route and clarify agreement and counted under `route_excluded`.
"""

from collections import Counter
import json
from pathlib import Path
import statistics

COARSE = {"chat": "direct", "local_knowledge": "direct"}
STAGES = ("material", "kb_search", "decision", "tools", "answer")
FAILED_STATES = ("failed", "timeout", "cancelled", "queue_timeout", "rejected_busy")


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))
    return ordered[index]


def _dist(values: list[float], digits: int = 1) -> dict:
    return {
        "n": len(values),
        "p50": round(percentile(values, 0.5), digits) if values else None,
        "p95": round(percentile(values, 0.95), digits) if values else None,
        "mean": round(statistics.mean(values), digits) if values else None,
    }


def _rate(hits: int, n: int) -> dict:
    return {"n": n, "agree": hits, "rate": round(hits / n, 3) if n else None}


def route_metrics(rows: list[dict]) -> dict:
    fine = coarse = clarify = reply = 0
    n_route = n_clarify = n_reply = 0
    confusion: dict[str, Counter] = {}
    clar_expected = clar_done = no_clar = false_clar = 0
    for row in rows:
        checks = row["score"]["checks"]
        route = checks.get("route")
        if route and route["expected"]:
            n_route += 1
            fine += route["actual"] == route["expected"]
            want = COARSE.get(route["expected"], route["expected"])
            got = COARSE.get(route["actual"], route["actual"])
            coarse += want == got
            confusion.setdefault(route["expected"], Counter())[str(route["actual"])] += 1
        c = checks.get("clarify")
        if c:
            n_clarify += 1
            clarify += c["pass"]
            if c["expected"]:
                clar_expected += 1
                clar_done += c["actual"]
            else:
                no_clar += 1
                false_clar += c["actual"]
        r = checks.get("should_reply")
        if r:
            n_reply += 1
            reply += r["pass"]
    return {
        "route_fine": _rate(fine, n_route),
        "route_coarse": _rate(coarse, n_route),
        "clarify": _rate(clarify, n_clarify),
        "effective_clarify_rate": round(clar_done / clar_expected, 3) if clar_expected else None,
        "clarify_expected": clar_expected,
        "false_clarify_rate": round(false_clar / no_clar, 3) if no_clar else None,
        "should_reply": _rate(reply, n_reply),
        "route_confusion_expected_to_actual": {
            k: dict(sorted(v.items())) for k, v in sorted(confusion.items())
        },
    }


def context_metrics(rows: list[dict]) -> dict:
    need = hit = selected = forbidden = 0
    macro = []
    for row in rows:
        checks = row["score"]["checks"]
        req = checks.get("required_context")
        if req is None:
            continue
        expected = req["expected"]
        got = len(expected) - len(req["missing"])
        need, hit = need + len(expected), hit + got
        if expected:
            macro.append(got / len(expected))
        selected += row["score"]["context_selected"]
        forbidden += len(checks.get("forbidden_context", {}).get("used", []))
    return {
        "required_recall_micro": round(hit / need, 3) if need else None,
        "required_recall_macro": round(statistics.mean(macro), 3) if macro else None,
        "forbidden_share": round(forbidden / selected, 3) if selected else None,
        "selected_per_request": round(selected / len(rows), 1) if rows else None,
    }


def load_exclusions(path: Path) -> dict[str, str]:
    """case_id -> reason; a missing file means no exclusions."""
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != "route-exclusions-1":
        raise ValueError("unsupported route exclusion file")
    return {str(k): str(v["reason"]) for k, v in data["cases"].items()}


def web_metrics(rows: list[dict]) -> dict:
    """Per-run web tool usage from LiveAgent records (searches, reads, cost, sources)."""
    searches, reads, costs, sources = [], [], [], []
    statuses = Counter()
    used = 0
    for r in rows:
        web = (r.get("live") or {}).get("web") or {}
        if not web:
            continue
        used += 1
        searches.append(web.get("searches", 0))
        reads.append(web.get("reads", 0))
        costs.append(web.get("cost_cny", 0.0))
        sources.append(len(web.get("sources") or []))
        statuses.update(web.get("statuses") or {})
    return {
        "runs_with_web": used,
        "searches_per_run": _dist(searches, 2),
        "reads_per_run": _dist(reads, 2),
        "sources_per_run": _dist(sources, 1),
        "cost_cny_total": round(sum(costs), 4),
        "call_statuses": dict(sorted(statuses.items())),
    }


def subject_metrics(rows: list[dict]) -> dict:
    """D-17: decisions flagged subject_uncertain and their verification outcomes."""
    flagged = searched = 0
    statuses = Counter()
    for r in rows:
        live = r.get("live") or {}
        if not (live.get("decision") or {}).get("subject_uncertain"):
            continue
        flagged += 1
        check = live.get("subject_check") or {}
        searched += bool(check.get("triggered"))
        statuses[str(check.get("status"))] += 1
    return {"flagged": flagged, "searched": searched, "statuses": dict(sorted(statuses.items()))}


def supplement_metrics(rows: list[dict]) -> dict:
    """D-18: runs with a Gemini supplement trigger, reasons, outcomes, cost and latency."""
    reasons, statuses = Counter(), Counter()
    called, latencies, cost = 0, [], 0.0
    runs = 0
    for r in rows:
        sup = (r.get("live") or {}).get("supplement")
        if not sup:
            continue
        runs += 1
        reasons[str(sup.get("reason"))] += 1
        statuses[str(sup.get("status"))] += 1
        if sup.get("triggered"):
            called += 1
            if sup.get("latency_s") is not None:
                latencies.append(sup["latency_s"])
        cost += sup.get("cost_cny") or 0.0
    return {
        "runs": runs,
        "called": called,
        "reasons": dict(sorted(reasons.items())),
        "statuses": dict(sorted(statuses.items())),
        "latency_s": _dist(latencies, 2),
        "cost_cny_total": round(cost, 4),
    }


def summarize_live(rows: list[dict], exclusions: dict[str, str] | None = None) -> dict:
    exclusions = exclusions or {}
    done = [r for r in rows if r.get("status") == "completed"]
    lat = [r["live"]["latency_total_s"] for r in done if r["live"].get("latency_total_s")]
    stages = {
        s: _dist([r["live"]["stages"][s] for r in done if s in r["live"].get("stages", {})], 2)
        for s in STAGES
    }
    tok_in, tok_out, tok_think, costs = [], [], [], []
    calls = Counter()
    for r in rows:
        records = (r.get("live") or {}).get("calls") or []
        tok_in.append(sum(c["usage"]["input"] for c in records))
        tok_out.append(sum(c["usage"]["output"] for c in records))
        tok_think.append(sum(c["usage"]["thinking"] or 0 for c in records))
        costs.append(sum(c["cost_cny"] for c in records))
        for c in records:
            calls[f"{c['stage']}:{c['model']}:{c['status']}"] += 1
    failed = [
        r for r in rows if r.get("anchor_state") in FAILED_STATES or r["status"] != "completed"
    ]
    # Run deadline, or a run that failed because its model calls timed out.
    timeouts = [
        r
        for r in rows
        if r.get("anchor_state") == "timeout" or "timeout" in str(r.get("anchor_reason") or "")
    ]
    first = [r["live"].get("decision_first_try") for r in done if "decision_first_try" in r["live"]]
    rule_fallback = sum(
        1 for r in done if (r["live"].get("decision") or {}).get("fallback") == "rule"
    )
    answer_fallback = sum(1 for r in done if r["live"].get("answer_fallback_used"))
    jev_rows = [
        r for r in done if isinstance(r["live"].get("jev"), dict) and r["live"]["jev"].get("choice")
    ]
    jev_agree = sum(
        r["live"]["jev"]["choice"] == r["score"]["checks"]["route"]["expected"] for r in jev_rows
    )
    jev_vs_decision = sum(
        r["live"]["jev"]["choice"] == (r["live"].get("decision") or {}).get("route")
        for r in jev_rows
    )
    scored = [r for r in rows if r.get("score")]
    routed = [r for r in scored if r["case_id"] not in exclusions]
    reviewed = [r for r in routed if r.get("reviewed")]
    return {
        "cases": len(rows),
        "completed": len(done),
        "routing": route_metrics(routed),
        "routing_reviewed": route_metrics(reviewed),
        "reviewed_cases": len(reviewed),
        "route_excluded": dict(
            Counter(exclusions[r["case_id"]] for r in scored if r["case_id"] in exclusions)
        ),
        "web": web_metrics(rows),
        "subject_check": subject_metrics(rows),
        "supplement": supplement_metrics(rows),
        "answer_seconds_by_thinking": {
            level: _dist(
                [
                    r["live"]["stages"]["answer"]
                    for r in done
                    if str(r["live"].get("answer_thinking")) == level
                    and "answer" in r["live"].get("stages", {})
                ],
                2,
            )
            for level in sorted(
                {
                    str(r["live"]["answer_thinking"])
                    for r in done
                    if r["live"].get("answer_thinking")
                }
            )
        },
        "context": context_metrics(scored),
        "latency_seconds_total": _dist(lat),
        "latency_seconds_by_stage": stages,
        "tokens_per_request": {
            "input_mean": round(statistics.mean(tok_in)) if tok_in else None,
            "output_mean": round(statistics.mean(tok_out)) if tok_out else None,
            "thinking_mean": round(statistics.mean(tok_think)) if tok_think else None,
        },
        "cost_cny": {
            "total": round(sum(costs), 4),
            "per_request_mean": round(statistics.mean(costs), 5) if costs else None,
            "per_request_p95": round(percentile(costs, 0.95), 5) if costs else None,
        },
        "failure_rate": round(len(failed) / len(rows), 3) if rows else None,
        "timeout_rate": round(len(timeouts) / len(rows), 3) if rows else None,
        "failure_reasons": dict(Counter(r.get("anchor_reason") or r["status"] for r in failed)),
        "decision_first_try_success": _rate(sum(bool(x) for x in first), len(first)),
        "decision_rule_fallback": rule_fallback,
        "answer_fallback_used": answer_fallback,
        "jev_shadow": {
            "n": len(jev_rows),
            "agree_with_annotation": jev_agree,
            "agree_with_decision": jev_vs_decision,
        },
        "calls": dict(sorted(calls.items())),
        "isolation": {
            k: sum(r["score"]["isolation"][k] for r in scored)
            for k in (scored[0]["score"]["isolation"] if scored else {})
        },
        "sent_to_qq": sum(r.get("sent_to_qq", 0) for r in rows),
    }
