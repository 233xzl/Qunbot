"""Link replay anchors to old-pipeline (SnowLuma + AstrBot) observations, read-only.

Sources:
- bot replies: the bot's own messages in the same group (user_id == self_id);
- AstrBot meme_knowledge diagnostics.jsonl: per-request phase rows (no group field);
- AstrBot data_v4.db provider_stats: model calls, scoped by the group in `umo`.

Diagnostics cover every group, so request matching is by time window and is
recorded with an explicit confidence. Only aggregate numbers, phase names and
whitelisted exception class names are exported; raw request ids are pseudonymized
by the caller and `umo` values are reduced to an in-group boolean.
"""

from __future__ import annotations

import json
import re
import sqlite3
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPLY_WINDOW_SECONDS = 300
SLOW_MS = 60_000
# AstrBot's host clock (diagnostics/provider_stats) and SnowLuma message times are
# not the same clock; the observed offset drifted by tens of seconds. Matching
# therefore estimates a local offset instead of assuming zero.
MAX_CLOCK_OFFSET = 120.0
# |request duration - observed reply delay| for an offset seed candidate.
SEED_DURATION_TOLERANCE = 4.0
SEED_NEIGHBORS = 7
# |(finished.time - reply time) - local offset| for an accepted match.
OFFSET_TOLERANCE = 4.0
START_TOLERANCE = 8.0
AMBIGUITY_MARGIN = 2.0
PROVIDER_SLACK = 3.0
REASON_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,63}$")
KNOWLEDGE_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.:-]{1,64}$")
FAILURE_PHASES = ("vision_failed", "web_blocked", "web_failed", "teaching_failed")
TEACHING_PHASES = (
    "teaching_input",
    "teaching_parse",
    "teaching_parse_fallback",
    "teaching_committed",
    "teaching_failed",
)


def _reply_ids(event: dict) -> list[str]:
    message = event.get("message")
    if not isinstance(message, list):
        return []
    return [
        str(s["data"]["id"])
        for s in message
        if isinstance(s, dict)
        and s.get("type") == "reply"
        and isinstance(s.get("data"), dict)
        and s["data"].get("id") is not None
    ]


def _is_bot(row: Any) -> bool:
    return str(row.event.get("user_id")) == str(row.event.get("self_id"))


# ---------------------------------------------------------------------------
# Bot replies


def link_bot_replies(
    rows: list[Any], anchors: list[Any], window_seconds: int = REPLY_WINDOW_SECONDS
) -> dict[str, dict]:
    """Return {anchor.source_row: {"row", "rule", "confidence", ...}}.

    Rule 1 (`reply_segment_to_anchor`): a bot message in the window whose top-level
    reply segment targets the anchor. Rule 2 (`first_unquoted_bot_message_after_anchor`):
    otherwise the first bot message in the window that quotes nothing and is not
    claimed by another anchor. Each bot message is claimed at most once.
    """
    ordered = sorted(rows, key=lambda r: r.order)
    bots = [r for r in ordered if _is_bot(r)]
    anchor_orders = sorted(a.order for a in anchors)
    claimed: set[str] = set()
    result: dict[str, dict] = {}

    def window(anchor: Any) -> list[Any]:
        group = str(anchor.event.get("group_id"))
        return [
            r
            for r in bots
            if str(r.event.get("group_id")) == group
            and r.order > anchor.order
            and r.timestamp <= anchor.timestamp + window_seconds
        ]

    for anchor in sorted(anchors, key=lambda a: a.order):
        target = str(anchor.event.get("message_id"))
        for row in window(anchor):
            if row.source_row not in claimed and target in _reply_ids(row.event):
                claimed.add(row.source_row)
                result[anchor.source_row] = {
                    "row": row,
                    "rule": "reply_segment_to_anchor",
                    "confidence": "high",
                    "intervening_anchor_count": 0,
                }
                break
    for anchor in sorted(anchors, key=lambda a: a.order):
        if anchor.source_row in result:
            continue
        for row in window(anchor):
            if row.source_row in claimed or _reply_ids(row.event):
                continue
            intervening = sum(anchor.order < o < row.order for o in anchor_orders)
            claimed.add(row.source_row)
            result[anchor.source_row] = {
                "row": row,
                "rule": "first_unquoted_bot_message_after_anchor",
                "confidence": "medium" if intervening == 0 else "low",
                "intervening_anchor_count": intervening,
            }
            break
    return result


# ---------------------------------------------------------------------------
# Diagnostics


@dataclass
class DiagRequest:
    request_id: str
    rows: list[dict] = field(default_factory=list)

    @property
    def finished(self) -> dict | None:
        return next((r for r in reversed(self.rows) if r.get("phase") == "finished"), None)

    @property
    def end(self) -> float:
        done = self.finished
        return float(done["time"]) if done else max(float(r["time"]) for r in self.rows)

    @property
    def start(self) -> float:
        """finished.ms covers work done before the first logged row."""
        done = self.finished
        first = min(float(r["time"]) for r in self.rows)
        if done and isinstance(done.get("ms"), (int, float)):
            return min(first, float(done["time"]) - float(done["ms"]) / 1000)
        return first


def load_diagnostics(path: Path) -> tuple[list[DiagRequest], dict]:
    requests: dict[str, DiagRequest] = {}
    counts: Counter = Counter()
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            counts["lines"] += 1
            try:
                row = json.loads(line)
                stamp = float(row["time"])
                request_id = str(row["request"])
            except (ValueError, KeyError, TypeError):
                counts["invalid_skipped"] += 1
                continue
            row["time"] = stamp
            requests.setdefault(request_id, DiagRequest(request_id)).rows.append(row)
    for request in requests.values():
        request.rows.sort(key=lambda r: r["time"])
    counts["requests"] = len(requests)
    return sorted(requests.values(), key=lambda r: r.start), dict(counts)


def summarize_request(request: DiagRequest) -> dict:
    phases = Counter(str(r.get("phase")) for r in request.rows)
    done = request.finished
    reasons = sorted(
        {
            str(r["reason"])
            for r in request.rows
            if isinstance(r.get("reason"), str) and REASON_PATTERN.match(r["reason"])
        }
    )

    def maximum(key: str) -> int | None:
        values = [r[key] for r in request.rows if isinstance(r.get(key), int)]
        return max(values) if values else None

    latency = done.get("ms") if done and isinstance(done.get("ms"), (int, float)) else None
    flags = [p for p in FAILURE_PHASES if phases[p]]
    if latency is not None and latency > SLOW_MS:
        flags.append("slow_over_60s")
    if done is None:
        flags.append("no_finished_row")
    tokens_in = sum(
        r["input_tokens"] for r in request.rows if isinstance(r.get("input_tokens"), int)
    )
    tokens_out = sum(
        r["output_tokens"] for r in request.rows if isinstance(r.get("output_tokens"), int)
    )
    knowledge_ids: list[str] = []
    for row in request.rows:
        if row.get("phase") == "local_retrieval" and isinstance(row.get("ids"), list):
            for value in row["ids"]:
                if isinstance(value, str) and KNOWLEDGE_ID_PATTERN.match(value):
                    if value not in knowledge_ids:
                        knowledge_ids.append(value)
    return {
        "phase_counts": dict(sorted(phases.items())),
        "retrieved_knowledge_ids": knowledge_ids[:20],
        "local_only": bool(phases["local_retrieval"])
        and not phases["web"]
        and not phases["vision"]
        and not phases["vision_failed"],
        "url_fetch_calls": sum(r.get("kind") == "fetch" for r in request.rows),
        "latency_ms": latency,
        "search_count": (done or {}).get("search", maximum("search")),
        "fetch_count": (done or {}).get("fetch", maximum("fetch")),
        "web_calls": phases["web"],
        "web_blocked": phases["web_blocked"] > 0,
        "vision_status": (
            "failed"
            if phases["vision_failed"]
            else "succeeded"
            if phases["vision"]
            else "not_attempted"
        ),
        "vision_fallback": phases["vision_fallback"] > 0,
        "local_retrieval_images": maximum("images"),
        "group_context_selected": maximum("group_context_selected"),
        "account_lookup": any(r.get("bilibili_lookup") is True for r in request.rows),
        "teaching": any(phases[p] for p in TEACHING_PHASES),
        "failure_flags": flags,
        "failure_reasons": reasons,
        "diag_input_tokens": tokens_in if tokens_in else None,
        "diag_output_tokens": tokens_out if tokens_out else None,
    }


def image_extraction(summary: dict) -> dict:
    return {
        "status": summary["vision_status"],
        "fallback_used": summary["vision_fallback"],
        "local_retrieval_images": summary["local_retrieval_images"],
        "failure_reasons": [
            r for r in summary["failure_reasons"] if summary["vision_status"] == "failed"
        ],
        "source": "diagnostics_jsonl",
    }


@dataclass
class AnchorWindow:
    key: str
    anchor_time: float
    reply_time: float | None


def _nearby(requests: list[DiagRequest], center: float, radius: float) -> list[DiagRequest]:
    return [r for r in requests if abs(r.start - center) <= radius]


def estimate_offsets(windows: list[AnchorWindow], requests: list[DiagRequest]) -> dict[str, dict]:
    """Estimate the diagnostics-minus-message clock offset near each anchor.

    Seeds are anchors with exactly one finished request whose duration matches the
    observed reply delay; the local offset is the median of the nearest seeds.
    """
    seeds: list[tuple[float, float]] = []
    for window in windows:
        if window.reply_time is None:
            continue
        delay = window.reply_time - window.anchor_time
        offsets = [
            r.end - window.reply_time
            for r in _nearby(requests, window.anchor_time, 2 * MAX_CLOCK_OFFSET)
            if r.finished is not None
            and abs(r.start - window.anchor_time) <= MAX_CLOCK_OFFSET
            and abs(r.end - window.reply_time) <= MAX_CLOCK_OFFSET
            and abs((r.end - r.start) - delay) <= SEED_DURATION_TOLERANCE
        ]
        if len(offsets) == 1:
            seeds.append((window.anchor_time, offsets[0]))
    result = {}
    for window in windows:
        nearest = sorted(seeds, key=lambda s: abs(s[0] - window.anchor_time))[:SEED_NEIGHBORS]
        if not nearest:
            result[window.key] = {"offset": 0.0, "seed_count": 0, "spread": None}
            continue
        values = sorted(v for _, v in nearest)
        middle = values[len(values) // 2]
        result[window.key] = {
            "offset": round(middle, 3),
            "seed_count": len(values),
            "spread": round(values[-1] - values[0], 3),
        }
    return result


def _candidate_cost(window: AnchorWindow, request: DiagRequest, offset: float) -> float | None:
    start_error = request.start - (window.anchor_time + offset)
    if window.reply_time is None:
        # Without a reply only the request start can be compared: weak evidence.
        return 20.0 + abs(start_error) if abs(start_error) <= START_TOLERANCE else None
    finish_error = request.end - (window.reply_time + offset)
    if request.finished is None:
        if abs(start_error) > START_TOLERANCE or finish_error > OFFSET_TOLERANCE:
            return None
        return 10.0 + abs(start_error)
    if abs(finish_error) > OFFSET_TOLERANCE or start_error < -START_TOLERANCE:
        return None
    if start_error > START_TOLERANCE + REPLY_WINDOW_SECONDS:
        return None
    return abs(finish_error) + 0.05 * abs(start_error)


def link_diagnostics(
    windows: list[AnchorWindow],
    requests: list[DiagRequest],
    offsets: dict[str, dict] | None = None,
) -> dict[str, dict]:
    """Global one-to-one greedy assignment by cost; confidence reflects ambiguity."""
    offsets = offsets if offsets is not None else estimate_offsets(windows, requests)
    pairs = []
    per_anchor: dict[str, list[float]] = {}
    per_request: Counter = Counter()
    for window in windows:
        offset = offsets.get(window.key, {}).get("offset", 0.0)
        for request in _nearby(requests, window.anchor_time + offset, 2 * MAX_CLOCK_OFFSET):
            cost = _candidate_cost(window, request, offset)
            if cost is not None:
                pairs.append((cost, window.key, request.request_id, window, request))
                per_anchor.setdefault(window.key, []).append(cost)
                per_request[request.request_id] += 1
    pairs.sort(key=lambda p: (p[0], p[1], p[2]))
    used_requests: set[str] = set()
    result: dict[str, dict] = {}
    for cost, key, request_id, window, request in pairs:
        if key in result or request_id in used_requests:
            continue
        used_requests.add(request_id)
        costs = sorted(per_anchor[key])
        clear = len(costs) == 1 or costs[1] - cost >= AMBIGUITY_MARGIN
        exclusive = per_request[request_id] == 1
        seeded = offsets.get(key, {}).get("seed_count", 0) > 0
        if window.reply_time is None or request.finished is None:
            confidence = "low"
        elif clear and exclusive and seeded:
            confidence = "high"
        elif clear:
            confidence = "medium"
        else:
            confidence = "low"
        result[key] = {
            "request": request,
            "cost": round(cost, 3),
            "confidence": confidence,
            "candidate_count": len(costs),
            "competing_anchor_count": per_request[request_id] - 1,
            "clock_offset_seconds": offsets.get(key, {}).get("offset", 0.0),
            "rule": (
                "finish_near_bot_reply_offset_corrected"
                if window.reply_time is not None and request.finished is not None
                else "unfinished_request_before_reply"
                if window.reply_time is not None
                else "start_near_anchor_no_reply"
            ),
        }
    return result


# ---------------------------------------------------------------------------
# Provider stats


@dataclass
class ProviderCall:
    start: float
    end: float
    in_group: bool
    status: str
    model: str | None
    token_input_other: int
    token_input_cached: int
    token_output: int


def umo_in_group(umo: Any, group_id: str) -> bool:
    return str(group_id) in re.split(r"[:_!]", str(umo or ""))


def load_provider_stats(db_path: Path, group_id: str) -> tuple[list[ProviderCall], dict]:
    con = sqlite3.connect(Path(db_path).resolve().as_uri() + "?mode=ro", uri=True)
    calls = []
    try:
        con.execute("PRAGMA query_only=ON")
        con.execute("PRAGMA temp_store=MEMORY")
        query = (
            "SELECT umo,status,provider_model,token_input_other,token_input_cached,"
            "token_output,start_time,end_time FROM provider_stats"
        )
        for umo, status, model, other, cached, output, start, end in con.execute(query):
            calls.append(
                ProviderCall(
                    float(start),
                    float(end),
                    umo_in_group(umo, group_id),
                    str(status),
                    str(model) if model else None,
                    int(other or 0),
                    int(cached or 0),
                    int(output or 0),
                )
            )
    finally:
        con.close()
    calls.sort(key=lambda c: c.start)
    counts = {"rows": len(calls), "in_group_rows": sum(c.in_group for c in calls)}
    return calls, counts


def provider_calls_between(
    calls: list[ProviderCall], start: float, end: float, window: str
) -> dict:
    """In-group calls fully inside [start, end] (AstrBot clock), plus an overlap count."""
    inside = [
        c for c in calls if c.start >= start - PROVIDER_SLACK and c.end <= end + PROVIDER_SLACK
    ]
    mine = [c for c in inside if c.in_group]
    return {
        "call_count": len(mine),
        "other_group_overlap_count": len(inside) - len(mine),
        "statuses": dict(sorted(Counter(c.status for c in mine).items())),
        "models": sorted({c.model for c in mine if c.model}),
        "token_input_other": sum(c.token_input_other for c in mine),
        "token_input_cached": sum(c.token_input_cached for c in mine),
        "token_output": sum(c.token_output for c in mine),
        "model_seconds": round(sum(c.end - c.start for c in mine), 3),
        "window": window,
    }


CONFIDENCE_ORDER = ["low", "medium", "high"]


def adjust_diag_confidence(confidence: str, provider: dict | None) -> str:
    """Group-scoped provider calls inside the request span corroborate the match;
    only other-group calls inside it contradict it."""
    if not provider:
        return confidence
    if provider["call_count"] and not provider["other_group_overlap_count"]:
        return "high" if confidence != "low" else "medium"
    if not provider["call_count"] and provider["other_group_overlap_count"]:
        return {"high": "medium"}.get(confidence, "low")
    return confidence


def combine_link(
    bot: dict | None,
    diag_confidence: str | None,
    provider: dict | None,
    diagnostics_covered: bool,
) -> dict:
    """Overall link status, weakest-source confidence, and reasons for gaps."""
    sources = []
    reasons = []
    levels = []
    if bot:
        sources.append("bot_reply")
        levels.append(bot["confidence"])
    else:
        reasons.append("no_bot_message_in_reply_window")
    if diag_confidence:
        sources.append("diagnostics")
        levels.append(diag_confidence)
    else:
        reasons.append(
            "no_matching_diagnostics_request"
            if diagnostics_covered
            else "anchor_outside_diagnostics_coverage"
        )
    if provider and provider["call_count"]:
        sources.append("provider_stats")
    else:
        reasons.append("no_in_group_provider_call_in_window")
    if bot and diag_confidence:
        status = "linked"
    elif sources:
        status = "partial"
    else:
        status = "unlinked"
    return {
        "status": status,
        "confidence": min(levels, key=CONFIDENCE_ORDER.index) if levels else None,
        "sources": sources,
        "missing_reasons": reasons,
    }
