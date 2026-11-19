"""Read-only, local-only SnowLuma replay extraction (stdlib; no network access).

Run --help for usage. Real payloads never belong in tracked fixtures, even after
pseudonymization. This is an evidence exporter, NOT an admission/event normalizer.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import re
import secrets
import sqlite3
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
RAW_ROOT = ROOT / "tests/replay/raw"
PRIVATE_ROOT = ROOT / "tests/replay/private"
SCHEMA_VERSION = "1.0"
PREFIXES = {
    "account": "u",
    "group": "g",
    "message": "m",
    "nickname": "n",
    "group_name": "gn",
    "media": "media",
    "row": "row",
    "opaque": "opaque",
}
LIMITATIONS = [
    "pseudonymized_not_anonymous_local_only",
    "known_aliases_only_residual_identification_risk",
    "timestamp_sequence_order_not_verified_receive_order",
    "database_snapshot_may_omit_deleted_or_recalled_events",
    "followup_horizon_is_bounded_not_proof_of_no_later_reply",
    "explicit_reply_edge_is_not_proof_of_answer_quality",
    "no_diagnostics_or_provider_stats_joined_metrics_unknown",
    "images_not_downloaded_or_opened_no_content_hash",
    "unknown_segments_redacted_not_lossless_wire_payload",
    "owner_blacklist_and_live_configuration_not_inspected",
]


def seconds(value: Any) -> float:
    number = float(value)
    return number / 1000 if number > 10**12 else number


def segments(event: dict) -> list[dict]:
    value = event.get("message", [])
    return [s for s in value if isinstance(s, dict)] if isinstance(value, list) else []


def top_level_at_bot(event: dict) -> bool:
    """Never parse raw_message/CQ strings, nested replies, or forwarded messages."""
    bot = event.get("self_id")
    return (
        bot is not None
        and event.get("post_type") == "message"
        and event.get("message_type") == "group"
        and event.get("group_id") is not None
        and event.get("user_id") is not None
        and str(event["user_id"]) != str(bot)
        and any(
            s.get("type") == "at"
            and isinstance(s.get("data"), dict)
            and str(s["data"].get("qq")) == str(bot)
            for s in segments(event)
        )
    )


def reply_ids(event: dict) -> list[str]:
    return [
        str(s["data"]["id"])
        for s in segments(event)
        if s.get("type") == "reply"
        and isinstance(s.get("data"), dict)
        and s["data"].get("id") is not None
    ]


def media_locator(data: dict) -> Any:
    return data.get("file") or data.get("file_id") or data.get("id") or data.get("url")


def event_key(event: dict) -> tuple[str, str]:
    return str(event.get("group_id")), str(event.get("message_id"))


def plain_text(event: dict) -> str:
    return " ".join(
        str(s.get("data", {}).get("text", ""))
        for s in segments(event)
        if s.get("type") == "text" and isinstance(s.get("data"), dict)
    )


@dataclass
class SourceRow:
    event: dict
    timestamp: float
    sequence: int
    source_row: str
    sequence_authoritative: bool = True

    @property
    def order(self) -> tuple[float, int, str]:
        return self.timestamp, self.sequence, self.source_row


def read_rows(db_path: Path, group_id: str) -> tuple[list[SourceRow], dict]:
    """Select only the authorized group, with a second payload scope check.

    No immutable=1: that could ignore live WAL contents. temp_store=MEMORY avoids
    SQLite sorting spilling real chat to OS temp files. A read transaction gives
    one consistent view; no backup, writable connection, or production DDL.
    """
    con = sqlite3.connect(db_path.resolve().as_uri() + "?mode=ro", uri=True)
    counts: Counter = Counter()
    rows = []
    try:
        con.execute("PRAGMA query_only=ON")
        con.execute("PRAGMA temp_store=MEMORY")
        con.execute("BEGIN")
        query = (
            "SELECT message_hash,timestamp,sequence,sequence_authoritative,data "
            "FROM messages WHERE is_group=1 AND session_id=?"
        )
        for row_id, timestamp, sequence, authoritative, data in con.execute(query, (group_id,)):
            counts["rows_in_sql_scope"] += 1
            try:
                event = json.loads(data)
            except (ValueError, TypeError):
                counts["invalid_json_skipped"] += 1
                continue
            if not isinstance(event, dict) or str(event.get("group_id")) != str(group_id):
                counts["payload_scope_mismatch_skipped"] += 1
                continue
            if event.get("message_type") != "group" or not isinstance(event.get("message"), list):
                counts["non_array_or_non_message_skipped"] += 1
                continue
            if any(event.get(k) is None for k in ("message_id", "user_id", "self_id", "time")):
                counts["missing_identity_or_time_skipped"] += 1
                continue
            try:
                rows.append(
                    SourceRow(
                        event, seconds(timestamp), int(sequence), str(row_id), bool(authoritative)
                    )
                )
            except (ValueError, TypeError, OverflowError):
                counts["invalid_time_or_sequence_skipped"] += 1
        con.rollback()
    finally:
        con.close()
    rows.sort(key=lambda r: r.order)
    counts["usable_rows"] = len(rows)
    counts["top_level_at_anchors"] = sum(top_level_at_bot(r.event) for r in rows)
    return rows, dict(counts)


def build_index(rows: list[SourceRow]) -> dict[tuple[str, str], SourceRow]:
    """Ambiguous duplicate IDs are deliberately not resolved (also scope by group)."""
    buckets: dict[tuple[str, str], list[SourceRow]] = {}
    for row in rows:
        buckets.setdefault(event_key(row.event), []).append(row)
    return {key: values[0] for key, values in buckets.items() if len(values) == 1}


def select_context(
    rows: list[SourceRow], anchor: SourceRow, minutes: int = 30, minimum_messages: int = 80
) -> list[SourceRow]:
    """Union of the previous N messages and all previous M minutes, same group."""
    previous = sorted(
        (
            r
            for r in rows
            if str(r.event.get("group_id")) == str(anchor.event["group_id"])
            and r.order < anchor.order
        ),
        key=lambda r: r.order,
    )
    by_count = previous[-minimum_messages:] if minimum_messages else []
    threshold = anchor.timestamp - minutes * 60
    selected = {r.source_row for r in by_count}
    selected.update(r.source_row for r in previous if r.timestamp >= threshold)
    return [r for r in previous if r.source_row in selected]


def collect_followup(
    rows: list[SourceRow], anchor: SourceRow, minutes: int = 30
) -> tuple[list[SourceRow], list[tuple[SourceRow, str]]]:
    """Explicit same-group reply chains are evidence; timing alone stays uncertain."""
    group = str(anchor.event["group_id"])
    future = sorted(
        (
            r
            for r in rows
            if str(r.event.get("group_id")) == group
            and r.order > anchor.order
            and r.timestamp <= anchor.timestamp + minutes * 60
        ),
        key=lambda r: r.order,
    )
    index = build_index(rows)
    related = {event_key(anchor.event)} if event_key(anchor.event) in index else set()
    bot_results = []
    for row in future:
        targets = {(group, target) for target in reply_ids(row.event)}
        direct = (group, str(anchor.event["message_id"])) in targets and bool(related)
        chain = any(target in related and target in index for target in targets)
        if direct or chain:
            related.add(event_key(row.event))
        if str(row.event.get("user_id")) == str(anchor.event["self_id"]):
            status = (
                "explicit_reply" if direct else "explicit_reply_chain" if chain else "uncertain"
            )
            bot_results.append((row, status))
    return future, bot_results


class Pseudonymizer:
    """Domain-separated stable HMACs. Message IDs include their group scope."""

    def __init__(self, key: bytes):
        if len(key) < 32:
            raise ValueError("HMAC key must contain at least 32 bytes")
        self.key = key
        self.replacements: dict[str, str] = {}
        self.alias_pattern: re.Pattern | None = None

    def alias(self, domain: str, value: Any, scope: str = "") -> str:
        payload = json.dumps(
            [domain, scope, str(value)], ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
        digest = hmac.new(self.key, payload, hashlib.sha256).hexdigest()[:24]
        return f"{PREFIXES[domain]}_{digest}"

    def message_id(self, value: Any, group: Any) -> str:
        return self.alias("message", value, str(group))

    def media_alias(self, data: dict) -> str:
        """Pseudonym of the source locator; NOT a content hash."""
        return self.alias("media", media_locator(data) or json.dumps(data, sort_keys=True))

    def learn(self, rows: list[SourceRow]) -> None:
        """Learn only structured identities in the authorized group's snapshot."""
        # Names first; numeric identity mappings take priority over numeric nicknames.
        identities: list[tuple[str, Any]] = []
        for row in rows:
            o = row.event
            sender = o.get("sender") if isinstance(o.get("sender"), dict) else {}
            for name in (sender.get("nickname"), sender.get("card")):
                if isinstance(name, str) and name.strip():
                    self.replacements[name] = self.alias("nickname", name)
            if isinstance(o.get("group_name"), str) and o["group_name"].strip():
                self.replacements[o["group_name"]] = self.alias("group_name", o["group_name"])
            identities.extend(("account", o.get(k)) for k in ("user_id", "self_id", "operator_id"))
            identities.extend([("account", sender.get("user_id")), ("group", o.get("group_id"))])
            for seg in segments(o):
                data = seg.get("data", {})
                if seg.get("type") == "at" and isinstance(data, dict) and data.get("qq") != "all":
                    identities.append(("account", data.get("qq")))
        for domain, value in identities:
            if value is not None and str(value):
                self.replacements[str(value)] = self.alias(domain, value)
        if self.replacements:
            self.alias_pattern = re.compile(
                "|".join(re.escape(s) for s in sorted(self.replacements, key=len, reverse=True))
            )

    def text(self, value: Any) -> str:
        text = str(value)
        # Redact transport secrets/locators before alias substitution can split them.
        text = re.sub(r"(?i)(?:https?|ftp|file)://[^\s<>\]\)\"']+", "[URL_REMOVED]", text)
        text = re.sub(r"(?i)www\.[^\s<>\]\)\"']+", "[URL_REMOVED]", text)
        text = re.sub(
            r"(?i)\b(?:bearer\s+|(?:api[_-]?key|access[_-]?token|token|secret|password)"
            r"\s*[:=]\s*[\"']?)[A-Za-z0-9_./+=:-]+",
            "[SECRET_REMOVED]",
            text,
        )
        text = re.sub(
            r"\b(?:sk-[A-Za-z0-9_-]{10,}|eyJ[A-Za-z0-9_.-]{20,})\b", "[SECRET_REMOVED]", text
        )
        text = re.sub(r"(?:[A-Za-z]:[\\/]|\\\\)[^\s<>\"']+", "[PATH_REMOVED]", text)
        text = re.sub(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", "[EMAIL_REMOVED]", text)
        text = re.sub(
            r"(?i)[\w\-()\u4e00-\u9fff]+\.(?:png|jpe?g|gif|webp|bmp|svg|"
            r"mp[34]|mov|wav|ogg|pdf|docx?|xlsx?|pptx?|zip|rar|7z|exe|py|json|txt|csv)\b",
            "[FILENAME_REMOVED]",
            text,
        )
        text = re.sub(r"\b[A-Za-z0-9_+/=-]{32,}\b", "[OPAQUE_REMOVED]", text)
        if self.alias_pattern:
            text = self.alias_pattern.sub(lambda match: self.replacements[match.group()], text)
        # Unrecognized long numeric identifiers are removed too, not asserted anonymous.
        return re.sub(r"(?<![\w])\d{5,20}(?![\w])", "[NUMBER_REMOVED]", text)

    def sanitize_segments(self, original: list, group: Any, depth: int = 0) -> list[dict]:
        safe = []
        for segment in original:
            if not isinstance(segment, dict):
                safe.append({"type": "redacted", "data": {"reason": "invalid_segment"}})
                continue
            kind = segment.get("type")
            data = segment.get("data") if isinstance(segment.get("data"), dict) else {}
            if kind == "text":
                safe.append({"type": "text", "data": {"text": self.text(data.get("text", ""))}})
            elif kind == "at":
                target = data.get("qq")
                if target is None:
                    safe.append({"type": "redacted", "data": {"reason": "missing_at_target"}})
                else:
                    safe.append(
                        {
                            "type": "at",
                            "data": {
                                "qq": "all" if target == "all" else self.alias("account", target)
                            },
                        }
                    )
            elif kind == "reply":
                if data.get("id") is None:
                    safe.append({"type": "redacted", "data": {"reason": "missing_reply_target"}})
                    continue
                value = {"id": self.message_id(data["id"], group)}
                # A quoted at stays nested: it must never become a top-level trigger.
                if isinstance(data.get("message"), list) and depth < 8:
                    value["message"] = self.sanitize_segments(data["message"], group, depth + 1)
                safe.append({"type": "reply", "data": value})
            elif kind in {"image", "video", "record", "file"}:
                value = {
                    "media_id": self.media_alias(data),
                    "availability": "unavailable",
                    "reason": "not_acquired_no_network",
                    "content_sha256": None,
                    "local_metadata": {},
                }
                for field in ("width", "height", "size", "file_size", "sub_type"):
                    raw = data.get(field)
                    if (
                        isinstance(raw, (int, float))
                        and not isinstance(raw, bool)
                        and 0 <= raw <= 10**12
                    ):
                        value["local_metadata"][field] = raw
                safe.append({"type": kind, "data": value})
            elif kind == "face" and str(data.get("id", "")).isdigit():
                safe.append({"type": "face", "data": {"id": str(data["id"])}})
            else:
                # JSON cards, forwards, XML and unknown extensions can hide URLs/tokens.
                known = (
                    kind if kind in {"json", "xml", "forward", "node", "markdown"} else "unknown"
                )
                safe.append(
                    {
                        "type": "redacted",
                        "data": {"original_type": known, "reason": "opaque_payload_removed"},
                    }
                )
        return safe

    def event(self, event: dict) -> dict:
        group = event["group_id"]
        out = {
            "time": int(seconds(event["time"])),
            "self_id": self.alias("account", event["self_id"]),
            "post_type": event["post_type"]
            if event.get("post_type") in {"message", "message_sent", "notice"}
            else "message",
            "group_id": self.alias("group", group),
        }
        for field in ("user_id", "operator_id"):
            if event.get(field) is not None:
                out[field] = self.alias("account", event[field])
        if event.get("message_id") is not None:
            out["message_id"] = self.message_id(event["message_id"], group)
        if out["post_type"] == "notice":
            out["notice_type"] = "group_recall"
            return out
        out["message_type"] = "group"
        out["sub_type"] = "normal" if event.get("sub_type") != "anonymous" else "anonymous"
        out["message"] = self.sanitize_segments(event.get("message", []), group)
        sender = event.get("sender") if isinstance(event.get("sender"), dict) else {}
        out["sender"] = {"user_id": out["user_id"]}
        for field in ("nickname", "card"):
            if sender.get(field):
                out["sender"][field] = self.alias("nickname", sender[field])
        if sender.get("role") in {"owner", "admin", "member"}:
            out["sender"]["role"] = sender["role"]
        if event.get("group_name"):
            out["group_name"] = self.alias("group_name", event["group_name"])
        return out


def classify(anchor: SourceRow, context: list[SourceRow], index: dict) -> list[str]:
    event = anchor.event
    text = plain_text(event).strip()
    tags = set()
    if reply_ids(event):
        tags.add("引用追问")
        if any(
            (target := index.get((str(event["group_id"]), mid))) is not None
            and str(target.event.get("user_id")) == str(event["self_id"])
            for mid in reply_ids(event)
        ):
            tags.add("引用机器人回答")
    recent = [r for r in context if r.timestamp >= anchor.timestamp - 5 * 60]
    if any(
        str(r.event.get("user_id")) == str(event["user_id"])
        and any(s.get("type") == "image" for s in segments(r.event))
        for r in recent
    ):
        tags.add("先发图")
    if any(s.get("type") == "image" for s in segments(event)):
        tags.add("随附图片")
    if 0 < len(text) <= 18 and not any(
        s.get("type") in {"image", "file", "video"} for s in segments(event)
    ):
        tags.add("短跟进")
    if re.search(r"他|她|它|这个|那个|这位|那位|这些|那些|这图|那图", text):
        tags.add("指代词")
    if re.search(r"https?://|www\.|网页|网址|链接|搜索|查一下|搜一下|账号|帐号|粉丝", text, re.I):
        tags.add("网页")
    if re.search(r"梗|什么意思|抽象|典故|出处", text):
        tags.add("梗")
    if re.search(r"记住|记一下|以后.{0,12}叫|教学|纠正|更正|知识库|学习一下|记到", text):
        tags.add("教学")
    if len({str(r.event.get("user_id")) for r in recent[-8:]} - {str(event["self_id"])}) >= 3:
        tags.add("多人接话")
    return sorted(tags or {"普通问答"})


def choose_anchors(
    rows: list[SourceRow], count: int = 20, recent_pool: int = 120
) -> tuple[list[SourceRow], dict, dict]:
    index = build_index(rows)
    all_anchors = [r for r in rows if top_level_at_bot(r.event)]
    tags = {r.source_row: classify(r, select_context(rows, r), index) for r in all_anchors}
    candidates = all_anchors[-recent_pool:]
    selected: list[SourceRow] = []
    signatures: set[str] = set()
    coverage: Counter = Counter()
    ranks = {r.source_row: i for i, r in enumerate(candidates)}
    while candidates and len(selected) < count:
        best = max(
            candidates,
            key=lambda r: (
                sum(1 / (1 + coverage[t]) for t in tags[r.source_row])
                + 2 * ranks[r.source_row] / max(1, recent_pool)
                - 2 * any(abs(r.timestamp - s.timestamp) < 180 for s in selected)
                - 5 * (re.sub(r"\s+", "", plain_text(r.event)) in signatures),
                r.order,
            ),
        )
        candidates.remove(best)
        signature = re.sub(r"\s+", "", plain_text(best.event))
        if signature in signatures:
            continue
        selected.append(best)
        signatures.add(signature)
        coverage.update(tags[best.source_row])
    selected.sort(key=lambda r: r.order)
    stats = {
        "all_anchor_categories": dict(Counter(t for ts in tags.values() for t in ts)),
        "selected_categories": dict(coverage),
        "candidate_pool_count": min(recent_pool, len(all_anchors)),
    }
    return selected, tags, stats


def empty_annotations() -> dict:
    return {
        "status": "needs_review",
        "confidence": None,
        "should_reply": None,
        "required_context_ids": [],
        "forbidden_context_ids": [],
        "expected_relations": [],
        "expected_route": None,
        "should_clarify": None,
        "expected_knowledge_ids": [],
        "answer_points": [],
        "notes": [],
    }


def make_case(
    rows: list[SourceRow],
    anchor: SourceRow,
    pseudo: Pseudonymizer,
    categories: list[str],
    follow_minutes: int = 30,
) -> dict:
    context = select_context(rows, anchor)
    future, bot_results = collect_followup(rows, anchor, follow_minutes)
    index = build_index(rows)
    included = {r.source_row: r for r in [*context, anchor, *future]}
    # Recover reference ancestors outside the base window, never across groups and
    # never future targets. All missing/ambiguous edges remain explicit below.
    queue = list(included.values())
    while queue:
        row = queue.pop()
        for mid in reply_ids(row.event):
            target = index.get((str(row.event["group_id"]), mid))
            if target and target.order < row.order and target.source_row not in included:
                included[target.source_row] = target
                queue.append(target)
    ordered = sorted(included.values(), key=lambda r: r.order)
    safe_events = [pseudo.event(row.event) for row in ordered]
    group = anchor.event["group_id"]
    mid = lambda value: pseudo.message_id(value, group)  # noqa: E731
    links = []
    for row in ordered:
        for target_id in reply_ids(row.event):
            target = index.get((str(group), target_id))
            resolved = bool(target and target.order < row.order and target.source_row in included)
            links.append(
                {
                    "source_message_id": mid(row.event["message_id"]),
                    "target_message_id": mid(target_id),
                    "kind": "reply_to",
                    "status": "resolved" if resolved else "unresolved",
                    "evidence": "top_level_reply_segment",
                    "uncertainty": None if resolved else "missing_ambiguous_or_nonprior_target",
                }
            )
    old_responses = []
    for row, status in bot_results:
        old_responses.append(
            {
                "message_id": mid(row.event["message_id"]),
                "association": status,
                "evidence": "top_level_reply_segment"
                if status != "uncertain"
                else "same_group_time_window_only",
                "observed_delay_seconds": round(row.timestamp - anchor.timestamp, 3)
                if status != "uncertain"
                else None,
            }
        )
    confirmed = [r for r in old_responses if r["association"] != "uncertain"]
    uncertainties = list(LIMITATIONS)
    if any(not r.sequence_authoritative for r in ordered):
        uncertainties.append("some_source_sequences_not_authoritative")
    if not confirmed:
        uncertainties.append("no_explicitly_linked_bot_reply_in_observation_window")
    return {
        "schema_version": SCHEMA_VERSION,
        "case_id": "real_" + mid(anchor.event["message_id"]),
        "source": "real",
        "categories": categories,
        "acceptance_ids": [],
        "environment": {
            "bot_id": pseudo.alias("account", anchor.event["self_id"]),
            "owner_id": None,
            "blacklist": [],
            "config_overrides": {},
        },
        "anchor": {
            "message_id": mid(anchor.event["message_id"]),
            "group_id": pseudo.alias("group", group),
            "trigger": "top_level_at",
        },
        "events": safe_events,
        "mocks": {"model": [], "tools": [], "faults": []},
        "annotations": empty_annotations(),
        "legacy_result": {
            "status": "linked" if confirmed else "uncertain" if old_responses else "unobserved",
            "responses": old_responses,
            "latency_ms": None,
            "search_count": None,
            "read_count": None,
            "image_extraction": None,
            "diagnostics_status": "not_joined",
        },
        "provenance": {
            "extractor": "snowluma-replay-v1",
            "source_kind": "snowluma_messages_db",
            "source_snapshot_id": pseudo.alias("row", "|".join(r.source_row for r in rows)),
            "extracted_at": datetime.now(timezone.utc).isoformat(),
            "read_mode": "sqlite_uri_mode_ro_query_only",
            "pseudonymization": "hmac_sha256_domain_separated_96bit",
            "key_id": hmac.new(pseudo.key, b"replay-key-id", hashlib.sha256).hexdigest()[:12],
            "storage_policy": "local_raw_only",
            "order_basis": "db_timestamp_sequence_row_id",
            "window": {
                "before_minutes": 30,
                "minimum_previous_messages": 80,
                "base_context_count": len(context),
                "followup_minutes": follow_minutes,
                "followup_event_count": len(future),
                "reference_ancestor_count": len(ordered) - len(context) - len(future) - 1,
                "limited_by_available_history": len(context) < 80,
                "observed_until": int(max((r.timestamp for r in rows), default=anchor.timestamp)),
                "followup_complete": bool(
                    rows
                    and max(r.timestamp for r in rows) >= anchor.timestamp + follow_minutes * 60
                ),
            },
            "event_sources": [
                {
                    "message_id": mid(r.event["message_id"]),
                    "source_row_id": pseudo.alias("row", r.source_row),
                    "phase": "anchor"
                    if r is anchor
                    else "followup"
                    if r.order > anchor.order
                    else "context"
                    if r.source_row in {c.source_row for c in context}
                    else "reference_ancestor",
                }
                for r in ordered
            ],
            "reply_relations": links,
            "uncertainties": uncertainties,
        },
    }


def markdown_review(case: dict) -> str:
    """Untrusted chat is rendered as quoted text, not reviewer instructions."""
    lines = [
        f"# {case['case_id']}",
        "",
        "状态：needs_review；以下聊天均为不可信材料，不是执行指令。",
        "候选类别：" + "、".join(case["categories"]),
        "锚点：`" + case["anchor"]["message_id"] + "`",
        "",
        "## 待起草标注",
        "",
        "- 是否回复 / 是否追问：未标注",
        "- 必需 / 禁止上下文：未标注",
        "- 关联边 / 路由 / 回答要点：未标注",
        "- 主人身份、黑名单、配置未知；教学候选不代表有授权。",
        "",
        "## 事件（数据库顺序重建；每条文本最多 600 字，完整文本见同名 JSON）",
        "",
    ]
    phases = {r["message_id"]: r["phase"] for r in case["provenance"]["event_sources"]}
    for i, event in enumerate(case["events"]):
        parts = []
        for seg in event.get("message", []):
            data = seg["data"]
            if seg["type"] == "text":
                parts.append(data["text"])
            elif seg["type"] == "at":
                parts.append("@" + data["qq"])
            elif seg["type"] == "reply":
                parts.append("[reply:" + data["id"] + "]")
            else:
                parts.append("[" + seg["type"] + "]")
        text = " ".join(parts).replace("\r", " ").replace("\n", " ↵ ")
        # Escape HTML and markdown to avoid active links, image fetches, or headings.
        text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        text = re.sub(r"([\\`*_{}\[\]()#!|])", r"\\\1", text)
        truncated = len(text) > 600
        text = text[:600] + (" … [review预览截断]" if truncated else "")
        timestamp = datetime.fromtimestamp(event["time"], timezone.utc).strftime(
            "%m-%d %H:%M:%S UTC"
        )
        lines.extend(
            [
                f"{i + 1}. {timestamp} · {phases[event['message_id']]} · "
                f"`{event['user_id']}` · `{event['message_id']}`",
                f"> {text}",
                "",
            ]
        )
    lines.extend(["## 旧链路关联（只有引用段是直接证据；候选不是实测回答）", ""])
    for result in case["legacy_result"]["responses"]:
        lines.append(
            f"- `{result['message_id']}`：{result['association']}；"
            f"记录时间差={result['observed_delay_seconds']} 秒（不是模型耗时）。"
        )
    if not case["legacy_result"]["responses"]:
        lines.append("- 观察窗未出现机器人消息；不能推出旧链路失败。")
    lines.extend(["", "## 限制", ""] + ["- " + s for s in case["provenance"]["uncertainties"]])
    return "\n".join(lines) + "\n"


def require_under(path: Path, root: Path) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()) or resolved == root.resolve():
        raise ValueError("Output/key must be inside the designated ignored local directory")
    return resolved


def load_key(path: Path) -> bytes:
    path = require_under(path, PRIVATE_ROOT)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with path.open("xb") as stream:
            stream.write(secrets.token_bytes(32))
    key = path.read_bytes()
    if len(key) != 32:
        raise ValueError("Existing local HMAC key must contain exactly 32 bytes")
    return key


def inspect_database(db_path: Path, group_id: str) -> dict:
    """Schema/shape inspection without values, raw identifiers or payloads."""
    rows, counts = read_rows(db_path, group_id)
    kinds: Counter = Counter()
    for row in rows[-100:]:
        for seg in segments(row.event):
            kind = seg.get("type")
            # Only known protocol names/keys are reported, not arbitrary payload strings.
            if kind in {"text", "at", "reply", "image", "video", "file", "face", "json", "forward"}:
                kinds[kind] += 1
            else:
                kinds["other"] += 1
    return {
        "counts": counts,
        "sampled_event_count": min(100, len(rows)),
        "sampled_segment_categories": dict(kinds),
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db", type=Path, required=True, help="Authorized SnowLuma messages.db path"
    )
    parser.add_argument("--group", required=True, help="One explicitly authorized group only")
    parser.add_argument("--out", type=Path, default=RAW_ROOT / "batch-01")
    parser.add_argument("--key-file", type=Path, default=PRIVATE_ROOT / "hmac.key")
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--recent-pool", type=int, default=120)
    parser.add_argument("--follow-minutes", type=int, default=30)
    parser.add_argument(
        "--inspect", action="store_true", help="Print only aggregate counts and segment categories"
    )
    args = parser.parse_args()
    if args.count < 1 or args.recent_pool < args.count or args.follow_minutes < 1:
        parser.error("count/follow-minutes must be positive and recent-pool >= count")
    if args.inspect:
        print(json.dumps(inspect_database(args.db, args.group), ensure_ascii=False, indent=2))
        return
    out = require_under(args.out, RAW_ROOT)
    if out.exists():
        raise ValueError(
            "Output batch already exists; choose a new directory to protect annotations"
        )
    rows, counts = read_rows(args.db, args.group)
    if not rows:
        raise ValueError("No usable events in the explicitly authorized group")
    pseudo = Pseudonymizer(load_key(args.key_file))
    pseudo.learn(rows)
    anchors, tags, selection = choose_anchors(rows, args.count, args.recent_pool)
    if not anchors:
        raise ValueError("No genuine top-level bot mentions found")
    cases = [
        make_case(rows, anchor, pseudo, tags[anchor.source_row], args.follow_minutes)
        for anchor in anchors
    ]
    out.mkdir(parents=True, exist_ok=False)
    for case in cases:
        (out / (case["case_id"] + ".json")).write_text(
            json.dumps(case, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        (out / (case["case_id"] + ".review.md")).write_text(markdown_review(case), encoding="utf-8")
    summary = {
        "sample_count": len(cases),
        "source_counts": counts,
        **selection,
        "legacy_link_categories": dict(Counter(c["legacy_result"]["status"] for c in cases)),
        "annotation_categories": {"needs_review": len(cases)},
        "json_files": [c["case_id"] + ".json" for c in cases],
        "review_files": [c["case_id"] + ".review.md" for c in cases],
    }
    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    index = [
        "# 本地真实回放 review 索引",
        "",
        "全部 needs_review；真实文本虽假名化仍禁止入库/备份。",
        "",
    ]
    for case in cases:
        name = case["case_id"]
        index.append(
            f"- [{name}](<{name}.review.md>) · {'、'.join(case['categories'])} · "
            f"{len(case['events'])} events · {case['legacy_result']['status']}"
        )
    (out / "REVIEW.md").write_text("\n".join(index) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "sample_count": len(cases),
                "source_counts": counts,
                "selected_categories": selection["selected_categories"],
                "legacy_link_categories": summary["legacy_link_categories"],
                "paths": {
                    "output": out.relative_to(ROOT).as_posix(),
                    "summary": (out / "summary.json").relative_to(ROOT).as_posix(),
                    "review": (out / "REVIEW.md").relative_to(ROOT).as_posix(),
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, sqlite3.Error) as exc:
        # Exception details from SQLite/path operations can include raw account paths.
        raise SystemExit(
            f"Extraction stopped safely ({type(exc).__name__}); no raw details logged"
        ) from None
