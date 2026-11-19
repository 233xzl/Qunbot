"""Build a stratified local replay batch with images and old-pipeline links (T1.2/T1.3/T1.5).

Read-only sources: SnowLuma messages.db (one authorized group), SnowLuma OneBot
HTTP read actions (`get_image` only, for refreshed image URLs), AstrBot
diagnostics.jsonl and data_v4.db provider_stats. Output stays under the
git-ignored tests/replay/raw/. Console output is aggregate counts only.

Run --help for usage; real paths/IDs are passed as arguments, never hard-coded.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from collections import Counter
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import extract_snowluma as base  # noqa: E402
import legacy_link as legacy  # noqa: E402
import replay_media as media  # noqa: E402

SCHEMA_VERSION = "1.1"
EXTRACTOR = "snowluma-replay-v2-batch"
QUOTA_PREFIX = "配额:"
OVERFLOW = "超额补充"
FAILURE_TAGS = (
    "旧链路图片失败",
    "旧链路超时",
    "旧链路搜索拦截",
    "旧链路网页失败",
    "旧链路教学失败",
)
NO_MEDIA_TAGS = {"先发图", "随附图片", "引用图片", "引用追问"}
WEB_TAGS = {"网页", "外部事实词", "卡片分享", "旧链路账号查询", "旧链路网页读取"}
# Broader keyword lists than the batch-01 classifier; still candidates, not labels.
EXTERNAL_FACT_PATTERN = re.compile(
    r"https?://|www\.|网页|网址|链接|搜索|查一下|搜一下|查查|搜搜|账号|帐号|粉丝|是谁|谁是|官网|"
    r"[bB]站|[uU][pP]主|微博|抖音|推特|油管|主页|最新|新闻|直播|BV\d|av\d"
)
MEME_PATTERN = re.compile(r"梗|啥意思|什么意思|是啥|是什么|啥梗|解释|含义|出处|出自|典故|来源|抽象")


def _quota_rules() -> list[tuple[str, int, Callable[[set[str]], bool]]]:
    return [
        ("引用机器人回答的追问", 6, lambda t: "引用机器人回答" in t),
        ("先发图后@", 6, lambda t: bool({"先发图", "引用图片"} & t)),
        ("多人接话/话题交叉/同名改名", 6, lambda t: bool({"多人接话", "同名或改名"} & t)),
        ("无材料简短追问", 5, lambda t: "无材料短问" in t),
        ("指代不明", 3, lambda t: "指代词" in t and not (NO_MEDIA_TAGS & t)),
        ("网页/URL/账号查询", 6, lambda t: bool(WEB_TAGS & t)),
        (
            "本地梗解释",
            8,
            lambda t: (
                bool({"梗", "梗解释词"} & t)
                or ("旧链路仅本地检索" in t and not ((NO_MEDIA_TAGS | {"短跟进"}) & t))
            ),
        ),
        ("主人教学", 3, lambda t: bool({"教学", "旧链路教学"} & t)),
        ("旧链路失败或超时", 6, lambda t: bool(set(FAILURE_TAGS) & t)),
    ]


QUOTA_RULES = _quota_rules()
QUOTAS = {name: quota for name, quota, _ in QUOTA_RULES}

LIMITATIONS_REMOVED = {
    "no_diagnostics_or_provider_stats_joined_metrics_unknown",
    "images_not_downloaded_or_opened_no_content_hash",
}
LIMITATIONS_ADDED = [
    "legacy_diagnostics_joined_by_clock_offset_corrected_time_window_cross_group_ambiguity",
    "images_fetched_via_onebot_get_image_not_visually_reviewed",
    "non_image_media_and_opaque_forwards_still_unavailable",
]


# ---------------------------------------------------------------------------
# Candidate tags


def has_image(segments: Any) -> bool:
    if not isinstance(segments, list):
        return False
    for seg in segments:
        if not isinstance(seg, dict):
            continue
        if seg.get("type") == "image":
            return True
        data = seg.get("data") if isinstance(seg.get("data"), dict) else {}
        if seg.get("type") == "reply" and has_image(data.get("message")):
            return True
    return False


def display_name(event: dict) -> str | None:
    sender = event.get("sender") if isinstance(event.get("sender"), dict) else {}
    for key in ("card", "nickname"):
        value = sender.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def extra_tags(anchor: Any, context: list[Any], index: dict, base_tags: set[str]) -> set[str]:
    event = anchor.event
    group = str(event["group_id"])
    tags = set()
    targets = [index.get((group, mid)) for mid in base.reply_ids(event)]
    if any(t is not None and has_image(base.segments(t.event)) for t in targets) or any(
        s.get("type") == "reply" and has_image((s.get("data") or {}).get("message"))
        for s in base.segments(event)
    ):
        tags.add("引用图片")
    if any(s.get("type") in {"json", "xml", "share"} for s in base.segments(event)) or any(
        t is not None
        and any(s.get("type") in {"json", "xml", "share"} for s in base.segments(t.event))
        for t in targets
    ):
        tags.add("卡片分享")
    window = [*context, anchor]
    recent_users = {str(r.event.get("user_id")) for r in window[-20:]}
    names_by_user: dict[str, set[str]] = {}
    users_by_name: dict[str, set[str]] = {}
    for row in window:
        user = str(row.event.get("user_id"))
        name = display_name(row.event)
        if name is None or user == str(event["self_id"]):
            continue
        names_by_user.setdefault(user, set()).add(name)
        users_by_name.setdefault(name, set()).add(user)
    renamed = any(len(v) > 1 for u, v in names_by_user.items() if u in recent_users)
    same_name = any(len(v & recent_users) > 1 for v in users_by_name.values())
    if renamed or same_name:
        tags.add("同名或改名")
    if "短跟进" in base_tags and not ({"引用追问", "先发图", "随附图片"} & base_tags):
        tags.add("无材料短问")
    text = base.plain_text(event)
    if EXTERNAL_FACT_PATTERN.search(text):
        tags.add("外部事实词")
    if MEME_PATTERN.search(text):
        tags.add("梗解释词")
    return tags


def legacy_tags(summary: dict | None, has_bot_reply: bool) -> set[str]:
    tags = set() if has_bot_reply else {"旧链路未回复"}
    if not summary:
        return tags
    flags = set(summary["failure_flags"])
    mapping = {
        "vision_failed": "旧链路图片失败",
        "slow_over_60s": "旧链路超时",
        "no_finished_row": "旧链路超时",
        "web_blocked": "旧链路搜索拦截",
        "web_failed": "旧链路网页失败",
        "teaching_failed": "旧链路教学失败",
    }
    tags.update(label for flag, label in mapping.items() if flag in flags)
    if summary["teaching"]:
        tags.add("旧链路教学")
    if summary["account_lookup"]:
        tags.add("旧链路账号查询")
    if summary["web_calls"]:
        tags.add("旧链路联网")
    if summary["url_fetch_calls"]:
        tags.add("旧链路网页读取")
    if summary["local_only"]:
        tags.add("旧链路仅本地检索")
    return tags


def quota_categories(tags: set[str]) -> list[str]:
    return [name for name, _, rule in QUOTA_RULES if rule(tags)]


# ---------------------------------------------------------------------------
# Stratified selection


def select_with_quotas(
    anchors: list[Any],
    tags: dict[str, set[str]],
    keep: set[str],
    quotas: dict[str, int],
    target: int,
    signature: Callable[[Any], str],
    bonus: Callable[[Any], float] = lambda a: 0.0,
    spacing_seconds: float = 180,
) -> tuple[list[Any], dict[str, str], dict]:
    """Keep required anchors, then fill each quota from the scarcest category first.

    Each selected anchor counts toward exactly one category. Returns the selected
    anchors (time order), {source_row: category}, and per-category stats.
    """
    eligible = {
        a.source_row: [c for c in quota_categories(tags[a.source_row]) if c in quotas]
        for a in anchors
    }
    by_row = {a.source_row: a for a in anchors}
    missing = keep - set(by_row)
    if missing:
        raise ValueError(f"{len(missing)} required anchors are not in the source snapshot")
    filled: Counter = Counter()
    assignment: dict[str, str] = {}
    selected: list[Any] = []
    signatures: set[str] = set()
    ranks = {a.source_row: i / max(1, len(anchors) - 1) for i, a in enumerate(anchors)}

    def take(anchor: Any, category: str) -> None:
        selected.append(anchor)
        assignment[anchor.source_row] = category
        if category in quotas:
            filled[category] += 1
        if signature(anchor):
            signatures.add(signature(anchor))

    pool_sizes = {c: sum(c in eligible[a.source_row] for a in anchors) for c in quotas}
    kept = sorted(
        (by_row[k] for k in keep), key=lambda a: (len(eligible[a.source_row]) or 99, a.order)
    )
    for anchor in kept:
        options = eligible[anchor.source_row]
        open_options = [c for c in options if filled[c] < quotas[c]]
        if open_options:
            # Spend a kept anchor on the category with the fewest real candidates.
            category = min(open_options, key=lambda c: (pool_sizes[c], options.index(c)))
        else:
            category = options[0] if options else OVERFLOW
        take(anchor, category)

    def score(anchor: Any, category: str | None) -> tuple:
        row = anchor.source_row
        crowded = any(abs(anchor.timestamp - s.timestamp) < spacing_seconds for s in selected)
        covered = Counter(t for s in selected for t in tags[s.source_row])
        novelty = sum(1 for t in tags[row] if not covered[t])
        failure_novelty = 0
        if category == "旧链路失败或超时":
            seen = {
                t
                for s in selected
                if assignment.get(s.source_row) == category
                for t in tags[s.source_row]
            }
            failure_novelty = sum(1 for t in FAILURE_TAGS if t in tags[row] and t not in seen)
        value = (
            bonus(anchor)
            + 0.5 * ranks[row]
            + 0.2 * novelty
            + 1.5 * failure_novelty
            - 3.0 * crowded
            - 0.2 * max(0, len(eligible[row]) - 1)
        )
        return value, anchor.order

    def available(category: str | None) -> list[Any]:
        chosen = {s.source_row for s in selected}
        return [
            a
            for a in anchors
            if a.source_row not in chosen
            and (category is None and eligible[a.source_row] or category in eligible[a.source_row])
            and not (signature(a) and signature(a) in signatures)
        ]

    for category in sorted(quotas, key=lambda c: (pool_sizes[c], list(quotas).index(c))):
        while filled[category] < quotas[category]:
            options = available(category)
            if not options:
                break
            take(max(options, key=lambda a: score(a, category)), category)
    while len(selected) < target:
        options = available(None)
        if not options:
            break
        take(max(options, key=lambda a: score(a, None)), OVERFLOW)
    selected.sort(key=lambda a: a.order)
    stats = {
        category: {
            "quota": quotas[category],
            "selected": filled[category],
            "shortfall": max(0, quotas[category] - filled[category]),
            "eligible_candidates": pool_sizes[category],
        }
        for category in quotas
    }
    stats[OVERFLOW] = {"selected": sum(v == OVERFLOW for v in assignment.values())}
    return selected, assignment, stats


# ---------------------------------------------------------------------------
# Case enrichment


def walk_media(segments: list[dict]):
    for seg in segments:
        if seg.get("type") in {"image", "video", "record", "file"}:
            yield seg
        elif seg.get("type") == "reply" and isinstance(seg["data"].get("message"), list):
            yield from walk_media(seg["data"]["message"])


def raw_image_keys(rows: list[Any], pseudo: base.Pseudonymizer) -> dict[str, Any]:
    """media_id -> raw get_image file key (kept in memory only)."""
    keys: dict[str, Any] = {}

    def visit(segments: Any) -> None:
        if not isinstance(segments, list):
            return
        for seg in segments:
            if not isinstance(seg, dict) or not isinstance(seg.get("data"), dict):
                continue
            if seg.get("type") == "image":
                keys.setdefault(pseudo.media_alias(seg["data"]), seg["data"].get("file"))
            elif seg.get("type") == "reply":
                visit(seg["data"].get("message"))

    for row in rows:
        visit(base.segments(row.event))
    return keys


def apply_media(case: dict, refs: dict[str, dict]) -> None:
    for event in case["events"]:
        for seg in walk_media(event.get("message", [])):
            data = seg["data"]
            if seg["type"] != "image":
                data["reason"] = "non_image_media_not_fetched"
                continue
            ref = refs.get(data["media_id"]) or media.unavailable("fetch_disabled")
            data["media_ref"] = dict(ref)
            data["availability"] = ref["status"]
            data["reason"] = ref["reason"]
            data["content_sha256"] = ref["sha256"]


def fill_legacy(
    case: dict,
    anchor: Any,
    pseudo: base.Pseudonymizer,
    bot: dict | None,
    diag: dict | None,
    offset: dict,
    calls: list[legacy.ProviderCall],
    diagnostics_range: tuple[float, float] | None,
) -> None:
    group = anchor.event["group_id"]
    result = case["legacy_result"]
    reply_time = bot["row"].timestamp if bot else None
    result["bot_reply"] = (
        {
            "message_id": pseudo.message_id(bot["row"].event["message_id"], group),
            "rule": bot["rule"],
            "confidence": bot["confidence"],
            "delay_seconds": round(bot["row"].timestamp - anchor.timestamp, 3),
            "window_seconds": legacy.REPLY_WINDOW_SECONDS,
            "intervening_anchor_count": bot["intervening_anchor_count"],
        }
        if bot
        else None
    )
    if diag:
        request = diag["request"]
        summary = legacy.summarize_request(request)
        provider = legacy.provider_calls_between(
            calls, request.start, request.end, "diagnostics_request_span"
        )
        confidence = legacy.adjust_diag_confidence(diag["confidence"], provider)
        result["diagnostics"] = {
            "request_ref": pseudo.alias("opaque", "diagnostics-request:" + request.request_id),
            "rule": diag["rule"],
            "confidence": confidence,
            "candidate_count": diag["candidate_count"],
            "competing_anchor_count": diag["competing_anchor_count"],
            "clock_offset_seconds": diag["clock_offset_seconds"],
            "clock_offset_spread_seconds": offset.get("spread"),
            **summary,
        }
        result["latency_ms"] = summary["latency_ms"]
        result["search_count"] = summary["search_count"]
        result["read_count"] = summary["fetch_count"]
        result["image_extraction"] = legacy.image_extraction(summary)
        result["diagnostics_status"] = "joined"
    else:
        shift = offset.get("offset", 0.0)
        end = (reply_time or anchor.timestamp + 120) + shift
        provider = legacy.provider_calls_between(
            calls, anchor.timestamp + shift - 2, end, "anchor_to_reply_offset_corrected"
        )
        confidence = None
        result["diagnostics"] = None
        result["diagnostics_status"] = "unmatched"
    result["provider_calls"] = provider
    covered = bool(
        diagnostics_range
        and diagnostics_range[0] - 600 <= anchor.timestamp <= diagnostics_range[1] + 600
    )
    result["link"] = legacy.combine_link(bot, confidence, provider, covered)


def relevant_message_ids(case: dict, annotation: dict) -> set[str]:
    anchor = case["anchor"]["message_id"]
    ids = {anchor, *annotation.get("required_context_ids", [])}
    ids.update(
        r["target_message_id"]
        for r in case["provenance"]["reply_relations"]
        if r["source_message_id"] == anchor and r["status"] == "resolved"
    )
    return ids


def reannotation_status(case: dict, annotation: dict) -> str | None:
    """Clarify drafts whose decisive images were unavailable and are now fetched."""
    if annotation.get("expected_route") != "clarify":
        return None
    ids = relevant_message_ids(case, annotation)
    images = [
        seg["data"]
        for event in case["events"]
        if event["message_id"] in ids
        for seg in walk_media(event.get("message", []))
        if seg["type"] == "image"
    ]
    if not images:
        return None
    available = sum(i["availability"] == "available" for i in images)
    if available == len(images):
        return "images_now_available"
    return "images_partially_available" if available else None


def finalize_case(case: dict, categories: list[str]) -> None:
    case["schema_version"] = SCHEMA_VERSION
    case["categories"] = categories
    case["provenance"]["extractor"] = EXTRACTOR
    kept = [u for u in case["provenance"]["uncertainties"] if u not in LIMITATIONS_REMOVED]
    case["provenance"]["uncertainties"] = kept + [u for u in LIMITATIONS_ADDED if u not in kept]


def review_markdown(case: dict) -> str:
    """Base review plus T1.3 and media sections; no URLs or active images."""
    text = base.markdown_review(case)
    result = case["legacy_result"]
    lines = ["", "## 旧链路结果（T1.3 关联；时间窗匹配附把握度）", ""]
    link = result["link"]
    lines.append(
        f"- 关联：{link['status']}；把握度={link['confidence']}；来源={'、'.join(link['sources'])}"
    )
    if link["missing_reasons"]:
        lines.append("- 缺失：" + "、".join(link["missing_reasons"]))
    bot = result["bot_reply"]
    if bot:
        lines.append(
            f"- 机器人回复 `{bot['message_id']}`：规则={bot['rule']}；"
            f"记录时间差={bot['delay_seconds']} 秒"
        )
    diag = result["diagnostics"]
    if diag:
        lines.append(
            f"- 诊断：耗时={diag['latency_ms']} ms；搜索={diag['search_count']}；"
            f"读取={diag['fetch_count']}；图片提取={diag['vision_status']}；"
            f"失败标记={'、'.join(diag['failure_flags']) or '无'}；时钟偏移≈{diag['clock_offset_seconds']} 秒"
        )
    provider = result["provider_calls"]
    lines.append(
        f"- 模型调用：{provider['call_count']} 次；输入={provider['token_input_other']}"
        f"+缓存{provider['token_input_cached']}；输出={provider['token_output']}"
    )
    lines.extend(["", "## 图片（内容地址，相对 tests/replay/raw/）", ""])
    for event in case["events"]:
        for seg in walk_media(event.get("message", [])):
            ref = seg["data"].get("media_ref")
            if not ref:
                continue
            where = f"`{ref['path']}`" if ref["path"] else ref["reason"]
            lines.append(f"- `{event['message_id']}`：{ref['status']} · {where}")
    return text + "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI


def load_sidecar(path: Path) -> dict:
    sidecar = json.loads(path.read_text(encoding="utf-8"))
    if sidecar.get("schema_version") != "annotations-sidecar-1":
        raise ValueError("unsupported annotation sidecar")
    return sidecar


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True, help="Authorized SnowLuma messages.db")
    parser.add_argument("--group", required=True, help="One explicitly authorized group only")
    parser.add_argument("--out", type=Path, required=True, help="New dir under tests/replay/raw/")
    parser.add_argument("--carry-from", type=Path, required=True, help="Previous batch dir")
    parser.add_argument("--diagnostics", type=Path, required=True, help="AstrBot diagnostics.jsonl")
    parser.add_argument("--astrbot-db", type=Path, required=True, help="AstrBot data_v4.db")
    parser.add_argument("--onebot-config", type=Path, help="SnowLuma onebot_<account>.json")
    parser.add_argument("--no-media", action="store_true", help="Skip get_image/downloads")
    parser.add_argument("--media-root", type=Path, default=base.RAW_ROOT / "media")
    parser.add_argument("--key-file", type=Path, default=base.PRIVATE_ROOT / "hmac.key")
    parser.add_argument("--target", type=int, default=50)
    parser.add_argument("--rate", type=float, default=5.0, help="get_image requests/second")
    parser.add_argument("--max-image-bytes", type=int, default=media.MAX_IMAGE_BYTES)
    parser.add_argument("--follow-minutes", type=int, default=30)
    return parser


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.target < 1 or not 0 < args.rate <= 5:
        parser.error("target must be positive and rate within (0, 5]")
    if not args.no_media and not args.onebot_config:
        parser.error("--onebot-config is required unless --no-media")
    out = base.require_under(args.out, base.RAW_ROOT)
    media_root = base.require_under(args.media_root, base.RAW_ROOT)
    carry = base.require_under(args.carry_from, base.RAW_ROOT)
    if out.exists():
        raise ValueError("Output batch already exists; choose a new directory")
    previous = load_sidecar(carry / "annotations.draft.json")

    rows, source_counts = base.read_rows(args.db, args.group)
    if not rows:
        raise ValueError("No usable events in the explicitly authorized group")
    pseudo = base.Pseudonymizer(base.load_key(args.key_file))
    pseudo.learn(rows)
    index = base.build_index(rows)
    anchors = [r for r in rows if base.top_level_at_bot(r.event)]
    case_id = {
        a.source_row: "real_" + pseudo.message_id(a.event["message_id"], a.event["group_id"])
        for a in anchors
    }
    by_case = {v: k for k, v in case_id.items()}
    keep_ids = set(previous["cases"])
    if not keep_ids <= set(by_case):
        raise ValueError("Some carried-over case ids are not anchors in this snapshot")

    # Old-pipeline links for every anchor (needed for failure-category candidates).
    bot_links = legacy.link_bot_replies(rows, anchors)
    requests, diag_counts = legacy.load_diagnostics(args.diagnostics)
    calls, provider_counts = legacy.load_provider_stats(args.astrbot_db, args.group)
    windows = [
        legacy.AnchorWindow(
            a.source_row,
            a.timestamp,
            bot_links[a.source_row]["row"].timestamp if a.source_row in bot_links else None,
        )
        for a in anchors
    ]
    offsets = legacy.estimate_offsets(windows, requests)
    diag_links = legacy.link_diagnostics(windows, requests, offsets)
    diag_range = (
        (min(r.start for r in requests), max(r.end for r in requests)) if requests else None
    )

    tags: dict[str, set[str]] = {}
    for anchor in anchors:
        context = base.select_context(rows, anchor)
        found = set(base.classify(anchor, context, index))
        found |= extra_tags(anchor, context, index, found)
        diag = diag_links.get(anchor.source_row)
        found |= legacy_tags(
            legacy.summarize_request(diag["request"]) if diag else None,
            anchor.source_row in bot_links,
        )
        if len(found) > 1:
            found.discard("普通问答")
        tags[anchor.source_row] = found

    def signature(anchor: Any) -> str:
        # Same text with a different quote/image structure is a different case.
        text = re.sub(r"\s+", "", base.plain_text(anchor.event))
        if not text:
            return ""
        shape = "".join(
            sorted({str(s.get("type")) for s in base.segments(anchor.event)} - {"text", "at"})
        )
        return text + "|" + shape

    def bonus(anchor: Any) -> float:
        return 1.0 * (anchor.source_row in bot_links) + 1.0 * (anchor.source_row in diag_links)

    selected, assignment, quota_stats = select_with_quotas(
        anchors,
        tags,
        {by_case[c] for c in keep_ids},
        QUOTAS,
        args.target,
        signature,
        bonus,
    )

    cases = []
    for anchor in selected:
        labels = [QUOTA_PREFIX + assignment[anchor.source_row], *sorted(tags[anchor.source_row])]
        case = base.make_case(rows, anchor, pseudo, labels, args.follow_minutes)
        if case["case_id"] != case_id[anchor.source_row]:
            raise ValueError("case id derivation mismatch")
        finalize_case(case, labels)
        fill_legacy(
            case,
            anchor,
            pseudo,
            bot_links.get(anchor.source_row),
            diag_links.get(anchor.source_row),
            offsets.get(anchor.source_row, {}),
            calls,
            diag_range,
        )
        cases.append(case)

    # Images: fetch each referenced media id once (cache shared across batches).
    keys = raw_image_keys(rows, pseudo)
    wanted = sorted(
        {
            seg["data"]["media_id"]
            for case in cases
            for event in case["events"]
            for seg in walk_media(event.get("message", []))
            if seg["type"] == "image"
        }
    )
    client = None
    if not args.no_media:
        client = media.OneBotReadClient.from_config(
            args.onebot_config, limiter=media.RateLimiter(args.rate)
        )
    fetcher = media.MediaFetcher(
        client,
        media_root,
        max_bytes=args.max_image_bytes,
        download_limiter=media.RateLimiter(args.rate),
    )
    refs = {}
    for i, media_id in enumerate(wanted, 1):
        refs[media_id] = fetcher.fetch(media_id, keys.get(media_id))
        if i % 25 == 0:
            fetcher.save_index()
    fetcher.save_index()
    for case in cases:
        apply_media(case, refs)

    # Annotation sidecar: carried drafts unchanged; new cases left for annotation.
    new_ids = sorted(c["case_id"] for c in cases if c["case_id"] not in keep_ids)
    sidecar = {k: v for k, v in previous.items() if k != "cases"}
    sidecar["carried_over_from"] = carry.name
    sidecar["cases"] = {
        c["case_id"]: previous["cases"].get(c["case_id"]) or base.empty_annotations()
        for c in sorted(cases, key=lambda c: c["case_id"])
    }
    reannotate = {}
    for case in cases:
        if case["case_id"] in keep_ids:
            status = reannotation_status(case, previous["cases"][case["case_id"]])
            if status:
                reannotate[case["case_id"]] = status
    sidecar["needs_annotation_cases"] = new_ids
    sidecar["needs_reannotation_cases"] = sorted(
        k for k, v in reannotate.items() if v == "images_now_available"
    )

    out.mkdir(parents=True, exist_ok=False)
    for case in cases:
        name = case["case_id"]
        (out / f"{name}.json").write_text(
            json.dumps(case, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        (out / f"{name}.review.md").write_text(review_markdown(case), encoding="utf-8")
    (out / "annotations.draft.json").write_text(
        json.dumps(sidecar, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    summary = build_summary(
        cases,
        anchors,
        tags,
        assignment,
        quota_stats,
        bot_links,
        diag_links,
        refs,
        fetcher,
        reannotate,
        keep_ids,
        new_ids,
        source_counts,
        diag_counts,
        provider_counts,
    )
    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_index(out, cases)
    print(
        json.dumps(
            {
                "sample_count": summary["sample_count"],
                "quota": {k: v.get("selected") for k, v in quota_stats.items()},
                "media": summary["media"]["unique_images"],
                "legacy_link_rate": summary["legacy"]["selected"]["link_rate"],
                "output": out.relative_to(base.ROOT).as_posix(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def build_summary(
    cases,
    anchors,
    tags,
    assignment,
    quota_stats,
    bot_links,
    diag_links,
    refs,
    fetcher,
    reannotate,
    keep_ids,
    new_ids,
    source_counts,
    diag_counts,
    provider_counts,
) -> dict:
    segment_status: Counter = Counter()
    for case in cases:
        for event in case["events"]:
            for seg in walk_media(event.get("message", [])):
                if seg["type"] == "image":
                    segment_status[seg["data"]["availability"]] += 1
    unique_status = Counter(r["status"] for r in refs.values())
    reasons = Counter(r["reason"] for r in refs.values() if r["status"] != "available")
    files = {r["sha256"]: r["bytes"] for r in refs.values() if r["status"] == "available"}
    selected_links = Counter(c["legacy_result"]["link"]["status"] for c in cases)
    all_status = Counter()
    for anchor in anchors:
        has_bot = anchor.source_row in bot_links
        has_diag = anchor.source_row in diag_links
        all_status[
            "linked" if has_bot and has_diag else "partial" if has_bot or has_diag else "unlinked"
        ] += 1

    def rate(counter: Counter, total: int) -> float:
        return round(counter["linked"] / total, 4) if total else 0.0

    return {
        "generator": "scripts/replay/build_batch.py",
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sample_count": len(cases),
        "carried_over_count": len(keep_ids),
        "new_case_count": len(new_ids),
        "source_counts": source_counts,
        "quota": quota_stats,
        "quota_assignment": {
            c["case_id"]: c["categories"][0].removeprefix(QUOTA_PREFIX) for c in cases
        },
        "selected_tag_counts": dict(
            sorted(Counter(t for c in cases for t in c["categories"][1:]).items())
        ),
        "all_anchor_tag_counts": dict(
            sorted(Counter(t for ts in tags.values() for t in ts).items())
        ),
        "all_anchor_count": len(anchors),
        "media": {
            "image_segments": dict(segment_status),
            "unique_images": dict(unique_status),
            "unavailable_reasons": dict(reasons),
            "unique_content_files": len(files),
            "total_bytes": sum(files.values()),
            "total_mb": round(sum(files.values()) / 1024 / 1024, 2),
            "fetch_stats": dict(sorted(fetcher.stats.items())),
        },
        "legacy": {
            "selected": {
                "status_counts": dict(selected_links),
                "link_rate": rate(selected_links, len(cases)),
                "bot_reply_rules": dict(
                    Counter(
                        (c["legacy_result"]["bot_reply"] or {}).get("rule", "none") for c in cases
                    )
                ),
                "diagnostics_confidence": dict(
                    Counter(
                        (c["legacy_result"]["diagnostics"] or {}).get("confidence", "unmatched")
                        for c in cases
                    )
                ),
                "provider_matched": sum(
                    c["legacy_result"]["provider_calls"]["call_count"] > 0 for c in cases
                ),
                "not_fully_linked": [
                    {
                        "case_id": c["case_id"],
                        "status": c["legacy_result"]["link"]["status"],
                        "reasons": c["legacy_result"]["link"]["missing_reasons"],
                    }
                    for c in cases
                    if c["legacy_result"]["link"]["status"] != "linked"
                ],
            },
            "all_anchors": {
                "status_counts": dict(all_status),
                "link_rate": rate(all_status, len(anchors)),
                "bot_reply_found": len(bot_links),
                "diagnostics_matched": len(diag_links),
            },
            "diagnostics_source_counts": diag_counts,
            "provider_stats_source_counts": provider_counts,
            "definition": "linked = bot reply identified AND diagnostics request matched",
        },
        "needs_reannotation": sorted(
            k for k, v in reannotate.items() if v == "images_now_available"
        ),
        "images_partially_available_clarify": sorted(
            k for k, v in reannotate.items() if v == "images_partially_available"
        ),
        "reannotation_rule": (
            "carried draft with expected_route=clarify whose anchor, resolved anchor reply "
            "target or required_context_ids contain images, all of which are now available"
        ),
        "needs_annotation": new_ids,
        "json_files": [c["case_id"] + ".json" for c in cases],
        "review_files": [c["case_id"] + ".review.md" for c in cases],
    }


def write_index(out: Path, cases: list[dict]) -> None:
    lines = [
        "# 本地真实回放 review 索引（batch-02）",
        "",
        "真实文本虽假名化仍禁止入库/备份；图片位于 tests/replay/raw/media/。",
        "",
    ]
    for case in cases:
        name = case["case_id"]
        link = case["legacy_result"]["link"]
        lines.append(
            f"- [{name}](<{name}.review.md>) · {case['categories'][0]} · "
            f"{len(case['events'])} events · 旧链路 {link['status']}/{link['confidence']}"
        )
    (out / "REVIEW.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, sqlite3.Error, KeyError) as exc:
        raise SystemExit(
            f"Batch build stopped safely ({type(exc).__name__}); no raw details logged"
        ) from None
