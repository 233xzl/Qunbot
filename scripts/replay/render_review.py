"""Render an offline HTML review site for a replay batch and its draft annotations.

Output goes to <batch_dir>/review/: index.html (single-page app shell, index table,
review guide) plus cases/<case_id>.js with pre-escaped conversation HTML. Opening
index.html by double-click works without network access. Images are referenced by
relative path into the shared media directory; nothing is copied.

Only counts and project-relative paths are printed.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import random
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
RAW_ROOT = ROOT / "tests/replay/raw"
DEFAULT_MEDIA_ROOT = RAW_ROOT / "media"

PRIORITY_SEED = 20261107
PRIORITY_RANDOM_COUNT = 10
KEEP_BEFORE_ANCHOR = 8
FOCUS_RADIUS = 2
PREVIEW_CHARS = 60
LOCAL_TZ = timezone(timedelta(hours=8))
REVIEW_FORMAT = "replay-review-1"

# Route labels follow docs/REPLAY-ANNOTATION.md section 3; the schema keeps the
# field a free string, so this list is the allowed set for owner review.
ROUTES = {
    "clarify": "先追问补材料",
    "local_knowledge": "本地知识检索",
    "web": "联网查询",
    "chat": "直接聊天回答",
    "teaching": "主人教学",
}
VERDICTS = {"agree": "同意", "modify": "需修改", "unsure": "无法判断"}

ALIAS_NAMES = list("甲乙丙丁戊己庚辛壬癸子丑寅卯辰巳午未申酉戌亥")
ALIAS_COLORS = [
    "#1f77b4",
    "#2ca02c",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#17becf",
    "#bcbd22",
    "#d62728",
    "#3949ab",
    "#00897b",
    "#6d4c41",
    "#7b1fa2",
]
BOT_ALIAS = "机器人"
BOT_COLOR = "#555"

MEDIA_REASONS = {
    "not_acquired_no_network": "首批未下载图片",
    "fetch_disabled": "构建时未获取图片",
    "synthetic_fixture": "合成样例无图片",
    "fetched_via_onebot_get_image": "已获取",
    "get_image_failed": "获取图片接口失败",
    "no_download_url": "没有下载地址",
    "oversize": "超过 10 MB 上限",
    "download_http_error": "下载失败（HTTP 错误）",
    "download_failed": "下载失败",
    "empty_content": "下载内容为空",
    "not_image": "文件不是图片",
    "missing_file_key": "缺少文件键",
    "non_image_media_not_fetched": "非图片媒体未下载",
    "local_file_missing": "本地文件缺失",
    "invalid_media_path": "图片路径无效",
}
MEDIA_KIND = {"image": "图片", "video": "视频", "record": "语音", "file": "文件"}
REDACTED_KIND = {"forward": "合并转发", "node": "转发节点", "json": "卡片", "xml": "卡片"}
ROLE_LABELS = {"owner": "群主", "admin": "管理员"}
CONFIDENCE = {"high": "高", "medium": "中", "low": "低"}
RELATION_KIND = {
    "reply_to": "引用",
    "answers": "回答",
    "refers_to": "指向",
    "corrects": "纠正",
    "continues": "接续",
}
RELATION_STATUS = {"expected": "期望", "resolved": "已解析", "unresolved": "未解析"}
BOT_RULES = {
    "reply_segment_to_anchor": "机器人消息引用了提问",
    "first_unquoted_bot_message_after_anchor": "提问后第一条未引用其他消息的机器人消息",
}
LINK_STATUS = {"linked": "完整关联", "partial": "部分关联", "unlinked": "未关联"}
MISSING_REASONS = {
    "no_bot_message_in_reply_window": "回复窗口内没有机器人消息",
    "no_matching_diagnostics_request": "没有匹配的诊断请求",
    "anchor_outside_diagnostics_coverage": "提问时间不在诊断记录范围内",
    "no_in_group_provider_call_in_window": "时间段内没有本群模型调用记录",
}
VISION = {"succeeded": "成功", "failed": "失败", "not_attempted": "未尝试"}
FAILURE_FLAGS = {
    "vision_failed": "识图失败",
    "web_blocked": "联网被拦截",
    "web_failed": "网页读取失败",
    "teaching_failed": "教学失败",
    "slow_over_60s": "耗时超过 60 秒",
    "no_finished_row": "没有完成记录",
}
WEEKDAYS = "一二三四五六日"
SAFE_ID = re.compile(r"^[A-Za-z0-9_-]+$")
MEDIA_NAME = re.compile(r"^[0-9a-f]{64}\.(jpg|png|gif|webp|bmp)$")


def esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def fingerprint(annotation: dict | None) -> str:
    payload = json.dumps(annotation, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def is_drafted(annotation: dict | None) -> bool:
    return bool(annotation) and annotation.get("status") in ("draft", "reviewed")


def js_json(value: Any) -> str:
    # Safe both as a JS literal and inside an inline <script> element.
    text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return text.replace("<", "\\u003c").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")


def num(value: Any) -> str:
    return "-" if value is None else str(value)


def local_time(ts: int | float) -> datetime:
    return datetime.fromtimestamp(ts, tz=LOCAL_TZ)


# ---------------------------------------------------------------- loading


def load_cases(batch_dir: Path) -> list[dict]:
    paths = sorted([*batch_dir.glob("real_*.json"), *batch_dir.glob("synthetic-*.json")])
    cases = []
    for path in paths:
        case = json.loads(path.read_text(encoding="utf-8"))
        if not SAFE_ID.match(str(case.get("case_id", ""))):
            raise ValueError("unsafe case id")
        cases.append(case)
    return cases


def load_sidecar(path: Path | None) -> dict | None:
    if path is None or not path.is_file():
        return None
    sidecar = json.loads(path.read_text(encoding="utf-8"))
    if sidecar.get("schema_version") != "annotations-sidecar-1":
        raise ValueError("unsupported annotation sidecar")
    return sidecar


def annotation_for(case: dict, sidecar: dict | None) -> dict | None:
    if sidecar is not None and case["case_id"] in sidecar.get("cases", {}):
        return sidecar["cases"][case["case_id"]]
    return case.get("annotations")


# ---------------------------------------------------------------- analysis


def is_bot_event(event: dict, bot_id: str) -> bool:
    return event.get("post_type") == "message_sent" or event.get("user_id") == bot_id


def assign_aliases(events: list[dict], bot_id: str) -> dict[str, str]:
    """Map user ids to 甲/乙/丙… by first appearance as speaker, then as @-target."""
    order: list[str] = []
    seen = {bot_id}
    for event in events:
        if event.get("post_type") == "notice":
            continue
        uid = event.get("user_id")
        if uid and uid not in seen and not is_bot_event(event, bot_id):
            seen.add(uid)
            order.append(uid)
    for event in events:
        for seg in iter_segments(event.get("message", [])):
            if seg.get("type") == "at":
                uid = seg.get("data", {}).get("qq")
                if uid and uid != "all" and uid not in seen:
                    seen.add(uid)
                    order.append(uid)
    aliases = {bot_id: BOT_ALIAS}
    for index, uid in enumerate(order):
        aliases[uid] = alias_name(index)
    return aliases


def alias_name(index: int) -> str:
    if index < len(ALIAS_NAMES):
        return ALIAS_NAMES[index]
    return f"成员{index + 1}"


def alias_color(alias: str, aliases: dict[str, str]) -> str:
    if alias == BOT_ALIAS:
        return BOT_COLOR
    names = [a for a in aliases.values() if a != BOT_ALIAS]
    try:
        index = names.index(alias)
    except ValueError:
        index = 0
    return ALIAS_COLORS[index % len(ALIAS_COLORS)]


def iter_segments(segments: list[dict]):
    for seg in segments or []:
        yield seg
        if seg.get("type") == "reply":
            yield from iter_segments(seg.get("data", {}).get("message") or [])


def top_reply_target(event: dict) -> str | None:
    for seg in event.get("message", []) or []:
        if seg.get("type") == "reply":
            return seg.get("data", {}).get("id")
    return None


def reply_ancestry(events: list[dict], anchor_id: str) -> list[str]:
    """Follow top-level reply segments from the anchor; returns ancestor ids nearest first."""
    by_id = {e["message_id"]: e for e in events if "message_id" in e}
    chain: list[str] = []
    current = by_id.get(anchor_id)
    seen = {anchor_id}
    while current is not None:
        target = top_reply_target(current)
        if not target or target in seen or target not in by_id:
            break
        seen.add(target)
        chain.append(target)
        current = by_id[target]
    return chain


def visible_indices(
    count: int,
    anchor_index: int,
    focus: set[int],
    keep: set[int],
    before: int = KEEP_BEFORE_ANCHOR,
    radius: int = FOCUS_RADIUS,
) -> set[int]:
    """Indices left expanded: the anchor and `before` messages ahead of it, `keep`
    exactly, and `radius` neighbours around each focus index (marks, ancestors)."""
    visible = set(range(max(0, anchor_index - before), anchor_index + 1))
    visible |= {i for i in keep if 0 <= i < count}
    for i in focus:
        visible |= set(range(max(0, i - radius), min(count, i + radius + 1)))
    return {i for i in visible if 0 <= i < count}


def fold_runs(count: int, visible: set[int]) -> list[tuple[str, int, int]]:
    """Group indices into ("show", i, i) and ("fold", start, end_inclusive) runs."""
    runs: list[tuple[str, int, int]] = []
    i = 0
    while i < count:
        if i in visible:
            runs.append(("show", i, i))
            i += 1
            continue
        start = i
        while i < count and i not in visible:
            i += 1
        runs.append(("fold", start, i - 1))
    return runs


def select_priority(annotations: dict[str, dict | None]) -> dict[str, list[str]]:
    drafted = sorted(cid for cid, ann in annotations.items() if is_drafted(ann))
    low = [cid for cid in drafted if (annotations[cid] or {}).get("confidence") == "low"]
    sample = random.Random(PRIORITY_SEED).sample(drafted, min(PRIORITY_RANDOM_COUNT, len(drafted)))
    return {"low_confidence": low, "random": sorted(sample)}


# ---------------------------------------------------------------- rendering


def resolve_image(data: dict, media_root: Path) -> tuple[str | None, str]:
    """Return (media file name, reason); the name is None when the image cannot be shown."""
    ref = data.get("media_ref") or {}
    reason = ref.get("reason") or data.get("reason") or "unknown"
    if ref.get("status") != "available" or not ref.get("path"):
        return None, reason
    name = str(ref["path"]).rsplit("/", 1)[-1]
    if not MEDIA_NAME.match(name):
        return None, "invalid_media_path"
    if not (media_root / name).is_file():
        return None, "local_file_missing"
    return name, reason


def image_html(data: dict, media_root: Path, href: str) -> str:
    sticker = (data.get("local_metadata") or {}).get("sub_type") == 1
    kind = "表情图" if sticker else "图片"
    name, reason = resolve_image(data, media_root)
    if name is None:
        label = MEDIA_REASONS.get(reason, reason)
        return f'<span class="img-missing">原图不可用（{esc(kind)}：{esc(label)}）</span>'
    ref = data.get("media_ref") or {}
    size = ""
    if ref.get("width") and ref.get("height"):
        size = f"{int(ref['width'])}×{int(ref['height'])}"
    cls = "thumb sticker" if sticker else "thumb"
    return (
        f'<figure class="img"><img class="{cls}" loading="lazy" '
        f'src="{esc(href.rstrip("/") + "/" + name)}" alt="{kind}" title="点击放大 {esc(size)}">'
        + (f"<figcaption>{kind}</figcaption>" if sticker else "")
        + "</figure>"
    )


def image_counts(events: list[dict], media_root: Path) -> dict[str, int]:
    counts = {"available": 0, "missing": 0}
    for event in events:
        for seg in iter_segments(event.get("message", [])):
            if seg.get("type") == "image":
                name, _ = resolve_image(seg.get("data", {}), media_root)
                counts["available" if name else "missing"] += 1
    return counts


class CaseRenderer:
    def __init__(self, case: dict, annotation: dict | None, media_root: Path, media_href: str):
        self.case = case
        self.annotation = annotation if is_drafted(annotation) else None
        self.events = case["events"]
        self.bot_id = case["environment"]["bot_id"]
        self.aliases = assign_aliases(self.events, self.bot_id)
        self.media_root = media_root
        self.media_href = media_href.rstrip("/")
        self.by_id = {e["message_id"]: e for e in self.events if "message_id" in e}
        self.index_of = {e["message_id"]: i for i, e in enumerate(self.events) if "message_id" in e}
        self.anchor_id = case["anchor"]["message_id"]
        self.anchor_index = self.index_of[self.anchor_id]
        self.anchor_user = self.by_id[self.anchor_id].get("user_id")
        self.ancestors = reply_ancestry(self.events, self.anchor_id)
        legacy = case.get("legacy_result") or {}
        bot_reply = legacy.get("bot_reply") or {}
        self.bot_reply_id = bot_reply.get("message_id")
        if not self.bot_reply_id:
            direct = [
                r["message_id"]
                for r in legacy.get("responses") or []
                if r.get("association") == "explicit_reply"
            ]
            self.bot_reply_id = direct[0] if direct else None
        if self.bot_reply_id not in self.by_id:
            self.bot_reply_id = None
        ann = self.annotation or {}
        self.required = set(ann.get("required_context_ids", []))
        self.forbidden = set(ann.get("forbidden_context_ids", []))

    # -- names
    def alias(self, uid: str | None) -> str:
        if uid is None:
            return "未知"
        return self.aliases.get(uid, "未知成员")

    def badge(self, uid: str | None) -> str:
        name = self.alias(uid)
        color = alias_color(name, self.aliases)
        extra = " asker" if uid == self.anchor_user and uid != self.bot_id else ""
        return f'<span class="alias{extra}" style="--c:{color}">{esc(name)}</span>'

    # -- segments
    def preview(self, segments: list[dict], limit: int | None = PREVIEW_CHARS) -> str:
        parts = []
        for seg in segments or []:
            kind = seg.get("type")
            data = seg.get("data", {})
            if kind == "text":
                parts.append(data.get("text", ""))
            elif kind == "at":
                qq = data.get("qq")
                parts.append("@全体成员" if qq == "all" else "@" + self.alias(qq))
            elif kind == "image":
                sub = (data.get("local_metadata") or {}).get("sub_type")
                parts.append("[表情图]" if sub == 1 else "[图片]")
            elif kind in MEDIA_KIND:
                parts.append(f"[{MEDIA_KIND[kind]}]")
            elif kind == "face":
                parts.append("[表情]")
            elif kind == "redacted":
                parts.append(f"[{REDACTED_KIND.get(data.get('original_type'), '内容')}已移除]")
        text = " ".join(" ".join(p.split()) for p in parts if p.strip())
        if limit is not None and len(text) > limit:
            text = text[:limit] + "…"
        return text or "（无文字）"

    def image_html(self, data: dict) -> str:
        return image_html(data, self.media_root, self.media_href)

    def segment_html(self, seg: dict, in_quote: bool = False) -> str:
        kind = seg.get("type")
        data = seg.get("data", {})
        if kind == "text":
            return f'<span class="t">{esc(data.get("text", ""))}</span>'
        if kind == "at":
            qq = data.get("qq")
            if qq == "all":
                return '<span class="at">@全体成员</span>'
            cls = "at at-bot" if qq == self.bot_id else "at"
            return f'<span class="{cls}">@{esc(self.alias(qq))}</span>'
        if kind == "reply":
            return self.quote_html(data, in_quote)
        if kind == "image":
            return self.image_html(data)
        if kind in MEDIA_KIND:
            reason = MEDIA_REASONS.get(data.get("reason"), data.get("reason", ""))
            return f'<span class="chip">[{MEDIA_KIND[kind]}：{esc(reason)}]</span>'
        if kind == "face":
            return '<span class="chip face">[QQ表情]</span>'
        if kind == "redacted":
            what = REDACTED_KIND.get(data.get("original_type"), "不透明内容")
            return f'<span class="chip">[{esc(what)}内容已移除，无法查看]</span>'
        return '<span class="chip">[未知消息段]</span>'

    def quote_html(self, data: dict, in_quote: bool) -> str:
        target = data.get("id")
        nested = data.get("message") or []
        if target in self.by_id:
            event = self.by_id[target]
            cls = "quote ancestor" if target in self.ancestors else "quote"
            who = self.badge(event.get("user_id"))
            text = esc(self.preview(event.get("message", [])))
            return (
                f'<button type="button" class="{cls}" data-jump="{esc(target)}" '
                f'title="点击跳到被引用的消息">↪ 引用 {who}：{text}</button>'
            )
        text = esc(self.preview(nested)) if nested else "原消息不在本样本中"
        return f'<span class="quote missing">↪ 引用（目标不在样本内）：{text}</span>'

    def body_html(self, event: dict) -> str:
        segments = event.get("message", []) or []
        quotes = [s for s in segments if s.get("type") == "reply"]
        rest = [s for s in segments if s.get("type") != "reply"]
        return "".join(self.segment_html(s) for s in quotes) + (
            '<div class="body">' + "".join(self.segment_html(s) for s in rest) + "</div>"
        )

    def message_html(self, index: int) -> str:
        event = self.events[index]
        mid = event.get("message_id", "")
        when = local_time(event.get("time", 0))
        if event.get("post_type") == "notice":
            return (
                f'<div class="sys" id="msg-{esc(mid)}">{when:%H:%M:%S} '
                f"{self.badge(event.get('operator_id'))} 撤回了一条消息</div>"
            )
        uid = event.get("user_id")
        bot = is_bot_event(event, self.bot_id)
        classes = ["msg", "bot" if bot else "user"]
        tags = []
        phase = "before" if index < self.anchor_index else "after"
        if mid == self.anchor_id:
            classes.append("anchor")
            phase = "anchor"
            tags.append('<span class="tag tag-anchor">本次提问</span>')
        if mid in self.ancestors:
            classes.append("ancestor")
            tags.append('<span class="tag tag-anc">提问的引用链</span>')
        if mid == self.bot_reply_id:
            classes.append("legacy-reply")
            tags.append('<span class="tag tag-legacy">旧机器人对本提问的回复</span>')
        if uid == self.anchor_user and not bot:
            classes.append("by-asker")
        role = ROLE_LABELS.get((event.get("sender") or {}).get("role"))
        if role:
            tags.append(f'<span class="tag">{role}</span>')
        draft = ""
        if phase == "before":
            if mid in self.required:
                draft = "required"
            elif mid in self.forbidden:
                draft = "forbidden"
        marker = (
            f'<button type="button" class="mark" data-mark-id="{esc(mid)}" '
            'title="点击切换：必需 → 禁止 → 无"></button>'
            if phase == "before"
            else ""
        )
        if phase == "after" and mid != self.bot_reply_id:
            marker = (
                '<span class="mark-fixed" title="提问之后的消息按规则不能作为上下文">之后</span>'
            )
        return (
            f'<div class="{" ".join(classes)}" id="msg-{esc(mid)}" data-id="{esc(mid)}" '
            f'data-phase="{phase}" data-draft="{draft}">'
            f'<div class="meta">{marker}{self.badge(uid)}'
            f'<span class="time" title="北京时间 {when:%Y-%m-%d %H:%M:%S}">{when:%H:%M:%S}</span>'
            f'{"".join(tags)}<span class="mid">#{esc(mid[-6:])}</span></div>'
            f'<div class="bubble">{self.body_html(event)}</div></div>'
        )

    def conversation_html(self) -> str:
        focus = {self.index_of[m] for m in self.ancestors}
        focus |= {
            self.index_of[m]
            for m in self.required | self.forbidden
            if m in self.index_of and self.index_of[m] < self.anchor_index
        }
        keep = {self.anchor_index}
        if self.bot_reply_id:
            keep.add(self.index_of[self.bot_reply_id])
        visible = visible_indices(len(self.events), self.anchor_index, focus, keep)
        parts = []
        last_date = None
        divider_done = False
        for kind, start, end in fold_runs(len(self.events), visible):
            if start > self.anchor_index and not divider_done:
                parts.append(
                    '<div class="divider">以下是提问之后的消息：仅用于对照旧机器人表现，'
                    "按规则不能作为回答上下文</div>"
                )
                divider_done = True
            first = local_time(self.events[start].get("time", 0))
            if first.date() != last_date:
                last_date = first.date()
                parts.append(
                    f'<div class="date">{first:%Y-%m-%d} 星期{WEEKDAYS[first.weekday()]}</div>'
                )
            if kind == "show":
                parts.append(self.message_html(start))
                continue
            last = local_time(self.events[end].get("time", 0))
            inner = "".join(self.message_html(i) for i in range(start, end + 1))
            count = end - start + 1
            parts.append(
                f'<div class="fold" data-count="{count}"><button type="button" class="fold-btn">'
                f"⋯ 展开 {count} 条未标记消息（{first:%H:%M}–{last:%H:%M}）</button>"
                f'<div class="fold-body" hidden>{inner}</div></div>'
            )
            last_date = last.date()
        return "".join(parts)

    def header_html(self, priority: list[str], reannotate: bool) -> str:
        case = self.case
        anchor = self.by_id[self.anchor_id]
        cats = case.get("categories", [])
        quota = [c for c in cats if c.startswith("配额:")]
        others = [c for c in cats if not c.startswith("配额:")]
        chips = "".join(f'<span class="cat quota">{esc(c[3:])}</span>' for c in quota)
        chips += "".join(f'<span class="cat">{esc(c)}</span>' for c in others)
        ann = self.annotation
        conf = (
            f"草稿把握度：{CONFIDENCE.get(ann.get('confidence'), '未填')}"
            if ann
            else "草稿：待起草"
        )
        flags = "".join(f'<span class="flag">{esc(p)}</span>' for p in priority)
        if reannotate:
            flags += '<span class="flag warn">需重标：草稿写于看不到图片时</span>'
        when = local_time(anchor.get("time", 0))
        return (
            f'<div class="case-title"><code>{esc(case["case_id"])}</code>'
            f'<span class="conf">{esc(conf)}</span>{flags}</div>'
            f'<div class="cats">候选类别：{chips or "无"}</div>'
            f'<div class="question"><div class="q-label">本次提问 · {self.badge(self.anchor_user)}'
            f" · {when:%Y-%m-%d %H:%M:%S}（北京时间）"
            f'<button type="button" class="jump-anchor" data-jump="{esc(self.anchor_id)}">'
            f"在对话中定位</button></div>"
            f'<div class="q-body">{self.body_html(anchor)}</div></div>'
        )

    def relation_html(self, rel: dict) -> str:
        def ref(mid: str) -> str:
            if mid == self.anchor_id:
                return "本次提问"
            event = self.by_id.get(mid)
            if event is None:
                return "样本外消息"
            return (
                f'<button type="button" class="ref" data-jump="{esc(mid)}">'
                f"{self.badge(event.get('user_id'))}：{esc(self.preview(event.get('message', []), 40))}"
                "</button>"
            )

        kind = RELATION_KIND.get(rel.get("kind"), rel.get("kind"))
        status = RELATION_STATUS.get(rel.get("status"), rel.get("status"))
        unc = rel.get("uncertainty")
        return (
            f"<li>{ref(rel.get('source_message_id', ''))} <b>{esc(kind)}</b> "
            f"{ref(rel.get('target_message_id', ''))} <span class=muted>（{esc(status)}）</span>"
            + (f'<div class="muted">依据：{esc(rel.get("evidence", ""))}</div>')
            + (f'<div class="muted">不确定：{esc(unc)}</div>' if unc else "")
            + "</li>"
        )

    def annotation_panel(self) -> str:
        ann = self.annotation
        if not ann:
            return (
                '<section class="panel"><h3>草稿标注</h3><p class="pending">待起草</p>'
                '<p class="muted">此样本还没有草稿。可以只写意见供起草参考，或先跳过。</p></section>'
            )

        def yesno(value):
            return "是" if value is True else "否" if value is False else "未定"

        route = ann.get("expected_route")
        route_text = f"{ROUTES.get(route, '未知路由')}（{route}）" if route else "未填"
        rows = [
            ("是否应回复", yesno(ann.get("should_reply"))),
            ("路由", route_text),
            ("是否先追问", yesno(ann.get("should_clarify"))),
            ("把握度", CONFIDENCE.get(ann.get("confidence"), "未填")),
            ("必需上下文", f"{len(self.required)} 条（对话中绿色标记）"),
            (
                "禁止上下文",
                f"提问前 {sum(1 for m in self.forbidden if self.index_of.get(m, 10**9) < self.anchor_index)} 条"
                "（红色标记）；提问之后的消息按规则全部禁止",
            ),
        ]
        body = "".join(f"<tr><th>{esc(k)}</th><td>{esc(v)}</td></tr>" for k, v in rows)
        rels = ann.get("expected_relations") or []
        rel_html = (
            "<ul class=rel>" + "".join(self.relation_html(r) for r in rels) + "</ul>"
            if rels
            else '<p class="muted">无</p>'
        )
        points = "".join(f"<li>{esc(p)}</li>" for p in ann.get("answer_points") or [])
        notes = "".join(f"<li>{esc(n)}</li>" for n in ann.get("notes") or [])
        knowledge = ann.get("expected_knowledge_ids") or []
        return (
            f'<section class="panel"><h3>草稿标注</h3><table class="kv">{body}</table>'
            f"<h4>期望关联</h4>{rel_html}"
            f"<h4>回答要点</h4>{'<ol>' + points + '</ol>' if points else '<p class=muted>无</p>'}"
            f"<h4>备注与局限</h4>{'<ul>' + notes + '</ul>' if notes else '<p class=muted>无</p>'}"
            + (
                f'<p class="muted">期望知识条目：{esc(", ".join(knowledge))}</p>'
                if knowledge
                else ""
            )
            + "</section>"
        )

    def legacy_panel(self) -> str:
        legacy = self.case.get("legacy_result") or {}
        parts = ['<section class="panel"><h3>旧机器人表现（仅供参考，不是标准答案）</h3>']
        bot_reply = legacy.get("bot_reply")
        if self.bot_reply_id and not bot_reply:
            # v1.0 cases: only the protocol reply edge is known.
            event = self.by_id[self.bot_reply_id]
            parts.append(
                f'<div class="legacy-answer">{self.body_html(event)}</div>'
                '<p class="muted">关联依据：机器人消息引用了提问'
                + ("（v1.0 未关联诊断与耗时）" if "bot_reply" not in legacy else "")
                + f' <button type="button" class="ref" data-jump="{esc(self.bot_reply_id)}">'
                "在对话中定位</button></p>"
            )
        elif self.bot_reply_id:
            event = self.by_id[self.bot_reply_id]
            parts.append(
                f'<div class="legacy-answer">{self.body_html(event)}</div>'
                f'<p class="muted">关联依据：{esc(BOT_RULES.get(bot_reply.get("rule"), bot_reply.get("rule")))}；'
                f"把握度 {esc(CONFIDENCE.get(bot_reply.get('confidence'), '-'))}；"
                f"提问后 {esc(bot_reply.get('delay_seconds'))} 秒"
                + (
                    f"；期间还有 {int(bot_reply.get('intervening_anchor_count'))} 条其他提问"
                    if bot_reply.get("intervening_anchor_count")
                    else ""
                )
                + f' <button type="button" class="ref" data-jump="{esc(self.bot_reply_id)}">在对话中定位</button></p>'
            )
        elif "bot_reply" in legacy:
            parts.append('<p class="muted">回复窗口内没有找到旧机器人的回复。</p>')
        else:
            parts.append('<p class="muted">此批次（v1.0）未关联旧机器人回复与诊断。</p>')
        rows = []
        diag = legacy.get("diagnostics")
        if diag:
            ms = diag.get("latency_ms")
            rows += [
                ("诊断匹配", f"把握度 {CONFIDENCE.get(diag.get('confidence'), '-')}"),
                ("耗时", f"{ms / 1000:.1f} 秒" if isinstance(ms, (int, float)) else "未知"),
                (
                    "搜索 / 读网页",
                    f"{num(diag.get('search_count'))} / {num(diag.get('fetch_count'))} 次",
                ),
                (
                    "联网",
                    f"{diag.get('web_calls', 0)} 次"
                    + ("（被拦截）" if diag.get("web_blocked") else ""),
                ),
                (
                    "识图",
                    VISION.get(diag.get("vision_status"), "-")
                    + ("（使用了兜底）" if diag.get("vision_fallback") else ""),
                ),
                ("本地检索命中", f"{len(diag.get('retrieved_knowledge_ids') or [])} 条"),
                (
                    "诊断记录 token",
                    f"输入 {num(diag.get('diag_input_tokens'))} / "
                    f"输出 {num(diag.get('diag_output_tokens'))}",
                ),
            ]
            flags = [FAILURE_FLAGS.get(f, f) for f in diag.get("failure_flags") or []]
            if flags:
                rows.append(("异常", "、".join(flags)))
            if diag.get("failure_reasons"):
                rows.append(("异常类型", "、".join(diag["failure_reasons"])))
        elif "diagnostics" in legacy:
            rows.append(("诊断", "未匹配到诊断请求"))
        calls = legacy.get("provider_calls")
        if calls:
            rows += [
                (
                    "本群模型调用",
                    f"{calls.get('call_count')} 次，模型耗时 {calls.get('model_seconds')} 秒",
                ),
                (
                    "模型 token",
                    f"输入 {calls.get('token_input_other')}（缓存 {calls.get('token_input_cached')}）"
                    f" / 输出 {calls.get('token_output')}",
                ),
            ]
            if calls.get("models"):
                rows.append(("模型", "、".join(calls["models"])))
            if calls.get("other_group_overlap_count"):
                rows.append(("同时段他群调用", f"{calls['other_group_overlap_count']} 次"))
        link = legacy.get("link")
        if link:
            text = (
                f"{LINK_STATUS.get(link.get('status'), link.get('status'))}，"
                f"把握度 {CONFIDENCE.get(link.get('confidence'), '-')}"
            )
            missing = [MISSING_REASONS.get(r, r) for r in link.get("missing_reasons") or []]
            if missing:
                text += "；缺少：" + "、".join(missing)
            rows.append(("关联总体", text))
        if rows:
            parts.append(
                '<table class="kv">'
                + "".join(f"<tr><th>{esc(k)}</th><td>{esc(v)}</td></tr>" for k, v in rows)
                + "</table>"
            )
        parts.append("</section>")
        return "".join(parts)

    def payload(self, priority: list[str], reannotate: bool) -> dict:
        pre_ids = [e["message_id"] for e in self.events[: self.anchor_index] if "message_id" in e]
        ann = self.annotation or {}
        return {
            "id": self.case["case_id"],
            "anchor_id": self.anchor_id,
            "drafted": self.annotation is not None,
            "draft": {
                "route": ann.get("expected_route"),
                "should_clarify": ann.get("should_clarify"),
                "required": [m for m in pre_ids if m in self.required],
                "forbidden": [m for m in pre_ids if m in self.forbidden],
            },
            "markable": pre_ids,
            "header_html": self.header_html(priority, reannotate),
            "conv_html": self.conversation_html(),
            "panel_html": self.annotation_panel() + self.legacy_panel(),
        }


# ---------------------------------------------------------------- site


def media_href(media_root: Path, review_dir: Path) -> str:
    try:
        rel = os.path.relpath(media_root.resolve(), review_dir.resolve())
    except ValueError:
        return media_root.resolve().as_uri()
    return rel.replace(os.sep, "/")


def display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.name


def render_index(boot: dict, rows: list[dict], batch_label: str, limitations: list[str]) -> str:
    priority = boot["priority"]
    table = []
    for n, row in enumerate(rows, 1):
        flags = "".join(f'<span class="flag">{esc(p)}</span>' for p in row["priority"])
        if row["reannotate"]:
            flags += '<span class="flag warn">需重标</span>'
        table.append(
            f'<tr data-case="{esc(row["id"])}" data-priority="{1 if row["priority"] else 0}">'
            f"<td>{n}</td>"
            f'<td><a href="#case={esc(row["id"])}"><code>{esc(row["id"][-8:])}</code></a></td>'
            f"<td>{esc(row['quota'])}</td>"
            f'<td class="q">{esc(row["question"])}</td>'
            f"<td>{esc(row['draft'])}</td><td>{esc(row['confidence'])}</td><td>{flags}</td>"
            f'<td class="status">未复核</td></tr>'
        )
    lim = "".join(f"<li>{esc(x)}</li>" for x in limitations)
    guide = GUIDE_HTML.replace("{{RESULT_PATH}}", esc(boot["result_path"]))
    pri_list = (
        f"<p class=muted>低把握草稿（{len(priority['low_confidence'])}）："
        f"{esc('、'.join(c[-8:] for c in priority['low_confidence']) or '无')}；"
        f"固定随机抽检（种子 {PRIORITY_SEED}，{len(priority['random'])} 条）："
        f"{esc('、'.join(c[-8:] for c in priority['random']) or '无')}</p>"
    )
    return (
        PAGE_HTML.replace("{{TITLE}}", esc(f"标注复核 · {batch_label}"))
        .replace("{{CSS}}", CSS)
        .replace("{{JS}}", JS)
        .replace("{{BOOT}}", js_json(boot))
        .replace("{{BATCH}}", esc(batch_label))
        .replace("{{GUIDE}}", guide)
        .replace("{{PRIORITY}}", pri_list)
        .replace("{{LIMITATIONS}}", f"<h3>草稿整体局限</h3><ul>{lim}</ul>" if lim else "")
        .replace("{{ROWS}}", "".join(table))
    )


def check_output_location(batch_dir: Path, cases: list[dict]):
    if any(c.get("source") == "real" for c in cases):
        if not batch_dir.resolve().is_relative_to(RAW_ROOT.resolve()):
            raise ValueError("real cases may only be rendered inside tests/replay/raw/")


def render_batch(batch_dir: Path, sidecar_path: Path | None, media_root: Path) -> dict:
    cases = load_cases(batch_dir)
    if not cases:
        raise ValueError("no cases")
    check_output_location(batch_dir, cases)
    sidecar = load_sidecar(sidecar_path)
    review_dir = batch_dir / "review"
    case_dir = review_dir / "cases"
    case_dir.mkdir(parents=True, exist_ok=True)
    annotations = {c["case_id"]: annotation_for(c, sidecar) for c in cases}
    priority = select_priority(annotations)
    reannotate = set((sidecar or {}).get("needs_reannotation_cases") or [])
    href = media_href(media_root, review_dir)
    rows, index_cases = [], []
    stats = {"available": 0, "missing": 0}
    written = set()
    for case in cases:
        cid = case["case_id"]
        ann = annotations[cid]
        flags = []
        if cid in priority["low_confidence"]:
            flags.append("优先：低把握")
        if cid in priority["random"]:
            flags.append("优先：随机抽检")
        renderer = CaseRenderer(case, ann, media_root, href)
        data = renderer.payload(flags, cid in reannotate)
        for key, value in image_counts(case["events"], media_root).items():
            stats[key] += value
        name = f"{cid}.js"
        (case_dir / name).write_text(f"window.__reviewCase({js_json(data)});\n", encoding="utf-8")
        written.add(name)
        drafted = is_drafted(ann)
        quota = next((c[3:] for c in case.get("categories", []) if c.startswith("配额:")), "")
        if not quota:
            quota = "、".join(case.get("categories", [])[:2])
        anchor = renderer.by_id[renderer.anchor_id]
        rows.append(
            {
                "id": cid,
                "quota": quota,
                "question": renderer.preview(anchor.get("message", []), 40),
                "draft": "草稿" if drafted else "待起草",
                "confidence": CONFIDENCE.get((ann or {}).get("confidence"), "-")
                if drafted
                else "-",
                "priority": flags,
                "reannotate": cid in reannotate,
            }
        )
        index_cases.append(
            {
                "id": cid,
                "drafted": drafted,
                "fingerprint": fingerprint(ann),
                "priority": bool(flags),
            }
        )
    for stale in case_dir.glob("*.js"):
        if stale.name not in written and SAFE_ID.match(stale.stem):
            stale.unlink()
    previous_priority = None
    priority_path = review_dir / "priority.json"
    if priority_path.is_file():
        previous_priority = json.loads(priority_path.read_text(encoding="utf-8"))
    priority_record = {"seed": PRIORITY_SEED, **priority}
    priority_path.write_text(
        json.dumps(priority_record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    boot = {
        "format": REVIEW_FORMAT,
        "batch": batch_dir.resolve().name,
        "generated_at": datetime.now(LOCAL_TZ).isoformat(timespec="seconds"),
        "routes": [[k, v] for k, v in ROUTES.items()],
        "verdicts": VERDICTS,
        "cases": index_cases,
        "priority": priority,
        "result_path": display_path(batch_dir / "review-result.json"),
    }
    limitations = (sidecar or {}).get("limitations") or []
    (review_dir / "index.html").write_text(
        render_index(boot, rows, batch_dir.resolve().name, limitations), encoding="utf-8"
    )
    return {
        "cases": len(cases),
        "drafted": sum(1 for a in annotations.values() if is_drafted(a)),
        "priority_low": len(priority["low_confidence"]),
        "priority_random": len(priority["random"]),
        "priority_changed": previous_priority is not None and previous_priority != priority_record,
        "images_available": stats["available"],
        "images_missing": stats["missing"],
        "out": display_path(review_dir / "index.html"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch_dir", type=Path)
    parser.add_argument(
        "--annotations", type=Path, help="default: <batch_dir>/annotations.draft.json"
    )
    parser.add_argument("--media-root", type=Path, default=DEFAULT_MEDIA_ROOT)
    args = parser.parse_args()
    sidecar = args.annotations or args.batch_dir / "annotations.draft.json"
    if args.annotations and not args.annotations.is_file():
        parser.error("annotation sidecar not found")
    try:
        result = render_batch(args.batch_dir, sidecar, args.media_root)
    except (OSError, ValueError, KeyError, TypeError, StopIteration):
        parser.error("cannot render batch (details omitted)")
    print(
        f"cases={result['cases']} drafted={result['drafted']} "
        f"priority_low={result['priority_low']} priority_random={result['priority_random']} "
        f"priority_changed={str(result['priority_changed']).lower()} "
        f"images_available={result['images_available']} images_missing={result['images_missing']}"
    )
    print(f"out={result['out']}")
    return 0


# ---------------------------------------------------------------- static assets

GUIDE_HTML = """
<section class="guide">
<h2>如何复核（每条约 3–5 分钟）</h2>
<ol>
<li><b>先看提问</b>：顶部黄色框是本次 @机器人 的提问，橙色描边的名字是提问者。人名已换成 甲、乙、丙…（每条样本内固定），「机器人」是旧机器人。时间均为北京时间。</li>
<li><b>检查上下文标记</b>：绿色「必需」= 回答时必须看到的历史消息；红色「禁止」= 不应使用的消息；其余未判定。请判断：必需的是否真的需要？有没有漏掉提问引用或指代的图片、前文？有没有把无关的话题当成必需？紫色竖线是提问的引用链。</li>
<li><b>检查右侧草稿</b>：机器人此时是否应先追问补材料；路由（追问 / 本地知识 / 联网 / 聊天 / 教学）是否合适；回答要点与备注是否合理。旧机器人的回答只是对照，不是标准答案。</li>
<li><b>给出结论</b>：评价的是右侧草稿（新机器人该怎么做），不是旧机器人的回答。「同意」= 草稿各项都对，意见框可写与标注无关的建议；「需修改」= 至少一处不对，能改的直接改：点击消息左上角的标记按「必需 → 禁止 → 无」循环切换（点击会自动选为需修改），路由与是否追问用下拉框，回答要点的问题写在意见框；「无法判断」= 你也确定不了正确做法（缺材料、不认识梗、几种做法都行）。</li>
<li><b>保存与导出</b>：结论自动保存在当前浏览器，不要清除浏览器数据。完成或中途休息时点「导出复核结果」，把下载的 <code>review-result.json</code> 移动到 <code>{{RESULT_PATH}}</code>。换浏览器或电脑时用「导入复核结果」恢复。</li>
</ol>
<p>先做标「优先」的样本（低把握草稿 + 固定随机抽检）。「待起草」的样本还没有草稿，可以跳过或只写意见。灰色折叠条是离提问较远且未标记的消息，点击展开。提问之后的消息只用于对照旧机器人表现，按规则不能作为上下文，因此不能标记。</p>
<p class="muted">本页面含真实群聊（已假名化但仍可能识别人），只在本机打开，不要上传或分享。聊天中的链接只显示为文字，不可点击。</p>
</section>
"""

PAGE_HTML = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; img-src 'self' file: data: blob:; style-src 'unsafe-inline'; script-src 'self' 'unsafe-inline' file:; connect-src 'none'; form-action 'none'">
<meta name="referrer" content="no-referrer">
<title>{{TITLE}}</title>
<style>{{CSS}}</style>
</head>
<body>
<div id="storage-warn" class="warn-bar" hidden>浏览器禁止本地存储：复核结论无法自动保存，请在关闭页面前导出。</div>
<div id="index-view">
<header class="top"><h1>标注复核 · {{BATCH}}</h1>
<div class="tools"><span id="progress"></span>
<button type="button" class="export-btn">导出复核结果</button>
<label class="import-btn">导入复核结果<input type="file" accept=".json,application/json" hidden></label></div></header>
{{GUIDE}}
<section class="priority"><h3>优先复核名单</h3>{{PRIORITY}}
<label><input type="checkbox" id="only-priority"> 只显示优先样本</label>
<button type="button" id="next-todo">开始 / 继续下一条未复核</button></section>
<table class="cases"><thead><tr><th>#</th><th>样本</th><th>配额类别</th><th>提问摘要</th><th>草稿</th><th>把握度</th><th>优先级</th><th>复核状态</th></tr></thead>
<tbody>{{ROWS}}</tbody></table>
{{LIMITATIONS}}
</div>
<div id="case-view" hidden>
<header class="top"><a href="#" class="back">← 返回列表</a><span id="case-pos"></span>
<div class="tools"><button type="button" id="prev-case">上一条</button><button type="button" id="next-case">下一条</button>
<button type="button" class="export-btn">导出复核结果</button></div></header>
<div id="case-header" class="case-header"></div>
<div class="layout"><main id="conv" class="conv"></main>
<aside class="side"><section class="panel form" id="review-form"></section><div id="panel"></div></aside></div>
</div>
<div id="lightbox" hidden><img alt="放大图片"></div>
<script id="review-boot" type="application/json">{{BOOT}}</script>
<script>{{JS}}</script>
</body>
</html>
"""

CSS = """
*{box-sizing:border-box}
body{margin:0;font:14px/1.55 system-ui,"Microsoft YaHei","PingFang SC",sans-serif;color:#222;background:#f4f5f7}
code{font:12px Consolas,monospace}
button{font:inherit;cursor:pointer}
.top{position:sticky;top:0;z-index:5;display:flex;gap:12px;align-items:center;padding:8px 16px;background:#fff;border-bottom:1px solid #ddd}
.top h1{font-size:18px;margin:0}
.tools{margin-left:auto;display:flex;gap:8px;align-items:center}
.import-btn,.export-btn,#next-todo{border:1px solid #888;border-radius:4px;padding:3px 10px;background:#fafafa}
.guide,.priority,#index-view>h3,#index-view>ul{max-width:1100px;margin:12px auto;padding:0 16px}
.guide{background:#fffbe6;border:1px solid #f0d97a;border-radius:6px;padding:8px 20px}
.guide h2{font-size:16px;margin:6px 0}
table.cases{max-width:1100px;margin:12px auto;border-collapse:collapse;background:#fff;width:calc(100% - 32px)}
table.cases th,table.cases td{border:1px solid #ddd;padding:4px 6px;text-align:left;vertical-align:top}
table.cases td.q{max-width:360px;word-break:break-all}
tr.hidden-row{display:none}
.status.agree{color:#2e7d32;font-weight:600}.status.modify{color:#c62828;font-weight:600}
.status.unsure{color:#8d6e00;font-weight:600}.status.stale{color:#e65100;font-weight:600}
.flag{display:inline-block;margin:1px 3px;padding:0 6px;border-radius:9px;background:#e3f2fd;color:#0d47a1;font-size:12px}
.flag.warn{background:#fff3e0;color:#e65100}
.muted{color:#777;font-size:12px}
.warn-bar{background:#c62828;color:#fff;padding:6px 16px}
.case-header{padding:8px 16px;background:#fff;border-bottom:1px solid #ddd}
.case-title{display:flex;gap:10px;align-items:center;flex-wrap:wrap}
.conf{font-weight:600}
.cat{display:inline-block;margin:1px 3px;padding:0 6px;border:1px solid #ccc;border-radius:3px;font-size:12px}
.cat.quota{border-color:#1565c0;color:#1565c0;font-weight:600}
.question{margin-top:6px;padding:8px 12px;background:#fff8d6;border:2px solid #f2b705;border-radius:6px}
.q-label{font-size:12px;color:#555;display:flex;gap:6px;align-items:center}
.q-body{font-size:16px}
.jump-anchor,.ref{border:none;background:none;color:#1565c0;padding:0 2px;text-align:left}
.layout{display:flex;gap:12px;padding:12px 16px;align-items:flex-start}
.conv{flex:1 1 60%;min-width:0}
.side{flex:0 0 400px;position:sticky;top:52px;max-height:calc(100vh - 60px);overflow:auto}
.panel{background:#fff;border:1px solid #ddd;border-radius:6px;padding:8px 12px;margin-bottom:10px}
.panel h3{font-size:15px;margin:4px 0 8px}.panel h4{font-size:13px;margin:10px 0 4px}
.panel ol,.panel ul{padding-left:20px;margin:4px 0}
.kv{border-collapse:collapse;width:100%}.kv th{width:96px;text-align:left;color:#555;font-weight:normal;vertical-align:top;padding:2px 4px}.kv td{padding:2px 4px}
.pending{font-size:18px;font-weight:700;color:#e65100}
.legacy-answer{background:#f1f1f1;border-radius:6px;padding:6px 8px;max-height:260px;overflow:auto}
.date{text-align:center;color:#888;font-size:12px;margin:10px 0}
.divider{margin:14px 0;padding:6px;text-align:center;background:#eceff1;color:#455a64;border-radius:4px;font-size:12px}
.sys{text-align:center;color:#888;font-size:12px;margin:6px 0}
.msg{margin:6px 0;max-width:86%}
.msg.bot{margin-left:auto}
.meta{display:flex;gap:6px;align-items:center;font-size:12px;color:#666;flex-wrap:wrap}
.msg.bot .meta{justify-content:flex-end}
.alias{display:inline-block;padding:0 6px;border-radius:3px;color:#fff;background:var(--c);font-weight:600;font-size:12px}
.alias.asker{outline:2px solid #ff9800;outline-offset:1px}
.time{color:#999}.mid{color:#bbb;font-size:11px}
.tag{padding:0 5px;border-radius:3px;background:#eee;color:#555;font-size:11px}
.tag-anchor{background:#f2b705;color:#000;font-weight:700}
.tag-anc{background:#ede7f6;color:#4527a0}
.tag-legacy{background:#e0e0e0;color:#333}
.bubble{display:inline-block;max-width:100%;background:#fff;border:1px solid #ddd;border-radius:8px;padding:6px 10px;word-break:break-word}
.msg.bot .bubble{background:#e8f0fe;border-color:#c6d6f5}
.msg.by-asker .bubble{border-color:#ffb74d}
.msg.anchor .bubble{background:#fff8d6;border:2px solid #f2b705;box-shadow:0 0 0 3px #fff1b8}
.msg.ancestor .bubble{border-left:5px solid #7e57c2}
.msg.legacy-reply .bubble{border:2px dashed #78909c}
.msg[data-phase=after]:not(.legacy-reply){opacity:.65}
.msg.mark-required .bubble{border:2px solid #2e7d32;background:#f1f8e9}
.msg.mark-forbidden .bubble{border:2px solid #c62828;background:#ffebee}
.t{white-space:pre-wrap}
.body{display:inline}
.mark{border:1px solid #aaa;border-radius:3px;background:#fafafa;font-size:11px;padding:0 5px;color:#666}
.mark.required{background:#2e7d32;border-color:#2e7d32;color:#fff;font-weight:700}
.mark.forbidden{background:#c62828;border-color:#c62828;color:#fff;font-weight:700}
.mark.changed::after{content:" *";}
.mark-fixed{font-size:11px;color:#999;border:1px dashed #bbb;border-radius:3px;padding:0 4px}
.draft-was{font-size:11px;color:#e65100}
.at{color:#1565c0}.at-bot{font-weight:600}
.chip{display:inline-block;margin:0 2px;padding:0 5px;border-radius:3px;background:#eceff1;color:#455a64;font-size:12px}
.quote{display:block;width:100%;margin:0 0 4px;padding:3px 8px;border:none;border-left:3px solid #b0bec5;background:#f5f7f8;color:#455a64;text-align:left;font-size:12px;border-radius:3px}
.quote.ancestor{border-left-color:#7e57c2;background:#f3effa}
.quote.missing{cursor:default}
.img{display:inline-block;margin:2px 4px 2px 0;vertical-align:top;text-align:center}
.img figcaption{font-size:10px;color:#888}
.thumb{max-width:220px;max-height:180px;border-radius:4px;border:1px solid #ccc;cursor:zoom-in;display:block}
.thumb.sticker{max-width:90px;max-height:90px}
.img-missing{display:inline-block;padding:4px 8px;border:1px dashed #c62828;border-radius:4px;color:#c62828;font-size:12px;background:#fff5f5}
.fold{margin:6px 0}
.fold-btn{width:100%;border:1px dashed #b0bec5;background:#eceff1;color:#546e7a;border-radius:4px;padding:3px;font-size:12px}
.flash .bubble{animation:flash 1.6s ease-out}
@keyframes flash{0%{box-shadow:0 0 0 6px #42a5f5}100%{box-shadow:none}}
.form label{margin-right:10px}
.form .row{margin:6px 0}
.form textarea{width:100%;min-height:80px;font:inherit}
.form select{font:inherit}
.form .verdicts label{display:inline-block;padding:3px 10px;border:1px solid #bbb;border-radius:4px;margin:2px}
.form .saved{font-size:12px;color:#2e7d32}
.form .stale{color:#e65100;font-weight:600}
#lightbox{position:fixed;inset:0;background:rgba(0,0,0,.85);z-index:20;display:flex;align-items:center;justify-content:center;cursor:zoom-out}
#lightbox[hidden]{display:none}
#lightbox img{max-width:96vw;max-height:96vh;background:#fff}
@media (max-width:900px){.layout{flex-direction:column}.side{position:static;flex:auto;width:100%;max-height:none}}
"""

JS = r"""
(function () {
  'use strict';
  var BOOT = JSON.parse(document.getElementById('review-boot').textContent);
  var BATCH = BOOT.batch;
  var VERDICTS = BOOT.verdicts;
  var ORDER = BOOT.cases.map(function (c) { return c.id; });
  var META = {};
  BOOT.cases.forEach(function (c) { META[c.id] = c; });
  var loaded = {};
  var pending = null;
  var current = null;
  var memoryStore = {};
  var storageOk = true;
  try { localStorage.setItem('replay-review:probe', '1'); localStorage.removeItem('replay-review:probe'); }
  catch (e) { storageOk = false; document.getElementById('storage-warn').hidden = false; }

  function key(id) { return 'replay-review:' + BATCH + ':' + id; }
  function getRec(id) {
    var raw = storageOk ? localStorage.getItem(key(id)) : memoryStore[id];
    if (!raw) return null;
    try { return JSON.parse(raw); } catch (e) { return null; }
  }
  function putRec(id, rec) {
    rec.updated_at = new Date().toISOString();
    var raw = JSON.stringify(rec);
    if (storageOk) localStorage.setItem(key(id), raw); else memoryStore[id] = raw;
  }
  function status(id) {
    var rec = getRec(id);
    if (!rec || !rec.verdict) return ['todo', '未复核'];
    if (rec.draft_fingerprint !== META[id].fingerprint) return ['stale', '草稿已变，需重看'];
    return [rec.verdict, VERDICTS[rec.verdict] || rec.verdict];
  }

  // ---------------------------------------------------------------- index
  function refreshIndex() {
    var done = 0, pdone = 0, ptotal = 0;
    document.querySelectorAll('table.cases tbody tr').forEach(function (tr) {
      var id = tr.getAttribute('data-case');
      var st = status(id);
      var cell = tr.querySelector('.status');
      cell.textContent = st[1];
      cell.className = 'status ' + st[0];
      var isP = tr.getAttribute('data-priority') === '1';
      if (isP) ptotal++;
      if (st[0] !== 'todo' && st[0] !== 'stale') { done++; if (isP) pdone++; }
      tr.classList.toggle('hidden-row', document.getElementById('only-priority').checked && !isP);
    });
    document.getElementById('progress').textContent =
      '已复核 ' + done + ' / ' + ORDER.length + '，优先 ' + pdone + ' / ' + ptotal;
  }
  function nextTodo(after) {
    var onlyP = document.getElementById('only-priority').checked;
    var start = after ? ORDER.indexOf(after) + 1 : 0;
    var list = ORDER.slice(start).concat(ORDER.slice(0, start));
    var pick = function (preferPriority) {
      for (var i = 0; i < list.length; i++) {
        var id = list[i], st = status(id)[0];
        if ((st === 'todo' || st === 'stale') && (!preferPriority || META[id].priority)) return id;
      }
      return null;
    };
    return pick(true) || (onlyP ? null : pick(false));
  }

  // ---------------------------------------------------------------- case view
  function loadCase(id) {
    if (loaded[id]) { renderCase(loaded[id]); return; }
    pending = id;
    var s = document.createElement('script');
    s.src = 'cases/' + encodeURIComponent(id) + '.js';
    s.onerror = function () {
      document.getElementById('conv').textContent = '无法加载样本文件，请重新运行渲染脚本。';
    };
    document.body.appendChild(s);
  }
  window.__reviewCase = function (data) {
    loaded[data.id] = data;
    if (pending === data.id) { pending = null; renderCase(data); }
  };

  function effectiveMarks(data, rec) {
    if (rec && rec.verdict === 'modify' && rec.marks) return rec.marks;
    var m = {};
    data.draft.required.forEach(function (id) { m[id] = 'required'; });
    data.draft.forbidden.forEach(function (id) { m[id] = 'forbidden'; });
    return m;
  }
  function draftMarks(data) { return effectiveMarks(data, null); }

  function applyMarks() {
    var data = current, rec = getRec(data.id);
    var marks = effectiveMarks(data, rec), draft = draftMarks(data);
    document.querySelectorAll('#conv .mark').forEach(function (btn) {
      var id = btn.getAttribute('data-mark-id');
      var m = marks[id] || '', d = draft[id] || '';
      btn.className = 'mark ' + m + (m !== d ? ' changed' : '');
      btn.textContent = m === 'required' ? '必需' : m === 'forbidden' ? '禁止' : '标记';
      btn.title = '点击切换：必需 → 禁止 → 无' + (m !== d ? '（草稿：' + (d === 'required' ? '必需' : d === 'forbidden' ? '禁止' : '无') + '）' : '');
      var msg = btn.closest('.msg');
      msg.classList.toggle('mark-required', m === 'required');
      msg.classList.toggle('mark-forbidden', m === 'forbidden');
    });
    var counts = { required: 0, forbidden: 0 };
    Object.keys(marks).forEach(function (k) { if (counts[marks[k]] !== undefined) counts[marks[k]]++; });
    var el = document.getElementById('mark-count');
    if (el) el.textContent = '当前：必需 ' + counts.required + ' 条，提问前禁止 ' + counts.forbidden + ' 条';
  }

  function ensureRec(data) {
    var rec = getRec(data.id) || {};
    rec.draft_fingerprint = META[data.id].fingerprint;
    return rec;
  }
  function startModify(data, rec) {
    rec.verdict = 'modify';
    if (!rec.marks) rec.marks = draftMarks(data);
    if (rec.route === undefined) rec.route = data.draft.route || null;
    if (rec.should_clarify === undefined) rec.should_clarify = data.draft.should_clarify;
  }

  function el(tag, attrs, text) {
    var e = document.createElement(tag);
    Object.keys(attrs || {}).forEach(function (k) { e.setAttribute(k, attrs[k]); });
    if (text !== undefined) e.textContent = text;
    return e;
  }

  function buildForm(data) {
    var form = document.getElementById('review-form');
    form.textContent = '';
    var rec = getRec(data.id);
    form.appendChild(el('h3', {}, '复核结论'));
    if (rec && rec.verdict && rec.draft_fingerprint !== META[data.id].fingerprint) {
      form.appendChild(el('p', { 'class': 'stale' }, '草稿在你上次复核后已更新，请重新确认结论。'));
    }
    var vr = el('div', { 'class': 'row verdicts' });
    Object.keys(VERDICTS).forEach(function (v) {
      var lab = el('label');
      var r = el('input', { type: 'radio', name: 'verdict', value: v });
      if (rec && rec.verdict === v) r.checked = true;
      if (v === 'agree' && !data.drafted) { r.disabled = true; lab.title = '尚无草稿，无法同意'; }
      lab.appendChild(r); lab.appendChild(document.createTextNode(' ' + VERDICTS[v]));
      vr.appendChild(lab);
    });
    form.appendChild(vr);
    var mod = el('div', { 'class': 'modify' });
    var rr = el('div', { 'class': 'row' });
    rr.appendChild(el('span', {}, '路由：'));
    var sel = el('select', { id: 'route-select' });
    sel.appendChild(el('option', { value: '' }, '（未选）'));
    BOOT.routes.forEach(function (r) { sel.appendChild(el('option', { value: r[0] }, r[1] + '（' + r[0] + '）')); });
    rr.appendChild(sel); mod.appendChild(rr);
    var cr = el('div', { 'class': 'row' });
    cr.appendChild(el('span', {}, '是否先追问：'));
    [['true', '是'], ['false', '否']].forEach(function (p) {
      var lab = el('label'); var r = el('input', { type: 'radio', name: 'clarify', value: p[0] });
      lab.appendChild(r); lab.appendChild(document.createTextNode(' ' + p[1])); cr.appendChild(lab);
    });
    mod.appendChild(cr);
    var mr = el('div', { 'class': 'row' });
    mr.appendChild(el('div', { 'class': 'muted' }, '上下文：点击对话中消息左上角的标记按钮，按 必需 → 禁止 → 无 切换；带 * 表示与草稿不同。'));
    mr.appendChild(el('div', { id: 'mark-count' }));
    var reset = el('button', { type: 'button' }, '恢复草稿标记');
    mr.appendChild(reset); mod.appendChild(mr);
    form.appendChild(mod);
    form.appendChild(el('div', { 'class': 'row' }, '意见（可选，需修改时请写原因）：'));
    var ta = el('textarea', { id: 'comment' });
    ta.value = (rec && rec.comment) || '';
    form.appendChild(ta);
    var saved = el('div', { 'class': 'saved' });
    form.appendChild(saved);
    var nav = el('div', { 'class': 'row' });
    var nt = el('button', { type: 'button' }, '下一条未复核 →');
    nav.appendChild(nt); form.appendChild(nav);

    function sync() {
      var r = getRec(data.id) || {};
      var isMod = r.verdict === 'modify';
      mod.style.display = isMod ? '' : 'none';
      sel.value = isMod && r.route ? r.route : '';
      form.querySelectorAll('input[name=clarify]').forEach(function (x) {
        x.checked = isMod && String(r.should_clarify) === x.value;
      });
      saved.textContent = r.updated_at ? '已自动保存到本机浏览器（' + new Date(r.updated_at).toLocaleString() + '）' : '';
      applyMarks();
      refreshIndex();
    }
    function update(fn) { var r = ensureRec(data); fn(r); putRec(data.id, r); sync(); }
    vr.addEventListener('change', function (e) {
      update(function (r) {
        if (e.target.value === 'modify') startModify(data, r);
        else { r.verdict = e.target.value; }
      });
    });
    sel.addEventListener('change', function () { update(function (r) { r.route = sel.value || null; }); });
    cr.addEventListener('change', function (e) { update(function (r) { r.should_clarify = e.target.value === 'true'; }); });
    ta.addEventListener('input', function () { update(function (r) { r.comment = ta.value; }); });
    reset.addEventListener('click', function () { update(function (r) { r.marks = draftMarks(data); }); });
    nt.addEventListener('click', function () { var n = nextTodo(data.id); location.hash = n ? '#case=' + n : '#'; });
    sync();
    current.cycleMark = function (id) {
      update(function (r) {
        if (r.verdict !== 'modify') {
          startModify(data, r);
          form.querySelector('input[name=verdict][value=modify]').checked = true;
        }
        var m = r.marks[id] || '';
        var next = m === '' ? 'required' : m === 'required' ? 'forbidden' : '';
        if (next) r.marks[id] = next; else delete r.marks[id];
      });
    };
  }

  function renderCase(data) {
    current = data;
    document.getElementById('case-header').innerHTML = data.header_html;
    document.getElementById('conv').innerHTML = data.conv_html;
    document.getElementById('panel').innerHTML = data.panel_html;
    var i = ORDER.indexOf(data.id);
    document.getElementById('case-pos').textContent = '第 ' + (i + 1) + ' / ' + ORDER.length + ' 条 · ' + status(data.id)[1];
    buildForm(data);
    var a = document.getElementById('msg-' + data.anchor_id);
    if (a) a.scrollIntoView({ block: 'center' });
  }

  function jump(id) {
    var target = document.getElementById('msg-' + id);
    if (!target) return;
    var fold = target.closest('.fold-body');
    if (fold) { fold.hidden = false; fold.previousElementSibling.hidden = true; }
    target.scrollIntoView({ behavior: 'smooth', block: 'center' });
    target.classList.remove('flash'); void target.offsetWidth; target.classList.add('flash');
  }

  // ---------------------------------------------------------------- export / import
  function exportAll() {
    var out = { format: BOOT.format, batch: BATCH, rendered_at: BOOT.generated_at,
      exported_at: new Date().toISOString(), cases: {} };
    ORDER.forEach(function (id) {
      var r = getRec(id);
      if (!r || !r.verdict) return;
      var e = { verdict: r.verdict, comment: r.comment || '', draft_fingerprint: r.draft_fingerprint,
        updated_at: r.updated_at };
      if (r.verdict === 'modify') {
        var marks = r.marks || {};
        e.route = r.route || null;
        e.should_clarify = typeof r.should_clarify === 'boolean' ? r.should_clarify : null;
        e.required_context_ids = Object.keys(marks).filter(function (k) { return marks[k] === 'required'; });
        e.forbidden_context_ids = Object.keys(marks).filter(function (k) { return marks[k] === 'forbidden'; });
      }
      out.cases[id] = e;
    });
    var blob = new Blob([JSON.stringify(out, null, 2)], { type: 'application/json' });
    var a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'review-result.json';
    document.body.appendChild(a); a.click(); a.remove();
    setTimeout(function () { URL.revokeObjectURL(a.href); }, 5000);
    alert('已导出 ' + Object.keys(out.cases).length + ' 条结论。请把下载的 review-result.json 移动到 ' + BOOT.result_path);
  }
  function importFile(file) {
    var reader = new FileReader();
    reader.onload = function () {
      var data;
      try { data = JSON.parse(reader.result); } catch (e) { alert('文件不是有效的 JSON。'); return; }
      if (!data || data.format !== BOOT.format || data.batch !== BATCH || typeof data.cases !== 'object') {
        alert('文件格式或批次不匹配（需要批次 ' + BATCH + '）。'); return;
      }
      var ids = Object.keys(data.cases).filter(function (id) { return META[id]; });
      if (!confirm('将导入 ' + ids.length + ' 条结论，覆盖本浏览器中相同样本的结论。继续？')) return;
      ids.forEach(function (id) {
        var e = data.cases[id];
        if (!e || !VERDICTS[e.verdict]) return;
        var rec = { verdict: e.verdict, comment: String(e.comment || ''), draft_fingerprint: e.draft_fingerprint };
        if (e.verdict === 'modify') {
          rec.route = e.route || null;
          rec.should_clarify = typeof e.should_clarify === 'boolean' ? e.should_clarify : null;
          rec.marks = {};
          (e.required_context_ids || []).forEach(function (m) { rec.marks[m] = 'required'; });
          (e.forbidden_context_ids || []).forEach(function (m) { rec.marks[m] = 'forbidden'; });
        }
        putRec(id, rec);
      });
      route();
    };
    reader.readAsText(file);
  }

  // ---------------------------------------------------------------- routing / events
  function route() {
    var m = /^#case=([A-Za-z0-9_-]+)$/.exec(location.hash);
    var showCase = m && META[m[1]];
    document.getElementById('index-view').hidden = !!showCase;
    document.getElementById('case-view').hidden = !showCase;
    if (showCase) loadCase(m[1]); else { current = null; refreshIndex(); }
  }
  window.addEventListener('hashchange', route);
  document.addEventListener('click', function (e) {
    var t = e.target;
    var img = t.closest && t.closest('img.thumb');
    if (img) { var lb = document.getElementById('lightbox'); lb.querySelector('img').src = img.src; lb.hidden = false; return; }
    if (t.closest && t.closest('#lightbox')) { document.getElementById('lightbox').hidden = true; return; }
    var j = t.closest && t.closest('[data-jump]');
    if (j) { e.preventDefault(); jump(j.getAttribute('data-jump')); return; }
    var fb = t.closest && t.closest('.fold-btn');
    if (fb) { fb.hidden = true; fb.nextElementSibling.hidden = false; return; }
    var mk = t.closest && t.closest('button.mark');
    if (mk && current && current.cycleMark) { current.cycleMark(mk.getAttribute('data-mark-id')); return; }
    if (t.closest && t.closest('.export-btn')) { exportAll(); return; }
    if (t.id === 'next-todo') { var n = nextTodo(null); if (n) location.hash = '#case=' + n; else alert('没有未复核的样本了。'); return; }
    if (t.id === 'prev-case' || t.id === 'next-case') {
      var i = ORDER.indexOf(current ? current.id : '');
      var k = t.id === 'prev-case' ? i - 1 : i + 1;
      if (k >= 0 && k < ORDER.length) location.hash = '#case=' + ORDER[k];
    }
  });
  document.addEventListener('keydown', function (e) { if (e.key === 'Escape') document.getElementById('lightbox').hidden = true; });
  document.querySelector('.import-btn input').addEventListener('change', function (e) {
    if (e.target.files[0]) importFile(e.target.files[0]);
    e.target.value = '';
  });
  document.getElementById('only-priority').addEventListener('change', refreshIndex);
  route();
})();
"""


if __name__ == "__main__":
    raise SystemExit(main())
