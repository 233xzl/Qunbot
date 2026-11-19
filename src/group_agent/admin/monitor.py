"""Read-only monitor projections shared by the overview and request archive."""

from datetime import datetime
import json
from math import ceil
from typing import Literal

from ..usage import CN_TZ

Period = Literal["today", "yesterday", "24h", "7d", "14d", "30d"]
Dimension = Literal["total", "model", "group"]


def window(period: Period, now: float):
    midnight = (
        datetime.fromtimestamp(now, CN_TZ)
        .replace(hour=0, minute=0, second=0, microsecond=0)
        .timestamp()
    )
    if period == "24h":
        return now - 86400, now, 3600
    if period == "yesterday":
        return midnight - 86400, midnight, 3600
    if period == "today":
        return midnight, now, 3600
    return midnight - (int(period[:-1]) - 1) * 86400, now, 86400


def tables(con):
    return {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}


def ledger(con):
    if any(r[1] == "group_id" for r in con.execute("PRAGMA table_info(usage_ledger)")):
        return "usage_ledger"
    return "(SELECT u.*,r.group_id FROM usage_ledger u LEFT JOIN runs r USING(run_id))"


def names(con, bot_id):
    if "group_names" not in tables(con):
        return {}
    return dict(con.execute("SELECT group_id,name FROM group_names WHERE bot_id=?", (bot_id,)))


def group_title(gid, directory):
    return "未归属" if gid is None else directory.get(gid, "未知群聊")


# Older ledger rows predate service tags. Only classify a recognised task.
SERVICE = """CASE WHEN service='search' OR task IN ('web_search','search') THEN '搜索'
WHEN service='fetch' OR task IN ('web_fetch','fetch','bilibili_video_info') THEN '获取'
WHEN service IN ('generate','judge') OR task IN ('answer','decision','fallback','judge','generate')
THEN '对话' ELSE '其他' END"""


def usage(con, bot_id, period, dimension, gid, now):
    start, end, step = window(period, now)
    count = max(1, ceil((end - start) / step))
    labels = [
        datetime.fromtimestamp(start + i * step, CN_TZ).strftime(
            "%m-%d %H:%M" if step == 3600 else "%Y-%m-%d"
        )
        for i in range(count)
    ]
    result = dict(
        today_cost=0,
        cost=0,
        tokens=0,
        requests=0,
        avg_seconds=None,
        series=[],
        providers=[],
        days=[dict(day=d, cost=0) for d in labels],
    )
    if con is None:
        return result
    result["today_cost"] = con.execute(
        "SELECT COALESCE(SUM(cost_cny),0) FROM usage_ledger WHERE created_at>=? AND created_at<?",
        window("today", now)[:2],
    ).fetchone()[0]
    where = " WHERE created_at>=? AND created_at<?"
    params = [start, end]
    if gid is not None:
        where += " AND group_id IS ?"
        params.append(gid or None)
    result["requests"], result["avg_seconds"] = con.execute(
        "SELECT COUNT(*),AVG(CASE WHEN ended_at>=created_at THEN ended_at-created_at END) FROM runs"
        + where,
        params,
    ).fetchone()
    directory = names(con, bot_id) if dimension == "group" else {}
    rows = con.execute(
        f"""SELECT CAST((created_at-?)/? AS INTEGER) bucket, provider,model,
        {SERVICE} category,group_id,SUM(cost_cny) cost,SUM(input_tokens+output_tokens) tokens,
        COUNT(*) calls FROM {ledger(con)} {where} GROUP BY bucket,provider,model,category,group_id""",
        [start, step, *params],
    )
    series, providers = {}, {}
    for bucket, provider, model, category, group, cost, tokens, calls in rows:
        key, label = "total", "全部消费"
        if dimension == "model":
            key, label = (provider, model, category), f"{model or provider} · {category}"
        elif dimension == "group":
            key, label = group, group_title(group, directory)
        if key not in series:
            series[key] = dict(label=label, values=[0] * count, cost=0, calls=0, tokens=0)
        item = series[key]
        item["values"][bucket] += cost
        item["cost"] += cost
        item["calls"] += calls
        item["tokens"] += tokens
        p = providers.setdefault(
            (provider, model), dict(provider=provider, model=model, cost=0, calls=0, tokens=0)
        )
        p["calls"] += calls
        p["tokens"] += tokens
        p["cost"] += cost
        result["cost"] += cost
        result["tokens"] += tokens
        result["days"][bucket]["cost"] += cost
    result["series"] = sorted(series.values(), key=lambda s: -s["cost"])
    result["providers"] = list(providers.values())
    return result


SEGMENT_NAMES = {
    "image": "图片",
    "face": "表情",
    "record": "语音",
    "video": "视频",
    "file": "文件",
    "json": "卡片",
    "forward": "转发消息",
}


def message_text(segments, revoked, bot_id):
    if revoked:
        return "消息已撤回"
    if segments is None:
        return "消息已清理"
    parts = []
    for segment in json.loads(segments):
        kind, data = segment["type"], segment["data"]
        if kind == "text":
            parts.append(str(data.get("text", "")))
        elif kind == "at":
            if str(data.get("qq")) != bot_id:
                parts.append("@全体成员" if data.get("qq") == "all" else "@群成员")
        elif kind != "reply":
            parts.append("[" + SEGMENT_NAMES.get(kind, "消息附件") + "]")
    return "".join(parts).strip() or "[无文字消息]"


def requests(con, *, offset=0, limit=50, state="", gid=None, bounds=None, run_id=None):
    if con is None:
        return []
    filters, params = [], []
    for column, value in [("group_id", gid), ("state", state or None), ("run_id", run_id)]:
        if value is not None:
            filters.append(f"{column}=?")
            params.append(value)
    if bounds:
        filters += ["created_at>=?", "created_at<?"]
        params.extend(bounds[:2])
    where = " WHERE " + " AND ".join(filters) if filters else ""
    schema = tables(con)
    joins = ""
    name, member = "NULL", "NULL"
    if "group_names" in schema:
        joins += " LEFT JOIN group_names n ON n.bot_id=r.bot_id AND n.group_id=r.group_id"
        name = "n.name"
    if "request_members" in schema:
        joins += " LEFT JOIN request_members a ON a.seq=r.message_seq"
        member = "a.name"
    # Page the runs before reading message bodies or summing their indexed ledger rows.
    fields = (
        "r.*"
        if run_id
        else "r.run_id,r.state,r.bot_id,r.group_id,r.route,r.created_at,r.started_at,r.ended_at,r.error_kind"
    )
    replies = "o.kind reply_kind,o.text reply_text,o.state delivery," if run_id else ""
    if run_id:
        joins += " LEFT JOIN outbound o ON o.run_id=r.run_id"
    cursor = con.execute(
        f"""WITH page AS (SELECT * FROM runs {where} ORDER BY created_at DESC,run_id DESC LIMIT ? OFFSET ?)
        SELECT {fields},{replies} m.segments,m.revoked,{member} member_name,{name} group_name,
        (SELECT sum(cost_cny) FROM usage_ledger u WHERE u.run_id=r.run_id) cost
        FROM page r LEFT JOIN messages m ON m.seq=r.message_seq {joins} ORDER BY r.created_at DESC,r.run_id DESC""",
        [*params, limit, offset],
    )
    result = []
    for row in cursor:
        data = dict(row)
        text = message_text(data.pop("segments"), data.pop("revoked"), data["bot_id"])
        data["group_name"] = data["group_name"] or "未知群聊"
        if data["cost"] is None and data["state"] in ("rejected", "queued", "running"):
            data["cost"] = 0
        if run_id:
            data["message"] = text
            data["usage"] = [
                dict(r) for r in con.execute("SELECT * FROM usage_ledger WHERE run_id=?", (run_id,))
            ]
        else:
            data["preview"] = " ".join(text.split())[:128]
        result.append(data)
    return result
