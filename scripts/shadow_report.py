"""Shadow trial report: the old bot's (AstrBot) real replies next to the new framework's
recorded answers, one item per reply run of the `--serve` store (D-19).

Both databases are opened read-only (`mode=ro`, `PRAGMA temp_store=MEMORY`); nothing is
written to them. Output goes to an ignored directory (default `data/serve/report/`):

- `index.html`: per request the question, the quoted message, the old reply, the new
  answer and metrics, with an owner rating (新更好 / 差不多 / 旧更好 / 都不好 + note) kept in
  the browser's localStorage and exportable as JSON. Every text is HTML-escaped.
- `summary.json`, `summary.md`: aggregate numbers only (no content, no ids).

Pairing of the old reply (same rules as `scripts/replay/legacy_link.py`): a bot message in
the group within `--window` seconds after the question whose reply segment targets it;
otherwise the first unquoted bot message after it not claimed by another request. Bot
messages starting with `〔新版〕` (or recorded as `new_version` sends) are the new
framework's own sends and are never counted as old replies.

Usage:
    python scripts/shadow_report.py --serve-db data/serve/shadow-gen.sqlite3 \
        [--snowluma-db /path/to/snowluma/data/<account>/messages.db] [--out data/serve/report] \
        [--since 2026-11-09] [--until 2026-11-10T12:00] [--import-ratings ratings.json]

Without `--snowluma-db` the path is `<--snowluma-root>/<bot id of the serve store>/messages.db`.
Generating again overwrites the same files (idempotent for the same inputs).
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import html
import json
from pathlib import Path
import sqlite3
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent / "replay"))
from legacy_link import REPLY_WINDOW_SECONDS, link_bot_replies  # noqa: E402

CN_TZ = timezone(timedelta(hours=8))
NEW_PREFIX = "〔新版〕"
RATINGS = {
    "new_better": "新更好",
    "same": "差不多",
    "old_better": "旧更好",
    "both_bad": "都不好",
}


def connect_ro(path: Path) -> sqlite3.Connection:
    con = sqlite3.connect(f"file:{Path(path).as_posix()}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA temp_store=MEMORY")  # sorting must not need a temp file
    return con


def parse_time(value: str | None) -> float | None:
    """Epoch seconds, or an ISO date / datetime (naive values are Beijing time)."""
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        pass
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=CN_TZ)
    return parsed.timestamp()


def fmt_time(ts: float | None) -> str:
    if ts is None:
        return "-"
    return datetime.fromtimestamp(ts, CN_TZ).strftime("%Y-%m-%d %H:%M:%S")


def label(group_id: str) -> str:
    return "g" + hashlib.sha256(str(group_id).encode()).hexdigest()[:6]


def render_segments(segments: Any, bot_id: str) -> str:
    """Readable text of OneBot segments; ids are not shown (mentions become placeholders)."""
    if not isinstance(segments, list):
        return ""
    out = []
    for seg in segments:
        if not isinstance(seg, dict):
            continue
        kind, data = seg.get("type"), seg.get("data") or {}
        if kind == "text":
            out.append(str(data.get("text") or ""))
        elif kind == "at":
            out.append("@机器人 " if str(data.get("qq")) == bot_id else "@某人 ")
        elif kind == "image":
            out.append("[图片]")
        elif kind == "face":
            out.append("[表情]")
        elif kind == "reply":
            continue
        else:
            out.append(f"[{kind}]")
    return "".join(out).strip()


def reply_ids(segments: Any) -> list[str]:
    if not isinstance(segments, list):
        return []
    return [
        str(s["data"]["id"])
        for s in segments
        if isinstance(s, dict)
        and s.get("type") == "reply"
        and isinstance(s.get("data"), dict)
        and s["data"].get("id") is not None
    ]


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))
    return round(ordered[index], 1)


def stats(values: list[float]) -> dict:
    return {"n": len(values), "p50": percentile(values, 0.5), "p95": percentile(values, 0.95)}


def _columns(con: sqlite3.Connection, table: str) -> set[str]:
    return {r[1] for r in con.execute(f"PRAGMA table_info({table})")}


# -- old side -----------------------------------------------------------------------


@dataclass
class Row:
    """Shape expected by `legacy_link.link_bot_replies`."""

    event: dict
    order: tuple
    timestamp: float
    source_row: str


def _key(group_id: Any, message_id: Any) -> str:
    return f"{group_id}:{message_id}"


def load_snowluma(path: Path, groups: set[str], since: float, until: float) -> dict[str, Row]:
    if not Path(path).is_file():
        raise FileNotFoundError("SnowLuma message store not found")
    con = connect_ro(path)
    try:
        rows = con.execute(
            "SELECT sequence, timestamp, data FROM messages WHERE is_group=1"
            " AND timestamp BETWEEN ? AND ?",
            (int(since), int(until) + 1),
        ).fetchall()
    finally:
        con.close()
    out: dict[str, Row] = {}
    for row in rows:
        try:
            event = json.loads(row["data"])
        except (TypeError, ValueError):
            continue
        if not isinstance(event, dict) or str(event.get("group_id")) not in groups:
            continue
        t = event.get("time") if isinstance(event.get("time"), int) else row["timestamp"]
        key = _key(event.get("group_id"), event.get("message_id"))
        out[key] = Row(event, (t, row["sequence"] or 0), float(t), key)
    return out


def _is_request(event: dict, bot_id: str) -> bool:
    if str(event.get("user_id")) == bot_id:
        return False
    return any(
        isinstance(s, dict)
        and s.get("type") == "at"
        and str((s.get("data") or {}).get("qq")) == bot_id
        for s in event.get("message") or []
    )


# -- report -------------------------------------------------------------------------------


def build(
    serve_db: Path,
    snowluma_db: Path | None,
    *,
    snowluma_root: Path,
    since: float | None,
    until: float | None,
    window: int = REPLY_WINDOW_SECONDS,
) -> tuple[list[dict], dict]:
    serve = connect_ro(serve_db)
    try:
        bot_row = serve.execute("SELECT bot_id FROM messages LIMIT 1").fetchone()
        bot_id = str(bot_row[0]) if bot_row else ""
        purpose = "purpose" in _columns(serve, "runs")
        where = "WHERE r.purpose='reply'" if purpose else ""
        runs = serve.execute(
            "SELECT r.*, m.segments AS segments, m.revoked AS revoked,"
            " COALESCE(m.platform_time, m.received_at, r.created_at) AS asked_at,"
            " o.text AS answer, o.state AS outbound_state, o.recorded_at AS recorded_at"
            " FROM runs r LEFT JOIN messages m ON m.seq=r.message_seq"
            " LEFT JOIN outbound o ON o.run_id=r.run_id AND o.kind='answer'"
            f" {where} ORDER BY asked_at, r.run_id"
        ).fetchall()
        runs = [
            dict(r)
            for r in runs
            if (since is None or (r["asked_at"] or 0) >= since)
            and (until is None or (r["asked_at"] or 0) < until)
        ]
        cost = {
            r[0]: r[1]
            for r in serve.execute(
                "SELECT run_id, SUM(cost_cny) FROM usage_ledger WHERE run_id IS NOT NULL"
                " GROUP BY run_id"
            )
        }
        snapshots = {
            r["run_id"]: (json.loads(r["message_ids"]), json.loads(r["reasons"]))
            for r in serve.execute("SELECT run_id, message_ids, reasons FROM context_snapshots")
        }
        new_version = {}
        nv_ids: set[str] = set()
        for r in serve.execute("SELECT * FROM outbound WHERE kind='new_version'"):
            new_version[_key(r["group_id"], r["reply_to"])] = dict(r)
            if r["qq_message_id"]:
                nv_ids.add(_key(r["group_id"], r["qq_message_id"]))
        messages = {}
        for run in runs:
            for rid in reply_ids(json.loads(run["segments"] or "[]")):
                key = _key(run["group_id"], rid)
                if key in messages:
                    continue
                row = serve.execute(
                    "SELECT segments, revoked, user_id FROM messages WHERE bot_id=? AND"
                    " group_id=? AND message_id=?",
                    (run["bot_id"], run["group_id"], rid),
                ).fetchone()
                messages[key] = dict(row) if row else None
    finally:
        serve.close()

    old: dict[str, dict] = {}
    nv_seen = 0
    if runs:
        path = snowluma_db or snowluma_root / bot_id / "messages.db"
        groups = {str(r["group_id"]) for r in runs}
        lo = min(r["asked_at"] for r in runs) - 60
        hi = max(r["asked_at"] for r in runs) + window + 60
        rows = load_snowluma(path, groups, lo, hi)
        bots, anchors = [], {}
        for key, row in rows.items():
            event = row.event
            if str(event.get("user_id")) == bot_id:
                text = render_segments(event.get("message"), bot_id)
                if text.startswith(NEW_PREFIX) or key in nv_ids:
                    nv_seen += 1  # the new framework's own send, not an old reply
                    continue
                bots.append(row)
            elif _is_request(event, bot_id):
                anchors[key] = row
        for run in runs:  # a run's question must be an anchor even if SnowLuma misses it
            key = _key(run["group_id"], run["message_id"])
            if key not in anchors:
                t = float(run["asked_at"] or 0)
                event = {
                    "group_id": run["group_id"],
                    "message_id": run["message_id"],
                    "user_id": run["user_id"],
                    "self_id": bot_id,
                    "message": json.loads(run["segments"] or "[]"),
                }
                anchors[key] = Row(event, (t, -1), t, key)
        for key, match in link_bot_replies(bots, list(anchors.values()), window).items():
            row = match["row"]
            old[key] = {
                "text": render_segments(row.event.get("message"), bot_id),
                "time": row.timestamp,
                "rule": match["rule"],
                "confidence": match["confidence"],
            }

    items = []
    for run in runs:
        key = _key(run["group_id"], run["message_id"])
        asked = run["asked_at"]
        quoted = []
        for rid in reply_ids(json.loads(run["segments"] or "[]")):
            q = messages.get(_key(run["group_id"], rid))
            if q is None:
                quoted.append("（引用的消息不在本库）")
            elif q["revoked"]:
                quoted.append("（引用的消息已撤回）")
            else:
                who = "机器人" if q["user_id"] == bot_id else "群友"
                quoted.append(f"{who}：{render_segments(json.loads(q['segments']), bot_id)}")
        ids, reasons = snapshots.get(run["run_id"], ([], {}))
        relations = Counter(str(v).split("/")[0] for v in reasons.values())
        answered = run["recorded_at"] or run["ended_at"]
        o = old.get(key)
        nv = new_version.get(key)
        items.append(
            {
                "run_id": run["run_id"],
                "group": label(run["group_id"]),
                "asked_at": asked,
                "question": "（已撤回）"
                if run["revoked"]
                else render_segments(json.loads(run["segments"] or "[]"), bot_id),
                "quoted": quoted,
                "old": o,
                "new": run["answer"],
                "outbound_state": run["outbound_state"],
                "run_state": run["state"],
                "error_kind": run["error_kind"],
                "route": run["route"],
                "cost_mode": run["cost_mode"],
                "latency_new": round(answered - asked, 1) if answered and asked else None,
                "latency_old": round(o["time"] - asked, 1) if o and asked else None,
                "cost_cny": round(cost.get(run["run_id"], 0.0) or 0.0, 6),
                "context_count": len(ids),
                "context_reasons": dict(sorted(relations.items())),
                "new_version": None
                if nv is None
                else {"state": nv["state"], "error_kind": nv["error_kind"]},
            }
        )
    meta = {"bot_messages_new_version": nv_seen}
    return items, meta


def summarize(items: list[dict], meta: dict, since, until, ratings: dict | None) -> dict:
    paired = [i for i in items if i["old"]]
    new_lat = [i["latency_new"] for i in items if i["latency_new"] is not None]
    old_lat = [i["latency_old"] for i in paired if i["latency_old"] is not None]
    both = [i for i in paired if i["latency_new"] is not None and i["latency_old"] is not None]
    total_cost = sum(i["cost_cny"] for i in items)
    summary = {
        "window": {"since": fmt_time(since), "until": fmt_time(until)},
        "requests": len(items),
        "paired_with_old_reply": len(paired),
        "pairing_rate": round(len(paired) / len(items), 4) if items else None,
        "pair_rules": dict(Counter(i["old"]["rule"] for i in paired)),
        "pair_confidence": dict(Counter(i["old"]["confidence"] for i in paired)),
        "run_states": dict(Counter(i["run_state"] for i in items)),
        "outbound_states": dict(Counter(i["outbound_state"] or "none" for i in items)),
        "error_kinds": dict(Counter(i["error_kind"] for i in items if i["error_kind"])),
        "routes": dict(Counter(i["route"] or "none" for i in items)),
        "cost_modes": dict(Counter(i["cost_mode"] or "none" for i in items)),
        "latency_new_s": stats(new_lat),
        "latency_old_s": stats(old_lat),
        "latency_both_answered_s": {
            "n": len(both),
            "new": stats([i["latency_new"] for i in both]),
            "old": stats([i["latency_old"] for i in both]),
        },
        "cost_cny": {
            "total": round(total_cost, 4),
            "per_request": round(total_cost / len(items), 5) if items else None,
        },
        "context": {
            "mean_picks": round(sum(i["context_count"] for i in items) / len(items), 2)
            if items
            else None,
            "relations": dict(
                sum((Counter(i["context_reasons"]) for i in items), Counter()).most_common()
            ),
        },
        "new_version": {
            "commands_recorded": sum(1 for i in items if i["new_version"]),
            "states": dict(Counter(i["new_version"]["state"] for i in items if i["new_version"])),
            "bot_messages_seen": meta.get("bot_messages_new_version", 0),
        },
    }
    if ratings is not None:
        known = {i["run_id"] for i in items}
        rated = {k: v for k, v in ratings.items() if k in known and v.get("rating") in RATINGS}
        summary["ratings"] = {
            "rated": len(rated),
            "unrated": len(known) - len(rated),
            "counts": {
                RATINGS[k]: n for k, n in Counter(v["rating"] for v in rated.values()).items()
            },
            "with_note": sum(1 for v in rated.values() if str(v.get("note") or "").strip()),
        }
    return summary


def load_ratings(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    ratings = data.get("ratings", data) if isinstance(data, dict) else {}
    return {str(k): v for k, v in ratings.items() if isinstance(v, dict)}


def summary_markdown(s: dict) -> str:
    def st(x):
        return f"p50 {x['p50']} / p95 {x['p95']}（n={x['n']}）"

    lines = [
        "# 影子对照汇总",
        "",
        f"- 时间范围：{s['window']['since']} ~ {s['window']['until']}",
        f"- 请求数：{s['requests']}；与旧回复配对：{s['paired_with_old_reply']}"
        f"（配对率 {s['pairing_rate']}）；配对规则 {s['pair_rules']}",
        f"- run 状态：{s['run_states']}；出站状态：{s['outbound_states']}",
        f"- 失败/取消原因：{s['error_kinds']}",
        f"- 路线：{s['routes']}；费用档：{s['cost_modes']}",
        f"- 新框架耗时（秒，提问到记录）：{st(s['latency_new_s'])}",
        f"- 旧链路耗时（秒，AstrBot 回复时间减提问时间）：{st(s['latency_old_s'])}",
        f"- 两边都有回答的请求：新 {st(s['latency_both_answered_s']['new'])}；"
        f"旧 {st(s['latency_both_answered_s']['old'])}",
        f"- 费用：合计 ¥{s['cost_cny']['total']}，每请求 ¥{s['cost_cny']['per_request']}",
        f"- 上下文：平均选入 {s['context']['mean_picks']} 条；理由 {s['context']['relations']}",
        f"- 〔新版〕指令：{s['new_version']}",
    ]
    if "ratings" in s:
        r = s["ratings"]
        lines.append(
            f"- 主人打分：已打 {r['rated']}、未打 {r['unrated']}；{r['counts']}；带备注 {r['with_note']}"
        )
    return "\n".join(lines) + "\n"


# -- HTML ---------------------------------------------------------------------------------

STYLE = """
body{font-family:system-ui,"Microsoft YaHei",sans-serif;margin:16px;background:#f6f6f6;color:#222}
.item{background:#fff;border:1px solid #ddd;border-radius:6px;margin:12px 0;padding:10px}
.item.rated{border-left:6px solid #4a8}
.head{color:#666;font-size:12px;margin-bottom:6px}
.q{font-weight:600;white-space:pre-wrap}
.quote{color:#555;font-size:13px;white-space:pre-wrap;border-left:3px solid #ccc;padding-left:6px}
.cols{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:8px}
.col{background:#fafafa;border:1px solid #eee;padding:6px;white-space:pre-wrap}
.col h4{margin:0 0 4px 0;font-size:13px;color:#336}
.metrics{font-size:12px;color:#444;margin-top:6px}
.rate{margin-top:6px;font-size:13px}
.rate textarea{width:100%;height:36px}
.bar{position:sticky;top:0;background:#f6f6f6;padding:6px 0;z-index:2}
"""

SCRIPT = """
const KEY = "shadow-report-ratings-v1";
const load = () => { try { return JSON.parse(localStorage.getItem(KEY) || "{}"); }
                     catch (e) { return {}; } };
const save = (r) => localStorage.setItem(KEY, JSON.stringify(r));
function refresh() {
  const r = load(); let n = 0;
  document.querySelectorAll(".item").forEach((el) => {
    const v = r[el.dataset.run] || {};
    el.querySelectorAll("input[type=radio]").forEach((i) => { i.checked = i.value === v.rating; });
    const t = el.querySelector("textarea"); if (document.activeElement !== t) t.value = v.note || "";
    el.classList.toggle("rated", !!v.rating); if (v.rating) n++;
  });
  document.getElementById("count").textContent = n;
}
document.addEventListener("change", (e) => {
  const el = e.target.closest(".item"); if (!el) return;
  const r = load(); const v = r[el.dataset.run] || {};
  if (e.target.type === "radio") v.rating = e.target.value;
  r[el.dataset.run] = v; save(r); refresh();
});
document.addEventListener("input", (e) => {
  if (e.target.tagName !== "TEXTAREA") return;
  const el = e.target.closest(".item"); const r = load(); const v = r[el.dataset.run] || {};
  v.note = e.target.value; r[el.dataset.run] = v; save(r);
});
function exportRatings() {
  const blob = new Blob([JSON.stringify({version: 1, ratings: load()}, null, 1)],
                        {type: "application/json"});
  const a = document.createElement("a"); a.href = URL.createObjectURL(blob);
  a.download = "shadow-ratings.json"; a.click();
}
function importRatings(input) {
  const f = input.files[0]; if (!f) return;
  f.text().then((t) => { const d = JSON.parse(t); save(Object.assign(load(), d.ratings || d));
                          refresh(); });
}
function onlyUnrated(on) {
  document.querySelectorAll(".item").forEach((el) => {
    el.style.display = on && el.classList.contains("rated") ? "none" : "";
  });
}
refresh();
"""


def e(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def render_item(i: dict, index: int) -> str:
    old = i["old"]
    old_text = old["text"] if old else "（未找到旧机器人的回复）"
    old_meta = f"{old['rule']} / {old['confidence']}" if old else "-"
    new_text = i["new"] or f"（无回答：{i['run_state']} {i['error_kind'] or ''}）"
    quotes = "".join(f'<div class="quote">{e(q)}</div>' for q in i["quoted"])
    reasons = "，".join(f"{k}×{v}" for k, v in i["context_reasons"].items()) or "无"
    nv = i["new_version"]
    nv_text = f"；〔新版〕发送：{nv['state']} {nv['error_kind'] or ''}" if nv else ""
    radios = "".join(
        f'<label><input type="radio" name="r{index}" value="{e(k)}"> {e(v)}</label> '
        for k, v in RATINGS.items()
    )
    metrics = (
        f"run：{i['run_state']}{' / ' + i['error_kind'] if i['error_kind'] else ''}；"
        f"出站：{i['outbound_state'] or '-'}；路线：{i['route'] or '-'}；费用档：{i['cost_mode']}；"
        f"新耗时：{i['latency_new'] if i['latency_new'] is not None else '-'} 秒；"
        f"旧耗时：{i['latency_old'] if i['latency_old'] is not None else '-'} 秒；"
        f"费用：¥{i['cost_cny']}；上下文 {i['context_count']} 条（{reasons}）{nv_text}"
    )
    return (
        f'<div class="item" data-run="{e(i["run_id"])}">'
        f'<div class="head">#{index} · {e(fmt_time(i["asked_at"]))} · 群 {e(i["group"])}</div>'
        f'<div class="q">{e(i["question"])}</div>{quotes}'
        f'<div class="cols"><div class="col"><h4>旧回答（{e(old_meta)}）</h4>{e(old_text)}</div>'
        f'<div class="col"><h4>新回答</h4>{e(new_text)}</div></div>'
        f'<div class="metrics">{e(metrics)}</div>'
        f'<div class="rate">{radios}<textarea placeholder="备注"></textarea></div></div>'
    )


def render_html(items: list[dict], summary: dict) -> str:
    body = "\n".join(render_item(i, n) for n, i in enumerate(items, 1))
    head = (
        f"请求 {summary['requests']}，配对 {summary['paired_with_old_reply']}，"
        f"新耗时 p50 {summary['latency_new_s']['p50']} 秒，旧耗时 p50 "
        f"{summary['latency_old_s']['p50']} 秒，费用 ¥{summary['cost_cny']['total']}"
    )
    return (
        '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">'
        "<title>影子对照</title>"
        f"<style>{STYLE}</style></head><body>"
        f"<h2>影子对照：旧机器人 vs 新框架</h2><p>{e(head)}</p>"
        '<div class="bar">已打分 <span id="count">0</span> 条 · '
        '<button onclick="exportRatings()">导出打分 JSON</button> '
        '<label>导入 <input type="file" accept=".json" onchange="importRatings(this)"></label> '
        '<label><input type="checkbox" onchange="onlyUnrated(this.checked)"> 只看未打分</label>'
        f"</div>{body}<script>{SCRIPT}</script></body></html>\n"
    )


def write_report(out: Path, items: list[dict], summary: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "index.html").write_text(render_html(items, summary), encoding="utf-8")
    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (out / "summary.md").write_text(summary_markdown(summary), encoding="utf-8")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serve-db", type=Path, default=Path("data/serve/serve.sqlite3"))
    parser.add_argument("--snowluma-db", type=Path)
    parser.add_argument("--snowluma-root", type=Path, default=Path("data/snowluma"))
    parser.add_argument("--out", type=Path, default=Path("data/serve/report"))
    parser.add_argument("--since", help="question time from (epoch or ISO, Beijing time)")
    parser.add_argument("--until", help="question time before (epoch or ISO, Beijing time)")
    parser.add_argument("--window", type=int, default=REPLY_WINDOW_SECONDS)
    parser.add_argument("--import-ratings", type=Path, help="exported ratings JSON to summarise")
    args = parser.parse_args(argv)
    try:
        since, until = parse_time(args.since), parse_time(args.until)
    except ValueError:
        parser.error("--since/--until: expected epoch seconds or an ISO date/time")
    if not args.serve_db.exists():
        parser.error("--serve-db: not found")
    ratings = load_ratings(args.import_ratings) if args.import_ratings else None
    try:
        items, meta = build(
            args.serve_db,
            args.snowluma_db,
            snowluma_root=args.snowluma_root,
            since=since,
            until=until,
            window=args.window,
        )
    except FileNotFoundError as exc:
        parser.error(str(exc))
    summary = summarize(items, meta, since, until, ratings)
    write_report(args.out, items, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
