"""Compare live replay runs of different context strategies side by side (stage 5).

Usage:
    python scripts/replay/compare_live.py --run A=data/live/ctx-A --run B=data/live/full-2 \
        --run C=data/live/ctx-C --run D=data/live/ctx-D --out data/live/ctx-compare

Reads the per-case rows written by `run_live.py`, re-applies the current labels, and
writes (all under git-ignored `data/`, private):
- `summary.json`: per-strategy aggregates on the cases every run finished (common set)
  and on each run's own cases, plus per-case context sizes;
- `scorecard.md`: aggregate table only (no chat text);
- `review.html`: per case the anchor question, the quoted message, the annotation's
  answer points and each strategy's answer with route, context size, tokens, latency
  and cost, for the owner to judge answer quality locally. Do not upload it.
No model calls.
"""

import argparse
import html
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from group_agent.replay.live_scoring import load_exclusions, summarize_live  # noqa: E402
from run_live import DEFAULT_BATCH, EXCLUSIONS, load_rows, refresh_label_checks  # noqa: E402
from score import load_cases  # noqa: E402


def parse_runs(items: list[str]) -> dict[str, Path]:
    runs = {}
    for item in items:
        name, _, path = item.partition("=")
        if not name or not path:
            raise SystemExit("--run expects NAME=DIR")
        runs[name] = Path(path)
    return runs


def context_of(row: dict | None) -> tuple:
    return tuple(((row or {}).get("anchor_trace") or {}).get("context_ids") or [])


def row_stats(row: dict) -> dict:
    live = row.get("live") or {}
    calls = live.get("calls") or []
    material = live.get("material") or {}
    checks = (row.get("score") or {}).get("checks") or {}
    req = checks.get("required_context") or {}
    expected = req.get("expected") or []
    return {
        "route": (checks.get("route") or {}).get("actual"),
        "route_ok": (checks.get("route") or {}).get("pass"),
        "clarify_ok": (checks.get("clarify") or {}).get("pass"),
        "required": len(expected),
        "required_hit": len(expected) - len(req.get("missing") or []),
        "forbidden_used": len((checks.get("forbidden_context") or {}).get("used") or []),
        "context_messages": material.get("messages"),
        "images_attached": material.get("images_attached"),
        "material_chars": material.get("chars"),
        "input_tokens": sum(c["usage"]["input"] for c in calls),
        "output_tokens": sum(c["usage"]["output"] for c in calls),
        "latency_s": live.get("latency_total_s"),
        "cost_cny": round(sum(c["cost_cny"] for c in calls), 6),
        "state": row.get("anchor_state") or row.get("status"),
        "searches": (live.get("web") or {}).get("searches", 0),
        "supplement": (live.get("supplement") or {}).get("status"),
    }


def _mean(values: list) -> float | None:
    values = [v for v in values if v is not None]
    return round(statistics.mean(values), 2) if values else None


def aggregate(rows: list[dict], exclusions: dict) -> dict:
    s = summarize_live(rows, exclusions)
    stats = [row_stats(r) for r in rows]
    return {
        "cases": s["cases"],
        "completed": s["completed"],
        "route_fine": s["routing"]["route_fine"],
        "route_coarse": s["routing"]["route_coarse"],
        "clarify": s["routing"]["clarify"],
        "effective_clarify_rate": s["routing"]["effective_clarify_rate"],
        "false_clarify_rate": s["routing"]["false_clarify_rate"],
        "routing_reviewed": s["routing_reviewed"]["route_fine"],
        "context": s["context"],
        "context_messages_mean": _mean([x["context_messages"] for x in stats]),
        "images_attached_mean": _mean([x["images_attached"] for x in stats]),
        "material_chars_mean": _mean([x["material_chars"] for x in stats]),
        "latency_seconds_total": s["latency_seconds_total"],
        "tokens_per_request": s["tokens_per_request"],
        "cost_cny": s["cost_cny"],
        "web_runs": s["web"]["runs_with_web"],
        "web": s["web"],
        "supplement": s["supplement"],
        "failure_rate": s["failure_rate"],
        "timeout_rate": s["timeout_rate"],
        "isolation": s["isolation"],
        "sent_to_qq": s["sent_to_qq"],
    }


def _pct(x) -> str:
    return "—" if x is None else f"{x * 100:.1f}%"


def scorecard(summary: dict) -> str:
    lines = [
        "# 上下文策略实时对照（汇总，不含聊天内容）",
        "",
        f"共同完成的样本 {summary['common']['cases']} 条；路由评分排除见 route_exclusions。",
        "",
        *table(summary["common"]["by_strategy"]),
    ]
    differs = summary["context_differs"]
    if differs["by_strategy"]:
        lines += [
            "",
            f"其中各策略上下文不同的 {differs['cases']} 条：",
            "",
            *table(differs["by_strategy"]),
        ]
    return "\n".join(lines) + "\n"


def table(agg: dict) -> list[str]:
    names = list(agg)
    head = "| 指标 | " + " | ".join(names) + " |"
    sep = "| --- |" + " --- |" * len(names)

    def line(label, fn):
        return f"| {label} | " + " | ".join(fn(agg[n]) for n in names) + " |"

    return [
        head,
        sep,
        line("路由一致（细粒度）", lambda a: _pct(a["route_fine"]["rate"])),
        line("追问判断一致", lambda a: _pct(a["clarify"]["rate"])),
        line("必需上下文召回（按条）", lambda a: _pct(a["context"]["required_recall_micro"])),
        line("选入中禁止占比", lambda a: _pct(a["context"]["forbidden_share"])),
        line("每次上下文消息数", lambda a: str(a["context_messages_mean"])),
        line("每次附图数", lambda a: str(a["images_attached_mean"])),
        line(
            "耗时 p50 / p95（秒）",
            lambda a: f"{a['latency_seconds_total']['p50']} / {a['latency_seconds_total']['p95']}",
        ),
        line(
            "输入 / 输出 token",
            lambda a: (
                f"{a['tokens_per_request']['input_mean']} / "
                f"{a['tokens_per_request']['output_mean']}"
            ),
        ),
        line(
            "费用均值 / 合计（元）",
            lambda a: f"{a['cost_cny']['per_request_mean']} / {a['cost_cny']['total']}",
        ),
        line(
            "联网 run / 每 run 搜索（联网费用）",
            lambda a: (
                f"{a['web']['runs_with_web']} / {a['web']['searches_per_run']['mean']}"
                f"（{a['web']['cost_cny_total']} 元）"
            ),
        ),
        line(
            "Gemini 补充 触发 / 调用 / 结果",
            lambda a: (
                f"{a['supplement']['runs']} / {a['supplement']['called']} / "
                + ("、".join(f"{k} {v}" for k, v in a["supplement"]["statuses"].items()) or "—")
            ),
        ),
        line("失败率", lambda a: _pct(a["failure_rate"])),
    ]


CSS = (
    "body{font-family:system-ui,'Microsoft YaHei',sans-serif;margin:24px;background:#f6f6f6}"
    ".case{background:#fff;padding:12px 16px;margin:14px 0;border-left:6px solid #888}"
    ".case.diff{border-color:#c80}.meta{color:#555;font-size:13px}"
    ".grid{display:grid;grid-template-columns:repeat(var(--n),1fr);gap:10px}"
    ".col{border:1px solid #ddd;padding:8px;border-radius:4px}.bad{color:#c22}.ok{color:#272}"
    "pre{white-space:pre-wrap;font-family:inherit;margin:0}h3{font-size:14px;margin:4px 0}"
    "table{border-collapse:collapse;font-size:13px}td,th{border:1px solid #ccc;padding:2px 6px}"
)


def review_html(runs: dict[str, dict], cases: dict, order: list[str]) -> str:
    esc = html.escape
    names = list(runs)
    cards = []
    for cid in order:
        case = cases.get(cid)
        if case is None:
            continue
        ann = case["annotations"]
        rows = {n: runs[n].get(cid) for n in names}
        any_row = next((r for r in rows.values() if r), {})
        live0 = any_row.get("live") or {}
        contexts = {n: context_of(r) for n, r in rows.items()}
        differs = len(set(contexts.values())) > 1
        points = "".join(f"<li>{esc(str(p))}</li>" for p in ann.get("answer_points") or [])
        cols = []
        for n in names:
            r = rows[n]
            if not r:
                cols.append(f"<div class='col'><h3>{esc(n)}</h3><p class='meta'>未运行</p></div>")
                continue
            st = row_stats(r)
            live = r.get("live") or {}
            reasons = (r.get("anchor_trace") or {}).get("context_reasons") or {}
            reason_text = ""
            if reasons:
                counts: dict[str, int] = {}
                for v in reasons.values():
                    counts[v] = counts.get(v, 0) + 1
                reason_text = "；".join(f"{esc(k)}×{v}" for k, v in sorted(counts.items()))
                reason_text = f"<p class='meta'>选择理由：{reason_text}</p>"
            route_cls = "ok" if st["route_ok"] else "bad"
            cols.append(
                f"<div class='col'><h3>{esc(n)}</h3>"
                f"<p class='meta'>路线 <b class='{route_cls}'>{esc(str(st['route']))}</b>；"
                f"上下文 {st['context_messages']} 条、图 {st['images_attached']}；"
                f"必需 {st['required_hit']}/{st['required']}，禁止 {st['forbidden_used']}；"
                f"输入 {st['input_tokens']} / 输出 {st['output_tokens']} token；"
                f"{st['latency_s']} 秒；¥{st['cost_cny']}；{esc(str(st['state']))}；"
                f"搜索 {st['searches']} 次"
                + (f"，Gemini 补充 {esc(str(st['supplement']))}" if st["supplement"] else "")
                + "</p>"
                + f"{reason_text}<pre>{esc(live.get('answer') or '')}</pre></div>"
            )
        quoted = live0.get("quoted_text") or ""
        cards.append(
            f"<section class='case {'diff' if differs else ''}'>"
            f"<h2>{esc(cid)} {'(已复核)' if ann.get('status') == 'reviewed' else ''}"
            f"{' — 各策略上下文不同' if differs else ' — 上下文相同'}</h2>"
            f"<p class='meta'>标注路线 <b>{esc(str(ann.get('expected_route')))}</b>，"
            f"追问 {ann.get('should_clarify')}，必需上下文 "
            f"{len(ann.get('required_context_ids') or [])} 条</p>"
            f"<h3>提问</h3><pre>{esc(live0.get('anchor_text') or '')}</pre>"
            + (f"<h3>被引用</h3><pre>{esc(quoted)}</pre>" if quoted else "")
            + f"<h3>标注要点</h3><ul>{points}</ul>"
            f"<div class='grid' style='--n:{len(names)}'>{''.join(cols)}</div></section>"
        )
    return (
        "<!doctype html><html lang='zh-CN'><meta charset='utf-8'>"
        f"<title>上下文策略对照（本机私有）</title><style>{CSS}</style>"
        "<h1>上下文策略实时对照（本机私有，不要上传）</h1>"
        "<p>每条样本列出提问与各策略的回答。橙色边框：各策略选入的上下文不同（回答差异可能来自上下文）；"
        "灰色：上下文相同，回答差异只来自模型随机性。路线绿色为与标注一致。"
        "请按标注要点判断哪个回答更好。</p>" + "".join(cards) + "</html>"
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, help="NAME=DIR, repeatable")
    parser.add_argument("--batch", type=Path, default=DEFAULT_BATCH)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    out = args.out.resolve()
    if not out.is_relative_to((ROOT / "data").resolve()):
        parser.error("--out must be under data/ (git-ignored; outputs are private)")
    batch = args.batch
    sidecar, reviewed = batch / "annotations.draft.json", batch / "annotations.reviewed.json"
    cases = load_cases(
        batch, sidecar if sidecar.exists() else None, reviewed if reviewed.exists() else None
    )
    by_id = {c["case_id"]: c for c in cases}
    exclusions = load_exclusions(EXCLUSIONS)
    runs: dict[str, dict] = {}
    for name, path in parse_runs(args.run).items():
        rows = {}
        for row in load_rows(path):
            if row["case_id"] in by_id:
                refresh_label_checks(row, by_id[row["case_id"]])
                rows[row["case_id"]] = row
        runs[name] = rows
    common = sorted(set.intersection(*(set(r) for r in runs.values()))) if runs else []
    differs = [c for c in common if len({context_of(rows[c]) for rows in runs.values()}) > 1]
    summary = {
        "runs": {n: str(p) for n, p in parse_runs(args.run).items()},
        "common": {
            "cases": len(common),
            "by_strategy": {
                n: aggregate([rows[c] for c in common], exclusions) for n, rows in runs.items()
            },
        },
        # Cases where the strategies selected different context: only there can answer
        # differences come from context rather than model randomness.
        "context_differs": {
            "cases": len(differs),
            "by_strategy": {
                n: aggregate([rows[c] for c in differs], exclusions) for n, rows in runs.items()
            }
            if differs
            else {},
        },
        "own": {n: aggregate(list(rows.values()), exclusions) for n, rows in runs.items()},
        "per_case": {
            cid: {n: row_stats(rows[cid]) for n, rows in runs.items() if cid in rows}
            for cid in sorted(set().union(*(set(r) for r in runs.values())))
        },
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out / "scorecard.md").write_text(scorecard(summary), encoding="utf-8")
    order = sorted(set().union(*(set(r) for r in runs.values())))
    (out / "review.html").write_text(review_html(runs, by_id, order), encoding="utf-8")
    print(scorecard(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
