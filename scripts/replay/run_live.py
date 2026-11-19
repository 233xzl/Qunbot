"""Live-model replay over real cases (T1.9 live tier): real models, stubbed web, no sending.

Usage:
    python scripts/replay/run_live.py [batch_dir] --live --budget-cny 1.0
        [--limit N] [--cases id,id] [--strategy C|B|A|D] [--concurrency 2]
        [--decider deepseek|gemini] [--jev-shadow] [--out data/live/<name>] [--rerun]
        [--no-account-book]
    python scripts/replay/run_live.py --score-only --out data/live/<name>

Without `--live` it only lists the selected cases and the budget projection (no calls).
Every case's result is written to `<out>/cases/<case_id>.json` as soon as it finishes; a
rerun with the same `--out` skips finished cases (resumable). New cases stop being
started when spent + mean cost per case x unfinished cases would exceed `--budget-cny`;
each model call also reserves its estimated cost against the remaining budget.

Outputs (all under the git-ignored `data/` directory, private): per-case results with
answers, `summary.json` (aggregate scorecard incl. legacy baseline A on the same cases),
`scorecard.md` (aggregate only), `review.html` (side-by-side answers for local reading).
Keys and the relay URL are never printed. `sent_to_qq` is always 0.
"""

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime
import html
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from group_agent.billing import open_account_book, open_google_quota  # noqa: E402
from group_agent.config import (  # noqa: E402
    ConfigError,
    load_jev_credentials,
    load_model_credentials,
    load_settings,
)
from group_agent.knowledge_search import KnowledgeSnapshot  # noqa: E402
from group_agent.models import Ledger  # noqa: E402
from group_agent.replay.live import LiveAgent, LiveConfig  # noqa: E402
from group_agent.replay.live_scoring import load_exclusions, summarize_live  # noqa: E402
from group_agent.web import (  # noqa: E402
    SafeHttp,
    SupplementBreaker,
    build_supplement_client,
    build_web_client,
)
from group_agent.replay.runner import run_case  # noqa: E402
from group_agent.replay.scoring import score_case  # noqa: E402
from score import legacy_route, load_cases, score_legacy  # noqa: E402

DEFAULT_BATCH = ROOT / "tests/replay/raw/batch-02"
EXCLUSIONS = ROOT / "tests/replay/route_exclusions.json"
PRIOR_CASE_COST = 0.01  # CNY, used for the projection until real costs are known


def pick(cases: list[dict], ids: str | None, limit: int | None) -> list[dict]:
    if ids:
        wanted = [i.strip() for i in ids.split(",") if i.strip()]
        known = {c["case_id"]: c for c in cases}
        missing = [i for i in wanted if i not in known]
        if missing:
            raise SystemExit(f"unknown case ids: {len(missing)}")
        cases = [known[i] for i in wanted]
    return cases[:limit] if limit else cases


def legacy_text(case: dict) -> str:
    reply = (case.get("legacy_result") or {}).get("bot_reply") or {}
    for event in case["events"]:
        if str(event.get("message_id")) == str(reply.get("message_id")):
            return "".join(
                str((s.get("data") or {}).get("text", ""))
                for s in event.get("message") or []
                if isinstance(s, dict) and s.get("type") == "text"
            ).strip()
    return ""


def run_one(case: dict, config: LiveConfig, strategy: str, reviewed_ids: set[str]) -> dict:
    holder: dict = {}

    def factory(replay):
        holder["agent"] = LiveAgent(replay, config)
        return holder["agent"]

    started = time.perf_counter()
    row = {"case_id": case["case_id"], "reviewed": case["case_id"] in reviewed_ids}
    try:
        result = run_case(
            case,
            strategy,
            agent_factory=factory,
            **({"limits": config.runtime_limits} if config.runtime_limits else {}),
        )
    except Exception as exc:  # noqa: BLE001 - one broken case must not stop the batch
        row.update(status="crashed", error=type(exc).__name__, live={}, sent_to_qq=0)
        row["wall_s"] = round(time.perf_counter() - started, 2)
        return row
    agent = holder.get("agent")
    anchor = next((r for r in result["runs"] if r["message_id"] == result["anchor"]), None)
    live = {}
    if agent is not None and anchor is not None and anchor["run_id"] in agent.runs:
        live = agent.runs[anchor["run_id"]].record
    row.update(
        status="completed",
        score=score_case(case, result),
        anchor_state=anchor["state"] if anchor else None,
        anchor_reason=anchor["reason"] if anchor else None,
        anchor_trace=anchor,
        live=live,
        sent_to_qq=result["sent_to_qq"],
        cost_cny=round(sum(c["cost_cny"] for c in live.get("calls", [])), 6),
        wall_s=round(time.perf_counter() - started, 2),
    )
    return row


def build_config(args, settings) -> LiveConfig:
    try:
        gemini = settings.models[settings.model_roles.get("answer", "gemini")]
        deepseek = settings.models[settings.model_roles.get("fallback", "deepseek")]
    except KeyError as exc:
        raise SystemExit(f"settings: missing [models.{exc.args[0]}]") from None
    jev = settings.models.get("jev") if args.jev_shadow else None
    # The migrated knowledge base, read-only (legacy snapshots are migration input only).
    default_kb = settings.knowledge_db
    knowledge_path = args.knowledge or (default_kb if default_kb and default_kb.is_file() else None)
    web_client = web_http = supplement = None
    if args.web and settings.web_enabled and not args.decision_only:
        web_client = build_web_client(settings)
        web_http = SafeHttp()
        if not args.no_gemini_supplement:
            supplement = build_supplement_client(settings)  # D-18; None when [web] says off
    book = open_account_book(settings, "replay", enabled=getattr(args, "account_book", True))
    from group_agent.persona import prompt_options

    decision = (
        (gemini if args.decider == "gemini" else deepseek)
        if args.decider
        else settings.models[settings.model_roles.get("decision", gemini.name)]
    )
    return LiveConfig(
        config_revision=settings.revision,
        prompt_text=prompt_options(settings),
        runtime_limits=settings.limits,
        decision_model=decision,
        decision_creds=load_model_credentials(settings.credentials, decision),
        answer_tools=settings.answer_tool_loop,
        decision_mode=settings.decision_mode,
        answer_same_model_retry=settings.answer_same_model_retry,
        gemini=gemini,
        deepseek=deepseek,
        gemini_creds=load_model_credentials(settings.credentials, gemini),
        deepseek_creds=load_model_credentials(settings.credentials, deepseek),
        # Replay spends the same account as the bot: calls also go to the account usage book.
        ledger=Ledger(
            None,
            account=book,
            source="replay" if book is not None else None,
            model_ids={name: spec.model for name, spec in settings.models.items()},
            quota=open_google_quota(settings, book),
        ),
        media_root=args.media_root,
        knowledge=KnowledgeSnapshot(knowledge_path) if knowledge_path else None,
        decider=args.decider or ("gemini" if decision.protocol == "google-genai" else "deepseek"),
        decision_thinking=None if args.decision_thinking == "NONE" else args.decision_thinking,
        answer_thinking=None
        if args.answer_thinking == "NONE"
        else (args.answer_thinking or settings.answer_thinking),
        decision_only=args.decision_only,
        jev=jev,
        jev_creds=load_jev_credentials(settings.credentials) if jev else None,
        web_client=web_client,
        web_http=web_http,
        web_price_cny=settings.web.price_per_call,
        web_supplement=supplement,
        supplement_min_seconds=settings.web.supplement_min_seconds,
        supplement_force=args.force_gemini_supplement,
        supplement_breaker=SupplementBreaker(
            settings.web.supplement_breaker_failures, settings.web.supplement_breaker_seconds
        )
        if supplement is not None
        else None,
        fact_rule=not args.no_fact_rule,
        subject_rule=not args.no_subject_rule,
        answer_thinking_simple=None if args.simple_thinking == "NONE" else args.simple_thinking,
    )


def load_rows(out: Path) -> list[dict]:
    return [
        json.loads(p.read_text(encoding="utf-8")) for p in sorted((out / "cases").glob("*.json"))
    ]


def refresh_label_checks(row: dict, case: dict) -> None:
    """Re-apply current route/clarify labels to a stored row (labels may be corrected after
    a run; the anchor trace keeps what the run did). Mirrors replay.scoring.score_case."""
    checks = (row.get("score") or {}).get("checks")
    trace = row.get("anchor_trace")
    ann = case.get("annotations") or {}
    if not checks or trace is None or ann.get("status") not in ("draft", "reviewed"):
        return
    route = trace.get("route")
    want = ann.get("expected_route")
    checks["route"] = {"expected": want, "actual": route, "pass": route == want}
    if ann.get("should_clarify") is not None:
        actual, expected = bool(trace.get("clarify")), bool(ann["should_clarify"])
        checks["clarify"] = {"expected": expected, "actual": actual, "pass": actual == expected}


def write_reports(out: Path, cases: list[dict], meta: dict) -> dict:
    rows = load_rows(out)
    by_id = {c["case_id"]: c for c in cases}
    for row in rows:
        if row["case_id"] in by_id:
            refresh_label_checks(row, by_id[row["case_id"]])
    scored_cases = [by_id[r["case_id"]] for r in rows if r["case_id"] in by_id]
    reviewed_cases = [c for c in scored_cases if c["annotations"].get("status") == "reviewed"]
    exclusions = load_exclusions(EXCLUSIONS)
    summary = {
        "meta": meta,
        "new_pipeline": summarize_live(rows, exclusions),
        "legacy_baseline_A_same_cases": score_legacy(scored_cases),
        "legacy_baseline_A_reviewed": score_legacy(reviewed_cases),
    }
    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out / "scorecard.md").write_text(scorecard_md(summary), encoding="utf-8")
    (out / "review.html").write_text(review_html(rows, by_id), encoding="utf-8")
    return summary


def _pct(x) -> str:
    return "—" if x is None else f"{x * 100:.0f}%"


def scorecard_md(s: dict) -> str:
    new, old = s["new_pipeline"], s["legacy_baseline_A_same_cases"]
    r = new["routing"]
    lines = [
        "# 实时回放评分（汇总，不含聊天内容）",
        "",
        f"样本 {new['cases']}，完成 {new['completed']}，复核样本 {new['reviewed_cases']}。",
        "",
        "| 指标 | 新链路 | 旧链路 A |",
        "| --- | --- | --- |",
        f"| 路由一致（5 类细粒度） | {_pct(r['route_fine']['rate'])} | — |",
        f"| 路由一致（粗粒度） | {_pct(r['route_coarse']['rate'])} | "
        f"{_pct(old['route_agreement_coarse'])} |",
        f"| 追问判断一致 | {_pct(r['clarify']['rate'])} | — |",
        f"| 应追问时追问 | {_pct(r['effective_clarify_rate'])} | "
        f"{_pct(old['effective_clarify_rate'])} |",
        f"| 不应追问却追问 | {_pct(r['false_clarify_rate'])} | {_pct(old['false_clarify_rate'])} |",
        f"| 是否回复一致 | {_pct(r['should_reply']['rate'])} | — |",
        f"| 耗时 p50 / p95（秒） | {new['latency_seconds_total']['p50']} / "
        f"{new['latency_seconds_total']['p95']} | {old['latency_seconds']['p50']} / "
        f"{old['latency_seconds']['p95']} |",
        f"| 每请求输入 / 输出 token | {new['tokens_per_request']['input_mean']} / "
        f"{new['tokens_per_request']['output_mean']} | {old['tokens_per_request']['input_mean']} / "
        f"{old['tokens_per_request']['output_mean']} |",
        f"| 每请求费用均值 / p95（元） | {new['cost_cny']['per_request_mean']} / "
        f"{new['cost_cny']['per_request_p95']} | {old['cost_cny_per_request']['mean']} / "
        f"{old['cost_cny_per_request']['p95']} |",
        f"| 失败率 / 超时率 | {_pct(new['failure_rate'])} / {_pct(new['timeout_rate'])} | — |",
        f"| 结构化判断首次成功 | {_pct(new['decision_first_try_success']['rate'])} | — |",
        f"| 联网 run 数 / 搜索均值 / 读取均值 | {new['web']['runs_with_web']} / "
        f"{new['web']['searches_per_run']['mean']} / {new['web']['reads_per_run']['mean']} | — |",
        "",
        f"路由评分排除：{new['route_excluded'] or '无'}；联网费用 ¥{new['web']['cost_cny_total']}。",
        "",
        f"对象核实（D-17）：标记 {new['subject_check']['flagged']}，"
        f"搜索 {new['subject_check']['searched']}，结果 {new['subject_check']['statuses'] or '无'}。",
        "",
        f"Gemini 搜索补充（D-18）：触发条件 {new['supplement']['runs']} 次，"
        f"实际调用 {new['supplement']['called']}，原因 {new['supplement']['reasons'] or '无'}，"
        f"结果 {new['supplement']['statuses'] or '无'}，耗时 p50/p95 "
        f"{new['supplement']['latency_s']['p50']}/{new['supplement']['latency_s']['p95']} 秒，"
        f"费用 ¥{new['supplement']['cost_cny_total']}。",
        "",
        f"总费用 ¥{new['cost_cny']['total']}；隔离错误 {sum(new['isolation'].values())}；"
        f"sent_to_qq={new['sent_to_qq']}。",
    ]
    return "\n".join(lines) + "\n"


def review_html(rows: list[dict], by_id: dict) -> str:
    esc = html.escape
    cards = []
    for row in rows:
        case = by_id.get(row["case_id"])
        if case is None:
            continue
        ann = case["annotations"]
        live = row.get("live") or {}
        route = (row.get("score") or {}).get("checks", {}).get("route", {})
        decision = live.get("decision") or {}
        ok = route.get("expected") == route.get("actual")
        points = "".join(f"<li>{esc(str(p))}</li>" for p in ann.get("answer_points") or [])
        quoted = live.get("quoted_text") or ""
        web = live.get("web") or {}
        sources = ""
        if web:
            items = "".join(
                f"<li>{esc(s.get('title') or '')}（{esc(s.get('site') or s.get('kind') or '')}）"
                f" {esc(s.get('url') or '')}</li>"
                for s in web.get("sources") or []
            )
            sources = (
                f"<p class='meta'>联网：搜索 {web.get('searches')} 次，读取 {web.get('reads')} 次，"
                f"¥{web.get('cost_cny')}</p><ol class='meta'>{items}</ol>"
            )
        check = live.get("subject_check") or {}
        subject = f"；对象核实（D-17）{esc(str(check.get('status')))}" if check else ""
        sup = live.get("supplement") or {}
        if sup:
            subject += (
                f"；Gemini 补充（D-18）{esc(str(sup.get('reason')))}→{esc(str(sup.get('status')))}"
                f"，{sup.get('latency_s')} 秒"
            )
        cards.append(
            f"""<section class="case {"ok" if ok else "bad"}">
<h2>{esc(row["case_id"])} {"(已复核)" if row.get("reviewed") else ""}</h2>
<p class="meta">标注路线 <b>{esc(str(route.get("expected")))}</b>，新链路
<b>{esc(str(route.get("actual")))}</b>，旧链路（推断）<b>{esc(str(legacy_route(case)))}</b>；
追问 标注 {ann.get("should_clarify")} / 新 {decision.get("should_clarify")}；
状态 {esc(str(row.get("anchor_state") or row.get("status")))}
{esc(str(row.get("anchor_reason") or ""))}；耗时 {live.get("latency_total_s")} 秒；
费用 ¥{row.get("cost_cny")}；回答来源 {esc(str(live.get("answer_source")))}</p>
<p class="meta">判断理由：{esc(str(decision.get("reason", "")))}
{"；代码规则 " + esc(str(live.get("route_rule"))) if live.get("route_rule") else ""}
{subject}；
回答思考 {esc(str(live.get("answer_thinking")))}</p>
{sources}
<div class="grid">
<div><h3>提问</h3><pre>{esc(live.get("anchor_text", ""))}</pre>
{"<h3>被引用</h3><pre>" + esc(quoted) + "</pre>" if quoted else ""}</div>
<div><h3>新回答</h3><pre>{esc(live.get("answer", ""))}</pre></div>
<div><h3>旧回答</h3><pre>{esc(legacy_text(case))}</pre></div>
<div><h3>标注要点</h3><ul>{points}</ul></div>
</div></section>"""
        )
    return (
        "<!doctype html><html lang='zh-CN'><meta charset='utf-8'>"
        "<title>实时回放对照（本机私有）</title><style>"
        "body{font-family:system-ui,'Microsoft YaHei',sans-serif;margin:24px;background:#f6f6f6}"
        ".case{background:#fff;border-left:6px solid #2a7;padding:12px 16px;margin:12px 0}"
        ".case.bad{border-color:#d55}.meta{color:#555;font-size:13px}"
        ".grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}"
        "pre{white-space:pre-wrap;font-family:inherit;margin:0}h3{font-size:14px;margin:4px 0}"
        "</style><h1>实时回放对照（本机私有，不要上传）</h1>"
        "<p>绿色：路线与标注一致；红色：不一致。旧回答只是行为观测，标注要点才是评估依据。</p>"
        + "".join(cards)
        + "</html>"
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch", type=Path, nargs="?", default=DEFAULT_BATCH)
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--reviewed", type=Path)
    parser.add_argument("--settings", type=Path, default=ROOT / "config/settings.toml")
    parser.add_argument("--strategy", choices=["A", "B", "C", "D"], default="C")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--cases")
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--budget-cny", type=float)
    parser.add_argument(
        "--prior-case-cny",
        type=float,
        default=PRIOR_CASE_COST,
        help="per-case cost assumed by the projection until real costs are known",
    )
    parser.add_argument("--decider", choices=["gemini", "deepseek"], default=None)
    parser.add_argument(
        "--decision-thinking", default="LOW", help="Gemini decider thinking level (or NONE)"
    )
    parser.add_argument(
        "--answer-thinking",
        default=None,
        help="Gemini answer thinking: auto (HIGH with images/web/complex, else "
        "--simple-thinking), default (= config), a level, or NONE",
    )
    parser.add_argument(
        "--simple-thinking", default="LOW", help="auto mode level for plain answers"
    )
    parser.add_argument(
        "--no-web", dest="web", action="store_false", help="do not call web search/fetch"
    )
    parser.add_argument(
        "--no-fact-rule", action="store_true", help="disable the chat->web fact-check code rule"
    )
    parser.add_argument(
        "--no-subject-rule",
        action="store_true",
        help="disable the D-17 subject verification search for subject_uncertain decisions",
    )
    parser.add_argument(
        "--no-gemini-supplement",
        action="store_true",
        help="disable the D-18 Gemini supplement search (default follows [web] gemini_supplement)",
    )
    parser.add_argument(
        "--force-gemini-supplement",
        action="store_true",
        help="experiment: run the Gemini supplement on every web-route run (measurement only)",
    )
    parser.add_argument("--jev-shadow", action="store_true")
    parser.add_argument(
        "--decision-only", action="store_true", help="route decision only, no answer call"
    )
    parser.add_argument("--knowledge", type=Path)
    parser.add_argument("--media-root", type=Path, default=ROOT / "tests/replay/raw")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--rerun", action="store_true", help="ignore finished cases in --out")
    parser.add_argument("--live", action="store_true", help="actually call models")
    parser.add_argument(
        "--no-account-book",
        dest="account_book",
        action="store_false",
        help="do not append calls to the account-wide usage book",
    )
    parser.add_argument("--score-only", action="store_true")
    args = parser.parse_args(argv)

    batch = args.batch
    sidecar = args.annotations or (batch / "annotations.draft.json")
    reviewed = args.reviewed or (batch / "annotations.reviewed.json")
    cases = load_cases(
        batch, sidecar if sidecar.exists() else None, reviewed if reviewed.exists() else None
    )
    cases = [c for c in cases if c["annotations"].get("status") in ("draft", "reviewed")]
    selected = pick(cases, args.cases, args.limit)
    reviewed_ids = {c["case_id"] for c in cases if c["annotations"]["status"] == "reviewed"}
    out = (args.out or ROOT / f"data/live/{datetime.now():%Y%m%d-%H%M%S}").resolve()
    if not out.is_relative_to((ROOT / "data").resolve()):
        parser.error("--out must be under data/ (git-ignored; outputs are private)")
    meta = {
        "batch": batch.name,
        "strategy": args.strategy,
        "decider": args.decider,
        "decision_thinking": args.decision_thinking,
        "answer_thinking": args.answer_thinking,
        "jev_shadow": args.jev_shadow,
        "decision_only": args.decision_only,
        "budget_cny": args.budget_cny,
        "simple_thinking": args.simple_thinking,
        "web": args.web and not args.decision_only,
        "fact_rule": not args.no_fact_rule,
        "subject_rule": not args.no_subject_rule,
        "gemini_supplement": not args.no_gemini_supplement,
        "force_gemini_supplement": args.force_gemini_supplement,
    }
    if args.score_only:
        if (out / "meta.json").exists():  # keep the run's own settings and ledger
            meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
        summary = write_reports(out, cases, meta)
        print(json.dumps(summary["new_pipeline"], ensure_ascii=False, indent=2))
        return 0
    (out / "cases").mkdir(parents=True, exist_ok=True)
    finished = set() if args.rerun else {p.stem for p in (out / "cases").glob("*.json")}
    todo = [c for c in selected if c["case_id"] not in finished]
    prior_rows = [r for r in load_rows(out) if r["case_id"] in {c["case_id"] for c in selected}]
    prior_spent = sum(r.get("cost_cny") or 0 for r in prior_rows) if not args.rerun else 0.0
    print(
        f"cases selected={len(selected)} todo={len(todo)} prior_spent_cny={prior_spent:.4f} "
        f"strategy={args.strategy} decider={args.decider} out={out.relative_to(ROOT)}"
    )
    if not args.live:
        print(f"dry run: projected <= {len(todo) * args.prior_case_cny:.2f} CNY at prior estimate")
        return 0
    if args.budget_cny is None or args.budget_cny <= 0:
        parser.error("--live requires a positive --budget-cny")
    settings = load_settings(args.settings)
    try:
        config = build_config(args, settings)
    except ConfigError as exc:
        parser.error(str(exc))  # field names only, never values
    config.ledger.budget_cny = max(0.0, args.budget_cny - prior_spent)
    meta["gemini_supplement"] = config.web_supplement is not None
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    costs = [r["cost_cny"] for r in prior_rows if r.get("cost_cny")]
    stopped = None
    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool:
        pending: dict = {}
        queue = list(todo)
        while queue or pending:
            while queue and len(pending) < max(1, args.concurrency):
                mean = sum(costs) / len(costs) if costs else args.prior_case_cny
                unfinished = len(queue) + len(pending)
                projected = prior_spent + config.ledger.spent + mean * unfinished
                if projected > args.budget_cny:
                    stopped = (
                        f"projection {projected:.3f} CNY > budget {args.budget_cny} "
                        f"with {unfinished} unfinished"
                    )
                    queue.clear()
                    break
                case = queue.pop(0)
                pending[pool.submit(run_one, case, config, args.strategy, reviewed_ids)] = case
            if not pending:
                break
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                case = pending.pop(future)
                row = future.result()
                (out / "cases" / f"{case['case_id']}.json").write_text(
                    json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                if row.get("cost_cny"):
                    costs.append(row["cost_cny"])
                route = ((row.get("score") or {}).get("checks") or {}).get("route") or {}
                print(
                    f"{case['case_id']} {row['status']} state={row.get('anchor_state')} "
                    f"route={route.get('actual')}/{route.get('expected')} "
                    f"cost={row.get('cost_cny')} wall={row.get('wall_s')}s "
                    f"spent={prior_spent + config.ledger.spent:.4f}",
                    flush=True,
                )
                if row.get("anchor_reason") == "budget_exceeded":
                    stopped = stopped or "a model call was refused by the budget"
                    queue.clear()
    if config.ledger.account is not None:
        config.ledger.account.close()
    meta.update(stopped=stopped, ledger=config.ledger.totals())
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    summary = write_reports(out, cases, meta)
    print(json.dumps(summary["new_pipeline"], ensure_ascii=False, indent=2))
    if stopped:
        print(f"STOPPED: {stopped}")
    print(f"spent_this_session_cny={config.ledger.spent:.4f} out={out.relative_to(ROOT)}")
    return 2 if stopped else 0


if __name__ == "__main__":
    raise SystemExit(main())
