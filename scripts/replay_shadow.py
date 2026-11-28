"""Shadow-request replay (D-21 quality gate): answer every reply request of the shadow trial
again with the current code, on a copy of the shadow store cut at the question.

The shadow store is opened read-only (sqlite `mode=ro`) and copied with the sqlite backup
API to a work database (default `data/serve/replay-<name>.sqlite3`, recreated each run);
nothing is written to the shadow store. In the copy the shadow trial's own reply runs (and
their answer rows and context snapshots) are removed so each question can get a fresh run.

Each question goes through `Orchestrator.generate_for` (the regeneration path:
`RunFlow`, context strategy C, the store's visible history before the question, real
models through `serve.build_live_config`), with a host clock that starts at the question's
time so retention and the prompt's date see the question moment. The answer is handed
back, not recorded; there is no gateway and no send capability here (`sent_to_qq` is
always 0). Group info is unavailable (no gateway): the agent records it as such.

Output (ignored directory `data/serve/replay/<name>/`):
- `pairs.jsonl`: per request `id8` (last 8 chars of the platform message id), the question
  and quote summaries, the old bot's (AstrBot) reply via `scripts/shadow_report.build`, the
  shadow-trial answer, this run's answer, routes, fallback details, latency and cost;
- `summary.json`: counts, failures, answer/decision sources, fallback and degradation
  counts (run `degraded` labels), latency p50/p95, cost, mean answer lengths (no content).

Usage:
    python scripts/replay_shadow.py --name persona-v1 --budget-cny 1.0 [--limit 5]
        [--ids a1b2c3d4,...] [--concurrency 2] [--out data/serve/replay/<name>]
        [--serve-db data/serve/shadow-gen.sqlite3] [--media-dir <db stem>.media]
        [--settings config/settings.toml] [--no-web] [--no-account-book]

Every call is also appended to the account-wide usage book (`[billing] db_path`, source
"replay"); `--no-account-book` skips that.
"""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from contextvars import ContextVar
import json
from pathlib import Path
import sqlite3
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from group_agent.admission import AccessPolicy  # noqa: E402
from group_agent.billing import open_account_book, open_google_quota  # noqa: E402
from group_agent.config import load_settings  # noqa: E402
from group_agent.live_agent import LiveAgent  # noqa: E402
from group_agent.orchestrator import Orchestrator  # noqa: E402
from group_agent.policy import RuntimeLimits  # noqa: E402
from group_agent.runflow import CURRENT_RUN  # noqa: E402
from group_agent.serve import build_live_config, latest_knowledge, model_ids  # noqa: E402
from group_agent.store import Store, _message  # noqa: E402
from group_agent.usage import UsageLedger  # noqa: E402
from shadow_report import build, connect_ro, fmt_time, percentile  # noqa: E402

# The copy's usage ledger already holds the shadow trial's spending; the replay is limited
# by --budget-cny instead of the daily cap (which would put runs into degraded mode).
REPLAY_DAILY_CAP_CNY = 1000.0


def copy_store(src: Path, dst: Path) -> None:
    """Read-only backup of the shadow store into a fresh work database."""
    dst = dst.resolve()
    if dst == src.resolve():
        raise SystemExit("work database must differ from the shadow store")
    if not dst.name.startswith("replay-") or dst.suffix != ".sqlite3":
        raise SystemExit("work database name must be replay-<name>.sqlite3")
    for suffix in ("", "-wal", "-shm"):
        Path(str(dst) + suffix).unlink(missing_ok=True)
    dst.parent.mkdir(parents=True, exist_ok=True)
    source = connect_ro(src)
    target = sqlite3.connect(dst)
    try:
        source.backup(target)
    finally:
        source.close()
        target.close()


def clear_reply_runs(path: Path) -> list[dict]:
    """Remove the shadow trial's reply runs from the copy; returns them (run id, seq...)."""
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    try:
        runs = [
            dict(r)
            for r in con.execute(
                "SELECT r.run_id, r.message_seq, r.message_id, r.route, r.state, r.error_kind,"
                " r.detail, o.text AS answer FROM runs r LEFT JOIN outbound o"
                " ON o.run_id=r.run_id AND o.kind='answer' WHERE r.purpose='reply'"
            )
        ]
        ids = [(r["run_id"],) for r in runs]
        con.executemany("DELETE FROM outbound WHERE run_id=?", ids)
        con.executemany("DELETE FROM context_snapshots WHERE run_id=?", ids)
        con.executemany("DELETE FROM runs WHERE run_id=?", ids)
        con.commit()
    finally:
        con.close()
    return runs


def _detail(raw: str | None) -> dict:
    try:
        return json.loads(raw or "{}") or {}
    except ValueError:
        return {}


# Host clock per run: (question time, wall start). Set in the worker before
# `generate_for`; the run task and its callbacks inherit the context.
RUN_CLOCK: ContextVar[tuple[float, float] | None] = ContextVar("RUN_CLOCK", default=None)


def replay_clock() -> float:
    """Wall time shifted so each run's host clock starts at its question's time."""
    base = RUN_CLOCK.get()
    if base is None:
        return time.time()
    asked, started = base
    return asked + (time.time() - started)


async def replay_one(store, orchestrator, settings, run: dict) -> dict:
    row = store.message_row(run["message_seq"])
    question = _message(row)
    asked = row["platform_time"] if row["platform_time"] is not None else row["received_at"]
    started = time.time()
    RUN_CLOCK.set((float(asked), started))
    run_id, text = await orchestrator.generate_for(
        question, run["message_seq"], run_seq=run["message_seq"]
    )
    task = orchestrator._run_tasks.get(run_id)
    if task is not None:  # the run detail is written when the task finishes
        await asyncio.wait([task])
    for _ in range(3):
        await asyncio.sleep(0)
    latency = round(time.time() - started, 2)
    new = store.con.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
    detail = _detail(new["detail"] if new else None)
    live = detail.get("live") or {}
    if new is not None and new["error_kind"] == "fallback_reply":
        text = settings.fallback_reply
    cost = store.con.execute(
        "SELECT COALESCE(SUM(cost_cny), 0) FROM usage_ledger WHERE run_id=?", (run_id,)
    ).fetchone()[0]
    ok = [a for a in live.get("decision_attempts") or [] if a.endswith(":ok")]
    decision_source = ok[0].split(":")[0] if ok else None
    if live.get("decision_source") == "lightweight":
        decision_source = "lightweight"
    if (live.get("decision") or {}).get("fallback") == "rule":
        decision_source = "rule"
    return {
        "run_id": run_id,
        "text": text,
        "state": new["state"] if new else None,
        "error_kind": new["error_kind"] if new else None,
        "route": new["route"] if new else None,
        "intended_route": detail.get("intended_route"),
        "answer_source": live.get("answer_source"),
        "answer_attempts": live.get("answer_attempts"),
        "answer_fallback_used": bool(live.get("answer_fallback_used")),
        "answer_failed": live.get("answer_failed"),
        "decision_fallback": (live.get("decision") or {}).get("fallback"),
        "decision_source": decision_source,
        "web": live.get("web") or {},
        "tool_calls": detail.get("tool_calls") or [],
        "degraded": detail.get("degraded") or [],
        "latency_s": latency,
        "cost_cny": round(cost or 0.0, 6),
    }


def _mean_len(texts: list[str | None]) -> float | None:
    values = [len(t) for t in texts if t]
    return round(statistics.mean(values), 1) if values else None


def summarize(lines: list[dict], spent: float, meta: dict) -> dict:
    done = [
        x
        for x in lines
        if x["new"] and x["state"] == "succeeded" and x["error_kind"] != "fallback_reply"
    ]
    failed = [x for x in lines if x not in done]
    latencies = [x["latency_s"] for x in lines if x["latency_s"] is not None]
    both = [x for x in done if x["old"] and x["shadow_new"]]
    return {
        **meta,
        "requests": len(lines),
        "completed": len(done),
        "failed": len(failed),
        "failure_kinds": dict(Counter(x["error_kind"] or x["state"] or "none" for x in failed)),
        "fallback_reply": sum(1 for x in lines if x["error_kind"] == "fallback_reply"),
        "answer_sources": dict(Counter(x["answer_source"] or "none" for x in lines)),
        "answer_fallback_used": sum(1 for x in lines if x["answer_fallback_used"]),
        # Model family that wrote the answer ("deepseek:no_images" counts as deepseek).
        "answer_models": dict(Counter((x["answer_source"] or "none").split(":")[0] for x in lines)),
        "decision_sources": dict(Counter(x.get("decision_source") or "none" for x in lines)),
        "decision_rule_fallback": sum(1 for x in lines if x["decision_fallback"] == "rule"),
        # Runs with any degradation, and per label (answer_gemini_skipped,
        # answer_fallback:<source>, decision_gemini_skipped, decision_fallback:<model>, ...).
        "degraded_runs": sum(1 for x in lines if x.get("degraded")),
        "degraded": dict(Counter(label for x in lines for label in x.get("degraded") or [])),
        "routes": dict(Counter(x["route"] or "none" for x in lines)),
        "route_changed_vs_shadow": sum(
            1 for x in lines if x["route"] and x["shadow_route"] and x["route"] != x["shadow_route"]
        ),
        "latency_s": {
            "p50": percentile(latencies, 0.5),
            "p95": percentile(latencies, 0.95),
            "max": max(latencies) if latencies else None,
        },
        "cost_cny": round(spent, 4),
        "mean_chars": {
            "new": _mean_len([x["new"] for x in done]),
            "shadow_new": _mean_len([x["shadow_new"] for x in done]),
            "old": _mean_len([x["old"] for x in done]),
        },
        "mean_chars_where_all_three": {
            "pairs": len(both),
            "new": _mean_len([x["new"] for x in both]),
            "shadow_new": _mean_len([x["shadow_new"] for x in both]),
            "old": _mean_len([x["old"] for x in both]),
        },
        "sent_to_qq": 0,
    }


async def main_async(args) -> int:
    serve_db = Path(args.serve_db)
    name = args.name
    out = Path(args.out or f"data/serve/replay/{name}")
    work_db = Path(args.work_db or f"data/serve/replay-{name}.sqlite3")
    media_root = Path(args.media_dir or serve_db.with_name(serve_db.stem + ".media"))
    settings = load_settings(Path(args.settings))

    # Old (AstrBot) replies and the shadow answers, paired by scripts/shadow_report.
    items, _ = build(serve_db, None, snowluma_root=Path(args.snowluma_root), since=None, until=None)
    by_run = {i["run_id"]: i for i in items}

    copy_store(serve_db, work_db)
    runs = clear_reply_runs(work_db)
    runs.sort(key=lambda r: (by_run.get(r["run_id"], {}).get("asked_at") or 0, r["run_id"]))
    if args.ids:
        wanted = {s.strip() for s in args.ids.split(",") if s.strip()}
        runs = [r for r in runs if str(r["message_id"])[-8:] in wanted]
    if args.limit:
        runs = runs[: args.limit]

    store = Store(work_db)
    # Replay spends the same account as the bot: the account book is separate from the
    # (copied) work store, so copying it never copies account usage.
    book = open_account_book(
        settings, "replay", enabled=not args.no_account_book, run_id=CURRENT_RUN.get
    )
    usage = UsageLedger(
        store,
        daily_cap_cny=REPLAY_DAILY_CAP_CNY,
        budget_cny=args.budget_cny,
        model_ids=model_ids(settings, False),
        account=book,
        source="replay" if book is not None else None,
        quota=open_google_quota(settings, book),
    )
    config = build_live_config(
        settings,
        ledger=usage,
        media_root=media_root,
        dry_models=False,
        web=not args.no_web,
        group_info=None,
        knowledge_path=Path(args.knowledge) if args.knowledge else latest_knowledge(),
    )
    if args.decision_mode:
        config.decision_mode = args.decision_mode
    if args.answer_thinking:
        config.answer_thinking = args.answer_thinking
    evidence_dir = out / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)

    def capture_evidence(record: dict) -> None:
        # Private replay output only; regular serve diagnostics remain content-free.
        (evidence_dir / f"{record['id8']}.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    config.evidence_sink = capture_evidence
    # One host for all runs (a second Orchestrator would mark the first one's active runs
    # interrupted at start-up); `generate_for` = the regeneration path.
    orchestrator = Orchestrator(
        store,
        AccessPolicy.from_settings(settings),
        settings,
        agent_factory=lambda host: LiveAgent(host, config),
        usage=usage,
        limits=RuntimeLimits(deadline_seconds=settings.deadline_seconds),
        clock=replay_clock,
        prepare_wait_seconds=0.0,
    )
    out.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(args.concurrency)
    lines: list[dict | None] = [None] * len(runs)
    stopped = {"budget": False}

    async def worker(index: int, run: dict) -> None:
        async with sem:
            if stopped["budget"] or usage.spent >= args.budget_cny:
                stopped["budget"] = True
                result = {"state": "skipped", "error_kind": "budget_reached"}
            else:
                try:
                    result = await replay_one(store, orchestrator, settings, run)
                except Exception as exc:  # noqa: BLE001 - one bad item must not stop the run
                    result = {"state": "error", "error_kind": f"error:{type(exc).__name__}"}
            item = by_run.get(run["run_id"], {})
            shadow_detail = _detail(run["detail"])
            line = {
                "id8": str(run["message_id"])[-8:],
                "time": fmt_time(item.get("asked_at")),
                "question": item.get("question"),
                "quoted": item.get("quoted"),
                "old": (item.get("old") or {}).get("text"),
                "shadow_new": run["answer"],
                "new": result.get("text"),
                "shadow_route": run["route"],
                "shadow_state": [run["state"], run["error_kind"]],
                "shadow_source": (shadow_detail.get("live") or {}).get("answer_source"),
                "route": result.get("route"),
                "intended_route": result.get("intended_route"),
                "state": result.get("state"),
                "error_kind": result.get("error_kind"),
                "answer_source": result.get("answer_source"),
                "answer_attempts": result.get("answer_attempts"),
                "answer_fallback_used": result.get("answer_fallback_used", False),
                "answer_failed": result.get("answer_failed"),
                "decision_fallback": result.get("decision_fallback"),
                "decision_source": result.get("decision_source"),
                "web": result.get("web", {}),
                "tool_calls": result.get("tool_calls", []),
                "degraded": result.get("degraded", []),
                "latency_s": result.get("latency_s"),
                "cost_cny": result.get("cost_cny", 0.0),
                "sent_to_qq": 0,
            }
            lines[index] = line
            done = sum(1 for x in lines if x is not None)
            print(
                f"[{done}/{len(runs)}] {line['id8']} {line['state']} {line['route']}"
                f" {line['answer_source']} {line['latency_s']}s spent={usage.spent:.4f}",
                flush=True,
            )

    try:
        await asyncio.gather(*(worker(i, r) for i, r in enumerate(runs)))
    finally:
        await orchestrator.aclose()
        store.close()
        if book is not None:
            book.close()
    final = [x for x in lines if x is not None]
    (out / "pairs.jsonl").write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in final), encoding="utf-8"
    )
    meta = {
        "name": name,
        "budget_cny": args.budget_cny,
        "concurrency": args.concurrency,
        "web": not args.no_web,
        "budget_stopped": stopped["budget"],
        "shadow_reply_runs": len(by_run),
        "model_ids": {name: spec.model for name, spec in settings.models.items()},
        "model_roles": settings.model_roles,
        "answer_tool_loop": settings.answer_tool_loop,
        "decision_mode": getattr(config, "decision_mode", "model"),
        "answer_thinking": config.answer_thinking,
        "answer_same_model_retry": getattr(config, "answer_same_model_retry", None),
        **(
            {
                "search_provider": settings.web.search_provider,
                "search_model": settings.web.gemini_model,
            }
            if settings.web.search is None
            else {
                "web_bindings": {
                    key: getattr(settings.web, key)
                    for key in ("search", "search_fallback", "fetch", "fetch_fallback")
                }
            }
        ),
    }
    summary = summarize(final, usage.spent, meta)
    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--name", default="replay")
    p.add_argument("--serve-db", default="data/serve/shadow-gen.sqlite3")
    p.add_argument("--work-db", default=None)
    p.add_argument("--media-dir", default=None)
    p.add_argument("--settings", default="config/settings.toml")
    p.add_argument("--snowluma-root", default="data/snowluma")
    p.add_argument("--knowledge", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--ids", default="", help="comma-separated last 8 chars of message ids")
    p.add_argument("--budget-cny", type=float, required=True)
    p.add_argument("--concurrency", type=int, default=2)
    p.add_argument("--no-web", action="store_true")
    p.add_argument(
        "--no-account-book", action="store_true", help="do not append to the account usage book"
    )
    p.add_argument("--decision-mode", choices=("model", "lightweight"))
    p.add_argument("--answer-thinking", choices=("auto", "complex_only", "LOW", "HIGH"))
    args = p.parse_args(argv)
    if args.concurrency < 1 or args.concurrency > 3:
        p.error("--concurrency: 1-3 (the relay rate-limits around 4 parallel requests)")
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
