"""Account usage book report (B2): aggregates only, read-only unless adding an opening balance.

    python scripts/billing_report.py [--settings config/settings.toml] [--db <book.sqlite3>] [--json]
    python scripts/billing_report.py add-opening --channel vertex --period 2026-11
        --list-cost 3.2 --currency USD [--google-queries 120] [--note "console export 2026-11-14"]

The summary groups calls by Asia/Shanghai calendar month x channel x service and shows calls, tokens, Google search queries actually issued,
list cost per currency (original price, before any credit), the CNY estimate and the calls
whose Google query count is still unknown. Opening balances ("接入前已发生用量") are listed
separately and are not added into the usage rows. A last table gives, per billing cycle
(`[billing] billing_cycle_day`), the Vertex Google search queries used (opening balances
included), the free quota (`google_free_queries_per_month`) and the estimated chargeable
queries beyond it. Everything is an estimate from recorded
usage, not a Cloud Billing confirmed spend. No row content, prompts or ids are printed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from group_agent.billing import GOOGLE_QUOTA_CHANNEL, AccountBook  # noqa: E402
from group_agent.config import BillingSpec, ConfigError, load_settings  # noqa: E402


def money(values: dict) -> str:
    return " + ".join(f"{v:.4f} {k}" for k, v in sorted(values.items())) or "-"


def render(summary: dict) -> str:
    lines = ["Account usage book (estimates from recorded usage; Asia/Shanghai months)"]
    header = (
        f"{'month':<8}{'channel':<26}{'service':<9}{'calls':>6}{'in_tok':>10}{'out_tok':>10}"
        f"{'g_queries':>10}{'unrec':>6}  {'list cost':<22}{'cny_est':>10}"
    )
    lines += [header, "-" * len(header)]
    for r in summary["usage"]:
        lines.append(
            f"{r['month']:<8}{r['channel']:<26}{r['service']:<9}{r['calls']:>6}{r['input']:>10}"
            f"{r['output']:>10}{r['google_queries']:>10}{r['reconcile_calls']:>6}  "
            f"{money(r['list_cost']):<22}{r['cost_cny']:>10.4f}"
        )
    if not summary["usage"]:
        lines.append("(no usage rows)")
    lines.append("")
    lines.append("Opening balances (usage before the book existed)")
    for r in summary["opening"]:
        lines.append(
            f"{r['month']:<8}{r['channel']:<26}entries={r['entries']} "
            f"google_queries={r['google_queries']} list cost {money(r['list_cost'])}"
        )
    if not summary["opening"]:
        lines.append("(none)")
    quota = summary.get("google_quota")
    if quota:
        lines.append("")
        lines.append(
            f"Google search free quota, channel {quota['channel']} (estimate: cycle starts on day "
            f"{quota['cycle_day']} Asia/Shanghai; Google counts per Pacific billing month)"
        )
        header = (
            f"{'cycle start':<13}{'used':>8}{'free quota':>12}{'chargeable':>12}{'free taken':>12}"
        )
        lines += [header, "-" * len(header)]
        for r in quota["cycles"]:
            lines.append(
                f"{r['cycle_start']:<13}{r['used']:>8}{quota['free_quota']:>12}"
                f"{r['chargeable']:>12}{r['free_recorded']:>12}"
            )
        if not quota["cycles"]:
            lines.append("(no Google search queries)")
    return "\n".join(lines)


def quota_summary(book: AccountBook, free_quota: int, cycle_day: int) -> dict:
    """Per billing cycle: queries used, the free quota and the estimated chargeable queries
    (used beyond the quota; they are billed at the list price)."""
    cycles = book.google_cycles(GOOGLE_QUOTA_CHANNEL, cycle_day)
    for r in cycles:
        r["chargeable"] = max(0, r["used"] - free_quota)
    return {
        "channel": GOOGLE_QUOTA_CHANNEL,
        "cycle_day": cycle_day,
        "free_quota": free_quota,
        "cycles": cycles,
    }


def billing_settings(args: argparse.Namespace) -> tuple[int, int]:
    """(free queries per month, cycle day) from settings; defaults when unreadable."""
    default = BillingSpec()
    try:
        spec = load_settings(args.settings).billing
    except (ConfigError, OSError):
        return default.google_free_queries_per_month, default.billing_cycle_day
    return spec.google_free_queries_per_month, spec.billing_cycle_day


def book_path(args: argparse.Namespace) -> Path:
    if args.db is not None:
        return args.db
    try:
        return load_settings(args.settings).billing.db_path
    except (ConfigError, OSError) as exc:
        raise SystemExit(f"settings: {type(exc).__name__} (pass --db)") from None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--settings", type=Path, default=ROOT / "config/settings.toml")
    parser.add_argument("--db", type=Path, help="account book path (default: [billing] db_path)")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("summary", help="monthly summary (default)")
    parser.add_argument("--json", action="store_true", help="summary as JSON")
    add = sub.add_parser("add-opening", help="add a manual opening-balance entry")
    add.add_argument("--channel", required=True)
    add.add_argument("--period", required=True, help="calendar month YYYY-MM")
    add.add_argument("--list-cost", type=float, default=0.0)
    add.add_argument("--currency", choices=("CNY", "USD"), required=True)
    add.add_argument("--google-queries", type=int, default=0)
    add.add_argument("--note", default="")
    args = parser.parse_args(argv)
    path = book_path(args)

    if args.command == "add-opening":
        book = AccountBook(path)
        try:
            entry = book.add_opening(
                args.channel,
                args.period,
                args.list_cost,
                args.currency,
                google_queries=args.google_queries,
                note=args.note,
            )
        except ValueError as exc:
            parser.error(str(exc))
        finally:
            book.close()
        print(f"opening balance entry {entry} added")
        return 0

    if not path.is_file():
        print("no account book yet")
        return 0
    book = AccountBook(path, read_only=True)
    try:
        free_quota, cycle_day = billing_settings(args)
        summary = {
            "usage": book.monthly_summary(),
            "opening": book.opening_summary(),
            "google_quota": quota_summary(book, free_quota, cycle_day),
        }
    finally:
        book.close()
    print(json.dumps(summary, ensure_ascii=False, indent=2) if args.json else render(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
