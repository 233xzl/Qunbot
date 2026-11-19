"""B3: monthly free Google Search grounding quota (Vertex), counted from the account book.

Tmp paths and fakes only; no network."""

import asyncio
from dataclasses import replace
import importlib.util
from pathlib import Path
import sqlite3
import sys
import threading

import pytest

from group_agent.billing import (
    ACCOUNT_SCHEMA_VERSION,
    AccountBook,
    GoogleFreeQuota,
    cycle_start,
    next_cycle_start,
    open_google_quota,
)
from group_agent.config import BillingSpec
from group_agent.models import BudgetExceeded, CallRecord, Ledger, Usage
from group_agent.store import Store
from group_agent.usage import UsageLedger
from test_gemini_search import GROUNDED, SPEC, FakeGenai, make_tool
from test_vertex_routing import FIXED, VERTEX_TABLES, write

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "billing_report_b3", ROOT / "scripts/billing_report.py"
)
report = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = report
spec.loader.exec_module(report)

NOV = 1_794_628_800.0  # 2026-11-14 12:00 Asia/Shanghai
DEC = NOV + 18 * 86400  # 2026-12-02 12:00
FEE_USD = 0.014
FEE = FEE_USD * 7.0  # CNY per query
TOKENS_CNY = 0.002


def search(queries: int | None, *, channel="vertex", service="search") -> CallRecord:
    """A settled Vertex search call at list price: tokens plus the per-query fee."""
    known = queries is not None
    record = CallRecord(
        task="web_search",
        model="gemini_search",
        mode="http",
        channel=channel,
        service=service,
        google_queries=queries,
        cost_cny=TOKENS_CNY + (queries or 0) * FEE,
        list_cost=(TOKENS_CNY / 7.0 + (queries or 0) * FEE_USD) if known else None,
        list_currency="USD",
    )
    record.usage = Usage(input=100, output=20, requests=1)
    return record


def make_quota(book, free=5, cycle_day=1, **kw) -> GoogleFreeQuota:
    return GoogleFreeQuota(
        book, free_per_month=free, cycle_day=cycle_day, fee_cny_per_query=FEE, **kw
    )


def rows(book):
    return book.con.execute("SELECT * FROM account_usage ORDER BY id").fetchall()


def test_queries_within_the_free_quota_cost_nothing_and_keep_list_cost(tmp_path):
    book = AccountBook(tmp_path / "a.sqlite3", clock=lambda: NOV)
    ledger = Ledger(None, account=book, source="serve", quota=make_quota(book, free=10))
    record = search(3)
    listed = record.list_cost
    ledger.settle(0.0, record)
    assert record.free_queries == 3
    assert record.cost_cny == pytest.approx(TOKENS_CNY)  # tokens stay, queries are free
    assert record.list_cost == listed and record.google_queries == 3
    (row,) = rows(book)
    assert (row["google_queries"], row["free_queries"]) == (3, 3)
    assert row["cost_cny"] == pytest.approx(TOKENS_CNY) and row["list_cost"] == pytest.approx(
        listed
    )
    assert ledger.spent == pytest.approx(TOKENS_CNY)
    book.close()


def test_a_call_crossing_the_boundary_is_split(tmp_path):
    book = AccountBook(tmp_path / "a.sqlite3", clock=lambda: NOV)
    quota = make_quota(book, free=5)
    ledger = Ledger(None, account=book, source="serve", quota=quota)
    first, second, third = search(3), search(4), search(2)
    for record in (first, second, third):
        ledger.settle(0.0, record)
    assert first.free_queries == 3 and first.cost_cny == pytest.approx(TOKENS_CNY)
    # 2 of 4 queries are still free; the other 2 pay the list price.
    assert second.free_queries == 2
    assert second.cost_cny == pytest.approx(TOKENS_CNY + 2 * FEE)
    assert third.free_queries == 0 and third.cost_cny == pytest.approx(TOKENS_CNY + 2 * FEE)
    assert [r["free_queries"] for r in rows(book)] == [3, 2, 0]
    assert quota.used() == 9 and quota.remaining() == 0
    book.close()


def test_quota_persists_across_reopen(tmp_path):
    path = tmp_path / "a.sqlite3"
    book = AccountBook(path, clock=lambda: NOV)
    Ledger(None, account=book, source="serve", quota=make_quota(book)).settle(0.0, search(3))
    book.close()
    book = AccountBook(path, clock=lambda: NOV)
    quota = make_quota(book)
    assert quota.used() == 3 and quota.remaining() == 2
    late = search(4)
    Ledger(None, account=book, source="serve", quota=quota).settle(0.0, late)
    assert late.free_queries == 2
    book.close()


def test_replay_probe_and_opening_balance_count_toward_the_same_quota(tmp_path):
    path = tmp_path / "a.sqlite3"
    book = AccountBook(path, clock=lambda: NOV)
    quota = make_quota(book, free=10)
    for source in ("replay", "probe"):  # other processes: their own Ledger, same book
        other = Ledger(None, account=book, source=source, quota=make_quota(book, free=10))
        other.settle(0.0, search(2))
    book.add_opening("vertex", "2026-11", 0.0, "USD", google_queries=3, note="before the book")
    book.add_opening("vertex", "2026-10", 0.0, "USD", google_queries=900)  # another cycle
    book.add_opening("relay", "2026-11", 0.0, "USD", google_queries=900)  # another channel
    serve = Ledger(None, account=book, source="serve", quota=quota)
    record = search(5)
    serve.settle(0.0, record)
    assert quota.used() == 12  # 2 + 2 + 3 opening + 5
    assert record.free_queries == 3  # 10 - (2 + 2 + 3) left
    # The report sees the same cycle (opening balance included).
    cycles = {c["cycle_start"]: c for c in book.google_cycles("vertex", 1)}
    assert cycles["2026-11-01"]["used"] == 12 and cycles["2026-10-01"]["used"] == 900
    book.close()


def test_cycle_rollover_resets_the_quota(tmp_path):
    clock = [NOV]
    book = AccountBook(tmp_path / "a.sqlite3", clock=lambda: clock[0])
    quota = make_quota(book, free=5)
    ledger = Ledger(None, account=book, source="serve", quota=quota)
    ledger.settle(0.0, search(5))
    assert quota.remaining() == 0
    clock[0] = DEC
    assert quota.remaining() == 5 and quota.used() == 0  # December cycle, no restart needed
    record = search(4)
    ledger.settle(0.0, record)
    assert record.free_queries == 4 and quota.remaining() == 1
    book.close()
    # A custom cycle day: 14 Nov is still in the cycle that started on 15 Oct.
    book = AccountBook(tmp_path / "b.sqlite3", clock=lambda: clock[0])
    clock[0] = NOV
    quota = make_quota(book, free=5, cycle_day=15)
    ledger = Ledger(None, account=book, source="serve", quota=quota)
    ledger.settle(0.0, search(5))
    assert quota.cycle.isoformat() == "2026-10-15" and quota.remaining() == 0
    clock[0] = NOV + 86400  # 15 Nov
    assert quota.cycle.isoformat() == "2026-11-15" and quota.remaining() == 5
    book.close()


def test_cycle_helpers():
    from datetime import date

    assert cycle_start(date(2026, 11, 14), 1) == date(2026, 11, 1)
    assert cycle_start(date(2026, 11, 14), 15) == date(2026, 10, 15)
    assert cycle_start(date(2026, 1, 3), 15) == date(2025, 12, 15)
    assert next_cycle_start(date(2026, 12, 1)) == date(2027, 1, 1)
    assert next_cycle_start(date(2026, 10, 15)) == date(2026, 11, 15)


def test_without_a_quota_or_book_calls_keep_list_pricing(tmp_path):
    record = search(3)
    plain = Ledger(None)
    plain.settle(0.0, record)
    assert record.free_queries is None and record.cost_cny == pytest.approx(TOKENS_CNY + 3 * FEE)
    book = AccountBook(tmp_path / "a.sqlite3", clock=lambda: NOV)
    booked = search(3)  # a book without a counter: still list price
    Ledger(None, account=book, source="serve").settle(0.0, booked)
    assert booked.free_queries is None and booked.cost_cny == pytest.approx(TOKENS_CNY + 3 * FEE)
    assert rows(book)[0]["free_queries"] is None
    book.close()
    assert open_google_quota(write_settings(tmp_path), None) is None  # --no-account-book, dry runs


def test_other_channels_and_unknown_query_counts_are_untouched(tmp_path):
    book = AccountBook(tmp_path / "a.sqlite3", clock=lambda: NOV)
    quota = make_quota(book, free=100)
    ledger = Ledger(None, account=book, source="serve", quota=quota)
    relay, unknown, fetch = search(3, channel="relay"), search(None), search(3, service="fetch")
    zero = search(0)
    for record in (relay, unknown, fetch, zero):
        ledger.settle(0.0, record)
    assert relay.free_queries is None and relay.cost_cny == pytest.approx(TOKENS_CNY + 3 * FEE)
    assert unknown.free_queries is None and unknown.cost_cny == pytest.approx(TOKENS_CNY)
    assert fetch.free_queries is None and zero.free_queries is None
    assert quota.used() == 0  # relay queries never use the Vertex quota
    book.close()


def test_daily_gate_no_longer_degrades_from_free_queries(tmp_path):
    def day_of_searches(quota, book):
        ledger = UsageLedger(
            Store(), daily_cap_cny=3.0, clock=lambda: NOV, account=book, source="serve", quota=quota
        )
        for _ in range(30):  # 120 queries: CNY 11.8 at list price against a 3.0 cap
            ledger.settle(0.0, search(4))
        return ledger

    list_priced = day_of_searches(None, None)
    assert list_priced.cost_mode(False) == "rejected"
    book = AccountBook(tmp_path / "a.sqlite3", clock=lambda: NOV)
    free = day_of_searches(make_quota(book, free=5000), book)
    assert free.cost_mode(False) == "normal" and free.cost_mode(True) == "normal"
    assert free.daily_spent() == pytest.approx(30 * TOKENS_CNY)
    # Past the quota the fee comes back: 120 free, then 4 chargeable queries per call.
    over = make_quota(book, free=120)
    ledger = UsageLedger(
        Store(), daily_cap_cny=3.0, clock=lambda: NOV, account=book, source="serve", quota=over
    )
    ledger.settle(0.0, search(4))
    assert ledger.daily_spent() == pytest.approx(TOKENS_CNY + 4 * FEE)
    book.close()


def test_concurrent_settles_never_both_take_the_last_free_queries(tmp_path):
    book = AccountBook(tmp_path / "a.sqlite3", clock=lambda: NOV)
    quota = make_quota(book, free=5)
    ledger = Ledger(None, account=book, source="serve", quota=quota)
    records = [search(1) for _ in range(12)]
    start, errors = threading.Barrier(len(records)), []

    def worker(record):
        try:
            start.wait()
            ledger.settle(0.0, record)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(r,)) for r in records]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert sum(r.free_queries for r in records) == 5  # exactly the quota, never more
    assert sum(r["free_queries"] for r in rows(book)) == 5 and quota.used() == 12
    assert sum(1 for r in records if r.cost_cny == pytest.approx(TOKENS_CNY + FEE)) == 7
    book.close()


def web_client(fake):
    from group_agent.gemini_search import GeminiSearchClient

    return GeminiSearchClient(
        replace(SPEC, channel="vertex", price_currency="USD", cny_per_unit=7.0),
        None,
        model="m",
        client_factory=lambda t: fake,
        fee_per_query=FEE,
        fee_list_per_query=FEE_USD,
        fee_currency="USD",
    )


def test_reservation_assumes_free_queries_while_the_quota_covers_the_estimate(tmp_path):
    client = web_client(FakeGenai(GROUNDED))
    token_only, listed = client.search_estimate(10), client.search_estimate(3)
    assert listed == pytest.approx(client.search_estimate_cny)
    assert listed - token_only == pytest.approx(4 * FEE) and client.search_estimate(4) == token_only
    cap = (token_only + listed) / 2  # a budget the free estimate fits and the list one does not
    book = AccountBook(tmp_path / "a.sqlite3", clock=lambda: NOV)

    # Plenty of free queries: the search is not blocked by a fee it would not be charged.
    quota = make_quota(book, free=10)
    tool = make_tool(client, ledger=Ledger(cap, quota=quota))
    assert asyncio.run(tool.search("python"))
    record = tool.records[0]
    assert record.google_queries == 2 and record.free_queries == 2
    assert record.cost_cny < token_only

    # Fewer than 4 free queries left: reserved at list price, so the same budget refuses.
    low = make_quota(book, free=3)
    with pytest.raises(BudgetExceeded):
        asyncio.run(make_tool(client, ledger=Ledger(cap, quota=low)).search("python"))
    # Without a quota the behaviour is unchanged.
    with pytest.raises(BudgetExceeded):
        asyncio.run(make_tool(client, ledger=Ledger(cap)).search("python"))
    book.close()


def test_account_book_v1_migrates_to_v2_and_old_rows_have_null(tmp_path):
    path = tmp_path / "a.sqlite3"
    book = AccountBook(path, clock=lambda: NOV)
    book.append(search(2), source="serve")
    book.close()
    with sqlite3.connect(path) as con:  # turn it into a v1 file
        con.execute("DROP INDEX account_usage_channel_day")
        con.execute("ALTER TABLE account_usage DROP COLUMN free_queries")
        con.execute("PRAGMA user_version=1")
    with pytest.raises(sqlite3.OperationalError):
        sqlite3.connect(path).execute("SELECT free_queries FROM account_usage")
    old = AccountBook(path, read_only=True)  # a report never migrates
    assert not old.has_free_queries and old.google_cycles("vertex", 1)[0]["used"] == 2
    old.close()
    book = AccountBook(path, clock=lambda: NOV)
    try:
        assert book.con.execute("PRAGMA user_version").fetchone()[0] == ACCOUNT_SCHEMA_VERSION == 2
        assert rows(book)[0]["free_queries"] is None and rows(book)[0]["google_queries"] == 2
        new = search(1)
        Ledger(None, account=book, source="serve", quota=make_quota(book)).settle(0.0, new)
        assert rows(book)[1]["free_queries"] == 1
    finally:
        book.close()


def write_settings(tmp_path, billing=""):
    tmp_path.mkdir(parents=True, exist_ok=True)
    return write(tmp_path, billing + VERTEX_TABLES + FIXED)


def test_billing_config_defaults_and_validation(tmp_path):
    default = BillingSpec()
    assert (default.google_free_queries_per_month, default.billing_cycle_day) == (5000, 1)
    settings = write_settings(tmp_path)
    assert settings.billing.google_free_queries_per_month == 5000
    assert settings.billing.billing_cycle_day == 1
    custom = write_settings(
        tmp_path / "c", "[billing]\ngoogle_free_queries_per_month = 0\nbilling_cycle_day = 15\n"
    )
    assert (custom.billing.google_free_queries_per_month, custom.billing.billing_cycle_day) == (
        0,
        15,
    )
    from group_agent.config import ConfigError

    for bad in (
        "google_free_queries_per_month = -1",
        "google_free_queries_per_month = 1.5",
        "billing_cycle_day = 0",
        "billing_cycle_day = 29",
        "billing_cycle_day = true",
    ):
        with pytest.raises(ConfigError, match="billing"):
            write_settings(tmp_path / "d", f"[billing]\n{bad}\n")
    book = AccountBook(tmp_path / "a.sqlite3", clock=lambda: NOV)
    quota = open_google_quota(settings, book)
    assert quota.free_per_month == 5000 and quota.fee_cny_per_query == pytest.approx(
        settings.billing.google_search_usd_per_1000 / 1000 * settings.billing.usd_cny
    )
    book.close()


def test_report_shows_used_free_and_chargeable_per_cycle(tmp_path):
    clock = [NOV]
    book = AccountBook(tmp_path / "a.sqlite3", clock=lambda: clock[0])
    quota = make_quota(book, free=5)
    ledger = Ledger(None, account=book, source="serve", quota=quota)
    ledger.settle(0.0, search(4))
    ledger.settle(0.0, search(4))
    clock[0] = DEC
    ledger.settle(0.0, search(2))
    book.add_opening("vertex", "2026-12", 0.0, "USD", google_queries=1)
    summary = report.quota_summary(book, 5, 1)
    assert summary["cycles"] == [
        {
            "cycle_start": "2026-11-01",
            "used": 8,
            "opening": 0,
            "free_recorded": 5,
            "chargeable": 3,
        },
        {
            "cycle_start": "2026-12-01",
            "used": 3,
            "opening": 1,
            "free_recorded": 2,
            "chargeable": 0,
        },
    ]
    text = report.render({"usage": [], "opening": [], "google_quota": summary})
    assert "chargeable" in text and "2026-11-01" in text and "estimate" in text
    book.close()
