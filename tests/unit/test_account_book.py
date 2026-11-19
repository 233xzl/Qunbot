"""B2: the account-wide usage book, its Ledger hook and the billing report (tmp paths only)."""

import asyncio
import importlib.util
import json
from pathlib import Path
import sqlite3
import sys
import threading

import pytest

from group_agent.billing import ACCOUNT_SCHEMA_VERSION, PROCESS_ID, AccountBook
from group_agent.models import CallRecord, Ledger, LiveModel, Usage
from group_agent.store import Store
from group_agent.usage import UsageLedger
from test_live_replay import GEMINI, text_model

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("billing_report", ROOT / "scripts/billing_report.py")
report = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = report
spec.loader.exec_module(report)

# 2026-11-14 12:00 and 2026-12-02 12:00, Asia/Shanghai.
NOV = 1_794_628_800.0
DEC = NOV + 18 * 86400


def call(model="gemini", *, channel="relay", service="generate", cost=0.01, **kw) -> CallRecord:
    record = CallRecord(
        task="t",
        model=model,
        mode="text",
        status="ok",
        channel=channel,
        service=service,
        cost_cny=cost,
        list_cost=kw.pop("list_cost", cost),
        list_currency=kw.pop("list_currency", "CNY"),
        latency_s=0.5,
        **kw,
    )
    record.usage = Usage(input=100, cached=10, output=20, thinking=5, requests=1)
    return record


def rows(book: AccountBook) -> list[sqlite3.Row]:
    return book.con.execute("SELECT * FROM account_usage ORDER BY id").fetchall()


def test_book_persists_across_reopen_and_has_its_own_schema(tmp_path):
    path = tmp_path / "billing" / "account.sqlite3"  # parent is created
    book = AccountBook(path, clock=lambda: NOV)
    book.append(call(), source="serve", model_id="gemini-real")
    book.close()
    book = AccountBook(path, clock=lambda: NOV)
    try:
        assert book.con.execute("PRAGMA user_version").fetchone()[0] == ACCOUNT_SCHEMA_VERSION
        assert book.con.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        tables = {r[0] for r in book.con.execute("SELECT name FROM sqlite_master")}
        assert {"account_usage", "opening_balance", "grants"} <= tables
        (row,) = rows(book)
        assert (row["source"], row["alias"], row["model"], row["day_cn"]) == (
            "serve",
            "gemini",
            "gemini-real",
            "2026-11-14",
        )
        assert (row["input_tokens"], row["cached_tokens"], row["output_tokens"]) == (100, 10, 20)
        assert (row["thinking_tokens"], row["requests"], row["process_id"]) == (5, 1, PROCESS_ID)
        assert row["created_at"] == NOV and row["reconcile"] == 0
        book.append(call(), source="probe")
        assert book.counts()["account_usage"] == 2
    finally:
        book.close()


def test_book_is_a_separate_file_from_the_serve_store(tmp_path):
    store_path, book_path = tmp_path / "serve.sqlite3", tmp_path / "account.sqlite3"
    store = Store(store_path)
    book = AccountBook(book_path)
    ledger = UsageLedger(store, account=book, source="serve", clock=lambda: NOV)
    ledger.settle(0.0, call())
    store.close()
    book.close()
    # The serve database knows nothing of the account book, so copying it copies no quota.
    with sqlite3.connect(store_path) as con:
        names = {r[0] for r in con.execute("SELECT name FROM sqlite_master")}
        assert "usage_ledger" in names
        assert not names & {"account_usage", "opening_balance", "grants"}
        assert con.execute("SELECT count(*) FROM usage_ledger").fetchone()[0] == 1
    assert store_path != book_path
    with sqlite3.connect(book_path) as con:
        assert con.execute("SELECT count(*) FROM account_usage").fetchone()[0] == 1
        names = {r[0] for r in con.execute("SELECT name FROM sqlite_master")}
        assert "runs" not in names and "usage_ledger" not in names


def test_unknown_source_and_future_schema_are_refused(tmp_path):
    book = AccountBook(tmp_path / "a.sqlite3")
    with pytest.raises(ValueError):
        book.append(call(), source="manual")
    book.close()
    future = tmp_path / "future.sqlite3"
    with sqlite3.connect(future) as con:
        con.execute("PRAGMA user_version=99")
    with pytest.raises(ValueError):
        AccountBook(future)
    with pytest.raises(ValueError):
        Ledger(1.0, account=object())  # a book needs its source


def test_concurrent_settles_from_threads_are_all_recorded(tmp_path):
    book = AccountBook(tmp_path / "a.sqlite3")
    ledger = Ledger(None, account=book, source="replay")
    errors = []

    def worker(n):
        try:
            for _ in range(25):
                ledger.settle(0.0, call(cost=0.001))
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    try:
        assert not errors
        assert book.counts()["account_usage"] == 200
        assert ledger.counts["account_rows"] == 200 and not ledger.counts["account_write_failed"]
        assert ledger.spent == pytest.approx(0.2)
        total = book.con.execute("SELECT SUM(cost_cny) FROM account_usage").fetchone()[0]
        assert total == pytest.approx(0.2)
    finally:
        book.close()


def test_ledger_hook_records_exactly_once_per_settle(tmp_path):
    book = AccountBook(tmp_path / "a.sqlite3")
    base = Ledger(1.0, account=book, source="replay", model_ids={"gemini": "real-id"})
    store = Store()
    usage = UsageLedger(store, account=book, source="serve", clock=lambda: NOV)
    live = LiveModel(GEMINI, lambda: text_model(), base)
    asyncio.run(live.generate("answer", "指令", ["材料"], timeout=5))
    usage_live = LiveModel(GEMINI, lambda: text_model(), usage)
    asyncio.run(usage_live.generate("answer", "指令", ["材料"], timeout=5))
    usage.settle(0.0, call())
    try:
        got = [(r["source"], r["alias"], r["model"]) for r in rows(book)]
        # One row per settle: the base ledger, the UsageLedger call, and the direct settle.
        assert got == [
            ("replay", "gemini", GEMINI.model),
            ("serve", "gemini", GEMINI.model),
            ("serve", "gemini", "gemini"),
        ]
        assert base.counts["account_rows"] == 1 and usage.counts["account_rows"] == 2
        assert usage.counts["rows"] == 2  # serve store rows are independent of the book
        first = rows(book)[0]
        assert (first["channel"], first["service"], first["list_currency"]) == (
            "relay",
            "generate",
            "CNY",
        )
    finally:
        book.close()


def test_account_write_failure_never_breaks_the_call(tmp_path):
    book = AccountBook(tmp_path / "a.sqlite3")
    book.close()  # every append now fails
    ledger = Ledger(1.0, account=book, source="serve")
    live = LiveModel(GEMINI, lambda: text_model(), ledger)
    text, record = asyncio.run(live.generate("answer", "指令", ["材料"], timeout=5))
    assert text == "好的" and record.cost_cny > 0
    assert ledger.counts["account_write_failed"] == 1 and not ledger.counts["account_rows"]
    assert ledger.spent == pytest.approx(record.cost_cny) and ledger.pending == 0
    store = Store()
    usage = UsageLedger(store, account=book, source="serve")
    usage.settle(0.0, call())
    assert usage.counts["account_write_failed"] == 1 and usage.counts["rows"] == 1


def test_google_search_without_query_count_is_marked_for_reconciliation(tmp_path):
    book = AccountBook(tmp_path / "a.sqlite3")
    ledger = Ledger(None, account=book, source="serve")
    ledger.settle(
        0.0, call("gemini_search", channel="vertex", service="search", google_queries=None)
    )
    ledger.settle(0.0, call("gemini_search", channel="vertex", service="search", google_queries=2))
    ledger.settle(0.0, call("deepseek_web", channel="official:api.deepseek.com", service="search"))
    try:
        flags = [(r["google_queries"], r["reconcile"], r["reconcile_note"]) for r in rows(book)]
        assert flags == [
            (None, 1, "google_query_count_unknown"),
            (2, 0, None),
            (None, 0, None),
        ]
    finally:
        book.close()


# -- opening balances and the report ---------------------------------------------------


def test_opening_balance_validation(tmp_path):
    book = AccountBook(tmp_path / "a.sqlite3", clock=lambda: NOV)
    try:
        entry = book.add_opening("vertex", "2026-11", 3.2, "USD", google_queries=120, note="x")
        assert entry == 1
        row = book.con.execute("SELECT * FROM opening_balance").fetchone()
        assert (row["channel"], row["period"], row["google_queries"], row["entered_at"]) == (
            "vertex",
            "2026-11",
            120,
            NOV,
        )
        for args in (
            ("", "2026-11", 1.0, "USD"),
            ("vertex", "2026-13", 1.0, "USD"),
            ("vertex", "2026-11-01", 1.0, "USD"),
            ("vertex", "2026-11", -1.0, "USD"),
            ("vertex", "2026-11", 1.0, "EUR"),
        ):
            with pytest.raises(ValueError):
                book.add_opening(*args)
        assert book.counts()["opening_balance"] == 1
    finally:
        book.close()


def seed(path, with_opening=True):
    clock = [NOV]
    book = AccountBook(path, clock=lambda: clock[0])
    ledger = Ledger(None, account=book, source="serve")
    for _ in range(2):
        ledger.settle(
            0.0,
            call(
                "vertex_hydrogen", channel="vertex", list_cost=0.02, list_currency="USD", cost=0.144
            ),
        )
    ledger.settle(
        0.0,
        call(
            "gemini_search",
            channel="vertex",
            service="search",
            list_cost=0.03,
            list_currency="USD",
            cost=0.216,
            google_queries=3,
        ),
    )
    ledger.settle(
        0.0, call("deepseek_web", channel="official:api.deepseek.com", service="fetch", cost=0.005)
    )
    clock[0] = DEC
    ledger.settle(
        0.0,
        call("vertex_hydrogen", channel="vertex", list_cost=0.01, list_currency="USD", cost=0.072),
    )
    if with_opening:
        book.add_opening("vertex", "2026-11", 1.5, "USD", google_queries=40, note="console")
    book.close()


def test_monthly_summary_groups_by_month_channel_service(tmp_path):
    path = tmp_path / "a.sqlite3"
    seed(path)
    book = AccountBook(path, read_only=True)
    try:
        summary = {(r["month"], r["channel"], r["service"]): r for r in book.monthly_summary()}
        assert set(summary) == {
            ("2026-11", "vertex", "generate"),
            ("2026-11", "vertex", "search"),
            ("2026-11", "official:api.deepseek.com", "fetch"),
            ("2026-12", "vertex", "generate"),
        }
        gen = summary[("2026-11", "vertex", "generate")]
        assert gen["calls"] == 2 and gen["input"] == 200 and gen["output"] == 40
        assert gen["list_cost"] == {"USD": 0.04} and gen["cost_cny"] == pytest.approx(0.288)
        search = summary[("2026-11", "vertex", "search")]
        assert search["google_queries"] == 3 and search["reconcile_calls"] == 0
        assert summary[("2026-11", "official:api.deepseek.com", "fetch")]["list_cost"] == {
            "CNY": 0.005
        }
        assert summary[("2026-12", "vertex", "generate")]["calls"] == 1
        (opening,) = book.opening_summary()
        assert opening["month"] == "2026-11" and opening["list_cost"] == {"USD": 1.5}
        assert opening["google_queries"] == 40 and opening["entries"] == 1
        with pytest.raises(sqlite3.OperationalError):  # the report connection is read-only
            book.con.execute("DELETE FROM account_usage")
    finally:
        book.close()


def test_billing_report_cli(tmp_path, capsys):
    path = tmp_path / "a.sqlite3"
    seed(path, with_opening=False)
    assert report.main(["--db", str(path), "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert len(data["usage"]) == 4 and data["opening"] == []
    assert report.main(["--db", str(path)]) == 0
    text = capsys.readouterr().out
    assert "2026-11" in text and "vertex" in text and "0.0400 USD" in text and "(none)" in text
    code = report.main(
        [
            "--db",
            str(path),
            "add-opening",
            "--channel",
            "vertex",
            "--period",
            "2026-11",
            "--list-cost",
            "2.5",
            "--currency",
            "USD",
            "--google-queries",
            "10",
            "--note",
            "before the book",
        ]
    )
    assert code == 0 and "added" in capsys.readouterr().out
    report.main(["--db", str(path), "--json"])
    data = json.loads(capsys.readouterr().out)
    assert data["opening"][0]["list_cost"] == {"USD": 2.5}
    with pytest.raises(SystemExit):  # bad period is a usage error, nothing is written
        report.main(
            [
                "--db",
                str(path),
                "add-opening",
                "--channel",
                "v",
                "--period",
                "2026-1",
                "--currency",
                "CNY",
            ]
        )
    # Aggregates only: no run ids, no content columns in the output.
    assert "run_id" not in json.dumps(data) and "task" not in json.dumps(data)


def test_report_never_creates_a_missing_book(tmp_path, capsys):
    missing = tmp_path / "none.sqlite3"
    assert report.main(["--db", str(missing)]) == 0
    assert "no account book" in capsys.readouterr().out
    assert not missing.exists()
