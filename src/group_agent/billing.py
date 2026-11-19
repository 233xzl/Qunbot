"""Account-wide usage book (B2): every settled model and web call of the bot's billing account.

`AccountBook` is its own SQLite file (`[billing] db_path`), deliberately not the serve store:
copying a serve or replay database must never copy the account's usage or, later, its quota.
Online serving, replay runs and probes all append to the same book (`source` says which), so
Google's monthly free search quota and the credits (B3) can be tracked per account.

Tables (PRAGMA user_version = ACCOUNT_SCHEMA_VERSION):

- `account_usage`: one row per settled call, in the call's original currency (`list_cost` in
  `list_currency`, never reduced by credits) plus the CNY estimate every gate uses.
  `reconcile` = 1 marks a call whose Google query count is unknown (to be reserved
  conservatively and reconciled against Cloud Billing by B3).
- `opening_balance`: manual baseline entries for usage that happened before the book existed
  ("接入前已发生用量以可追溯期初数补入"); reports add them to the month they name.
- `grants`: reserved for B3 (one-time and monthly credits). Nothing reads or writes it yet.

`Ledger(account=book, source=...)` appends on every settle. One short transaction per row;
the caller counts a failed append instead of failing the model call.

Schema v2 (B3) adds `account_usage.free_queries`: the queries of a Vertex search call that
`GoogleFreeQuota` estimated to fall in the monthly free quota (NULL = not applied, and every
v1 row). `list_cost` keeps the full list price for reconciliation; `cost_cny` has the free
queries' fee taken out, so it is the figure the daily cash gate sees.

`GoogleFreeQuota` is the free-query counter. Used queries of a cycle = `google_queries` of
every Vertex `account_usage` row of the cycle (serve, replay and probe share the account)
plus the cycle's `opening_balance` entries. The cycle starts on `[billing] billing_cycle_day`
of the Asia/Shanghai calendar; Google counts per billing month in Pacific time, so the
counter is an estimate, to be reconciled against Cloud Billing. An opening balance belongs to
the cycle that starts in its `period` month.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
import re
import sqlite3
import threading
import time
from uuid import uuid4

from .models import CallRecord

CN_TZ = timezone(timedelta(hours=8))  # Asia/Shanghai has no DST (as usage.CN_TZ)
ACCOUNT_SCHEMA_VERSION = 2
GOOGLE_QUOTA_CHANNEL = "vertex"  # only this channel has the free monthly search quota
SOURCES = ("serve", "replay", "probe")
PERIOD = re.compile(r"[0-9]{4}-(?:0[1-9]|1[0-2])")
# One id per process: rows of one serve/replay/probe process can be told apart and summed.
PROCESS_ID = uuid4().hex[:12]

ACCOUNT_SCHEMA = """
CREATE TABLE IF NOT EXISTS account_usage (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at REAL NOT NULL,            -- UTC epoch seconds
    day_cn TEXT NOT NULL,                -- Asia/Shanghai calendar day, YYYY-MM-DD
    source TEXT NOT NULL CHECK(source IN ('serve', 'replay', 'probe')),
    process_id TEXT NOT NULL, run_id TEXT,
    channel TEXT, service TEXT,
    alias TEXT NOT NULL,                 -- configured alias or web backend name
    model TEXT, task TEXT, status TEXT,
    input_tokens INTEGER NOT NULL DEFAULT 0, cached_tokens INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0, thinking_tokens INTEGER,
    requests INTEGER NOT NULL DEFAULT 0,
    google_queries INTEGER,              -- NULL = unknown or not a Google search
    list_cost REAL, list_currency TEXT,  -- original-currency list price of the call
    cost_cny REAL NOT NULL DEFAULT 0, latency_s REAL,
    reconcile INTEGER NOT NULL DEFAULT 0, reconcile_note TEXT,
    free_queries INTEGER                 -- v2: queries estimated free (NULL = not applied)
);
CREATE INDEX IF NOT EXISTS account_usage_day ON account_usage(day_cn);
CREATE INDEX IF NOT EXISTS account_usage_channel_day ON account_usage(channel, day_cn);
CREATE TABLE IF NOT EXISTS opening_balance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    channel TEXT NOT NULL,
    period TEXT NOT NULL,                -- calendar month the baseline belongs to, YYYY-MM
    list_cost REAL NOT NULL DEFAULT 0, list_currency TEXT NOT NULL,
    google_queries INTEGER NOT NULL DEFAULT 0,
    note TEXT, entered_at REAL NOT NULL
);
-- B3 (credits and free-quota deduction) will use this table; nothing writes it yet.
CREATE TABLE IF NOT EXISTS grants (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    channel TEXT NOT NULL,
    kind TEXT NOT NULL CHECK(kind IN ('one_time', 'monthly')),
    amount REAL NOT NULL, currency TEXT NOT NULL,
    starts TEXT, expires TEXT, note TEXT
);
"""


class AccountBook:
    """Thread-safe append-only usage book; see the module docstring."""

    def __init__(
        self,
        path: Path | str,
        *,
        process_id: str | None = None,
        clock: Callable[[], float] = time.time,
        read_only: bool = False,
        run_id: Callable[[], str | None] = lambda: None,
    ):
        """`run_id` names the current run for each row (serve passes `runflow.CURRENT_RUN.get`;
        not imported here so probes need no gateway modules)."""
        self.path, self.process_id, self.clock = Path(path), process_id or PROCESS_ID, clock
        self.run_id = run_id
        self._lock = threading.Lock()
        if read_only:
            # Reports never create or migrate the file.
            self.con = sqlite3.connect(f"{self.path.resolve().as_uri()}?mode=ro", uri=True)
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.con = sqlite3.connect(str(self.path), timeout=30, check_same_thread=False)
        self.con.row_factory = sqlite3.Row
        try:
            version = self.con.execute("PRAGMA user_version").fetchone()[0]
            if version not in (0, 1, ACCOUNT_SCHEMA_VERSION) or (read_only and version == 0):
                raise ValueError("unsupported account book schema version")
            columns = {r[1] for r in self.con.execute("PRAGMA table_info(account_usage)")}
            # A read-only handle may see a v1 file it must not migrate.
            self.has_free_queries = "free_queries" in columns or not read_only
            if not read_only:
                self.con.execute("PRAGMA journal_mode=WAL")
                # v1 -> v2: old rows keep NULL free_queries.
                migrate = ""
                if columns and "free_queries" not in columns:
                    migrate = "ALTER TABLE account_usage ADD COLUMN free_queries INTEGER;\n"
                self.con.executescript(
                    f"BEGIN IMMEDIATE;\n{migrate}{ACCOUNT_SCHEMA}\n"
                    f"PRAGMA user_version={ACCOUNT_SCHEMA_VERSION};\nCOMMIT;"
                )
        except Exception:
            self.con.close()
            raise

    def close(self) -> None:
        with self._lock:
            self.con.close()

    # -- writes -------------------------------------------------------------------------
    def append(self, record: CallRecord, *, source: str, model_id: str | None = None) -> int:
        """Append one settled call; returns the row id. Raises on write failure (the
        `Ledger` hook counts it and carries on)."""
        if source not in SOURCES:
            raise ValueError("unknown usage source")
        now = self.clock()
        usage = record.usage
        unknown_queries = (
            record.service == "search"
            and record.model == "gemini_search"
            and record.google_queries is None
        )
        row = (
            now,
            datetime.fromtimestamp(now, CN_TZ).strftime("%Y-%m-%d"),
            source,
            self.process_id,
            self.run_id(),
            record.channel,
            record.service,
            record.model,
            model_id or record.model,
            record.task,
            record.status,
            usage.input,
            usage.cached,
            usage.output,
            usage.thinking,
            usage.requests,
            record.google_queries,
            record.list_cost,
            record.list_currency,
            record.cost_cny,
            round(record.latency_s, 3),
            1 if unknown_queries else 0,
            "google_query_count_unknown" if unknown_queries else None,
            record.free_queries,
        )
        with self._lock, self.con:
            cursor = self.con.execute(
                "INSERT INTO account_usage(created_at,day_cn,source,process_id,run_id,channel,"
                "service,alias,model,task,status,input_tokens,cached_tokens,output_tokens,"
                "thinking_tokens,requests,google_queries,list_cost,list_currency,cost_cny,"
                "latency_s,reconcile,reconcile_note,free_queries) VALUES "
                "(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                row,
            )
            return int(cursor.lastrowid)

    def add_opening(
        self,
        channel: str,
        period: str,
        list_cost: float,
        list_currency: str,
        *,
        google_queries: int = 0,
        note: str = "",
    ) -> int:
        """Add a manual baseline for usage that happened before the book existed."""
        if not channel or not PERIOD.fullmatch(period):
            raise ValueError("opening balance needs a channel and a YYYY-MM period")
        if list_currency not in ("CNY", "USD"):
            raise ValueError("opening balance currency: expected CNY or USD")
        if list_cost < 0 or google_queries < 0:
            raise ValueError("opening balance amounts must not be negative")
        with self._lock, self.con:
            cursor = self.con.execute(
                "INSERT INTO opening_balance(channel,period,list_cost,list_currency,"
                "google_queries,note,entered_at) VALUES (?,?,?,?,?,?,?)",
                (channel, period, list_cost, list_currency, google_queries, note, self.clock()),
            )
            return int(cursor.lastrowid)

    # -- reads (aggregates only) ----------------------------------------------------------
    def monthly_summary(self) -> list[dict]:
        """Per Asia/Shanghai month x channel x service: calls, tokens, Google queries, list
        cost per currency, CNY estimate and the calls still awaiting reconciliation."""
        with self._lock:
            rows = self.con.execute(
                "SELECT substr(day_cn,1,7) AS month, COALESCE(channel,'?') AS channel,"
                " COALESCE(service,'?') AS service, list_currency, count(*) AS calls,"
                " SUM(input_tokens) AS input, SUM(cached_tokens) AS cached,"
                " SUM(output_tokens) AS output, SUM(COALESCE(thinking_tokens,0)) AS thinking,"
                " SUM(COALESCE(google_queries,0)) AS queries, SUM(list_cost) AS listed,"
                " SUM(cost_cny) AS cny, SUM(reconcile) AS reconcile"
                " FROM account_usage GROUP BY 1,2,3,4 ORDER BY 1,2,3,4"
            ).fetchall()
        out: dict[tuple, dict] = {}
        for r in rows:
            item = out.setdefault(
                (r["month"], r["channel"], r["service"]),
                {
                    "month": r["month"],
                    "channel": r["channel"],
                    "service": r["service"],
                    "calls": 0,
                    "input": 0,
                    "cached": 0,
                    "output": 0,
                    "thinking": 0,
                    "google_queries": 0,
                    "reconcile_calls": 0,
                    "list_cost": {},
                    "cost_cny": 0.0,
                },
            )
            item["calls"] += r["calls"]
            item["input"] += r["input"]
            item["cached"] += r["cached"]
            item["output"] += r["output"]
            item["thinking"] += r["thinking"]
            item["google_queries"] += r["queries"]
            item["reconcile_calls"] += r["reconcile"]
            item["cost_cny"] += r["cny"] or 0.0
            if r["list_currency"] is not None:
                item["list_cost"][r["list_currency"]] = item["list_cost"].get(
                    r["list_currency"], 0.0
                ) + (r["listed"] or 0.0)
        for item in out.values():
            item["cost_cny"] = round(item["cost_cny"], 6)
            item["list_cost"] = {k: round(v, 6) for k, v in item["list_cost"].items()}
        return list(out.values())

    def opening_summary(self) -> list[dict]:
        """Opening balances per month x channel, list cost per currency."""
        with self._lock:
            rows = self.con.execute(
                "SELECT period AS month, channel, list_currency, SUM(list_cost) AS listed,"
                " SUM(google_queries) AS queries, count(*) AS entries FROM opening_balance"
                " GROUP BY 1,2,3 ORDER BY 1,2,3"
            ).fetchall()
        out: dict[tuple, dict] = {}
        for r in rows:
            item = out.setdefault(
                (r["month"], r["channel"]),
                {
                    "month": r["month"],
                    "channel": r["channel"],
                    "entries": 0,
                    "google_queries": 0,
                    "list_cost": {},
                },
            )
            item["entries"] += r["entries"]
            item["google_queries"] += r["queries"]
            item["list_cost"][r["list_currency"]] = round(r["listed"], 6)
        return list(out.values())

    def counts(self) -> dict:
        with self._lock:
            return {
                name: self.con.execute(f"SELECT count(*) FROM {name}").fetchone()[0]
                for name in ("account_usage", "opening_balance", "grants")
            }

    def google_queries_used(self, channel: str, start: date, end: date) -> int:
        """Google search queries of `channel` in [start, end) by Asia/Shanghai day (every
        source), plus the opening-balance entries of the period named by `start`'s month."""
        with self._lock:
            used = self.con.execute(
                "SELECT COALESCE(SUM(google_queries), 0) FROM account_usage"
                " WHERE channel=? AND day_cn>=? AND day_cn<?",
                (channel, start.isoformat(), end.isoformat()),
            ).fetchone()[0]
            opening = self.con.execute(
                "SELECT COALESCE(SUM(google_queries), 0) FROM opening_balance"
                " WHERE channel=? AND period=?",
                (channel, start.strftime("%Y-%m")),
            ).fetchone()[0]
        return int(used) + int(opening)

    def google_cycles(self, channel: str, cycle_day: int) -> list[dict]:
        """Per billing cycle of `channel`: queries used (usage rows plus opening balances),
        of which opening, and the free queries the quota counter recorded."""
        free_sql = "SUM(COALESCE(free_queries, 0))" if self.has_free_queries else "0"
        with self._lock:
            rows = self.con.execute(
                f"SELECT day_cn, SUM(COALESCE(google_queries, 0)), {free_sql} FROM account_usage"
                " WHERE channel=? GROUP BY day_cn",
                (channel,),
            ).fetchall()
            openings = self.con.execute(
                "SELECT period, SUM(google_queries) FROM opening_balance WHERE channel=?"
                " GROUP BY period",
                (channel,),
            ).fetchall()
        cycles: dict[date, dict] = {}

        def entry(start: date) -> dict:
            return cycles.setdefault(
                start,
                {"cycle_start": start.isoformat(), "used": 0, "opening": 0, "free_recorded": 0},
            )

        for day, queries, free in rows:
            item = entry(cycle_start(date.fromisoformat(day), cycle_day))
            item["used"] += int(queries)
            item["free_recorded"] += int(free)
        for period, queries in openings:
            item = entry(date(int(period[:4]), int(period[5:]), cycle_day))
            item["used"] += int(queries)
            item["opening"] += int(queries)
        return [cycles[k] for k in sorted(cycles)]


def cycle_start(day: date, cycle_day: int) -> date:
    """First day of the billing cycle containing `day` (cycle_day is 1..28)."""
    if day.day >= cycle_day:
        return day.replace(day=cycle_day)
    return (day.replace(day=1) - timedelta(days=1)).replace(day=cycle_day)


def next_cycle_start(start: date) -> date:
    return (start.replace(day=1) + timedelta(days=32)).replace(day=start.day)


class GoogleFreeQuota:
    """Estimate of the account's remaining free Google Search grounding queries (B3).

    Used queries of the cycle are read from the `AccountBook` (see the module docstring), so
    they survive restarts and include replay and probe processes. An in-process cache serves
    `remaining()` (the reservation estimate); `settling()` re-reads the book under the lock,
    so two threads settling near the boundary never both take the last free queries. If a
    book append fails the cache keeps the queries (it never drops below its own takes).
    """

    def __init__(
        self,
        book: AccountBook,
        *,
        free_per_month: int = 5000,
        cycle_day: int = 1,
        fee_cny_per_query: float,
        channel: str = GOOGLE_QUOTA_CHANNEL,
        clock: Callable[[], float] | None = None,
    ):
        if not 1 <= cycle_day <= 28 or free_per_month < 0 or fee_cny_per_query < 0:
            raise ValueError("invalid free quota parameters")
        self.book, self.free_per_month, self.cycle_day = book, free_per_month, cycle_day
        self.fee_cny_per_query, self.channel = fee_cny_per_query, channel
        self.clock = clock or book.clock
        self._lock = threading.Lock()
        self._cycle: date | None = None
        self._used = 0
        with self._lock:
            self._sync(force=True)

    def _sync(self, *, force: bool) -> None:
        """Refresh the cycle and the used count (caller holds the lock). Within a cycle the
        count never drops below what this process already took."""
        today = datetime.fromtimestamp(self.clock(), CN_TZ).date()
        start = cycle_start(today, self.cycle_day)
        if start != self._cycle:
            self._cycle, self._used, force = start, 0, True
        if force:
            stored = self.book.google_queries_used(self.channel, start, next_cycle_start(start))
            self._used = max(self._used, stored)

    @property
    def cycle(self) -> date:
        with self._lock:
            self._sync(force=False)
            return self._cycle

    def used(self) -> int:
        with self._lock:
            self._sync(force=False)
            return self._used

    def remaining(self) -> int:
        """Free queries left this cycle, from the in-process cache."""
        with self._lock:
            self._sync(force=False)
            return max(0, self.free_per_month - self._used)

    def applies(self, record: CallRecord) -> bool:
        return (
            record.channel == self.channel
            and record.service == "search"
            and bool(record.google_queries)
            and record.google_queries > 0
        )

    @contextmanager
    def settling(self, record: CallRecord) -> Iterator[None]:
        """Hold the quota lock while the caller books `record`: the free share of its
        queries is taken out of `cost_cny` first (and noted in `free_queries`), then the
        caller appends the row, then the cache counts the queries."""
        with self._lock:
            queries = record.google_queries if self.applies(record) else 0
            if queries:
                self._sync(force=True)
                free = min(queries, max(0, self.free_per_month - self._used))
                record.free_queries = free
                record.cost_cny = max(0.0, record.cost_cny - free * self.fee_cny_per_query)
            try:
                yield
            finally:
                self._used += queries


def open_google_quota(settings, book: AccountBook | None) -> GoogleFreeQuota | None:
    """The free-quota counter for `book` from `[billing]`, or None without a book (dry runs,
    `--no-account-book`): those keep list pricing."""
    if book is None:
        return None
    billing = settings.billing
    return GoogleFreeQuota(
        book,
        free_per_month=billing.google_free_queries_per_month,
        cycle_day=billing.billing_cycle_day,
        fee_cny_per_query=billing.google_search_usd_per_1000 / 1000 * billing.usd_cny,
    )


def open_account_book(
    settings,
    source: str,
    *,
    enabled: bool = True,
    path: Path | None = None,
    run_id: Callable[[], str | None] = lambda: None,
) -> AccountBook | None:
    """The configured account book for a caller of `source`, or None when `enabled` is
    false (`--no-account-book`, dry runs). `path` overrides `[billing] db_path`."""
    if not enabled:
        return None
    if source not in SOURCES:
        raise ValueError("unknown usage source")
    return AccountBook(path or settings.billing.db_path, run_id=run_id)
