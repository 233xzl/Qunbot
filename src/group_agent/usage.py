"""Persistent usage ledger and the daily cost cap (T3.9, RUN-10, D-04, D-12).

`UsageLedger` is a `models.Ledger` (same reserve/settle interface used by `LiveModel`,
`JevClient` and `web.WebTool`) that also writes every settled call to the store's
`usage_ledger` table: run id (from `runflow.CURRENT_RUN`), provider (the configured route
name, e.g. gemini / deepseek / deepseek_web / gemini_search), model id, task, status,
token counts, cost and time. Days are natural days in Asia/Shanghai.

The daily total is loaded from the store when the ledger starts, so a restart keeps the
day's spending, and is recomputed at midnight. `cost_mode(is_owner)` applies D-04 through
`policy.cost_mode`: below 80 % normal; from 80 % degraded (web off, shorter context); from
100 % only the owner is served, others get no run and no notice (D-12).

An optional process budget (`budget_cny`, e.g. a smoke run's limit) is the base class's
hard cap: a call that could exceed it is refused before it is sent.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timedelta, timezone
import threading
import time
from typing import Any

from .models import BudgetExceeded, CallRecord, Ledger
from .policy import RuntimeLimits, cost_mode
from .runflow import CURRENT_GROUP, CURRENT_RUN
from .store import Store

CN_TZ = timezone(timedelta(hours=8))  # Asia/Shanghai has no DST


def cn_day(ts: float) -> str:
    return datetime.fromtimestamp(ts, CN_TZ).strftime("%Y-%m-%d")


class UsageLedger(Ledger):
    def __init__(
        self,
        store: Store,
        *,
        daily_cap_cny: float = 3.0,
        budget_cny: float | None = None,
        limits: RuntimeLimits = RuntimeLimits(),
        model_ids: dict[str, str] | None = None,
        clock: Callable[[], float] = time.time,
        account: Any = None,
        source: str | None = None,
        quota: Any = None,
    ):
        super().__init__(
            budget_cny, account=account, source=source, model_ids=model_ids, quota=quota
        )
        self.store, self.daily_cap_cny, self.limits = store, daily_cap_cny, limits
        self.clock = clock
        self._lock = threading.RLock()  # nested process and daily reservation checks
        self.group_caps = {}
        self.group_pending = {}
        self.owner_call = lambda: False
        self._day = cn_day(clock())
        self._daily = store.usage_total(self._day)
        self._group_daily = store.usage_by_group(self._day)
        self.restored_cny = self._daily  # spent today before this process started

    def reserve(self, estimate):
        check = getattr(self, "before_call", None)
        if check is not None:
            check()
        group = CURRENT_GROUP.get()
        owner = self.owner_call()
        with self._lock:
            ratio, scope = self._budget_ratio(group, estimate)
            if not owner and ratio > self.limits.cap_ratio:
                raise BudgetExceeded(f"projected spend exceeds {scope} daily budget")
            super().reserve(estimate)
            if group is not None:
                self.group_pending[group] = self.group_pending.get(group, 0.0) + estimate

    def set_profiles(self, profiles):
        self.group_caps = {
            gid: p["daily_cost_cny"]
            for gid, p in profiles.items()
            if p.get("enabled") and "daily_cost_cny" in p
        }

    def _budget_ratio(self, group, estimate=0.0):
        self._roll()
        total = (
            (self._daily + self.pending + estimate) / self.daily_cap_cny
            if self.daily_cap_cny > 0
            else 0.0
        )
        if group in self.group_caps:
            local = (
                self._group_daily.get(group, 0.0) + self.group_pending.get(group, 0.0) + estimate
            ) / self.group_caps[group]
            if local > total:
                return local, "group"
        return total, "global"

    def decision(self, is_owner, group):
        with self._lock:
            ratio, scope = self._budget_ratio(group)
            mode = cost_mode(ratio, is_owner, self.limits)
            return mode, scope if mode != "normal" else None

    def _roll(self) -> str:
        day = cn_day(self.clock())
        if day != self._day:
            self._day, self._daily = day, self.store.usage_total(day)
            self._group_daily = self.store.usage_by_group(day)
        return day

    def settle(self, estimate: float, record: CallRecord) -> None:
        group = CURRENT_GROUP.get()
        super().settle(estimate, record)
        now = self.clock()
        with self._lock:
            day = self._roll()
            self._daily += record.cost_cny
            if group is not None:
                self._group_daily[group] = self._group_daily.get(group, 0.0) + record.cost_cny
                pending = self.group_pending.get(group, 0.0) - estimate
                if pending > 1e-10:
                    self.group_pending[group] = pending
                else:
                    self.group_pending.pop(group, None)
        usage = record.usage
        row = {
            "run_id": CURRENT_RUN.get(),
            "group_id": group,
            "day": day,
            "created_at": now,
            "provider": record.model,
            "model": record.model_id or self.model_ids.get(record.model, record.model),
            "task": record.task,
            "mode": record.mode,
            "status": record.status,
            "input_tokens": usage.input,
            "cached_tokens": usage.cached,
            "output_tokens": usage.output,
            "thinking_tokens": usage.thinking,
            "requests": usage.requests,
            "cost_cny": record.cost_cny,
            "latency_s": round(record.latency_s, 3),
            "channel": record.channel,
            "service": record.service,
            "list_cost": record.list_cost,
            "list_currency": record.list_currency,
            "google_queries": record.google_queries,
        }
        try:
            self.store.record_usage(row)
            outcome = "rows"
        except Exception:  # noqa: BLE001 - the in-memory total still counts the call
            outcome = "write_failed"
        with self._lock:
            self.counts[outcome] += 1

    def daily_spent(self) -> float:
        with self._lock:
            self._roll()
            return self._daily

    def ratio(self) -> float:
        if self.daily_cap_cny <= 0:
            return 0.0
        return self.daily_spent() / self.daily_cap_cny

    def cost_mode(self, is_owner: bool, group: str | None = None) -> str:
        return self.decision(is_owner, group)[0]

    def report(self) -> dict:
        day = cn_day(self.clock())
        return {
            "day": day,
            "daily_cap_cny": self.daily_cap_cny,
            "daily_spent_cny": round(self.daily_spent(), 6),
            "restored_cny": round(self.restored_cny, 6),
            "session_spent_cny": round(self.spent, 6),
            "ratio": round(self.ratio(), 4),
            "cost_mode": self.cost_mode(is_owner=False),
            "by_provider": self.store.usage_summary(day),
        }
