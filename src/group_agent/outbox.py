"""Outbound intents: idempotent records, state machine and send-rate scheduling.

RUN-08: one record per idempotency key; RUN-09 / D-10: per-group minimum interval,
per-group and per-account per-minute caps. States follow requirements 3.2:
pending -> sending -> acked | uncertain | failed. Nothing here talks to QQ; the persistent
rows live in `store.Store` (outbound) and are sent by `live_send.LiveSender`.
`fit_message_text` bounds the length of one outgoing answer.
"""

from bisect import insort
from dataclasses import dataclass, field

from .policy import RuntimeLimits

WINDOW_SECONDS = 60.0
# One QQ message per answer (one outbound row, one send: idempotent per run). The old bot
# turned replies above 1,500 characters into a merged-forward message (AstrBot
# `forward_threshold`); the new framework does not split into several messages (that would
# need several sends per run and flood the group) and instead cuts at a sentence or line
# end before this limit and appends a visible mark. Real answers: p95 about 120 characters.
MAX_REPLY_CHARS = 1500
TRUNCATION_MARK = "……（后面太长，省略了）"
_CUT_AFTER = "\n。！？!?；;…"


def fit_message_text(text: str, limit: int = MAX_REPLY_CHARS) -> tuple[str, bool]:
    """`(text, truncated)`: surrounding whitespace removed, at most `limit` characters."""
    text = str(text).strip()
    if len(text) <= limit:
        return text, False
    room = max(1, limit - len(TRUNCATION_MARK))
    head = text[:room]
    cut = max(head.rfind(c) for c in _CUT_AFTER)
    if cut >= room * 0.6:  # a sentence end late enough; else cut mid-sentence
        head = head[: cut + 1]
    return head.rstrip() + TRUNCATION_MARK, True


STATES = ("pending", "sending", "acked", "uncertain", "failed")
_NEXT = {
    "pending": {"sending", "failed"},
    "sending": {"acked", "uncertain", "failed"},
    "acked": set(),
    "uncertain": set(),
    "failed": set(),
}


def _fits(times: list[float], t: float, limit: int) -> float | None:
    """None if adding `t` keeps every 60 s window within `limit`; else the next try time."""
    for start in sorted({*[x for x in times if t - WINDOW_SECONDS < x <= t], t}):
        if start > t:
            break
        count = sum(1 for x in times if start <= x < start + WINDOW_SECONDS) + 1
        if count > limit:
            return start + WINDOW_SECONDS
    return None


class SendRateLimiter:
    """Deterministic send schedule; `reserve()` returns when the send may happen."""

    def __init__(self, limits: RuntimeLimits = RuntimeLimits()):
        self.limits = limits
        self.by_group: dict[str, list[float]] = {}
        self.account: list[float] = []

    def reserve(self, group_id: str, now: float) -> float:
        group = self.by_group.setdefault(group_id, [])
        t = now
        while True:
            if group and t < group[-1] + self.limits.group_min_interval_seconds:
                t = group[-1] + self.limits.group_min_interval_seconds
                continue
            later = _fits(group, t, self.limits.group_per_minute)
            if later is None:
                later = _fits(self.account, t, self.limits.account_per_minute)
            if later is None:
                break
            t = later
        group.append(t)  # per-group reservations are monotonic by construction
        insort(self.account, t)
        return t


@dataclass
class OutboundRecord:
    key: str
    run_id: str
    kind: str  # answer | busy_notice
    group_id: str
    reply_to: str
    requester: str
    text: str
    state: str = "pending"
    scheduled_at: float | None = None
    sent_at: float | None = None
    error_kind: str | None = None
    history: list[tuple[float, str]] = field(default_factory=list)


class Outbox:
    """In-memory reference of outbound idempotency and state rules (SQLite later, T3.7)."""

    def __init__(self):
        self.records: dict[str, OutboundRecord] = {}

    def enqueue(self, record: OutboundRecord, now: float) -> tuple[OutboundRecord, bool]:
        existing = self.records.get(record.key)
        if existing is not None:
            return existing, False
        record.history.append((now, record.state))
        self.records[record.key] = record
        return record, True

    def transition(self, key: str, state: str, now: float, error_kind: str | None = None):
        record = self.records[key]
        if state not in _NEXT[record.state]:
            raise ValueError(f"outbound: invalid transition {record.state}->{state}")
        record.state = state
        record.error_kind = error_kind or record.error_kind
        record.history.append((now, state))
        return record
