"""Per-group multi-turn answer sessions (`[session]`, off by default).

Hydrogen-class models reuse a cached prefix only when a request continues an earlier
request together with that request's own response, so a group keeps one conversation:

    instructions + tools
    [user]  first turn: selected context, the request and its notes
    [model] the response as returned (tool calls, tool results, signatures kept)
    [user]  next turn: group messages since the previous turn, the new request
    ...

Every request is still its own run (boundary, budget, deadline, cancellation, sending).
A run starts from a snapshot (`SessionView`) of the group's session; the answer joins the
session only after it was delivered (`commit`), and only when no other run of the same
session committed in between (concurrent runs branch from the same turn; the first
delivered answer joins, the others reach later turns as ordinary bot messages). Failed,
cancelled, unsent and fallback answers never join.

Speaker labels (群友A ...) and message numbers (#m1 ...) are stable within a session, so
earlier turns are never rewritten. A session restarts when it was idle too long, its last
request grew past `max_tokens`, it is older than `max_age_minutes`, the model, instructions
or tools changed (`key`), or a message it contains was recalled or its author blocked.

Sessions live in memory: a restart begins new sessions (the provider cache lasts only a
few minutes anyway).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from typing import Any, Callable
from uuid import uuid4

from .config import SessionSpec

# Bump when the session rendering changes: old sessions restart instead of mixing styles.
RENDER_VERSION = "session-1"
BOT_LABEL = "机器人（你）"


def session_key(spec: Any, instructions: str, tools: list[str]) -> str:
    """Identity of what a cached prefix depends on besides the turns themselves."""
    text = "\x1f".join(
        [
            RENDER_VERSION,
            str(getattr(spec, "name", "")),
            str(getattr(spec, "model", "")),
            str(getattr(spec, "protocol", "")),
            instructions,
            *sorted(tools),
        ]
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


@dataclass
class GroupSession:
    session_id: str
    key: str
    started_at: float
    last_at: float
    group_id: str = ""
    version: int = 0  # committed turns
    history: list = field(default_factory=list)  # Pydantic AI ModelMessage list
    labels: dict[str, str] = field(default_factory=dict)  # user id -> 群友X
    names: dict[str, str] = field(default_factory=dict)  # card/nickname -> label
    numbers: dict[str, int] = field(default_factory=dict)  # message id -> n of #m<n>
    next_number: int = 1
    known: set[str] = field(default_factory=set)  # message ids shown in committed turns
    users: set[str] = field(default_factory=set)  # authors of known messages
    answer_ids: set[str] = field(default_factory=set)  # QQ ids of committed answers
    cutoff_time: float | None = None  # production time of the last turn's newest message
    tokens: int = 0  # input + output tokens of the last committed request


@dataclass
class SessionView:
    """One run's snapshot of a group session; the run mutates only its own copies."""

    group_id: str
    session_id: str
    base_version: int
    reason: str  # continued | new | idle | tokens | age | changed | invalid
    history: list
    labels: dict[str, str]
    names: dict[str, str]
    numbers: dict[str, int]
    next_number: int
    known: frozenset[str]
    answer_ids: frozenset[str]
    cutoff_time: float | None
    spec: SessionSpec
    # Filled by the run while rendering / answering.
    shown: set[str] = field(default_factory=set)  # message ids rendered in this turn
    shown_users: set[str] = field(default_factory=set)
    shown_until: float | None = None
    messages: list | None = None  # complete history after a successful session answer
    tokens: int = 0

    @property
    def fresh(self) -> bool:
        return not self.history

    @property
    def turn(self) -> int:
        return self.base_version + 1


class SessionManager:
    def __init__(self) -> None:
        self.sessions: dict[tuple[str, str], GroupSession] = {}
        self.counts: dict[str, int] = {}

    def _count(self, name: str) -> None:
        self.counts[name] = self.counts.get(name, 0) + 1

    def begin(
        self,
        bot_id: str,
        group_id: str,
        key: str,
        spec: SessionSpec,
        now: float,
        valid: Callable[[GroupSession], bool] = lambda _: True,
    ) -> SessionView:
        slot = (bot_id, group_id)
        current = self.sessions.get(slot)
        reason = "continued"
        if current is None:
            reason = "new"
        elif current.key != key:
            reason = "changed"
        elif now - current.last_at > spec.idle_minutes * 60:
            reason = "idle"
        elif now - current.started_at > spec.max_age_minutes * 60:
            reason = "age"
        elif current.tokens >= spec.max_tokens:
            reason = "tokens"
        elif current.version and not valid(current):
            reason = "invalid"
        if reason != "continued" or current is None:
            current = GroupSession(uuid4().hex[:12], key, now, now, group_id)
            current.labels = {bot_id: BOT_LABEL}
            self.sessions[slot] = current
        self._count(f"begin_{reason}")
        return SessionView(
            group_id=group_id,
            session_id=current.session_id,
            base_version=current.version,
            reason=reason,
            history=list(current.history),
            labels=dict(current.labels),
            names=dict(current.names),
            numbers=dict(current.numbers),
            next_number=current.next_number,
            known=frozenset(current.known),
            answer_ids=frozenset(current.answer_ids),
            cutoff_time=current.cutoff_time,
            spec=spec,
        )

    def commit(
        self, bot_id: str, view: SessionView, now: float, answer_id: str | None = None
    ) -> bool:
        """Join a delivered session answer; False when another run committed first or
        the session was replaced meanwhile."""
        current = self.sessions.get((bot_id, view.group_id))
        if (
            view.messages is None
            or current is None
            or current.session_id != view.session_id
            or current.version != view.base_version
        ):
            self._count("commit_skipped")
            return False
        current.history = list(view.messages)
        current.labels, current.names = view.labels, view.names
        current.numbers, current.next_number = view.numbers, view.next_number
        current.known = set(view.known) | view.shown
        current.users |= view.shown_users
        if answer_id:
            current.answer_ids = set(current.answer_ids) | {answer_id}
        if view.shown_until is not None:
            current.cutoff_time = max(current.cutoff_time or 0, view.shown_until)
        current.tokens = view.tokens
        current.version += 1
        current.last_at = now
        self._count("commit")
        return True

    def drop(self, bot_id: str, group_id: str) -> None:
        self.sessions.pop((bot_id, group_id), None)

    def status(self) -> dict:
        return {
            "groups": len(self.sessions),
            "turns": sum(s.version for s in self.sessions.values()),
            **self.counts,
        }
