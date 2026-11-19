"""Small typed boundaries independent of transport and model SDKs."""

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class Message:
    bot_id: str
    group_id: str
    message_id: str
    user_id: str
    segments: tuple[dict[str, Any], ...]

    @property
    def text(self) -> str:
        return "".join(str(s["data"].get("text", "")) for s in self.segments if s["type"] == "text")

    @property
    def reply_ids(self) -> tuple[str, ...]:
        return tuple(
            str(s["data"]["id"])
            for s in self.segments
            if s["type"] == "reply" and "id" in s["data"]
        )


def parse_group_message(event: Any, bot_id: str) -> Message | None:
    """Fail closed: CQ strings, nested mentions and malformed segments are not triggers."""
    if not isinstance(event, dict) or event.get("post_type") not in ("message", "message_sent"):
        return None
    if event.get("message_type") != "group" or str(event.get("self_id")) != bot_id:
        return None
    for key in ("group_id", "message_id", "user_id"):
        value = event.get(key)
        if isinstance(value, bool) or not isinstance(value, (str, int)) or not str(value):
            return None
    segs = event.get("message")
    if not isinstance(segs, list) or not all(
        isinstance(s, dict) and isinstance(s.get("type"), str) and isinstance(s.get("data"), dict)
        for s in segs
    ):
        return None
    return Message(
        bot_id,
        str(event["group_id"]),
        str(event["message_id"]),
        str(event["user_id"]),
        tuple(deepcopy(segs)),
    )


def has_top_level_at(message: Message) -> bool:
    return any(
        s["type"] == "at" and str(s["data"].get("qq")) == message.bot_id for s in message.segments
    )


@dataclass(frozen=True)
class Context:
    current: Message
    history: tuple[Message, ...]
    boundary: int


class ModelRunner(Protocol):
    async def run(self, context: Context) -> str: ...


class ContextStrategy(Protocol):
    def build(self, current: Message, history: list[Message], boundary: int) -> Context: ...


class Tool(Protocol):
    async def call(self, arguments: dict[str, Any]) -> dict[str, Any]: ...


class Sender(Protocol):
    """Prototype sender records an intent only, never a platform acknowledgement."""

    async def record(self, run_id: str, message: Message, text: str) -> None: ...


class MockModel:
    async def run(self, context: Context) -> str:
        return "[离线模拟] 已接收请求；未调用模型、工具或 QQ 发送接口。"


class FallbackModel:
    """Try runners in order; an exception (not cancellation) moves to the next (AC-11).

    `attempts` records `(index, error_kind | None)` per try for diagnostics.
    """

    def __init__(self, *runners: ModelRunner):
        if not runners:
            raise ValueError("FallbackModel needs at least one runner")
        self.runners = runners
        self.attempts: list[tuple[int, str | None]] = []

    async def run(self, context: Context) -> str:
        last: Exception | None = None
        for index, runner in enumerate(self.runners):
            try:
                text = await runner.run(context)
            except Exception as exc:  # CancelledError is BaseException and propagates
                self.attempts.append((index, type(exc).__name__))
                if getattr(exc, "reason", None) == "budget_exceeded":
                    raise
                last = exc
                continue
            self.attempts.append((index, None))
            return text
        raise last


class ExplicitContext:
    """Bounded same-scope reply ancestry baseline, not semantic topic resolution."""

    def __init__(self, limit: int = 80):
        self.limit = limit

    def build(self, current: Message, history: list[Message], boundary: int) -> Context:
        visible = {
            m.message_id: m
            for m in history
            if m.bot_id == current.bot_id and m.group_id == current.group_id
        }
        selected, pending, seen = [], list(current.reply_ids), {current.message_id}
        while pending and len(selected) < self.limit:
            key = pending.pop(0)
            if key in seen:
                continue
            seen.add(key)
            if key in visible:
                m = visible[key]
                selected.append(m)
                pending.extend(m.reply_ids)
        return Context(current, tuple(selected), boundary)
