"""OneBot v11 gateway over SnowLuma's forward WebSocket (T3.1, MIG-02).

SnowLuma's forward server has the "Universal" role: one connection carries pushed events
and action calls. This module owns that connection:

- Events are parsed into the project's message objects (`parse_event`) and handed to one
  consumer task in arrival order, so a handler may await actions without blocking the
  receive loop that delivers their responses.
- Actions are read-only and allowlisted (`READ_ONLY_ACTIONS`). Anything else (send_*,
  delete_msg, set_*, ...) is refused in code before a frame is built, both in
  `OneBotGateway.call` and in the connection wrapper that is the only object able to
  write a frame. With `allow_actions=False` (the shadow default) the connection is wrapped
  as receive-only and no frame is ever written.
- The single write exception (D-19): with `owner_send=True` the gateway issues exactly
  one `OwnerSendChannel` (`OWNER_SEND_ACTION` only), held by the owner's `#新版` command
  handler (`owner_command.py`). `call()` still refuses every non-read action.
- The cutover answer path (RUN-07..09): with `answer_send=True` the gateway issues exactly
  one `AnswerSendChannel` (`OWNER_SEND_ACTION` only, its own grant), held by
  `live_send.LiveSender`, which sends recorded answers after the run flow's pre-publish
  check and send-rate wait.
- The reaction (D-20): with `reaction=True` the gateway issues exactly one
  `ReactionChannel` (`REACTION_ACTION` = set_msg_emoji_like only, emoji fixed, set=true,
  rate-limited), held by `reaction.ProcessingReaction`.
- Every call gets a unique echo, a timeout, a token-bucket rate limit and a concurrency
  cap. Unknown or late responses are dropped and counted. On disconnect every pending call
  ends with `GatewayDisconnected`. Reconnects use capped exponential backoff.

Error objects carry a safe `kind` and the numeric retcode only; SnowLuma's error wording is
not kept.
"""

from __future__ import annotations

import asyncio
from collections import Counter, OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
import inspect
import itertools
import json
import time
from typing import Any
from uuid import uuid4

from .contracts import Message, parse_group_message

MAX_FRAME_BYTES = 8 * 1024 * 1024
# Read-only OneBot actions the agent may call. Sending, recalling, reactions and every
# set_* / admin action are deliberately absent; sending is a separate, approved task.
READ_ONLY_ACTIONS = frozenset(
    {
        "get_msg",
        "get_image",
        "get_forward_msg",
        "get_status",
        "get_login_info",
        "get_group_info",
        "get_group_list",
        "get_group_member_info",
        "get_group_member_list",
    }
)


class GatewayError(Exception):
    """A failed action call; `kind` and `retcode` are safe to record."""

    def __init__(self, kind: str, retcode: int | None = None):
        super().__init__(kind if retcode is None else f"{kind} (retcode {retcode})")
        self.kind, self.retcode = kind, retcode


class ActionRefused(GatewayError):
    """The action is not allowlisted, or actions are disabled on this gateway."""


class ActionTimeout(GatewayError):
    pass


class GatewayDisconnected(GatewayError):
    pass


class ActionFailed(GatewayError):
    """SnowLuma answered with a non-ok status, or the response had an unexpected shape."""


# D-19: the only write action, reachable only through `OwnerSendChannel` (one per
# gateway, issued once, and only when the gateway was built with `owner_send=True`).
OWNER_SEND_ACTION = "send_group_msg"
_OWNER_GRANT = object()  # module-private proof that a frame comes from OwnerSendChannel
# Cutover (D-21, RUN-07..09): ordinary answers, reachable only through `AnswerSendChannel`
# (one per gateway, issued once, only when the gateway was built with `answer_send=True`);
# same action, its own grant, held by `live_send.LiveSender`.
_ANSWER_GRANT = object()  # module-private proof that a frame comes from AnswerSendChannel
# D-20: the "processing" reaction, reachable only through `ReactionChannel` (one per
# gateway, issued once, only when the gateway was built with `reaction=True`).
REACTION_ACTION = "set_msg_emoji_like"
_REACTION_GRANT = object()  # module-private proof that a frame comes from ReactionChannel
_GRANTED_ACTIONS = {
    OWNER_SEND_ACTION: (_OWNER_GRANT, _ANSWER_GRANT),
    REACTION_ACTION: (_REACTION_GRANT,),
}


def _granted(action: Any, grant: object) -> bool:
    return (
        grant is not None
        and isinstance(action, str)
        and any(g is grant for g in _GRANTED_ACTIONS.get(action, ()))
    )


def check_action(action: Any) -> str:
    if not isinstance(action, str) or action not in READ_ONLY_ACTIONS:
        raise ActionRefused("action_not_allowed")
    return action


# -- connection wrappers --------------------------------------------------------------


class ReceiveOnly:
    """Expose only receiving and closing of a connection; there is deliberately no send."""

    def __init__(self, connection: Any):
        self._connection = connection

    def __aiter__(self):
        return self._connection.__aiter__()

    async def close(self) -> None:
        await self._connection.close()


class ReadOnlyActions(ReceiveOnly):
    """Receive, close, and write allowlisted read-only action frames; the exceptions are
    `OWNER_SEND_ACTION` carrying the module-private grant of `OwnerSendChannel` or of
    `AnswerSendChannel`, and `REACTION_ACTION` carrying the one of `ReactionChannel`."""

    async def send_action(
        self, action: str, params: dict[str, Any], echo: str, *, grant: object = None
    ) -> None:
        if not _granted(action, grant):
            check_action(action)
        if not isinstance(params, dict) or not isinstance(echo, str):
            raise ActionRefused("bad_request")
        frame = json.dumps({"action": action, "params": params, "echo": echo}, ensure_ascii=False)
        await self._connection.send(frame)


# -- rate limiting --------------------------------------------------------------------


class TokenBucket:
    """`rate` tokens per second up to `capacity`; `acquire` waits for one token."""

    def __init__(
        self,
        rate: float,
        capacity: float,
        clock: Callable[[], float] = time.monotonic,
    ):
        if rate <= 0 or capacity < 1:
            raise ValueError("token bucket needs rate > 0 and capacity >= 1")
        self.rate, self.capacity, self.clock = rate, capacity, clock
        self.tokens, self.updated = float(capacity), clock()

    def _refill(self) -> None:
        now = self.clock()
        self.tokens = min(self.capacity, self.tokens + (now - self.updated) * self.rate)
        self.updated = now

    def try_acquire(self) -> bool:
        self._refill()
        if self.tokens >= 1:
            self.tokens -= 1
            return True
        return False

    async def acquire(self) -> None:
        while not self.try_acquire():
            await asyncio.sleep((1 - self.tokens) / self.rate)


# -- event parsing --------------------------------------------------------------------


@dataclass(frozen=True)
class GroupMessageEvent:
    message: Message
    platform_time: int | None
    self_sent: bool  # post_type == "message_sent"


@dataclass(frozen=True)
class RecallEvent:
    group_id: str
    message_id: str
    user_id: str
    operator_id: str | None
    platform_time: int | None


@dataclass(frozen=True)
class MetaEvent:
    meta_event_type: str  # "heartbeat", "lifecycle", ...
    sub_type: str | None  # lifecycle: "connect" / "enable" / "disable"
    interval_ms: int | None  # heartbeat interval
    online: bool | None  # heartbeat status.online


@dataclass(frozen=True)
class NoticeEvent:
    notice_type: str
    sub_type: str | None
    group_id: str | None


@dataclass(frozen=True)
class IgnoredEvent:
    reason: str  # invalid, api_response, other_bot, private, request, unknown


Event = GroupMessageEvent | RecallEvent | MetaEvent | NoticeEvent | IgnoredEvent


def _int_or_none(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _id_or_none(value: Any) -> str | None:
    if isinstance(value, bool) or not isinstance(value, (str, int)) or not str(value):
        return None
    return str(value)


def _str_or_none(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def is_action_response(frame: Any) -> bool:
    return (
        isinstance(frame, dict)
        and "post_type" not in frame
        and any(k in frame for k in ("echo", "retcode", "status"))
    )


def parse_event(event: Any, bot_id: str) -> Event:
    """Classify one pushed OneBot frame. Fails closed: malformed input is `IgnoredEvent`."""
    if not isinstance(event, dict):
        return IgnoredEvent("invalid")
    if "echo" in event or "retcode" in event:
        return IgnoredEvent("api_response")
    post_type = event.get("post_type")
    if post_type == "meta_event":
        status = event.get("status")
        online = status.get("online") if isinstance(status, dict) else None
        return MetaEvent(
            str(event.get("meta_event_type", "unknown")),
            _str_or_none(event.get("sub_type")),
            _int_or_none(event.get("interval")),
            online if isinstance(online, bool) else None,
        )
    if str(event.get("self_id")) != bot_id:
        return IgnoredEvent("other_bot")
    if post_type in ("message", "message_sent"):
        if event.get("message_type") != "group":
            return IgnoredEvent("private")
        message = parse_group_message(event, bot_id)
        if message is None:
            return IgnoredEvent("invalid")
        return GroupMessageEvent(
            message, _int_or_none(event.get("time")), post_type == "message_sent"
        )
    if post_type == "notice":
        notice_type = event.get("notice_type", "unknown")
        if notice_type != "group_recall":
            return NoticeEvent(
                str(notice_type),
                _str_or_none(event.get("sub_type")),
                _id_or_none(event.get("group_id")),
            )
        ids = [_id_or_none(event.get(k)) for k in ("group_id", "message_id", "user_id")]
        if any(v is None for v in ids):
            return IgnoredEvent("invalid")
        group_id, message_id, user_id = ids
        return RecallEvent(
            group_id,
            message_id,
            user_id,
            _id_or_none(event.get("operator_id")),
            _int_or_none(event.get("time")),
        )
    if post_type == "request":
        return IgnoredEvent("request")
    return IgnoredEvent("unknown")


# -- gateway --------------------------------------------------------------------------


class OneBotGateway:
    """One forward WebSocket for events and read-only actions.

    `on_event(frame)` receives every pushed event dict (not action responses) in arrival
    order; it may be sync or async. Async handlers run on one consumer task, so they may
    call `self.call(...)`.
    """

    def __init__(
        self,
        url: str,
        token: str,
        on_event: Callable[[dict], Any] | None = None,
        *,
        allow_actions: bool = False,
        actions: frozenset[str] | None = None,
        connect: Callable[..., Any] | None = None,
        log: Callable[[str], None] = lambda _: None,
        reconnect_initial: float = 1.0,
        reconnect_max: float = 30.0,
        action_timeout: float = 10.0,
        rate_per_second: float = 5.0,
        burst: int = 5,
        max_concurrency: int = 4,
        max_queued_events: int = 10_000,
        drain_timeout: float = 5.0,
        clock: Callable[[], float] = time.monotonic,
        owner_send: bool = False,
        reaction: bool = False,
        answer_send: bool = False,
    ):
        self.url, self._token = url, token
        self.on_event = on_event
        self.allow_actions = allow_actions
        # D-19: whether `owner_send_channel()` may issue the single send channel.
        self._owner_send = bool(owner_send and allow_actions)
        self._owner_channel_issued = False
        # Cutover: whether `answer_send_channel()` may issue the single answer channel.
        self._answer_send = bool(answer_send and allow_actions)
        self._answer_channel_issued = False
        # D-20: whether `reaction_channel()` may issue the single reaction channel.
        self._reaction = bool(reaction and allow_actions)
        self._reaction_channel_issued = False
        self._clock = clock
        # Optional narrower allowlist for this gateway (e.g. the shadow: get_msg/get_image).
        if actions is not None and not frozenset(actions) <= READ_ONLY_ACTIONS:
            raise ValueError("actions must be a subset of READ_ONLY_ACTIONS")
        self.actions = READ_ONLY_ACTIONS if actions is None else frozenset(actions)
        self._connect = connect
        self.log = log
        self.reconnect_initial, self.reconnect_max = reconnect_initial, reconnect_max
        self.action_timeout = action_timeout
        self.drain_timeout = drain_timeout
        self.stats: Counter[str] = Counter()
        self._bucket = TokenBucket(rate_per_second, burst, clock)
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._events: asyncio.Queue = asyncio.Queue(max_queued_events)
        self._pending: dict[str, asyncio.Future] = {}
        self._expired: OrderedDict[str, None] = OrderedDict()
        self._echo_prefix = f"ga-{uuid4().hex[:10]}"
        self._echo_counter = itertools.count(1)
        self._connection: ReadOnlyActions | ReceiveOnly | None = None
        self._connected = asyncio.Event()

    @property
    def connected(self) -> bool:
        return self._connection is not None

    async def wait_connected(self, timeout: float) -> bool:
        try:
            await asyncio.wait_for(self._connected.wait(), timeout)
        except TimeoutError:
            return False
        return True

    # -- actions --------------------------------------------------------------------
    async def call(
        self, action: str, params: dict[str, Any] | None = None, *, timeout: float | None = None
    ) -> Any:
        """Call one allowlisted read-only action and return its `data`."""
        check_action(action)
        if not self.allow_actions:
            raise ActionRefused("actions_disabled")
        if action not in self.actions:
            raise ActionRefused("action_not_enabled")
        if params is not None and not isinstance(params, dict):
            raise ActionRefused("bad_request")
        return await self._request(action, params, timeout, None, None)

    def owner_send_channel(self) -> OwnerSendChannel:
        """Issue the single `OWNER_SEND_ACTION` channel of this gateway (D-19). Refused unless
        the gateway was built with `owner_send=True`, and refused on any second call, so
        exactly one holder (the `#新版` command handler) can ever send."""
        if not self._owner_send:
            raise ActionRefused("owner_send_disabled")
        if self._owner_channel_issued:
            raise ActionRefused("owner_send_channel_taken")
        self._owner_channel_issued = True
        return OwnerSendChannel(self, _OWNER_GRANT)

    def answer_send_channel(self) -> AnswerSendChannel:
        """Issue the single answer-sending channel of this gateway (cutover, RUN-07..09).
        Refused unless the gateway was built with `answer_send=True`, and on any second
        call, so exactly one holder (`live_send.LiveSender`) can ever send answers."""
        if not self._answer_send:
            raise ActionRefused("answer_send_disabled")
        if self._answer_channel_issued:
            raise ActionRefused("answer_send_channel_taken")
        self._answer_channel_issued = True
        return AnswerSendChannel(self, _ANSWER_GRANT)

    def reaction_channel(self, emoji_id: str, rate_per_second: float = 2.0) -> ReactionChannel:
        """Issue the single `REACTION_ACTION` channel of this gateway (D-20), with its emoji
        fixed. Refused unless the gateway was built with `reaction=True`, and on any second
        call, so exactly one holder (`reaction.ProcessingReaction`) can ever react."""
        if not self._reaction:
            raise ActionRefused("reaction_disabled")
        if self._reaction_channel_issued:
            raise ActionRefused("reaction_channel_taken")
        self._reaction_channel_issued = True
        return ReactionChannel(self, _REACTION_GRANT, emoji_id, rate_per_second, self._clock)

    async def _request(
        self,
        action: str,
        params: dict[str, Any] | None,
        timeout: float | None,
        grant: object,
        progress: dict | None,
    ) -> Any:
        self.stats["action_calls"] += 1
        echo: str | None = None
        try:
            async with asyncio.timeout(self.action_timeout if timeout is None else timeout):
                async with self._semaphore:
                    await self._bucket.acquire()
                    connection = self._connection
                    if not isinstance(connection, ReadOnlyActions):
                        raise GatewayDisconnected("not_connected")
                    echo = f"{self._echo_prefix}-{next(self._echo_counter)}"
                    future = asyncio.get_running_loop().create_future()
                    self._pending[echo] = future
                    if progress is not None:
                        progress["writing"] = True
                    try:
                        if grant is None:
                            await connection.send_action(action, dict(params or {}), echo)
                        else:
                            await connection.send_action(
                                action, dict(params or {}), echo, grant=grant
                            )
                    except GatewayError:
                        raise
                    except Exception:
                        raise GatewayDisconnected("send_failed") from None
                    response = await future
        except TimeoutError:
            self.stats["action_timeouts"] += 1
            raise ActionTimeout("timeout") from None
        except GatewayError as exc:
            self.stats[f"action_errors_{exc.kind}"] += 1
            raise
        finally:
            if echo is not None and self._pending.pop(echo, None) is not None:
                # Timed out or cancelled while waiting: a later response is "late".
                self._expired[echo] = None
                while len(self._expired) > 256:
                    self._expired.popitem(last=False)
        return self._unwrap(response)

    def _unwrap(self, response: Any) -> Any:
        if not isinstance(response, dict):
            raise ActionFailed("bad_response")
        retcode = _int_or_none(response.get("retcode"))
        if response.get("status") != "ok" or retcode != 0:
            self.stats["action_failed"] += 1
            raise ActionFailed("failed", retcode)
        self.stats["action_ok"] += 1
        return response.get("data")

    def _resolve(self, response: dict) -> None:
        echo = response.get("echo")
        future = self._pending.pop(echo, None) if isinstance(echo, str) else None
        if future is not None and not future.done():
            self.stats["responses"] += 1
            future.set_result(response)
        elif isinstance(echo, str) and echo in self._expired:
            self.stats["late_responses"] += 1
        else:
            self.stats["unmatched_responses"] += 1

    def _fail_pending(self, kind: str) -> None:
        pending, self._pending = self._pending, {}
        for future in pending.values():
            if not future.done():
                future.set_exception(GatewayDisconnected(kind))

    # -- connection loop ------------------------------------------------------------
    async def run(self, stop: asyncio.Event) -> Counter[str]:
        """Receive until `stop` is set, reconnecting with capped backoff.

        Returns:
            Counters: connects, disconnects, connect_failures, frames, bad_frames, action
            and response counts.
        """
        connect = self._connect
        if connect is None:
            import websockets

            connect = websockets.connect
        consumer = asyncio.create_task(self._consume())
        delay = self.reconnect_initial
        try:
            while not stop.is_set():
                try:
                    raw = await connect(
                        self.url,
                        additional_headers={"Authorization": f"Bearer {self._token}"},
                        max_size=MAX_FRAME_BYTES,
                        open_timeout=10,
                    )
                except Exception as exc:
                    self.stats["connect_failures"] += 1
                    self.log(f"connect failed: {type(exc).__name__}; retry in {delay:.0f}s")
                    await _sleep_or_stop(stop, delay)
                    delay = min(delay * 2, self.reconnect_max)
                    continue
                connection = ReadOnlyActions(raw) if self.allow_actions else ReceiveOnly(raw)
                self._connection = connection
                self._connected.set()
                self.stats["connects"] += 1
                delay = self.reconnect_initial
                self.log("connected")
                receiver = asyncio.create_task(self._receive(connection))
                stopper = asyncio.create_task(stop.wait())
                done, _ = await asyncio.wait(
                    {receiver, stopper}, return_when=asyncio.FIRST_COMPLETED
                )
                stopper.cancel()
                if receiver not in done:
                    receiver.cancel()
                    await asyncio.gather(receiver, return_exceptions=True)
                elif receiver.exception() is not None:
                    self.log(f"connection lost: {type(receiver.exception()).__name__}")
                self._connection = None
                self._connected.clear()
                self._fail_pending("disconnected")
                try:
                    await connection.close()
                except Exception:
                    pass
                self.stats["disconnects"] += 1
                if not stop.is_set():
                    self.log(f"disconnected; reconnect in {delay:.0f}s")
                    await _sleep_or_stop(stop, delay)
                    delay = min(delay * 2, self.reconnect_max)
        finally:
            self._connection = None
            self._connected.clear()
            self._fail_pending("stopped")
            # Handle events already received before stopping, within a bound.
            try:
                await asyncio.wait_for(self._events.join(), self.drain_timeout)
            except TimeoutError:
                self.stats["undrained_events"] += self._events.qsize()
            consumer.cancel()
            await asyncio.gather(consumer, return_exceptions=True)
        return self.stats

    async def _receive(self, connection: ReceiveOnly) -> None:
        async for frame in connection:
            self.stats["frames"] += 1
            try:
                payload = json.loads(frame)
            except (TypeError, ValueError):
                self.stats["bad_frames"] += 1
                continue
            if is_action_response(payload):
                self._resolve(payload)
                continue
            if self.on_event is None:
                continue
            try:
                self._events.put_nowait(payload)
            except asyncio.QueueFull:
                self.stats["dropped_events"] += 1

    async def _consume(self) -> None:
        while True:
            event = await self._events.get()
            try:
                result = self.on_event(event)
                if inspect.isawaitable(result):
                    await result
            except Exception as exc:
                self.stats["handler_errors"] += 1
                self.log(f"event handler failed: {type(exc).__name__}")
            finally:
                self._events.task_done()


@dataclass(frozen=True)
class SendResult:
    """Outcome of one group send: `sent` (platform message id when given), `failed`
    (known not delivered: refused, not connected, non-ok answer) or `uncertain` (the frame
    may have been written: timeout or connection lost afterwards; never retried)."""

    status: str
    message_id: str | None = None
    error_kind: str | None = None


class _GroupSendChannel:
    """`send_group_msg` only, with the given group and segments, no other action and no
    other parameters. Validation of the content is the holder's job."""

    _gateway: OneBotGateway
    _grant: object

    async def send_group_msg(
        self, group_id: str, segments: list[dict], *, timeout: float = 15.0
    ) -> SendResult:
        params = {"group_id": _platform_id(group_id), "message": segments}
        progress: dict = {}
        try:
            data = await self._gateway._request(
                OWNER_SEND_ACTION, params, timeout, self._grant, progress
            )
        except ActionTimeout:
            return SendResult("uncertain" if progress else "failed", None, "timeout")
        except GatewayDisconnected as exc:
            written = progress and exc.kind != "not_connected"
            return SendResult("uncertain" if written else "failed", None, exc.kind)
        except ActionFailed as exc:
            return SendResult("failed", None, f"failed_{exc.retcode}" if exc.retcode else exc.kind)
        except GatewayError as exc:
            return SendResult("failed", None, exc.kind)
        message_id = data.get("message_id") if isinstance(data, dict) else None
        if isinstance(message_id, bool) or not isinstance(message_id, (str, int)):
            return SendResult("sent", None, "no_message_id")
        return SendResult("sent", str(message_id), None)


class OwnerSendChannel(_GroupSendChannel):
    """The owner's write capability (D-19), held by `owner_command.OwnerCommandSender`."""

    def __init__(self, gateway: OneBotGateway, grant: object):
        if grant is not _OWNER_GRANT:
            raise ActionRefused("owner_send_disabled")
        self._gateway, self._grant = gateway, grant


class AnswerSendChannel(_GroupSendChannel):
    """The answer write capability (cutover, RUN-07..09), held by `live_send.LiveSender`."""

    def __init__(self, gateway: OneBotGateway, grant: object):
        if grant is not _ANSWER_GRANT:
            raise ActionRefused("answer_send_disabled")
        self._gateway, self._grant = gateway, grant


@dataclass(frozen=True)
class ReactionResult:
    """Outcome of one reaction: `ok`, or `failed` with a safe `error_kind` (retcode only)."""

    status: str
    error_kind: str | None = None
    retcode: int | None = None


class ReactionChannel:
    """The second write capability (D-20): `set_msg_emoji_like` only, with the emoji id
    selected from the two faces fixed at issue. Both outcomes share one token bucket.
    A limited notice waits for capacity; a processing reaction above the burst is skipped."""

    def __init__(
        self,
        gateway: OneBotGateway,
        grant: object,
        emoji_id: str,
        rate_per_second: float = 2.0,
        clock: Callable[[], float] = time.monotonic,
    ):
        if grant is not _REACTION_GRANT:
            raise ActionRefused("reaction_disabled")
        if not isinstance(emoji_id, str) or not emoji_id.isdigit() or len(emoji_id) > 6:
            raise ActionRefused("bad_emoji")
        self._gateway, self._grant, self._emoji_id = gateway, grant, emoji_id
        # QQ faces: closed mouth (8), or sleepy (25) if 8 is the processing face.
        self._limited_emoji_id = "25" if emoji_id == "8" else "8"
        self._bucket = TokenBucket(rate_per_second, 2, clock)

    async def set_msg_emoji_like(
        self, message_id: str, *, rate_limited: bool = False, timeout: float = 5.0
    ) -> ReactionResult:
        if isinstance(message_id, bool) or not isinstance(message_id, (str, int)):
            return ReactionResult("failed", "bad_message_id")
        if rate_limited:
            await self._bucket.acquire()
        elif not self._bucket.try_acquire():
            return ReactionResult("failed", "rate_limited")
        emoji_id = self._limited_emoji_id if rate_limited else self._emoji_id
        params = {"message_id": _platform_id(message_id), "emoji_id": emoji_id, "set": True}
        try:
            await self._gateway._request(REACTION_ACTION, params, timeout, self._grant, None)
        except ActionFailed as exc:
            return ReactionResult("failed", exc.kind, exc.retcode)
        except GatewayError as exc:
            return ReactionResult("failed", exc.kind)
        return ReactionResult("ok")


async def _sleep_or_stop(stop: asyncio.Event, seconds: float) -> None:
    try:
        await asyncio.wait_for(stop.wait(), timeout=seconds)
    except TimeoutError:
        pass


# -- typed read-only API --------------------------------------------------------------


def _platform_id(value: str | int) -> int | str:
    """OneBot expects numeric ids; keep non-numeric ids as given."""
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    text = str(value)
    try:
        return int(text)
    except ValueError:
        return text


@dataclass(frozen=True)
class ImageInfo:
    """`get_image` result. Download, hashing and dedup: `media.MediaCollector` (T3.3)."""

    local_path: str | None  # SnowLuma's cached file path, if any
    url: str | None  # refreshed download URL, if any
    file_size: int | None
    file_name: str | None


class OneBotApi:
    """Typed wrappers over the allowlisted actions.

    Group ids must come from the caller's run context, never from model arguments.
    """

    def __init__(self, gateway: Any):
        self.gateway = gateway

    async def _dict(self, action: str, params: dict | None = None, **kw) -> dict:
        data = await self.gateway.call(action, params, **kw)
        if not isinstance(data, dict):
            raise ActionFailed("bad_response")
        return data

    async def get_msg(self, message_id: str | int, **kw) -> dict:
        return await self._dict("get_msg", {"message_id": _platform_id(message_id)}, **kw)

    async def get_image(self, file: str, **kw) -> ImageInfo:
        if not isinstance(file, str) or not file:
            raise ActionRefused("bad_request")
        data = await self._dict("get_image", {"file": file}, **kw)
        path, url = data.get("file"), data.get("url")
        size = data.get("file_size")
        if isinstance(size, str) and size.isdigit():
            size = int(size)
        return ImageInfo(
            path if isinstance(path, str) and path and not path.startswith("http") else None,
            url if isinstance(url, str) and url.startswith(("http://", "https://")) else None,
            _int_or_none(size),
            _str_or_none(data.get("file_name")),
        )

    async def get_forward_msg(self, resource_id: str, **kw) -> list[dict]:
        """Read a forward resource by segment id (not the signed root message id)."""
        if not isinstance(resource_id, str) or not 0 < len(resource_id) <= 2048:
            raise ActionRefused("bad_request")
        data = await self._dict("get_forward_msg", {"id": resource_id}, **kw)
        if len(json.dumps(data, ensure_ascii=False).encode("utf-8")) > 1024 * 1024:
            raise ActionFailed("forward_oversize")
        nodes = data.get("messages")
        if not isinstance(nodes, list) or not all(isinstance(node, dict) for node in nodes):
            raise ActionFailed("bad_response")
        return nodes

    async def get_status(self, **kw) -> dict:
        return await self._dict("get_status", **kw)

    async def get_login_info(self, **kw) -> dict:
        return await self._dict("get_login_info", **kw)

    async def get_group_list(self) -> list[dict]:
        rows = await self.gateway.call("get_group_list", {})
        if not isinstance(rows, list) or not all(
            isinstance(row, dict) and _id_or_none(row.get("group_id")) is not None for row in rows
        ):
            raise ActionFailed("bad_response")
        return rows

    async def get_group_info(self, group_id: str, no_cache: bool = False, **kw) -> dict:
        params = {"group_id": _platform_id(group_id), "no_cache": no_cache}
        return await self._dict("get_group_info", params, **kw)

    async def get_group_member_info(
        self, group_id: str, user_id: str, no_cache: bool = False, **kw
    ) -> dict:
        params = {
            "group_id": _platform_id(group_id),
            "user_id": _platform_id(user_id),
            "no_cache": no_cache,
        }
        return await self._dict("get_group_member_info", params, **kw)

    async def get_group_member_list(
        self, group_id: str, no_cache: bool = False, **kw
    ) -> list[dict]:
        params = {"group_id": _platform_id(group_id), "no_cache": no_cache}
        data = await self.gateway.call("get_group_member_list", params, **kw)
        if not isinstance(data, list) or not all(isinstance(m, dict) for m in data):
            raise ActionFailed("bad_response")
        return data
