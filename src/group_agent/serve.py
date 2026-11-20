"""Online mode, record only: `group-agent --serve` (T3.4 / T3.7 / T3.9 wiring).

One forward WebSocket to the local SnowLuma (`gateway.OneBotGateway`, read-only action
allowlist), event intake with collection and admission (`intake.EventIntake`), image
collection and quote backfill, the run orchestrator (`orchestrator.Orchestrator`) with the
live agent (`live_agent.LiveAgent`), the persistent usage ledger and the periodic
retention cleanup. Answers end as `recorded_not_sent` outbound rows. The single exception
(D-19) is the owner's `#新版` command (`owner_command.NewVersionCommand`), which may send
one `〔新版〕` answer per question when `[serve] owner_command_send` / `--allow-owner-send`
is on; it alone receives the gateway's send channel. D-20: with `[serve]
processing_reaction` / `--reaction`, `reaction.ProcessingReaction` alone receives the
gateway's reaction channel and puts the "processing" emoji on admitted requests.
Cutover (D-21): with `[serve] live_send` / `--live-send`, `live_send.LiveSender` alone
receives the gateway's answer channel; the orchestrator hands it the run id of each
re-checked, rate-limited outbound row (`publisher`) and every answer is sent. The `#新版`
command is then switched off (it would answer a second time).

`dry_models=True` replaces the model clients with local fakes (`dry_model_builder`): no
credentials are loaded for models or search, every call costs 0, and the agent still goes
through material rendering, the structured route decision and the answer chain.
"""

from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack
from collections import Counter
from collections.abc import Callable
from dataclasses import replace
import json
from pathlib import Path
import re
import time
from typing import Any

from .admission import AccessPolicy, RequestQueue
from .backfill import ReplyBackfill
from .config import ModelCredentials, Settings, load_model_credentials, load_snowluma_credentials
from .gateway import READ_ONLY_ACTIONS, OneBotApi, OneBotGateway, RecallEvent, parse_event
from . import received_groups
from .group_info import GroupInfoCache, GroupInfoTool
from .intake import EventIntake
from .knowledge_base import KnowledgeBase
from .knowledge_search import KnowledgeSnapshot
from .live_agent import LiveAgent, LiveConfig
from .live_send import LiveSender
from .media import ImageSourcePolicy, MediaCollector
from .orchestrator import TEACHING_PENDING_TEXT, Orchestrator
from .owner_command import NewVersionCommand
from .owner_control import OwnerControl, PauseState
from .policy import COST_REJECTED
from .reaction import ProcessingReaction
from .retention import periodic_cleanup
from .self_sync import SelfSync, default_snowluma_db
from .store import Store
from .billing import open_account_book, open_google_quota
from .runflow import CURRENT_RUN
from .usage import UsageLedger, cn_day

ROOT = Path(__file__).resolve().parents[2]
DRY_ANSWER = "[离线模拟] 已接收请求；未调用模型、工具或 QQ 发送接口。"  # = contracts.MockModel
_GROUP_WORDS = re.compile(r"群主|管理员|多少人|人数|群名|头衔|名片|入群")


def _dry_decision(prompt: str) -> dict:
    request = prompt.rsplit("<request>", 1)[-1]
    return {
        "route": "chat",
        "should_clarify": False,
        "needs_group_info": bool(_GROUP_WORDS.search(request)),
        "reason": "离线模拟判断",
    }


def dry_model_builder(spec, creds) -> Any:
    """Local fake for every configured model: route decisions say chat (group info when
    the request asks for it), answers are the offline mock text; usage is zero."""
    from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
    from pydantic_ai.models.function import FunctionModel
    from pydantic_ai.profiles import ModelProfile
    from pydantic_ai.usage import RequestUsage

    def fn(messages, info):
        prompt = ""
        for message in messages:
            for part in getattr(message, "parts", []):
                content = getattr(part, "content", None)
                if isinstance(content, str):
                    prompt += content
                elif isinstance(content, list):
                    prompt += "".join(c for c in content if isinstance(c, str))
        mode = getattr(info.model_request_parameters, "output_mode", "text")
        if info.output_tools and mode == "tool":
            decision = _dry_decision(prompt)
            return ModelResponse(
                parts=[ToolCallPart(info.output_tools[0].name, decision)], usage=RequestUsage()
            )
        if mode in ("native", "prompted"):
            body = json.dumps(_dry_decision(prompt), ensure_ascii=False)
            return ModelResponse(parts=[TextPart(body)], usage=RequestUsage())
        return ModelResponse(parts=[TextPart(DRY_ANSWER)], usage=RequestUsage())

    return FunctionModel(fn, profile=ModelProfile(supports_json_schema_output=True))


def latest_knowledge(root: Path = ROOT) -> Path | None:
    """The migrated knowledge base (read-only use by replay tools); legacy snapshots under
    data/fixtures/ are migration input only and are no longer read at run time."""
    path = root / "data/knowledge/knowledge.sqlite3"
    return path if path.is_file() else None


def build_live_config(
    settings: Settings,
    *,
    ledger,
    media_root: Path,
    dry_models: bool,
    web: bool,
    group_info: Callable[[str, str], Any] | None,
    knowledge_path: Path | None,
    log: Callable[[str], None] = lambda _: None,
    knowledge: Any = None,
    previous: LiveConfig | None = None,
    reuse_web: bool = False,
) -> LiveConfig:
    """`knowledge`: the live `KnowledgeBase` (serve); else `knowledge_path` is opened as a
    read-only `KnowledgeSnapshot` (replay tools)."""
    from .models import ModelBreaker
    from .web import SafeHttp, SupplementBreaker, build_supplement_client, build_web_client

    web = web and settings.web_enabled

    try:
        roles = settings.model_roles
        gemini = settings.models[roles.get("answer", "gemini")]
        deepseek = settings.models[roles.get("fallback", "deepseek")]
        decision = settings.models[roles.get("decision", roles.get("answer", "gemini"))]
    except KeyError as exc:
        from .config import ConfigError

        raise ConfigError(f"models.{exc.args[0]}: required for --serve") from None
    web_client = web_http = supplement = breaker = None
    if dry_models:
        # Fakes cost nothing: FunctionModel estimates token counts, so zero the prices.
        free = {"price_input": 0.0, "price_output": 0.0, "price_cached_input": None}
        gemini, deepseek = replace(gemini, **free), replace(deepseek, **free)
        decision = replace(decision, **free)
        gemini_creds = deepseek_creds = ModelCredentials("https://dry-run.invalid", "dry-run")
        decision_creds = gemini_creds
        builder = dry_model_builder
    else:
        gemini_creds = load_model_credentials(settings.credentials, gemini)
        deepseek_creds = load_model_credentials(settings.credentials, deepseek)
        decision_creds = load_model_credentials(settings.credentials, decision)
        builder = None
        if reuse_web and previous is not None:
            web_client, web_http = previous.web_client, previous.web_http
            supplement, breaker = previous.web_supplement, previous.supplement_breaker
        elif web:
            web_client, web_http = build_web_client(settings), SafeHttp()
            supplement = build_supplement_client(settings)
            if supplement is not None:
                # One breaker per process, shared by all runs (D-18).
                breaker = SupplementBreaker(
                    settings.web.supplement_breaker_failures,
                    settings.web.supplement_breaker_seconds,
                )
    from .persona import prompt_options

    extra = {"model_builder": builder} if builder is not None else {}
    return LiveConfig(
        config_revision=settings.revision,
        prompt_text=prompt_options(settings),
        runtime_limits=settings.limits,
        gemini=gemini,
        deepseek=deepseek,
        gemini_creds=gemini_creds,
        deepseek_creds=deepseek_creds,
        decision_model=decision,
        decision_creds=decision_creds,
        answer_tools=settings.answer_tool_loop,
        decision_mode=settings.decision_mode,
        answer_thinking=settings.answer_thinking,
        answer_same_model_retry=settings.answer_same_model_retry,
        ledger=ledger,
        media_root=media_root,
        knowledge=knowledge
        if knowledge is not None
        else KnowledgeSnapshot(knowledge_path)
        if knowledge_path
        else None,
        web_client=web_client,
        web_http=web_http,
        web_price_cny=settings.web.price_per_call,
        web_supplement=supplement,
        supplement_min_seconds=settings.web.supplement_min_seconds,
        supplement_breaker=breaker,
        # D-20: one Gemini breaker per process, shared by every run (decision and answer).
        model_breaker=ModelBreaker(
            settings.model_breaker_failures, settings.model_breaker_cooldown_s, log=log
        ),
        group_info=group_info,
        teaching_text=TEACHING_PENDING_TEXT,
        fallback_reply_text=settings.fallback_reply or None,
        web_unavailable_note="本次联网搜索与网页读取不可用。",
        group_info_unavailable_note="群信息查询暂时不可用。",
        **extra,
    )


def model_ids(settings: Settings, dry_models: bool) -> dict[str, str]:
    if dry_models:
        return {name: "dry-run" for name in settings.models}
    ids = {name: spec.model for name, spec in settings.models.items()}
    web = settings.web
    if web.search is None:
        ids["gemini_search"] = web.gemini_model
        if web.gemini_url_context:
            ids["gemini_url_context"] = web.gemini_model
    else:  # V3: each backend's model comes from its own model table
        for key, backend in (("search", "gemini_search"), ("fetch", "gemini_url_context")):
            for alias in (getattr(web, key), getattr(web, f"{key}_fallback")):
                # "deepseek" is the web-tools keyword, never a model alias here.
                if alias != "deepseek" and alias in settings.models:
                    ids[backend] = settings.models[alias].model
    ids["deepseek_web"] = "deepseek-web-tools"
    return ids


def control_models(settings: Settings, dry_models: bool) -> dict[str, str]:
    """Answer and search model ids for the owner's `#状态` line (settings only)."""
    if dry_models:
        return {"answer_model": "dry-run", "search_model": "dry-run"}
    ids = model_ids(settings, dry_models)
    answer_alias = settings.model_roles.get("answer", "gemini")
    web = settings.web
    deepseek_search = (
        (web.search == "deepseek") if web.search is not None else (web.search_provider != "gemini")
    )
    search = ids["deepseek_web"] if deepseek_search else ids.get("gemini_search", "?")
    return {"answer_model": ids.get(answer_alias, answer_alias), "search_model": search}


async def offline_wait(stop):
    await stop.wait()
    return {}


async def run_serve(settings: Settings, **kwargs) -> dict:
    """Own all engine resources, including partial startup failures."""
    from .runtime import instance_lock

    async with AsyncExitStack() as stack:
        stack.enter_context(instance_lock(kwargs["db_path"]))
        return await _run_serve(settings, stack=stack, **kwargs)


async def _run_serve(
    settings: Settings,
    *,
    db_path: Path,
    media_root: Path,
    duration: float | None,
    dry_models: bool,
    budget_cny: float | None,
    web: bool = True,
    cleanup: bool = True,
    inject: list[dict] | None = None,
    knowledge_path: Path | None = None,
    connect: Callable[..., Any] | None = None,
    log: Callable[[str], None] = lambda _: None,
    stop: asyncio.Event | None = None,
    owner_send: bool | None = None,
    self_sync: bool | None = None,
    snowluma_db: Path | None = None,
    reaction: bool | None = None,
    live_send: bool | None = None,
    stop_file: Path | None = None,
    account_book: bool = True,
    owner_control: bool | None = None,
    on_ready: Callable[[], None] = lambda: None,
    stack: AsyncExitStack,
    on_session: Callable = lambda session: None,
    offline: bool = False,
) -> dict:
    """Run the online record-only mode until `duration` elapses or `stop` is set;
    returns a summary with counts only (no chat content, ids or secrets).

    `account_book` (default on) appends every settled call to the account-wide usage book
    (`billing.AccountBook`, `[billing] db_path`, source "serve"); dry runs never write it.
    A book that cannot be opened is logged and serving goes on without it.

    `owner_send` (default: `settings.owner_command_send`) lets the owner's `#新版` command
    send (D-19); the gateway then issues its single send channel straight to the command
    handler. Everything else stays record-only either way.

    `self_sync` (default: `settings.self_sync`) reads the bot account's own messages from
    SnowLuma's message store read-only (`self_sync.SelfSync`; path `snowluma_db`, else
    `settings.snowluma_db`, else derived from the bot id).

    `reaction` (default: `settings.processing_reaction`, D-20) puts the "processing" emoji on
    each admitted request; the gateway issues its single reaction channel straight to
    `reaction.ProcessingReaction`, which intake calls on admission.

    `live_send` (default: `settings.live_send`, the cutover) sends every answer through
    `live_send.LiveSender`, which alone gets the gateway's answer channel; the owner's
    `#新版` sending is then off. `stop_file`: the process stops gracefully once that file
    exists (the launcher's stop script creates it).

    `owner_control` (default: `settings.owner_control`) enables the owner's `#暂停` / `#恢复`
    / `#状态` commands (`owner_control.OwnerControl`): the pause flag is persisted in the
    store and loaded here; while paused no reply run starts. Its confirmation goes through
    the live answer sender (`LiveSender.deliver`), or is only recorded without live send."""
    owner_send = settings.owner_command_send if owner_send is None else owner_send
    owner_control = settings.owner_control if owner_control is None else owner_control
    self_sync = settings.self_sync if self_sync is None else self_sync
    reaction = settings.processing_reaction if reaction is None else reaction
    live_send = settings.live_send if live_send is None else live_send
    if live_send and owner_send:
        owner_send = False
        log("live_send is on: the owner's #新版 command does not send")
    from .runtime import ServeSession
    from .config import SnowLumaCredentials

    credentials = (
        SnowLumaCredentials("ws://127.0.0.1:1", "offline")
        if offline
        else load_snowluma_credentials(settings.credentials)
    )
    db_path.parent.mkdir(parents=True, exist_ok=True)
    media_root.mkdir(parents=True, exist_ok=True)
    store = Store(db_path)
    stack.callback(store.close)
    started = time.time()
    pause = PauseState(store, enabled=owner_control or offline, started=started)
    if pause.paused:
        log("owner control: starting paused (persisted); the owner sends #恢复 to resume")
    stop = stop or asyncio.Event()
    if duration:
        timer = asyncio.get_running_loop().call_later(duration, stop.set)
        stack.callback(timer.cancel)
    access = AccessPolicy.from_settings(settings)
    book = None
    try:
        book = open_account_book(
            settings, "serve", enabled=account_book and not dry_models, run_id=CURRENT_RUN.get
        )
    except Exception as exc:  # noqa: BLE001 - class name only; serving must not depend on it
        log(f"account book unavailable: {type(exc).__name__}")
    if book is not None:
        stack.callback(book.close)
    quota = None
    try:
        quota = open_google_quota(settings, book)  # B3: free monthly Google search queries
    except Exception as exc:  # noqa: BLE001 - without it searches keep their list price
        log(f"google free quota unavailable: {type(exc).__name__}")
    usage = UsageLedger(
        store,
        daily_cap_cny=settings.daily_cost_cny,
        limits=settings.limits,
        budget_cny=budget_cny,
        model_ids=model_ids(settings, dry_models),
        account=book,
        source="serve" if book is not None else None,
        quota=quota,
    )
    queue = RequestQueue(256)
    holder: dict[str, Any] = {}

    def on_event(event: Any) -> None:
        result = holder["intake"].intake(event)
        holder["groups"].observe(event, settings.bot_id)
        if (
            result.status == "notice_group_decrease"
            and str(event.get("user_id")) == settings.bot_id
        ):
            gid = str(event.get("group_id"))
            with store.con:
                received_groups.stop_boundary(store.con, settings.bot_id, gid, left=True)
            holder["orchestrator"].cancel_group(gid)
        holder["command"].observe(result)
        if owner_control:
            holder["control"].observe(result)
        if holder["sync"] is not None and result.seq is not None:
            holder["sync"].observe(result.message)
        if result.status in ("recall_revoked", "recall_tombstone"):
            parsed = parse_event(event, settings.bot_id)
            if isinstance(parsed, RecallEvent):
                holder["orchestrator"].on_recall(parsed.group_id, parsed.message_id)

    gateway = OneBotGateway(
        credentials.ws_url,
        credentials.token,
        on_event,
        allow_actions=True,
        actions=READ_ONLY_ACTIONS,
        connect=connect,
        log=log,
        owner_send=owner_send,
        reaction=reaction,
        answer_send=live_send,
    )
    api = OneBotApi(gateway)
    # Cutover: the answer channel (if enabled) goes straight into its one holder; the
    # orchestrator only gets the holder's `deliver` hook (run ids of its own rows).
    holder["live"] = (
        LiveSender(gateway.answer_send_channel(), store, access, log=log) if live_send else None
    )
    if holder["live"] is not None:
        stack.push_async_callback(holder["live"].aclose)
        holder["live"].set_pause_gate(pause.is_paused)
    # D-20: the reaction channel (if enabled) goes straight into its one holder; requests
    # the cost gate would reject get no "processing" emoji.
    holder["reaction"] = (
        ProcessingReaction(
            gateway.reaction_channel(settings.reaction_emoji_id),
            store,
            should_react=lambda r: usage.cost_mode(r.is_owner, r.message.group_id) != COST_REJECTED,
            log=log,
        )
        if reaction
        else None
    )
    if holder["reaction"] is not None:
        stack.push_async_callback(holder["reaction"].aclose)
    collection = settings.collection
    media = MediaCollector(
        store,
        api,
        media_root,
        policy=ImageSourcePolicy(collection.image_host_suffixes),
        max_bytes=collection.max_image_mb * 1024 * 1024,
        rate_per_second=collection.download_per_second,
        max_concurrency=collection.max_concurrent_downloads,
        max_pending=collection.max_pending_downloads,
        log=log,
    )

    stack.push_async_callback(media.aclose)

    def on_backfilled(seq: int, message: Any) -> None:
        media.schedule(seq, message)
        if message.user_id == settings.bot_id and not message.reply_ids:
            # An old-pipeline reply without a quote: pair it with its request (answers).
            row = store.message_row(seq)
            store.link_unquoted_bot_reply(seq, message, row["platform_time"] if row else None)

    backfill = ReplyBackfill(
        store,
        api,
        settings.bot_id,
        lambda: access.snapshot.blocked_users,
        on_stored=on_backfilled,
    )
    stack.push_async_callback(backfill.aclose)
    groups = GroupInfoCache(api)
    holder["groups"] = groups
    holder["sync"] = None
    if self_sync:
        holder["sync"] = SelfSync(
            store,
            settings.bot_id,
            snowluma_db or settings.snowluma_db or default_snowluma_db(settings.bot_id),
            excluded_groups=lambda: (
                access.snapshot.blocked_groups | access.snapshot.collection_off_groups
            ),
            interval=settings.self_sync_interval_s,
            on_stored=media.schedule,
            log=log,
        )
    if holder["sync"] is not None:
        stack.push_async_callback(holder["sync"].aclose)
    holder["intake"] = EventIntake(
        store,
        access,
        admit=True,
        sink=queue,
        media=media,
        on_stored=backfill.schedule,
        on_admitted=holder["reaction"].admitted if holder["reaction"] else None,
        control=(lambda m: holder["control"].classify(m)) if owner_control else None,
        paused=pause.is_paused,
    )
    kb_path = knowledge_path if knowledge_path is not None else settings.knowledge_db
    knowledge = KnowledgeBase(kb_path) if kb_path is not None else None
    if knowledge is not None:
        stack.callback(knowledge.close)
        counts = knowledge.counts()
        log(f"knowledge base: {counts['entries']} entries, {counts['revisions']} revisions")
        if not counts["entries"]:
            log("knowledge base is empty: run scripts/migrate_knowledge.py first")
    config = build_live_config(
        settings,
        ledger=usage,
        media_root=media_root,
        dry_models=dry_models,
        web=web,
        group_info=lambda group_id, user_id: GroupInfoTool(groups, group_id, user_id),
        knowledge_path=None,
        knowledge=knowledge,
        log=log,
    )
    orchestrator = Orchestrator(
        store,
        access,
        settings,
        agent_factory=lambda host: LiveAgent(host, config),
        usage=usage,
        limits=settings.limits,
        knowledge=knowledge,
        media_root=media_root,
        forward_fetch=None if offline else api.get_forward_msg,
        backfill=backfill,
        media=media,
        self_sync=holder["sync"],
        publisher=holder["live"].deliver if holder["live"] is not None else None,
        paused=pause.is_paused,
        log=log,
    )
    stack.push_async_callback(orchestrator.aclose)
    holder["orchestrator"] = orchestrator
    # Owner remote control: the confirmation is handed to the answer sender by row id (the
    # same hook the orchestrator gets); without live send it is only recorded.

    def status_info() -> dict:
        report = usage.report()
        return {
            "spent": report["daily_spent_cny"],
            "cap": report["daily_cap_cny"],
            "cost_mode": report["cost_mode"],
            **control_models(session.settings, dry_models),
        }

    holder["control"] = OwnerControl(
        store,
        access,
        pause,
        cancel_replies=orchestrator.cancel_replies,
        deliver=holder["live"].deliver if holder["live"] is not None else None,
        rate=orchestrator.rate,
        status_info=status_info,
        log=log,
    )
    stack.push_async_callback(holder["control"].aclose)
    # D-19: the send channel (if enabled) goes straight into the command handler, which
    # builds the only sender around it; nothing else in this process gets either.
    holder["command"] = NewVersionCommand(
        store,
        access,
        orchestrator,
        channel=gateway.owner_send_channel() if owner_send else None,
        limits=orchestrator.limits,
        log=log,
    )
    stack.push_async_callback(holder["command"].aclose)
    session = ServeSession(
        replace(settings, live_send=live_send),
        orchestrator,
        usage,
        pause,
        gateway,
        holder,
        config,
        media_root,
        dry_models,
        web,
        log,
        offline,
    )
    session.use_profiles(await session.prepare_profiles(settings, initial=True))
    on_session(session)
    cleanup_reports: list[dict] = []

    async def injector() -> None:
        if not inject:
            return
        if not offline:
            await gateway.wait_connected(15)
        for event in inject:
            on_event(event)

    async def stop_file_watch() -> None:
        while stop_file is not None and not stop.is_set():
            if stop_file.exists():
                log("stop file found: shutting down")
                stop.set()
                return
            try:
                await asyncio.wait_for(stop.wait(), 1.0)
            except TimeoutError:
                pass

    tasks = [
        asyncio.create_task(gateway.run(stop) if not offline else offline_wait(stop)),
        asyncio.create_task(orchestrator.serve(queue, stop)),
        asyncio.create_task(injector()),
        asyncio.create_task(stop_file_watch()),
    ]
    if not offline:
        tasks.append(
            asyncio.create_task(
                received_groups.sync_directory(
                    stop, store, access, gateway, api, orchestrator.cancel_group
                )
            )
        )
    if holder["sync"] is not None:
        tasks.append(asyncio.create_task(holder["sync"].run(stop)))
    if cleanup:
        tasks.append(
            asyncio.create_task(
                periodic_cleanup(
                    stop,
                    lambda: session.settings.retention.cleanup_interval_minutes * 60,
                    lambda: session.cleanup(),
                    on_report=cleanup_reports.append,
                    log=log,
                )
            )
        )
    try:
        on_ready()
        log(f"configuration adopted: {settings.revision or 'defaults'}")
        results = await asyncio.gather(*tasks)
        connection = results[0]
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await orchestrator.aclose()
    summary = {
        "mode": "serve_live_send" if session.settings.live_send else "serve_record_only",
        "live_send": session.settings.live_send,
        "live_sends": dict(holder["live"].counts) if holder["live"] is not None else None,
        "dry_models": dry_models,
        "web": web and not dry_models,
        "read_actions": sorted(gateway.actions),
        "started_at": started,
        "ended_at": time.time(),
        "connection": dict(connection),
        "events": dict(holder["intake"].counts),
        "requests": dict(queue.counts),
        "orchestrator": dict(orchestrator.counts),
        "interrupted_previous_runs": orchestrator.interrupted,
        "store": {
            k: v
            for k, v in store.summary().items()
            if k in ("messages", "reply_requests", "runs", "run_states", "outbound_records")
        },
        "outbound_states": store.summary()["outbound_states"],
        "session": session_counts(store, started),
        "usage": usage.report(),
        "account_book": None
        if book is None
        else {
            "rows": usage.counts["account_rows"],
            "write_failed": usage.counts["account_write_failed"],
        },
        "media": dict(media.counts),
        "backfill": dict(backfill.counts),
        "group_info_cache": dict(groups.counts),
        "cleanup_runs": len(cleanup_reports),
        "injected_events": len(inject or []),
        "owner_command_send": owner_send,
        "owner_commands": dict(holder["command"].counts),
        "owner_control": {
            "enabled": owner_control,
            "paused": pause.paused,
            "since": pause.since if owner_control else None,
            "counts": dict(holder["control"].counts),
            "events": store.summary()["control_events"],
        },
        "self_sync": dict(holder["sync"].counts) if holder["sync"] is not None else None,
        # D-20: per model key: state, times opened, seconds open, skipped calls, probes.
        "model_breaker": session.config.model_breaker.summary()
        if session.config.model_breaker
        else {},
        "processing_reaction": reaction,
        "reactions": dict(holder["reaction"].counts) if holder["reaction"] else None,
        "sent_to_qq": store.summary()["sent_to_qq"],
        "knowledge": knowledge.counts() if knowledge is not None else None,
    }
    return summary


def session_counts(store: Store, since: float) -> dict:
    """Run and outbound counts created at or after `since` (for a smoke report)."""
    runs = Counter(
        r[0] for r in store.con.execute("SELECT state FROM runs WHERE created_at >= ?", (since,))
    )
    outbound = Counter(
        r[0]
        for r in store.con.execute("SELECT state FROM outbound WHERE created_at >= ?", (since,))
    )
    return {"runs": dict(runs), "outbound": dict(outbound), "day": cn_day(since)}
