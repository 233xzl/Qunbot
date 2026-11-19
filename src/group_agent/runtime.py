"""In-process engine lifecycle and per-module configuration application."""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from dataclasses import replace
import os
from pathlib import Path
import time
from uuid import uuid4

from .config import ConfigError
from .config_service import ConfigService, flatten, plain


@contextmanager
def instance_lock(db_path):
    path = Path(str(db_path) + ".engine.lock")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as file:
        file.seek(0)
        if path.stat().st_size == 0:
            file.write(b"0")
            file.flush()
        file.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise ConfigError("engine: this database is already in use by an engine") from None
        try:
            yield
        finally:
            if os.name == "nt":
                file.seek(0)
                msvcrt.locking(file.fileno(), msvcrt.LK_UNLCK, 1)


IMMEDIATE = {
    "daily_cost_cny",
    "owner_id",
    "blocked_users",
    "blocked_groups",
    "web_enabled",
    "live_send",
}
NEXT_RUN = {
    "group_profiles",
    "models",
    "model_roles",
    "web",
    "prompts",
    "runtime",
    "deadline_seconds",
    "context_messages",
    "fallback_reply",
    "answer_tool_loop",
    "decision_mode",
    "answer_thinking",
    "answer_same_model_retry",
    "model_breaker_failures",
    "model_breaker_cooldown_s",
    "credential_refs",
}
IMMEDIATE_RUNTIME = {
    "global_active",
    "user_active",
    "global_queue",
    "user_queue",
    "group_min_interval_seconds",
    "group_per_minute",
    "account_per_minute",
    "degrade_ratio",
    "cap_ratio",
    "busy_notice_window_seconds",
}

META = {"revision", "published_revision", "config_store", "sources"}


def application_plan(before, after):
    a, b = flatten(plain(before)), flatten(plain(after))
    changes = []
    for key in sorted(a.keys() | b.keys()):
        top = key.split(".")[0]
        if top in META or a.get(key) == b.get(key):
            continue
        mode = "maintenance"
        if top in NEXT_RUN or key in {
            "billing.usd_cny",
            "billing.usd_cny_date",
            "billing.google_search_usd_per_1000",
        }:
            mode = "new_run"
        if (
            (top == "group_profiles" and key.endswith((".daily_cost_cny", ".enabled")))
            or top in IMMEDIATE
            or key == "collection.disabled_groups"
            or key in {"runtime." + name for name in IMMEDIATE_RUNTIME}
        ):
            mode = "immediate"
        if top == "retention":
            mode = "next_cleanup"
        if key == "live_send" and after.live_send and not before.live_send:
            mode = "maintenance"
        if top == "credential_refs" and "SNOWLUMA_" in key:
            mode = "maintenance"
        changes.append({"field": key, "apply": mode})
    return {"changes": changes, "maintenance": any(x["apply"] == "maintenance" for x in changes)}


class ServeSession:
    def __init__(
        self,
        settings,
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
    ):
        self.settings, self.orchestrator, self.usage = settings, orchestrator, usage
        self.pause, self.gateway, self.holder, self.config = pause, gateway, holder, config
        pause.enabled = True
        pause.reload()
        self.media_root, self.dry_models, self.web, self.log = media_root, dry_models, web, log
        self.offline = offline
        if holder["live"]:
            holder["live"].set_pause_gate(pause.is_paused)
            holder["live"].enabled = lambda: self.settings.live_send
        from .runflow import CURRENT_GROUP

        usage.before_call = self.check_call
        usage.owner_call = self.owner_call
        usage.web_allowed = lambda: (
            self.web
            and self.settings.web_enabled
            and self.usage.cost_mode(True, CURRENT_GROUP.get()) == "normal"
        )

    def cleanup(self, *, dry_run=False):
        from .retention import run_cleanup

        if self.orchestrator.active:
            return {"skipped_active": len(self.orchestrator.active)}
        return run_cleanup(
            self.orchestrator.store,
            self.settings.retention,
            media_root=self.media_root,
            dry_run=dry_run,
        )

    def owner_call(self):
        from .runflow import CURRENT_RUN

        run_id = CURRENT_RUN.get()
        active = self.orchestrator.active.get(run_id)
        return bool(
            active
            and (
                active[0].user_id == self.settings.owner_id or run_id in self.orchestrator._captures
            )
        )

    def check_call(self):
        from .runflow import CURRENT_RUN
        from .models import BudgetExceeded

        active = self.orchestrator.active.get(CURRENT_RUN.get())
        if active and self.orchestrator.check(active[0]):
            raise BudgetExceeded("runtime policy refuses this call")

    async def prepare_profiles(self, settings, *, reload_credentials=False, initial=False):
        agents = {}
        try:
            if not initial:
                agents[None] = await self.prepare(settings, reload_credentials=reload_credentials)
            for gid, profile in settings.group_profiles.items():
                if profile.get("enabled") and (
                    profile.get("prompts") or profile.get("model_roles")
                ):
                    agents[gid] = await self.prepare(
                        settings.for_group(gid), group_id=gid, reload_credentials=reload_credentials
                    )
            return agents
        except BaseException:
            for agent in agents.values():
                await self.discard(agent)
            raise

    async def prepare(self, settings, *, group_id=None, reload_credentials=False):
        from .serve import build_live_config
        from .live_agent import LiveAgent

        old = self.orchestrator.group_agents.get(group_id, self.orchestrator._agent)
        prior = (
            self.settings.for_group(group_id)
            if group_id in self.orchestrator.group_agents
            else self.settings
        )
        previous = old.config

        def search_models(value):
            aliases = {
                value.web.search,
                value.web.fetch,
                value.web.search_fallback,
                value.web.fetch_fallback,
            }
            if value.web.search is None:
                aliases.add("gemini")
            return {
                name: value.models[name]
                for name in aliases
                if name in value.models and name != "deepseek"
            }

        same_web = (
            not reload_credentials
            and settings.web == prior.web
            and search_models(settings) == search_models(prior)
            and settings.billing == prior.billing
            and settings.credentials == prior.credentials
            and settings.web_enabled == prior.web_enabled
            and settings.prompts.get("search") == prior.prompts.get("search")
        )
        config = build_live_config(
            settings,
            ledger=self.usage,
            media_root=self.media_root,
            dry_models=self.dry_models,
            web=self.web,
            group_info=previous.group_info,
            knowledge_path=None,
            knowledge=previous.knowledge,
            log=self.log,
            previous=previous,
            reuse_web=same_web,
        )
        if (
            settings.models == prior.models
            and settings.model_roles == prior.model_roles
            and not reload_credentials
            and settings.credentials == prior.credentials
        ):
            config = replace(config, model_breaker=previous.model_breaker)
        agent = LiveAgent(self.orchestrator, config)
        for attr, spec, creds in (
            ("gemini", "gemini", "gemini_creds"),
            ("deepseek", "deepseek", "deepseek_creds"),
            ("gemini_decider", "decision_model", "decision_creds"),
        ):
            candidate, old_model = getattr(agent, attr), getattr(old, attr)
            if (
                not reload_credentials
                and candidate.spec == old_model.spec
                and getattr(config, creds) == getattr(previous, creds)
                and candidate.breaker is old_model.breaker
                and candidate.thinking_level == old_model.thinking_level
            ):
                setattr(agent, attr, old_model)
        try:
            for model in self.orchestrator.model_resources(
                agent
            ) - self.orchestrator.model_resources(old):
                _ = model.model  # construct SDK clients; no request is sent
        except BaseException:
            await self.discard(agent)
            raise
        return agent

    async def discard(self, agent):
        for model in self.orchestrator.model_resources(agent) - self.orchestrator._model_resources:
            await model.aclose()

    def use_profiles(self, agents):
        self.orchestrator.group_agents = agents
        for agent in agents.values():
            self.orchestrator._model_resources.update(self.orchestrator.model_resources(agent))

    def adopt(self, settings, agents):
        agent = agents[None]
        self.use_profiles({gid: a for gid, a in agents.items() if gid is not None})
        self.settings, self.config = settings, agent.config
        self.orchestrator.update(settings, agent)
        for prepared in agents.values():
            prepared.config.model_breaker.threshold = settings.model_breaker_failures
            prepared.config.model_breaker.cooldown_s = settings.model_breaker_cooldown_s

    def control(self, action):
        control = self.holder["control"]
        control.apply(action, self.settings.bot_id, "web", uuid4().hex)

    def status(self):
        orch = self.orchestrator
        return {
            "state": "paused" if self.pause.paused else "running",
            "offline": self.offline,
            "gateway_connected": self.gateway.connected,
            "active": orch.scheduler.total_active,
            "queued": orch.scheduler.waiting_count(),
            "usage": self.usage.report(),
            "config_revision": self.settings.revision,
            "published_revision": self.settings.published_revision,
            "counts": dict(orch.counts),
            "started_at": self.pause.since,
        }


class RuntimeController:
    def __init__(self, settings, **options):
        self.settings, self.options = settings, options
        self.session = None
        self.task = None
        self.state, self.error = "stopped", None
        self.lock = asyncio.Lock()
        self.stop_event = asyncio.Event()
        self.ready = asyncio.Event()
        self.summary = None

    def status(self):
        result = {
            "state": self.state,
            "error": self.error,
            "config_revision": self.settings.revision,
            "published_revision": self.settings.published_revision,
            "overrides": [k for k, v in self.settings.sources.items() if v == "cli"],
        }
        if self.session and self.state == "running":
            result.update(self.session.status())
        return result

    def record(self, state):
        if self.settings.config_store:
            ConfigService(self.settings.config_store).record_runtime(self.settings, state)

    async def start(self):
        async with self.lock:
            if self.task and not self.task.done():
                return self.status()
            self.state, self.error = "starting", None
            self.stop_event, self.ready = asyncio.Event(), asyncio.Event()
            self.task = asyncio.create_task(self._run())
            await self.ready.wait()
            return self.status()

    async def _run(self):
        from .serve import run_serve

        def attached(session):
            self.session = session

        def ready():
            self.state = "running"
            self.record("adopted")
            callback()
            self.ready.set()

        try:
            options = dict(self.options)
            callback = options.pop("on_ready", lambda: None)
            opts = {
                "db_path": self.settings.serve_db,
                "media_root": self.settings.collection.media_dir
                or self.settings.serve_db.with_suffix(".media"),
                "duration": None,
                "dry_models": False,
                "budget_cny": None,
                **options,
            }
            self.summary = await run_serve(
                self.settings, **opts, stop=self.stop_event, on_ready=ready, on_session=attached
            )
            self.state = "stopped"
        except Exception as exc:
            self.state, self.error = (
                "failed",
                str(exc) if isinstance(exc, ConfigError) else type(exc).__name__,
            )
        finally:
            self.session = None
            self.ready.set()
            self.record(self.state)

    async def stop(self):
        async with self.lock:
            if self.task and not self.task.done():
                self.state = "draining"
                self.stop_event.set()
                await self.task
            return self.status()

    async def control(self, action):
        async with self.lock:
            if not self.session or self.state != "running":
                raise ConfigError("engine: start before pausing or resuming")
            self.session.control(action)
            return self.status()

    async def pause(self):
        return await self.control("pause")

    async def resume(self):
        return await self.control("resume")

    async def apply_config(self, settings, *, publish=None, reload_credentials=False):
        async with self.lock:
            plan = application_plan(self.settings, settings)
            session = self.session
            if session and reload_credentials and not session.offline:
                from .config import load_snowluma_credentials

                gateway_creds = load_snowluma_credentials(settings.credentials)
                if (gateway_creds.ws_url, gateway_creds.token) != (
                    session.gateway.url,
                    session.gateway._token,
                ):
                    plan["maintenance"] = True
                    plan["changes"].append(
                        {"field": "SnowLuma credentials", "apply": "maintenance"}
                    )
            agent = None
            if session and not plan["maintenance"]:
                try:
                    agent = await session.prepare_profiles(
                        settings, reload_credentials=reload_credentials
                    )
                except ConfigError:
                    raise
                except Exception as exc:
                    raise ConfigError(
                        f"client initialization failed: {type(exc).__name__}"
                    ) from None
            try:
                if publish is not None:
                    settings = publish()
                if session and plan["maintenance"]:
                    return {"state": "pending_maintenance", **plan}
                if session:
                    for prepared in agent.values():
                        prepared.config = replace(
                            prepared.config, config_revision=settings.revision
                        )
                    session.adopt(settings, agent)
                self.settings = settings
                self.record("adopted" if session else "stopped")
            except BaseException:
                if agent is not None and (
                    session is None or session.orchestrator._agent is not agent[None]
                ):
                    for prepared in agent.values():
                        await session.discard(prepared)
                raise
            if session:
                await session.orchestrator.retire_models()
            return {
                "state": "applied" if session else "saved",
                **plan,
                "revision": settings.published_revision,
                "applied_at": time.time(),
            }


async def run_controlled(settings, **options):
    controller = RuntimeController(settings, **options)
    try:
        await controller.start()
        await controller.task
    finally:
        await controller.stop()
    if controller.error:
        raise ConfigError(controller.error)
    return controller.summary
