"""Offline minimal vertical slice. No production sender or credentials are used."""

import asyncio
from dataclasses import dataclass

from .admission import AccessPolicy, snapshot_from_settings
from .config import Settings
from .contracts import (
    ContextStrategy,
    ExplicitContext,
    Message,
    MockModel,
    ModelRunner,
)
from .intake import EventIntake
from .store import Store

# Intake statuses kept as they are; everything else that is not stored maps to
# `ignored_invalid` for the replay harness (private, other bot, meta, notices).
_KEPT = {"trigger", "duplicate", "ignored_blocked", "ignored_collection_off"}


@dataclass(frozen=True)
class Admission:
    status: str
    message: Message | None = None
    seq: int | None = None


class RecordingSender:
    def __init__(self, store: Store):
        self.store = store

    async def record(self, run_id: str, message: Message, text: str) -> None:
        self.store.record_outbound(run_id, message, text)


class Pipeline:
    def __init__(
        self,
        settings: Settings,
        store: Store,
        model: ModelRunner | None = None,
        context: ContextStrategy | None = None,
    ):
        self.store = store
        self.access = AccessPolicy.from_settings(settings)
        self._settings = settings
        # Replay keeps no local receive time: ordering uses platform time, then seq.
        self.intake = EventIntake(store, self.access, clock=lambda: None)
        self.model = model if model is not None else MockModel()
        self.context = (
            context if context is not None else ExplicitContext(settings.context_messages)
        )
        self.sender = RecordingSender(store)

    @property
    def settings(self) -> Settings:
        return self._settings

    @settings.setter
    def settings(self, value: Settings) -> None:
        # Runtime permission updates (e.g. a blacklist change during replay).
        self._settings = value
        self.access.replace_snapshot(snapshot_from_settings(value))

    def admit(self, event: dict) -> Admission:
        """Admission and collection only: parse, block, dedup, recall, trigger decision.

        Status `trigger` means the message must start a reply run; every other status is
        final for this event. Shared by `process()` and the deterministic replay runner.
        """
        result = self.intake.intake(event)
        status = result.status
        if status == "stored":
            status = "collected"
        elif status.startswith("recall_"):
            pass
        elif status not in _KEPT:
            status = "ignored_invalid"
        return Admission(status, result.message, result.seq)

    async def process(self, event: dict) -> str:
        admission = self.admit(event)
        if admission.status != "trigger":
            return admission.status
        m, seq = admission.message, admission.seq
        run_id = self.store.start_run(seq)
        try:
            history = [
                h
                for h in self.store.history(m, seq)
                if h.user_id not in self.settings.blocked_users
            ]
            context = self.context.build(m, history, seq)
            async with asyncio.timeout(self.settings.deadline_seconds):
                text = await self.model.run(context)
            if not isinstance(text, str) or not text.strip():
                raise ValueError("empty model output")
            await self.sender.record(run_id, m, text)
            return "recorded"
        except asyncio.CancelledError:
            self.store.fail_run(run_id, "CancelledError", "cancelled")
            raise
        except Exception as exc:
            # Provider error messages can contain prompt material or credentials.
            self.store.fail_run(run_id, type(exc).__name__)
            return "failed"
