import asyncio

import pytest

from group_agent.admission import (
    AccessPolicy,
    AccessSnapshot,
    RecordOnly,
    RequestQueue,
    reply_candidate,
)
from group_agent.backfill import ReplyBackfill
from group_agent.config import Retention, Settings
from group_agent.contracts import Message
from group_agent.intake import EventIntake
from group_agent.pipeline import Pipeline
from group_agent.retention import run_cleanup
from group_agent.store import Store

AT_BOT = {"type": "at", "data": {"qq": "BOT"}}


def event(mid="m1", user="U1", group="G1", segments=None, **extra):
    data = {
        "post_type": "message",
        "message_type": "group",
        "self_id": "BOT",
        "group_id": group,
        "message_id": mid,
        "user_id": user,
        "time": 1_700_000_000,
        "message": segments if segments is not None else [AT_BOT],
    }
    data.update(extra)
    return data


def make(store=None, sink=None, **snapshot):
    store = store or Store()
    access = AccessPolicy(AccessSnapshot("BOT", **snapshot))
    sink = sink if sink is not None else RecordOnly()
    return store, access, sink, EventIntake(store, access, sink=sink, clock=lambda: 1.0)


@pytest.mark.parametrize(
    "sample,status",
    [
        (event(), "trigger"),
        (event(segments=[{"type": "text", "data": {"text": "@BOT [CQ:at,qq=BOT]"}}]), "stored"),
        (
            event(segments=[{"type": "reply", "data": {"id": "m0", "message": [AT_BOT]}}]),
            "stored",
        ),
        (event(segments=[{"type": "at", "data": {"qq": "all"}}]), "stored"),
        (event(user="BOT"), "stored"),
        (event(post_type="message_sent"), "stored"),
        (event(message_type="private"), "ignored_private"),
        (event(self_id="OTHER"), "ignored_other_bot"),
        (event(user="BAD"), "ignored_blocked"),
        (event(group="BADG"), "ignored_blocked"),
    ],
)
def test_admission_matrix(sample, status):
    store, _, sink, intake = make(
        owner_id="OWNER", blocked_users=frozenset({"BAD"}), blocked_groups=frozenset({"BADG"})
    )
    result = intake.intake(sample)
    assert result.status == status
    expected_requests = 1 if status == "trigger" else 0
    assert store.summary()["reply_requests"] == expected_requests
    assert sink.counts["recorded"] == expected_requests
    if status == "ignored_blocked":
        assert store.summary()["messages"] == 0


def test_request_carries_boundary_and_owner_flag():
    _, _, _, intake = make(owner_id="OWNER")
    result = intake.intake(event(user="OWNER"))
    request = result.request
    assert request.seq == result.seq and request.is_owner
    assert request.platform_time == 1_700_000_000 and request.received_at == 1.0
    assert request.key == ("BOT", "G1", "m1")
    assert not make(owner_id="OWNER")[3].intake(event(user="U2")).request.is_owner


def test_dedup_survives_restart_and_redelivery(tmp_path):
    path = tmp_path / "a.db"
    store, _, sink, intake = make(Store(path))
    assert intake.handle(event()) == "trigger"
    assert intake.handle(event()) == "duplicate"
    store.close()
    store, _, sink, intake = make(Store(path))
    assert intake.handle(event()) == "duplicate"  # replay after reconnect / restart
    assert sink.counts["recorded"] == 0
    assert store.summary()["reply_requests"] == 1
    store.close()


def test_redelivery_after_retention_removed_message_does_not_request_again():
    store, _, sink, intake = make()
    assert intake.handle(event(time=1000)) == "trigger"
    # Message expired from the buffer, request row kept for the related period.
    retention = Retention(recent_hours=1, related_days=7)
    store.con.execute("DELETE FROM messages")
    store.con.commit()
    assert intake.handle(event(time=1000)) == "duplicate"
    assert sink.counts["recorded"] == 1
    run_cleanup(store, retention, now=1000 + 8 * 86400)
    assert store.summary()["reply_requests"] == 0


async def test_backfilled_message_never_triggers():
    class Api:
        async def get_msg(self, message_id):
            return {
                "message_type": "group",
                "group_id": "G1",
                "message_id": message_id,
                "user_id": "U5",
                "time": 1_699_999_000,
                "message": [AT_BOT, {"type": "text", "data": {"text": "old"}}],
            }

    store, _, sink, intake = make()
    backfill = ReplyBackfill(store, Api(), "BOT", clock=lambda: 2.0)
    current = Message("BOT", "G1", "m2", "U2", ({"type": "reply", "data": {"id": "m1"}},))
    assert await backfill.resolve(current) == ["stored"]
    # The same message arriving later as a pushed event (with a real @) is a duplicate.
    assert intake.handle(event("m1", user="U5")) == "duplicate"
    assert sink.counts["recorded"] == 0 and store.summary()["reply_requests"] == 0


def test_runtime_blacklist_update_applies_to_next_event():
    store, access, _, intake = make()
    assert intake.handle(event("m1")) == "trigger"
    version = access.version
    access.update(blocked_users={"U1"})
    assert access.version == version + 1
    assert intake.handle(event("m2")) == "ignored_blocked"
    access.update(blocked_users=frozenset(), blocked_groups=["G1"])
    assert intake.handle(event("m3")) == "ignored_blocked"
    with pytest.raises(ValueError):
        access.update(bot_id="OTHER")


def test_collection_off_group_keeps_only_requests():
    store, _, sink, intake = make(collection_off_groups=frozenset({"G1"}))
    text = [{"type": "text", "data": {"text": "chat"}}]
    assert intake.handle(event("m1", segments=text)) == "ignored_collection_off"
    assert intake.handle(event("m2")) == "trigger"
    assert intake.handle(event("m3", group="G2", segments=text)) == "stored"
    assert store.lookup("BOT", "G1", "m1") is None
    assert store.lookup("BOT", "G1", "m2") is not None


def test_shadow_style_intake_without_admission_stores_mentions():
    store = Store()
    access = AccessPolicy(AccessSnapshot("BOT"))
    intake = EventIntake(store, access, admit=False, clock=lambda: 1.0)
    assert intake.handle(event()) == "stored"
    assert store.summary()["reply_requests"] == 0


async def test_request_queue_hand_off_and_full_queue():
    queue = RequestQueue(maxsize=1)
    store, _, _, intake = make(sink=queue)
    assert intake.handle(event("m1")) == "trigger"
    assert intake.handle(event("m2")) == "trigger"
    assert queue.counts == {"queued": 1, "dropped": 1}
    states = dict(store.con.execute("SELECT message_id, state FROM reply_requests").fetchall())
    assert states == {"m1": "queued", "m2": "dropped"}
    request = await asyncio.wait_for(queue.get(), 1)
    assert request.message.message_id == "m1"


def test_reply_candidate_rule():
    at = Message("BOT", "G1", "m", "U1", (AT_BOT,))
    assert reply_candidate(at, False)
    assert not reply_candidate(at, True)
    assert not reply_candidate(Message("BOT", "G1", "m", "BOT", (AT_BOT,)), False)


async def test_pipeline_settings_update_reaches_admission():
    from dataclasses import replace

    store = Store()
    pipeline = Pipeline(Settings(), store)
    assert pipeline.admit(event("m1", user="U1")).status == "trigger"
    pipeline.settings = replace(pipeline.settings, blocked_users=frozenset({"U1"}))
    assert pipeline.admit(event("m2", user="U1")).status == "ignored_blocked"
    assert pipeline.admit(event("m2", user="U1", message_type="private")).status == (
        "ignored_invalid"
    )
