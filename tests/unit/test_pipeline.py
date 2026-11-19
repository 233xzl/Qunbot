import asyncio
from dataclasses import replace

import pytest

from group_agent.app import main
from group_agent.config import Settings
from group_agent.contracts import ExplicitContext, Message, parse_group_message
from group_agent.pipeline import Pipeline
from group_agent.store import Store


def event(mid="m1", user="U1", group="G1", segments=None, **kwargs):
    return {
        "post_type": "message",
        "message_type": "group",
        "self_id": "BOT",
        "group_id": group,
        "message_id": mid,
        "user_id": user,
        "message": segments if segments is not None else [{"type": "at", "data": {"qq": "BOT"}}],
        **kwargs,
    }


@pytest.fixture
def store():
    value = Store()
    yield value
    value.close()


async def test_minimal_chain_and_idempotency(store):
    pipeline = Pipeline(Settings(), store)
    assert await pipeline.process(event()) == "recorded"
    assert await pipeline.process(event()) == "duplicate"
    row = store.con.execute("SELECT * FROM outbound").fetchone()
    assert (row["group_id"], row["reply_to"], row["requester"], row["state"], row["mode"]) == (
        "G1",
        "m1",
        "U1",
        "pending",
        "offline",
    )
    assert store.summary()["sent_to_qq"] == 0
    assert store.summary()["runs"] == 1
    assert await pipeline.process(event(group="G2")) == "recorded"


@pytest.mark.parametrize(
    "sample,status",
    [
        (event(segments=[{"type": "text", "data": {"text": "@BOT [CQ:at,qq=BOT]"}}]), "collected"),
        (
            event(
                segments=[
                    {
                        "type": "reply",
                        "data": {"id": "m0", "content": [{"type": "at", "data": {"qq": "BOT"}}]},
                    }
                ]
            ),
            "collected",
        ),
        (event(segments=[{"type": "at", "data": {"qq": "all"}}]), "collected"),
        (event(user="BOT"), "collected"),
        (event(message_type="private"), "ignored_invalid"),
        (event(self_id="OTHER_BOT"), "ignored_invalid"),
        (event(segments="[CQ:at,qq=BOT]"), "ignored_invalid"),
        (event(segments=[{"type": "at", "data": "BOT"}]), "ignored_invalid"),
        (event(user="BLOCKED"), "ignored_blocked"),
        (event(group="BLOCKED_GROUP"), "ignored_blocked"),
    ],
)
async def test_admission(store, sample, status):
    settings = replace(
        Settings(),
        blocked_users=frozenset({"BLOCKED"}),
        blocked_groups=frozenset({"BLOCKED_GROUP"}),
    )
    assert await Pipeline(settings, store).process(sample) == status
    assert store.summary()["runs"] == 0


async def test_reply_context_is_scoped_and_frozen(store):
    seen = []

    class Capture:
        async def run(self, context):
            seen.append(context)
            return "ok"

    pipeline = Pipeline(Settings(), store, model=Capture())
    await pipeline.process(event("base", segments=[{"type": "text", "data": {"text": "source"}}]))
    await pipeline.process(event("other", group="G2", segments=[]))
    await pipeline.process(
        event(
            "q",
            segments=[
                {"type": "at", "data": {"qq": "BOT"}},
                {"type": "reply", "data": {"id": "base"}},
                {"type": "reply", "data": {"id": "other"}},
            ],
        )
    )
    await pipeline.process(event("later", segments=[]))
    assert [m.message_id for m in seen[0].history] == ["base"]
    assert seen[0].boundary == 3


async def test_timeout_does_not_block_other_run(store):
    started = asyncio.Event()
    release = asyncio.Event()

    class Slow:
        async def run(self, context):
            if context.current.user_id == "SLOW":
                started.set()
                await release.wait()
            return "ok"

    pipeline = Pipeline(replace(Settings(), deadline_seconds=0.05), store, model=Slow())
    slow = asyncio.create_task(pipeline.process(event("slow", user="SLOW")))
    await started.wait()
    assert await pipeline.process(event("fast", user="FAST")) == "recorded"
    assert await slow == "failed"
    assert store.summary()["run_states"] == {"failed": 1, "recorded": 1}
    assert (
        store.con.execute("SELECT error_kind FROM runs WHERE state='failed'").fetchone()[0]
        == "TimeoutError"
    )


async def test_cancel_is_recorded_and_propagated(store):
    started = asyncio.Event()

    class Wait:
        async def run(self, context):
            started.set()
            await asyncio.Event().wait()

    task = asyncio.create_task(Pipeline(Settings(), store, model=Wait()).process(event()))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert store.summary()["run_states"] == {"cancelled": 1}
    assert store.summary()["outbound_records"] == 0


def test_reply_cycles_are_bounded():
    a = Message("BOT", "G1", "a", "U1", ({"type": "reply", "data": {"id": "b"}},))
    b = Message("BOT", "G1", "b", "U1", ({"type": "reply", "data": {"id": "a"}},))
    assert [m.message_id for m in ExplicitContext().build(a, [a, b], 3).history] == ["b"]


def test_message_is_detached_from_mutable_event():
    original = event(segments=[{"type": "text", "data": {"text": "before"}}])
    message = parse_group_message(original, "BOT")
    original["message"][0]["data"]["text"] = "after"
    assert message.text == "before"


@pytest.mark.parametrize(
    "environment", [{"blocked_users": "BLOCKED"}, {"bot_id": None}, [], {"blacklist": "BAD"}]
)
def test_cli_rejects_invalid_environment(tmp_path, capsys, environment):
    import json

    path = tmp_path / "case.json"
    path.write_text(json.dumps({"events": [], "environment": environment}), encoding="utf-8")
    with pytest.raises(SystemExit) as error:
        main(["--replay", str(path)])
    assert error.value.code == 2
    assert "expected readable JSON" in capsys.readouterr().err


def test_cli_schema_blacklist_and_unknown_owner(tmp_path, capsys):
    import json

    path = tmp_path / "case.json"
    path.write_text(
        json.dumps(
            {
                "events": [event()],
                "environment": {"bot_id": "BOT", "owner_id": None, "blacklist": ["U1"]},
            }
        ),
        encoding="utf-8",
    )
    assert main(["--replay", str(path)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["events"] == {"ignored_blocked": 1}
    assert result["outbound_records"] == 0


async def test_schema_nested_mentions_and_nonself_message_sent_do_not_trigger(store):
    pipeline = Pipeline(Settings(), store)
    nested = event(
        "nested",
        segments=[
            {
                "type": "reply",
                "data": {"id": "missing", "message": [{"type": "at", "data": {"qq": "BOT"}}]},
            }
        ],
    )
    assert await pipeline.process(nested) == "collected"
    assert await pipeline.process(event("sent", post_type="message_sent")) == "collected"
    assert store.summary()["runs"] == 0


async def test_same_message_id_across_groups_does_not_select_foreign_content(store):
    observed = []

    class Capture:
        async def run(self, context):
            observed.append(context)
            return "ok"

    pipeline = Pipeline(Settings(), store, model=Capture())
    await pipeline.process(
        event("shared", group="G1", segments=[{"type": "text", "data": {"text": "local"}}])
    )
    await pipeline.process(
        event("shared", group="G2", segments=[{"type": "text", "data": {"text": "foreign"}}])
    )
    await pipeline.process(
        event(
            "q",
            segments=[
                {"type": "at", "data": {"qq": "BOT"}},
                {"type": "reply", "data": {"id": "shared"}},
            ],
        )
    )
    assert [m.text for m in observed[0].history] == ["local"]


async def test_later_message_during_model_wait_cannot_enter_snapshot(store):
    started, release = asyncio.Event(), asyncio.Event()
    seen = []

    class Wait:
        async def run(self, context):
            seen.append(context)
            started.set()
            await release.wait()
            return "ok"

    pipeline = Pipeline(Settings(), store, model=Wait())
    task = asyncio.create_task(pipeline.process(event("q")))
    await started.wait()
    await pipeline.process(event("later", segments=[{"type": "text", "data": {"text": "future"}}]))
    release.set()
    assert await task == "recorded"
    assert seen[0].history == ()
    assert seen[0].boundary == 1


def test_cli_rejects_unimplemented_config_overrides(tmp_path, capsys):
    import json

    path = tmp_path / "case.json"
    path.write_text(
        json.dumps(
            {"events": [event()], "environment": {"config_overrides": {"admission": "bypass"}}}
        ),
        encoding="utf-8",
    )
    with pytest.raises(SystemExit) as error:
        main(["--replay", str(path)])
    assert error.value.code == 2
    assert "config_overrides are not implemented" in capsys.readouterr().err


def test_shadow_refuses_example_identity(capsys):
    # The example settings carry a placeholder bot id; shadow must not connect with it.
    with pytest.raises(SystemExit) as error:
        main(["--shadow"])
    assert error.value.code == 2
    assert "placeholder" in capsys.readouterr().err
