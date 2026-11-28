"""`--serve` end to end with a fake SnowLuma connection and dry models: nothing is sent."""

import asyncio
from dataclasses import replace
import json
import sqlite3

import pytest

from group_agent.app import main
from group_agent.config import ModelSpec, Settings
from group_agent.gateway import READ_ONLY_ACTIONS
from group_agent.serve import DRY_ANSWER, run_serve
from group_agent.store import Store

GEMINI = ModelSpec("gemini", "google-genai", "fake-gemini", "RELAY_BASE_URL", "K1", 30, 0.2, 1.0)
DEEPSEEK = ModelSpec("deepseek", "anthropic", "fake-deepseek", "RELAY_BASE_URL", "K2", 30, 0.4, 2.0)
MEMBERS = [
    {"user_id": 1, "nickname": "群主本人", "card": "老大", "role": "owner"},
    {"user_id": 2, "nickname": "管理", "card": "", "role": "admin"},
    {"user_id": 3, "nickname": "路人", "card": "", "role": "member"},
]


class FakeConnection:
    """SnowLuma forward WS stand-in: pushes events, answers read-only actions."""

    def __init__(self, events):
        self.incoming: asyncio.Queue = asyncio.Queue()
        self.sent: list[dict] = []
        for event in events:
            self.incoming.put_nowait(json.dumps(event, ensure_ascii=False))

    def __aiter__(self):
        return self

    async def __anext__(self):
        frame = await self.incoming.get()
        if frame is None:
            raise StopAsyncIteration
        return frame

    async def send(self, frame: str) -> None:
        request = json.loads(frame)
        self.sent.append(request)
        action, echo = request["action"], request["echo"]
        if action == "get_group_info":
            data = {"group_id": 1, "group_name": "测试群", "member_count": 3}
        elif action == "get_group_member_list":
            data = MEMBERS
        elif action == "send_group_msg":  # fake platform: an answer send
            data = {"message_id": 9001}
        else:
            response = {"status": "failed", "retcode": 100, "echo": echo}
            self.incoming.put_nowait(json.dumps(response))
            return
        response = {"status": "ok", "retcode": 0, "data": data, "echo": echo}
        self.incoming.put_nowait(json.dumps(response, ensure_ascii=False))

    async def close(self):
        self.incoming.put_nowait(None)


def message(mid, user, text):
    return {
        "post_type": "message",
        "message_type": "group",
        "self_id": "BOT",
        "group_id": "G1",
        "message_id": mid,
        "user_id": user,
        "time": 1_800_000_000,
        "message": [
            {"type": "at", "data": {"qq": "BOT"}},
            {"type": "text", "data": {"text": text}},
        ],
    }


def settings_for(tmp_path) -> Settings:
    secrets = tmp_path / "secrets.env"
    secrets.write_text("SNOWLUMA_WS_URL=ws://127.0.0.1:3001\nSNOWLUMA_WS_TOKEN=t\n", "utf-8")
    return Settings(
        bot_id="BOT",
        owner_id="OWNER",
        secrets_file=secrets,
        models={"gemini": GEMINI, "deepseek": DEEPSEEK},
    )


def test_serve_dry_models_records_answers_and_never_sends(tmp_path):
    events = [
        message("m1", "U1", "群主是谁"),
        message("m2", "U2", "你好"),
        message("m2", "U2", "你好"),  # redelivered
        {"post_type": "meta_event", "meta_event_type": "heartbeat", "self_id": "BOT"},
    ]
    conn = FakeConnection(events)
    db = tmp_path / "serve.sqlite3"

    async def connect(*args, **kwargs):
        return conn

    async def scenario():
        stop = asyncio.Event()

        async def watcher():
            for _ in range(200):
                await asyncio.sleep(0.05)
                if not db.exists():
                    continue
                probe = sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True)
                try:
                    done = probe.execute(
                        "SELECT count(*) FROM outbound WHERE state='recorded_not_sent'"
                    ).fetchone()[0]
                except sqlite3.OperationalError:
                    done = 0
                finally:
                    probe.close()
                if done >= 2:
                    break
            stop.set()

        watch = asyncio.create_task(watcher())
        summary = await run_serve(
            settings_for(tmp_path),
            db_path=db,
            media_root=tmp_path / "media",
            duration=None,
            dry_models=True,
            budget_cny=0.01,
            cleanup=False,
            knowledge_path=None,
            connect=connect,
            stop=stop,
        )
        await watch
        return summary

    summary = asyncio.run(scenario())
    assert summary["sent_to_qq"] == 0 and summary["mode"] == "serve_record_only"
    actions = {f["action"] for f in conn.sent}
    assert actions <= READ_ONLY_ACTIONS and not any(a.startswith("send") for a in actions)
    assert {"get_group_info", "get_group_member_list"} <= actions
    store = Store(db)
    rows = store.con.execute("SELECT * FROM outbound ORDER BY reply_to").fetchall()
    assert [(r["reply_to"], r["state"], r["text"]) for r in rows] == [
        ("m1", "recorded_not_sent", DRY_ANSWER),
        ("m2", "recorded_not_sent", DRY_ANSWER),
    ]
    runs = store.con.execute("SELECT * FROM runs ORDER BY message_id").fetchall()
    assert [r["state"] for r in runs] == ["succeeded", "succeeded"]
    detail = json.loads(runs[0]["detail"])
    assert {"name": "group_info", "status": "ok"} in detail["tool_calls"]
    assert summary["usage"]["daily_spent_cny"] == 0
    assert store.con.execute("SELECT count(*) FROM usage_ledger").fetchone()[0] >= 4
    assert summary["session"]["outbound"] == {"recorded_not_sent": 2}
    store.close()


async def test_status_and_reactions_after_group_rate_decision(tmp_path):
    questions = [message("m1", "U1", "你好"), message("m2", "OWNER", "再问")]
    stop = asyncio.Event()

    class RateConnection(FakeConnection):
        async def send(self, frame):
            req = json.loads(frame)
            if req["action"] == "set_msg_emoji_like":
                self.sent.append(req)
                self.incoming.put_nowait(
                    json.dumps({"status": "ok", "retcode": 0, "echo": req["echo"]})
                )
                if req["params"]["message_id"] == "m2":
                    self.incoming.put_nowait(json.dumps(message("s1", "OWNER", "#状态")))
            else:
                await super().send(frame)
                if sum(r["action"] == "send_group_msg" for r in self.sent) == 3:
                    stop.set()

    conn = RateConnection(questions + [questions[-1]])

    async def connect(*args, **kwargs):
        return conn

    settings = settings_for(tmp_path)
    limits = replace(settings.runtime, group_admit_max=1, group_min_interval_seconds=0)
    await asyncio.wait_for(
        run_serve(
            replace(settings, runtime=limits),
            db_path=tmp_path / "serve.sqlite3",
            media_root=tmp_path / "media",
            duration=None,
            dry_models=True,
            cleanup=False,
            budget_cny=0.01,
            knowledge_path=None,
            connect=connect,
            stop=stop,
            live_send=True,
            reaction=True,
            owner_control=True,
        ),
        timeout=15,
    )
    faces = [r["params"] for r in conn.sent if r["action"] == "set_msg_emoji_like"]
    assert [(f["message_id"], f["emoji_id"]) for f in faces] == [("m1", "424"), ("m2", "8")]
    replies = [r["params"]["message"] for r in conn.sent if r["action"] == "send_group_msg"]
    texts = {m[0]["data"]["id"]: m[1]["data"]["text"] for m in replies}
    assert texts["m2"].startswith("当前回答次数已达上限，预计 ")
    assert "已用 1/1，更新于 " in texts["s1"]
    assert texts["s1"].endswith(texts["m2"])


def test_only_the_answer_sender_holds_a_send_path():
    """RUN-08 / cutover: the only send entry is the gateway's one-shot answer channel, used
    only by live_send.py; run code (orchestrator/runflow/agent/usage) and serve hold no send
    action, no channel type and no gateway handle; the orchestrator only gets the answer
    holder's `deliver` hook."""
    import inspect
    from pathlib import Path

    from group_agent import gateway, live_agent, live_send, orchestrator, runflow, serve, usage
    from group_agent.store import OUTBOUND_STATES

    words = (
        "send_group_msg",
        "send_msg",
        "send_private_msg",
        "call_action",
        "SEND_ACTION",
        "AnswerSendChannel",
        "_ANSWER_GRANT",
        "grant=",
        "._request(",
    )
    for module in (orchestrator, runflow, live_agent, usage, serve):
        source = inspect.getsource(module)
        for word in words:
            assert word not in source, (module.__name__, word)
    for module in (orchestrator, runflow, usage, live_agent):  # run code: no channel/sender
        source = inspect.getsource(module)
        assert "gateway" not in source.replace("never calls the gateway", ""), module.__name__
        for word in (
            "answer_send_channel",
            "LiveSender",
            "import live_send",
            "from .live_send",
            "_channel",
        ):
            assert word not in source, (module.__name__, word)
    # serve only passes the channel straight into its one holder, once.
    serve_source = inspect.getsource(serve)
    assert serve_source.count("answer_send_channel(") == 1
    assert "LiveSender(gateway.answer_send_channel(), store, access, log=log)" in serve_source
    assert serve_source.count("LiveSender(") == 1
    # The send action exists in gateway.py (the channel) and its one holder.
    root = Path(gateway.__file__).parent
    holders = {p.name for p in root.rglob("*.py") if "send_group_msg" in p.read_text("utf-8")}
    assert holders == {"gateway.py", "live_send.py"}
    issuers = {p.name for p in root.rglob("*.py") if "answer_send_channel(" in p.read_text("utf-8")}
    assert issuers == {"gateway.py", "serve.py"}
    assert "send_group_msg(" in inspect.getsource(live_send.LiveSender)
    assert gateway.SEND_ACTION not in READ_ONLY_ACTIONS
    assert set(OUTBOUND_STATES) == {
        "pending",
        "recorded_not_sent",
        "sending",
        "sent",
        "uncertain",
        "failed",
    }
    assert not any(a.startswith(("send", "set", "delete")) for a in READ_ONLY_ACTIONS)


def test_only_the_reaction_holder_holds_the_reaction_path():
    """D-20: set_msg_emoji_like lives in gateway.py (channel) and reaction.py (its one
    holder); serve passes the channel straight into the holder, once; run code holds none."""
    import inspect
    from pathlib import Path

    from group_agent import gateway, live_agent, orchestrator, reaction, runflow, serve, usage

    for module in (orchestrator, runflow, live_agent, usage, serve):
        source = inspect.getsource(module)
        for word in (
            "set_msg_emoji_like",
            "REACTION_ACTION",
            "ReactionChannel",
            "_REACTION_GRANT",
            "grant=",
        ):
            assert word not in source, (module.__name__, word)
    for module in (orchestrator, runflow, usage, live_agent):
        source = inspect.getsource(module)
        for word in ("reaction_channel", "ProcessingReaction", "import reaction"):
            assert word not in source, (module.__name__, word)
    serve_source = inspect.getsource(serve)
    assert serve_source.count("reaction_channel(") == 1
    assert "gateway.reaction_channel(settings.reaction_emoji_id)," in serve_source
    root = Path(gateway.__file__).parent
    holders = {p.name for p in root.rglob("*.py") if "set_msg_emoji_like" in p.read_text("utf-8")}
    assert holders == {"gateway.py", "reaction.py"}
    users = {p.name for p in root.rglob("*.py") if "reaction_channel(" in p.read_text("utf-8")}
    assert users == {"gateway.py", "serve.py"}
    assert "set_msg_emoji_like(" in inspect.getsource(reaction.ProcessingReaction)
    assert gateway.REACTION_ACTION not in READ_ONLY_ACTIONS


def test_serve_cli_validation(capsys):
    with pytest.raises(SystemExit) as error:
        main(["--serve"])  # example settings: placeholder identity
    assert error.value.code == 2
    assert "placeholder" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        main(["--shadow", "--dry-models"])
    assert "--dry-models requires --serve" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        main(["--serve", "--allow-owner-send"])  # retired with the `#新版` command
    assert "unrecognized arguments" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        main(["--shadow", "--reaction"])
    assert "--reaction requires --serve" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        main(["--serve", "--reaction", "--no-reaction"])
    assert "exclude each other" in capsys.readouterr().err


def _serve_events(tmp_path, events, **kw):
    conn = FakeConnection(events)
    db = tmp_path / "serve.sqlite3"

    async def connect(*args, **kwargs):
        return conn

    async def scenario():
        stop = asyncio.Event()

        async def watcher():
            for _ in range(200):
                await asyncio.sleep(0.05)
                if not db.exists():
                    continue
                probe = sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True)
                try:
                    done = probe.execute(
                        "SELECT count(*) FROM outbound WHERE state='recorded_not_sent'"
                    ).fetchone()[0]
                except sqlite3.OperationalError:
                    done = 0
                finally:
                    probe.close()
                if done >= 2:
                    break
            stop.set()

        watch = asyncio.create_task(watcher())
        summary = await run_serve(
            settings_for(tmp_path),
            db_path=db,
            media_root=tmp_path / "media",
            duration=None,
            dry_models=True,
            budget_cny=0.01,
            cleanup=False,
            knowledge_path=None,
            connect=connect,
            stop=stop,
            **kw,
        )
        await watch
        return summary

    return asyncio.run(scenario()), conn, db


def test_serve_processing_reaction_once_per_admitted_request(tmp_path):
    """D-20: with the switch on, each admitted request gets one set_msg_emoji_like (fixed
    emoji, set=true) at intake; the fake platform refuses it, which is only recorded."""
    plain = {**message("m3", "U3", "随便聊"), "message": [{"type": "text", "data": {"text": "x"}}]}
    events = [message("m1", "U1", "你好"), message("m1", "U1", "你好"), plain]
    events.append(message("m2", "U2", "在吗"))
    summary, conn, db = _serve_events(tmp_path, events, reaction=True)
    reactions = [f for f in conn.sent if f["action"] == "set_msg_emoji_like"]
    assert [f["params"] for f in reactions] == [
        {"message_id": "m1", "emoji_id": "424", "set": True},
        {"message_id": "m2", "emoji_id": "424", "set": True},
    ]
    assert summary["processing_reaction"] is True and summary["sent_to_qq"] == 0
    assert summary["reactions"] == {"failed": 2}
    assert summary["model_breaker"]["gemini"]["state"] == "closed"
    store = Store(db)
    rows = store.con.execute("SELECT reaction FROM reply_requests ORDER BY message_id").fetchall()
    assert [r[0] for r in rows] == ["failed_failed_100", "failed_failed_100"]
    assert store.con.execute("SELECT count(*) FROM outbound").fetchone()[0] == 2  # runs unaffected
    store.close()


def test_serve_reaction_off_by_default(tmp_path):
    summary, conn, _ = _serve_events(
        tmp_path, [message("m1", "U1", "你好"), message("m2", "U2", "在吗")]
    )
    assert summary["processing_reaction"] is False and summary["reactions"] is None
    assert {f["action"] for f in conn.sent} <= READ_ONLY_ACTIONS
