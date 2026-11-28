"""Group sessions (`session.py`) and the per-group admission rate, with fake models only."""

import asyncio
from dataclasses import replace
import json

import pytest
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.models.function import FunctionModel

from group_agent.config import ConfigError, SessionSpec, parse_settings
from group_agent.config_service import validate_keys
from group_agent.live_agent import LiveAgent
from group_agent.policy import RuntimeLimits
from group_agent.session import SessionManager

from test_live_replay import USAGE, live_config
from test_orchestrator import Harness, ev, recall, wall


QUICK = RuntimeLimits(group_min_interval_seconds=0, group_per_minute=1000)


def prompt_text(message) -> str:
    """Text of a user request in a Pydantic AI message list."""
    assert isinstance(message, ModelRequest)
    out = []
    for part in message.parts:
        if isinstance(part, UserPromptPart):
            content = part.content
            out += (
                [content]
                if isinstance(content, str)
                else [c for c in content if isinstance(c, str)]
            )
    return "\n".join(out)


def session_harness(tmp_path, *, fail=lambda n: False, delay=0.0, **spec):
    h = Harness(limits=QUICK)
    h.orch._base_settings = replace(h.settings, session=SessionSpec(enabled=True, **spec))
    calls = []

    def gemini():
        async def fn(messages, info):
            calls.append(list(messages))
            n = len(calls)
            if delay:
                await asyncio.sleep(delay)
            if fail(n):
                raise RuntimeError("primary down")
            return ModelResponse(parts=[TextPart(f"回答{n}")], usage=USAGE)

        return FunctionModel(fn)

    def deepseek():
        async def fn(messages, info):
            return ModelResponse(parts=[TextPart("备用回答")], usage=USAGE)

        return FunctionModel(fn)

    config = live_config(tmp_path, gemini, deepseek, decision_mode="lightweight", answer_tools=True)
    h.orch.agent = LiveAgent(h.orch, config)
    return h, calls


def session_detail(h, mid):
    return json.loads(h.runs()[mid]["detail"])["live"]["session"]


async def test_next_request_continues_the_session_with_stable_turns(tmp_path):
    h, calls = session_harness(tmp_path)
    t = int(wall())
    h.feed(ev("q1", user="U1", text="第一问", t=t))
    await h.settle()
    h.feed(ev("c1", user="U2", text="插一句闲聊", at=False, t=t + 1))
    h.feed(ev("q2", user="U2", text="第二问", t=t + 2))
    await h.settle()
    first, second = calls
    # The earlier request and its own response are resent unchanged; one new turn follows.
    assert second[: len(first)] == first
    assert isinstance(second[len(first)], ModelResponse)
    assert second[len(first)].parts[0].content == "回答1"
    assert len(second) == len(first) + 2
    assert "[#m1]" in prompt_text(first[0]) and "第一问" in prompt_text(first[0])
    turn = prompt_text(second[-1])
    assert "插一句闲聊" in turn and "第二问" in turn
    assert "编号接续前几轮" in turn and "第一问" not in turn
    d1 = session_detail(h, "q1")
    assert (d1["turn"], d1["reason"], d1["answered"], d1["committed"]) == (1, "new", True, True)
    assert session_detail(h, "q2")["turn"] == 2
    assert session_detail(h, "q2")["reason"] == "continued"
    assert session_detail(h, "q2")["committed"] is True


async def test_a_fallback_answer_never_joins_the_session(tmp_path):
    h, calls = session_harness(tmp_path, fail=lambda n: n == 1)
    t = int(wall())
    h.feed(ev("q1", text="第一问", t=t))
    await h.settle()
    assert h.outbound()[0]["text"] == "备用回答"
    assert session_detail(h, "q1").get("committed") is None
    h.feed(ev("q2", text="第二问", t=t + 1))
    await h.settle()
    assert len(calls[-1]) == 1  # fresh: nothing from the failed turn
    assert session_detail(h, "q2")["committed"] is True


@pytest.mark.parametrize("change", ["recall", "idle", "tokens", "blocked"])
async def test_session_restarts(tmp_path, change):
    spec = {"max_tokens": 1000} if change == "tokens" else {}
    h, calls = session_harness(tmp_path, **spec)
    t = int(wall())
    h.feed(ev("q1", user="U1", text="第一问", t=t))
    await h.settle()
    if change == "recall":
        h.feed(recall("q1", user="U1"))
    elif change == "idle":
        h.orch.sessions.sessions[("BOT", "G1")].last_at -= 21 * 60
    elif change == "blocked":
        h.orch.apply_access(blocked_users={"U1"})
    h.feed(ev("q2", user="U3", text="第二问", t=t + 1))
    await h.settle()
    expected = {"recall": "invalid", "idle": "idle", "tokens": "tokens", "blocked": "invalid"}
    assert session_detail(h, "q2")["reason"] == expected[change]
    assert len(calls[-1]) == 1


async def test_concurrent_turns_branch_and_only_the_first_joins(tmp_path):
    h, calls = session_harness(tmp_path, delay=0.05)
    t = int(wall())
    h.feed(ev("q1", user="U1", text="第一问", t=t))
    await h.settle()
    h.feed(ev("q2", user="U2", text="第二问", t=t + 1))
    h.feed(ev("q3", user="U3", text="第三问", t=t + 1))
    await h.settle()
    assert len(calls[1]) == len(calls[2]) == 3  # both continue turn 1
    committed = [session_detail(h, mid)["committed"] for mid in ("q2", "q3")]
    assert sorted(committed) == [False, True]
    # The other pending request is never shown as material of its sibling.
    for call in calls[1:]:
        text = prompt_text(call[-1])
        assert ("第二问" in text) != ("第三问" in text)
    assert h.orch.sessions.sessions[("BOT", "G1")].version == 2


async def test_sessions_are_off_by_default(tmp_path):
    h, calls = session_harness(tmp_path)
    h.orch._base_settings = h.settings  # SessionSpec() is disabled
    h.feed(ev("q1", text="第一问"))
    await h.settle()
    h.feed(ev("q2", text="第二问"))
    await h.settle()
    assert [len(c) for c in calls] == [1, 1]
    assert "session" not in json.loads(h.runs()["q2"]["detail"])["live"]


def test_manager_rejects_a_commit_from_a_replaced_session():
    manager = SessionManager()
    spec = SessionSpec(enabled=True)
    view = manager.begin("BOT", "G1", "k1", spec, 0.0)
    view.messages = ["turn"]
    other = manager.begin("BOT", "G1", "k2", spec, 1.0)  # model or instructions changed
    assert other.reason == "changed"
    assert manager.commit("BOT", view, 2.0) is False
    other.messages = ["turn"]
    assert manager.commit("BOT", other, 2.0) is True


# -- configuration ---------------------------------------------------------------------


def test_session_settings_and_group_overrides(tmp_path):
    settings = parse_settings({"session": {"enabled": True, "max_tokens": 6000}}, tmp_path / "s")
    assert settings.session == SessionSpec(enabled=True, max_tokens=6000)
    with_profile = replace(
        settings,
        group_profiles={"G1": {"enabled": True, "session": {"enabled": False, "idle_minutes": 5}}},
    )
    assert with_profile.session_for_group("G1") == SessionSpec(
        enabled=False, idle_minutes=5, max_tokens=6000
    )
    assert with_profile.session_for_group("G2") == settings.session
    for bad in ({"enabled": 1}, {"max_tokens": 10}, {"unknown": 1}, {"idle_minutes": 0}):
        with pytest.raises(ConfigError):
            parse_settings({"session": bad}, tmp_path / "s")
        with pytest.raises(ConfigError):
            validate_keys({"group_profiles": {"G1": {"enabled": True, "session": bad}}})


def test_admission_settings_validate(tmp_path):
    limits = parse_settings(
        {"runtime": {"group_admit_max": 5, "group_admit_window_minutes": 3}}, tmp_path / "s"
    ).runtime
    assert (limits.group_admit_max, limits.group_admit_window_minutes) == (5, 3)
    with pytest.raises(ConfigError):
        parse_settings({"runtime": {"group_admit_max": -1}}, tmp_path / "s")
    with pytest.raises(ConfigError):
        parse_settings(
            {"runtime": {"group_admit_max": 1, "group_admit_window_minutes": 0}}, tmp_path / "s"
        )
    validate_keys({"group_profiles": {"G1": {"runtime": {"group_admit_max": 0}}}})
    with pytest.raises(ConfigError):
        validate_keys({"group_profiles": {"G1": {"runtime": {"group_admit_window_minutes": 0}}}})


# -- admission rate ----------------------------------------------------------------------


def fast(limits=None, **kw):
    delays = {f"{p}{i}": 0.0 for p in ("a", "b", "own") for i in range(8)} | {"own": 0.0}
    limits = replace(limits or RuntimeLimits(), group_min_interval_seconds=0, group_per_minute=1000)
    return Harness(limits=limits, agent_kw={"delays": delays}, **kw)


async def test_group_admission_rate_notifies_each_request_including_owner():
    h = fast(RuntimeLimits(group_admit_max=2))
    for i in range(4):
        h.feed(ev(f"a{i}", user=f"U{i}"))
    h.feed(ev("own", user="OWNER"))
    h.feed(ev("b0", group="G2"))
    await h.settle()
    runs = h.runs()
    assert [runs[f"a{i}"]["error_kind"] for i in range(4)] == [
        None,
        None,
        "group_rate_limited",
        "group_rate_limited",
    ]
    assert runs["a2"]["state"] == "rejected"
    assert runs["own"]["error_kind"] == "group_rate_limited"
    assert runs["b0"]["state"] == "succeeded"
    notices = [o for o in h.outbound() if o["kind"] == "busy_notice"]
    assert {n["reply_to"] for n in notices} == {"a2", "a3", "own"}
    window = h.orch.admission_window("G1", wall())
    assert window.used == 2
    assert window.limit_notice().startswith("当前回答次数已达上限，预计 ")
    assert window.status_text().endswith(window.limit_notice())
    assert all(n["text"] == window.limit_notice() for n in notices)
    assert h.feed(ev("a2", user="U2")) == "duplicate"
    assert len(h.outbound()) == 6
    assert sorted(h.agent.answered) == ["a0", "a1", "b0"]
    # The window passes (counted from stored runs): members are served again.
    h.store.con.execute("UPDATE runs SET created_at = created_at - 601")
    h.feed(ev("a5", user="U5"))
    await h.settle()
    assert h.runs()["a5"]["state"] == "succeeded"


async def test_group_override_of_the_admission_rate():
    h = fast()
    h.orch._base_settings = replace(
        h.settings, group_profiles={"G1": {"enabled": True, "runtime": {"group_admit_max": 1}}}
    )
    for i in range(2):
        h.feed(ev(f"a{i}", user=f"U{i}"))
        h.feed(ev(f"b{i}", user=f"U{i}", group="G2"))
    await h.settle()
    runs = h.runs()
    assert runs["a1"]["error_kind"] == "group_rate_limited"
    assert runs["b1"]["state"] == "succeeded"


async def test_rolling_window_persists_across_restart_and_lowered_limit(tmp_path):
    from group_agent.store import Store

    path = tmp_path / "messages.sqlite3"
    store = Store(path)
    h = fast(RuntimeLimits(group_admit_max=10), store=store)
    for i in range(10):
        h.feed(ev(f"a{i}", user=f"U{i}"))
    await h.settle()
    # Admissions span midnight in China. A fractional expiry rounds up in the notice.
    start = 57590.25  # 1970-01-01 23:59:50.25
    with store.con:
        for i in range(10):
            store.con.execute(
                "UPDATE runs SET created_at=? WHERE message_id=?", (start + i, f"a{i}")
            )
    store.close()
    store = Store(path)
    h = fast(RuntimeLimits(group_admit_max=3, group_admit_window_minutes=1), store=store)
    full = h.orch.admission_window("G1", start + 9)
    assert full.used == 10 and full.available_at == start + 7 + 60
    assert full.limit_notice() == "当前回答次数已达上限，预计 1970-01-02 00:00:58 恢复。"
    assert h.orch.admission_window("G1", full.available_at).available_at is None
    assert h.orch.admission_window("G2", start + 9).used == 0
    store.close()


# -- compaction: an ongoing conversation is carried over, not cut ------------------------

COMPACT = {"compaction": "recent", "max_tokens": 1000, "hard_max_tokens": 100_000}


def cool(h, seconds=200):
    h.orch.sessions.sessions[("BOT", "G1")].last_at -= seconds


async def test_cool_long_segment_continues_with_the_last_questions_and_answers(tmp_path):
    h, calls = session_harness(tmp_path, **COMPACT)
    t = int(wall())
    h.feed(ev("q1", user="U1", text="第一问", t=t))
    await h.settle()
    cool(h)
    h.feed(ev("q2", user="U2", text="那第二个呢", t=t + 1))
    await h.settle()
    assert len(calls[-1]) == 1  # rebuilt: no model history is resent
    text = prompt_text(calls[-1][0])
    assert "接续前面的对话" in text
    assert "[#m1] 群友A" in text and "第一问" in text  # same number and label as before
    assert "机器人（你）" in text and "回答1" in text
    assert "那第二个呢" in text
    detail = session_detail(h, "q2")
    assert detail["reason"] == "compacted_tokens" and detail["committed"] is True
    live = json.loads(h.runs()["q2"]["detail"])["live"]
    assert live["session_material"]["carried"] == 2
    # The next turn continues the compacted segment normally.
    h.feed(ev("q3", user="U1", text="第三问", t=t + 2))
    await h.settle()
    assert len(calls[-1]) == 3 and session_detail(h, "q3")["reason"] == "continued"


async def test_warm_segment_grows_past_the_soft_limit(tmp_path):
    h, calls = session_harness(tmp_path, **COMPACT)
    t = int(wall())
    h.feed(ev("q1", text="第一问", t=t))
    await h.settle()
    h.feed(ev("q2", text="第二问", t=t + 1))
    await h.settle()
    assert len(calls[-1]) == 3 and session_detail(h, "q2")["reason"] == "continued"


async def test_hard_limit_compacts_even_while_warm(tmp_path):
    h, calls = session_harness(tmp_path, **{**COMPACT, "hard_max_tokens": 1000})
    t = int(wall())
    h.feed(ev("q1", text="第一问", t=t))
    await h.settle()
    h.feed(ev("q2", text="第二问", t=t + 1))
    await h.settle()
    assert session_detail(h, "q2")["reason"] == "compacted_tokens"
    assert "回答1" in prompt_text(calls[-1][0])


@pytest.mark.parametrize("change", ["idle", "recall"])
async def test_idle_or_recalled_conversations_still_start_empty(tmp_path, change):
    h, calls = session_harness(tmp_path, **COMPACT)
    t = int(wall())
    h.feed(ev("q1", user="U1", text="第一问", t=t))
    await h.settle()
    if change == "idle":
        cool(h, 21 * 60)
    else:
        cool(h)
        h.feed(recall("q1", user="U1"))
    h.feed(ev("q2", user="U2", text="第二问", t=t + 1))
    await h.settle()
    assert session_detail(h, "q2")["reason"] == ("idle" if change == "idle" else "invalid")
    assert "接续前面的对话" not in prompt_text(calls[-1][0])
    assert "回答1" not in prompt_text(calls[-1][0])


async def test_chained_compaction_keeps_only_the_latest_turns(tmp_path):
    h, calls = session_harness(tmp_path, **{**COMPACT, "carry_turns": 1})
    t = int(wall())
    for i in range(1, 4):
        if i > 1:
            cool(h)
        h.feed(ev(f"q{i}", text=f"问题{i}号", t=t + i))
        await h.settle()
    text = prompt_text(calls[-1][0])
    assert session_detail(h, "q3")["reason"] == "compacted_tokens"
    assert "问题2号" in text and "回答2" in text
    assert "问题1号" not in text and "回答1" not in text


async def test_compaction_off_keeps_the_hard_cut(tmp_path):
    h, calls = session_harness(tmp_path, max_tokens=1000)
    t = int(wall())
    h.feed(ev("q1", text="第一问", t=t))
    await h.settle()
    cool(h)
    h.feed(ev("q2", text="第二问", t=t + 1))
    await h.settle()
    assert session_detail(h, "q2")["reason"] == "tokens"
    assert "回答1" not in prompt_text(calls[-1][0])


def test_carried_request_image_is_attached_before_other_context(tmp_path):
    from group_agent.live_agent import render_material
    from group_agent.session import SessionManager, Turn

    (tmp_path / "a.png").write_bytes(b"x")
    image = {"type": "image", "data": {"media_ref": {"status": "available", "path": "a.png"}}}
    request = {"message_id": "r1", "user_id": "U1", "time": 100, "message": [image]}
    other = {"message_id": "o1", "user_id": "U3", "time": 150, "message": [image]}
    manager = SessionManager()
    spec = SessionSpec(enabled=True, **{**COMPACT, "carry_turns": 2})
    first = manager.begin("BOT", "G1", "k", spec, 0.0)
    first.request, first.messages = request, ["turn"]
    manager.commit("BOT", first, 1.0, "a1", answer_text="图里是一只猫")
    manager.sessions[("BOT", "G1")].tokens = 5000
    view = manager.begin("BOT", "G1", "k", spec, 500.0)
    assert view.reason == "compacted_tokens" and len(view.carry) == 2
    anchor = {"message_id": "q2", "user_id": "U2", "time": 200, "message": []}
    text, attach, stats, _ = render_material(
        [*view.carry, other], anchor, "BOT", tmp_path, max_images=1, session=view
    )
    assert "接续的最近问答" in stats["image_labels"]["图片1"]
    assert "图里是一只猫" in text
    assert isinstance(manager.sessions[("BOT", "G1")].turns[0], Turn)
