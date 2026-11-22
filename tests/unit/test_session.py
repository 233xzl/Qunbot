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
from group_agent.policy import GROUP_RATE_NOTICE_TEXT, RuntimeLimits
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
            out += [content] if isinstance(content, str) else [c for c in content if isinstance(c, str)]
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

    config = live_config(
        tmp_path, gemini, deepseek, decision_mode="lightweight", answer_tools=True
    )
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


async def test_group_admission_rate_limits_members_not_the_owner():
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
    assert runs["own"]["state"] == runs["b0"]["state"] == "succeeded"
    notices = [o for o in h.outbound() if o["kind"] == "busy_notice"]
    assert [n["text"] for n in notices] == [GROUP_RATE_NOTICE_TEXT]
    assert sorted(h.agent.answered) == ["a0", "a1", "b0", "own"]
    # The window passes (counted from stored runs): members are served again.
    h.store.con.execute("UPDATE runs SET created_at = created_at - 601")
    h.orch._admit_notices.clear()
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
