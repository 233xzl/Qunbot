"""Strategy C additions of 2026-11-14: recent_chat, quote_context, longer continuation,
deadline-bounded asker_pending and the nudge (`urges`) rule; all ids are synthetic."""

import copy
import json
from pathlib import Path

from group_agent.context_strategies import DEFAULT_RULES, explain_c, is_nudge_text
from group_agent.live_agent import RECENT_CHAT_HEADING, nudge_note, render_material
from group_agent.policy import RuntimeLimits
from group_agent.replay import ISOLATION_KEYS, run_case, score_case

from test_context_strategies import at, ev, reply, text

CASE_DIR = Path(__file__).resolve().parents[1] / "replay" / "cases"
BOT_ONLY = DEFAULT_RULES - {"recent_chat"}


def picks(events, anchor="q", **kw):
    return {p.message_id: (p.relation, p.method) for p in explain_c(events, anchor, **kw)}


def req(mid, user, t, body, *, quote=None):
    segs = ([reply(quote)] if quote else []) + [at("u_bot"), text(body)]
    return ev(mid, user, t, *segs)


def say(mid, user, t, body):
    return ev(mid, user, t, text(body))


def answer(mid, t, to, body="答案"):
    return ev(mid, "u_bot", t, reply(to), text(body))


# -- A: nudge ---------------------------------------------------------------------------


def test_nudge_phrases():
    for phrase in (
        "回答我",
        " 回答我！",
        "回我",
        "快说",
        "快回答",
        "说话",
        "人呢",
        "在吗",
        "在不在",
        "？",
        "?",
    ):
        assert is_nudge_text(phrase), phrase
    for phrase in ("", "  ", "回答我这个问题吧", "崩铁17326是什么意思", "说话的艺术是什么"):
        assert not is_nudge_text(phrase), phrase


def test_case_a_nudge_picks_the_pending_request():
    events = [
        req("a", "u_1", 100, "崩铁17326是什么意思"),
        say("o", "u_2", 110, "路过"),
        req("q", "u_1", 132, "回答我"),
    ]
    got = picks(events, rules=BOT_ONLY)
    assert got == {"a": ("urges", "nudge_pending_request")}
    # The pending request is kept in front of everything else.
    assert picks(events)["a"][0] == "urges"


def test_nudge_also_pulls_the_pending_requests_quote_ancestry():
    events = [
        say("root", "u_2", 50, "原帖内容"),
        req("a", "u_1", 100, "这是什么", quote="root"),
        req("q", "u_1", 132, "回我"),
    ]
    got = picks(events, rules=BOT_ONLY)
    assert got == {"a": ("urges", "nudge_pending_request"), "root": ("reply_to", "urged_chain")}


def test_nudge_needs_a_pending_request_within_nudge_seconds_and_no_quote_or_media():
    old = [req("a", "u_1", 100, "问题"), req("q", "u_1", 100 + 301, "回答我")]
    assert "a" not in picks(old, rules=BOT_ONLY)
    answered = [req("a", "u_1", 100, "问题"), answer("b", 110, "a"), req("q", "u_1", 132, "回答我")]
    assert all(r != "urges" for r, _ in picks(answered, rules=BOT_ONLY).values())
    other_user = [req("a", "u_2", 100, "问题"), req("q", "u_1", 132, "回答我")]
    assert "a" not in picks(other_user, rules=BOT_ONLY)
    quoted = [req("a", "u_1", 100, "问题"), req("q", "u_1", 132, "回答我", quote="a")]
    assert picks(quoted, rules=BOT_ONLY)["a"][0] == "reply_to"  # the explicit quote wins
    media = [
        req("a", "u_1", 100, "问题"),
        ev("q", "u_1", 132, at("u_bot"), text("回答我"), {"type": "image", "data": {}}),
    ]
    assert "a" not in picks(media, rules=BOT_ONLY)


def test_nudge_skips_earlier_nudges_and_can_be_ablated():
    events = [
        req("a", "u_1", 100, "问题"),
        req("n1", "u_1", 120, "回答我"),
        req("q", "u_1", 140, "人呢"),
    ]
    assert picks(events, rules=BOT_ONLY)["a"][0] == "urges"
    assert "a" not in picks(events, rules=BOT_ONLY - {"urges"}, follow_seconds=0)


# -- B: long follow-up ------------------------------------------------------------------


def case_b(question_text="你从哪里知道银枝的这个玩法的"):
    return [
        req("q1", "u_1", 0, "列举最强5大轮椅"),
        answer("a1", 40, "q1", "第一名是永动时停银枝……"),
        req("f1", "u_1", 169, "介绍一下永动时停银枝", quote="a1"),
        say("n1", "u_3", 600, "别的话题"),
        req("q", "u_1", 684, question_text),
    ]


def test_case_b_long_followup_gets_answer_question_and_followup():
    got = picks(case_b(), rules=BOT_ONLY)
    assert got == {
        "a1": ("continues", "bot_answer_followup"),
        "q1": ("answers", "followup_chain"),
        "f1": ("continues", "asker_followup"),
    }


def test_old_answer_needs_a_real_link():
    # 10 minutes after the answer a self-contained request must not pull the old exchange.
    events = [
        req("q1", "u_1", 0, "列举最强5大轮椅"),
        answer("a1", 40, "q1", "第一名是永动时停银枝"),
        req("q", "u_1", 650, "帮我写一首关于春天的诗"),
    ]
    assert picks(events) == {}
    # A short text pulls it only within the 300 s window...
    assert "a1" in picks([*events[:2], req("q", "u_1", 250, "好的")], rules=BOT_ONLY)
    assert picks([*events[:2], req("q", "u_1", 650, "好的")]) == {}
    # ... follow-up wording still links an old answer ...
    assert "a1" in picks([*events[:2], req("q", "u_1", 650, "为啥")], rules=BOT_ONLY)
    # ... a bare 这 is not a strong back-reference for an old answer, 这个 is ...
    assert picks([*events[:2], req("q", "u_1", 650, "这游戏怎么下载安装")]) == {}
    assert "a1" in picks(
        [*events[:2], req("q", "u_1", 650, "这个游戏怎么下载安装")], rules=BOT_ONLY
    )
    # ... shared bigrams with the earlier answer do.
    shared = [*events[:2], req("q", "u_1", 650, "永动时停银枝是谁设计的")]
    assert "a1" in picks(shared, rules=BOT_ONLY)
    # Beyond continue_seconds nothing is picked even with a back-reference.
    gone = [*events[:2], req("q", "u_1", 1000, "你从哪里知道的")]
    assert picks(gone) == {}


def test_asker_followup_skips_answered_branches_and_inflight_runs():
    events = [
        req("q1", "u_1", 0, "问题一"),
        answer("a1", 30, "q1", "回答一"),
        req("f_done", "u_1", 60, "追问一", quote="a1"),
        answer("a2", 90, "f_done", "回答追问"),
        req("f_busy", "u_1", 270, "追问二", quote="a1"),
        req("q", "u_1", 300, "你说的追问一是什么意思"),
    ]
    got = picks(events, rules=BOT_ONLY)
    assert "f_busy" not in got  # still running: its own run
    assert got["a2"][0] == "continues"  # the latest answer to the asker is the one followed
    assert {"f_done", "q1", "a1"} <= set(got)  # ... with its question and ancestry


# -- asker_pending ----------------------------------------------------------------------


def test_asker_pending_is_bounded_by_the_run_deadline():
    events = [
        req("p", "u_1", 0, "还没回答的请求"),
        say("s", "u_1", 190, "补充一句"),
        req("q", "u_1", 200, "这是另一个请求"),
    ]
    kw = {"follow_seconds": 300, "rules": BOT_ONLY - {"urges"}}
    assert picks(events, **kw)["p"] == ("asker_recent", "follow_window")  # 200 s > 120 s
    assert "p" not in picks(events, pending_seconds=300, **kw)
    assert "p" in picks(events, asker_pending=False, pending_seconds=300, **kw)
    close = [req("p", "u_1", 100, "还没回答的请求"), req("q", "u_1", 200, "这是另一个请求")]
    assert "p" not in picks(close, **kw)  # 100 s: still possibly running
    assert "p" not in picks(close)  # recent_chat leaves it out too


# -- C: quote neighbourhood -------------------------------------------------------------


def case_c():
    return [
        say("far", "u_9", 1740 - 1740 - 200, "太早了"),
        say("b4", "u_3", -110, "更早一句"),
        say("b3", "u_4", -90, "前文三"),
        say("b2", "u_5", -60, "前文二"),
        say("b1", "u_3", -30, "前文一"),
        say("t", "u_2", 0, "4c来了"),
        say("a1", "u_3", 20, "后文一"),
        say("a2", "u_4", 60, "后文二"),
        say("a3", "u_5", 100, "后文三"),
        say("late", "u_6", 400, "太晚了"),
        req("q", "u_1", 1740, "is this true？", quote="t"),
    ]


def test_case_c_old_quote_gets_its_neighbourhood():
    got = picks(case_c(), rules=BOT_ONLY)
    assert got == {
        "t": ("reply_to", "quote_chain"),
        "b3": ("quote_context", "quote_neighbour_before"),
        "b2": ("quote_context", "quote_neighbour_before"),
        "b1": ("quote_context", "quote_neighbour_before"),
        "a1": ("quote_context", "quote_neighbour_after"),
        "a2": ("quote_context", "quote_neighbour_after"),
    }


def test_quote_context_applies_to_short_text_but_not_a_fresh_long_quote():
    short = [
        say("b", "u_3", 10, "前一句"),
        say("t", "u_2", 20, "4c来了"),
        say("x", "u_3", 25, "后一句"),
    ]
    short.append(req("q", "u_1", 40, "是真的吗", quote="t"))
    assert {"b", "x"} <= set(picks(short, rules=BOT_ONLY))
    long_fresh = copy.deepcopy(short)
    long_fresh[1] = say("t", "u_2", 20, "这是一句超过十个字符的比较长的话")
    assert set(picks(long_fresh, rules=BOT_ONLY)) == {"t"}
    assert set(picks(short, rules=BOT_ONLY - {"quote_context"})) == {"t"}


def test_quote_context_is_first_hop_only_and_skips_inflight_requests():
    events = [
        say("deep_n", "u_8", 5, "祖先的邻居"),
        say("root", "u_2", 10, "根"),
        say("t", "u_2", 3000, "短"),
        req("busy", "u_7", 3030, "别人的请求"),
        say("ok", "u_3", 3040, "后一句"),
        req("q", "u_1", 3100, "什么意思", quote="t"),
    ]
    events[2]["message"].insert(0, reply("root"))
    got = picks(events, rules=BOT_ONLY)
    assert "deep_n" not in got and "busy" not in got and "ok" in got


# -- D: recent_chat ---------------------------------------------------------------------


def test_recent_chat_window_cap_priority_and_exclusions():
    events = [
        say("old", "u_2", 100, "超出窗口"),
        *[say(f"r{i}", "u_2", 150 + i * 5, f"近期{i}") for i in range(10)],
        answer("rb", 205, "r3", "机器人的话"),
        req("busy", "u_3", 215, "别人刚发的请求"),
        req("done", "u_3", 120, "别人已回答的请求"),
        answer("done_a", 125, "done"),
        req("q", "u_1", 250, "这是什么"),
    ]
    got = picks(events)
    recent = [m for m, (r, _) in got.items() if r == "recent_chat"]
    assert len(recent) == 8  # recent_max
    assert "old" not in got and "busy" not in got  # outside window / in flight
    assert "rb" in recent  # bot messages are included
    assert "r0" not in recent and "r9" in recent  # the most recent ones are kept
    assert picks(events, recent_max=0).keys() == picks(events, rules=BOT_ONLY).keys()
    # Lowest priority: a tight budget drops the background slice first.
    quoting = [*events[:-1], req("q", "u_1", 250, "这是什么", quote="r3")]
    assert set(picks(quoting, max_messages=1)) == {"r3"}


def test_recent_chat_skips_picked_messages_and_media_and_other_groups():
    events = [
        ev("img", "u_5", 200, {"type": "image", "data": {}}),
        ev("cap", "u_6", 205, text("带图"), {"type": "image", "data": {}}),
        ev("other", "u_9", 210, text("别的群"), group="g_2"),
        say("t", "u_2", 215, "被引用"),
        req("q", "u_1", 250, "这是什么", quote="t"),
    ]
    got = picks(events)
    assert got["t"][0] == "reply_to"
    assert not {"img", "cap", "other"} & set(got)


# -- material rendering -----------------------------------------------------------------


def test_material_labels_recent_chat_quote_neighbours_and_urged_request(tmp_path):
    events = case_c()
    anchor = events[-1]
    chosen = explain_c(events, "q")
    reasons = {p.message_id: f"{p.relation}/{p.method}" for p in chosen}
    prior = [e for e in events[:-1] if str(e["message_id"]) in reasons]
    out, _, stats, _ = render_material(prior, anchor, "u_bot", tmp_path, reasons=reasons)
    material = out.split("</material>")[0]
    assert "［被引用消息 #" in material and "的前文］" in material and "的后文］" in material
    assert stats["recent_chat"] == 0 and RECENT_CHAT_HEADING not in material

    events = [say("r1", "u_2", 150, "路过的话"), req("a", "u_1", 190, "崩铁17326是什么意思")]
    events.append(req("q", "u_1", 210, "回答我"))
    chosen = explain_c(events, "q")
    reasons = {p.message_id: f"{p.relation}/{p.method}" for p in chosen}
    prior = [e for e in events[:-1] if str(e["message_id"]) in reasons]
    out, _, stats, _ = render_material(prior, events[-1], "u_bot", tmp_path, reasons=reasons)
    material = out.split("</material>")[0]
    assert RECENT_CHAT_HEADING in material
    heading_at = material.index(RECENT_CHAT_HEADING)
    assert material.index("崩铁17326") < heading_at < material.index("路过的话")
    assert "［提问人此前未回答的请求，现在被催促］" in material
    assert stats["recent_chat"] == 1 and stats["urged_index"] == 2  # "[#2]" in the note


def test_nudge_note_only_when_this_run_answers_the_earlier_request():
    class Trace:
        nudge = {"earlier_message_id": "a", "outcome": "answer_earlier"}

    class Material:
        stats = {"urged_index": 2}

    note = nudge_note(Trace(), Material())
    assert "[#2]" in note and "催促" in note
    Trace.nudge = {"earlier_message_id": "a", "outcome": "merged"}
    assert nudge_note(Trace(), Material()) is None
    Material.stats = {}
    Trace.nudge = {"earlier_message_id": "a", "outcome": "answer_earlier"}
    assert nudge_note(Trace(), Material()) is None


# -- run flow: nudge merge vs answer ----------------------------------------------------


def load_nudge_case() -> dict:
    return json.loads((CASE_DIR / "synthetic-nudge-merged.json").read_text(encoding="utf-8"))


def run_of(result, mid):
    return next(r for r in result["runs"] if r["message_id"] == mid)


def test_nudge_merged_while_the_earlier_run_is_active():
    case = load_nudge_case()
    result = run_case(case)
    nudge, earlier = run_of(result, "m_nudge"), run_of(result, "m_q1")
    assert nudge["state"] == "succeeded" and nudge["reason"] == "nudge_merged"
    assert nudge["route"] == "nudge_merged" and nudge["model_calls"] == []
    assert nudge["nudge"] == {
        "earlier_message_id": "m_q1",
        "earlier": "active",
        "outcome": "merged",
    }
    assert [o["reply_to"] for o in result["outbound"]] == ["m_q1"]  # no second answer
    assert earlier["state"] == "sent" and earlier["nudge"] is None
    assert run_of(result, "m_nudge")["context_ids"] == ["m_q1"]
    assert score_case(case, result)["isolation"] == {k: 0 for k in ISOLATION_KEYS}
    assert result["unused_mocks"] == [] and result["sent_to_qq"] == 0


def test_nudge_answers_the_earlier_question_when_that_run_ended_without_answer():
    case = load_nudge_case()
    # The earlier run overruns the 120 s deadline (expired, nothing sent); the nudge
    # arrives after that.
    case["mocks"]["model"][0]["delay_ms"] = 300000
    case["mocks"]["model"].append(
        {
            "name": "answer",
            "response": "17326 是活动编号（虚构）。",
            "delay_ms": 1000,
            "error": None,
        }
    )
    case["events"][1]["time"] += 100
    result = run_case(case, limits=RuntimeLimits(deadline_seconds=90))
    earlier, nudge = run_of(result, "m_q1"), run_of(result, "m_nudge")
    assert earlier["state"] == "timeout" and not [
        o for o in result["outbound"] if o["reply_to"] == "m_q1"
    ]
    assert nudge["nudge"]["earlier"] == "open" and nudge["nudge"]["outcome"] == "answer_earlier"
    assert nudge["route"] == "chat" and nudge["state"] == "sent"
    assert any(c["name"].startswith("answer") for c in nudge["model_calls"])
    assert [o["reply_to"] for o in result["outbound"]] == ["m_nudge"]
    assert "m_q1" in nudge["context_ids"]


def test_nudge_without_any_earlier_run_in_the_replay_is_answered():
    case = load_nudge_case()
    case["events"][0]["message"] = [{"type": "text", "data": {"text": "没有@的普通消息"}}]
    result = run_case(case)
    assert run_of(result, "m_nudge")["nudge"] is None  # not a pending *request*


# -- live 2026-11-14: quoting a caption must bring its author's image ---------------------


def test_quoted_caption_brings_the_authors_image_posted_just_before():
    image = {"type": "image", "data": {"file": "a.jpg", "url": "https://qpic.cn/x"}}
    events = [
        ev("img", "u_2", 800, image),
        say("cap", "u_2", 818, "这是某新闻下面的评论"),
        say("more", "u_2", 822, "味太大了"),
        say("other_img", "u_3", 854, "卧槽还有活人吗"),
        req("q", "u_1", 1000, "锐评一下上面的图", quote="cap"),
    ]
    got = picks(events)
    assert got["cap"][0] == "reply_to"
    assert got["img"] == ("continues", "author_burst")
    # Other members' messages never become image material through this rule.
    assert got.get("other_img", ("",))[0] != "continues"
    # The ordinary burst window (15 s) is unchanged elsewhere.
    narrow = picks(events, quote_burst_seconds=15)
    assert "img" not in narrow or narrow["img"][0] != "continues"
