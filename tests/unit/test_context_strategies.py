import importlib.util
from pathlib import Path

from group_agent.context_strategies import (
    DEFAULT_RULES,
    STRATEGIES,
    explain_c,
    strategy_a,
    strategy_b,
    strategy_c,
    strategy_d,
)

SPEC = importlib.util.spec_from_file_location("score", Path("scripts/replay/score.py"))
score = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(score)


def ev(mid, user, t, *segs, group="g_1"):
    return {
        "post_type": "message",
        "message_type": "group",
        "self_id": "u_bot",
        "group_id": group,
        "message_id": mid,
        "user_id": user,
        "time": t,
        "sender": {"user_id": user, "nickname": "n" + user},
        "message": list(segs) or [{"type": "text", "data": {"text": "x"}}],
    }


def reply(mid):
    return {"type": "reply", "data": {"id": mid}}


def at(qq):
    return {"type": "at", "data": {"qq": qq}}


def text(s):
    return {"type": "text", "data": {"text": s}}


EVENTS = [
    ev("m1", "u_1", 50),  # asker's old message, outside B's follow window
    ev("m2", "u_2", 110),  # other user, later quoted by the bot
    ev("m3", "u_bot", 120, reply("m2")),  # bot answer
    ev("m4", "u_3", 130),  # unrelated
    ev("m5", "u_9", 140, group="g_2"),  # other group
    ev("m6", "u_1", 190),  # asker, just before anchor
    ev("q", "u_1", 200, at("u_bot"), reply("m3"), text("why")),  # anchor
    ev("m7", "u_2", 210),  # after anchor
]


def test_a_excludes_bot_records_but_attaches_quote_and_involved_senders():
    chosen = strategy_a(EVENTS, "q")
    # Asker's messages plus the quoted bot message; the bot's sender id adds nothing
    # because legacy never recorded its own messages.
    assert chosen == ["m1", "m3", "m6"]


def test_a_per_sender_cap_and_record_cap():
    events = [ev(f"m{i}", "u_1", i) for i in range(10)] + [ev("q", "u_1", 99, at("u_bot"))]
    assert strategy_a(events, "q") == [f"m{i}" for i in range(5, 10)]
    assert strategy_a(events, "q", record_cap=3) == ["m7", "m8", "m9"]


def test_d_is_a_same_group_window_before_anchor():
    assert strategy_d(EVENTS, "q") == ["m1", "m2", "m3", "m4", "m6"]
    assert strategy_d(EVENTS, "q", k=2) == ["m4", "m6"]
    assert strategy_d(EVENTS, "q", minutes=1) == ["m6"]


def test_b_follows_quote_ancestry_and_recent_asker_lines():
    assert strategy_b(EVENTS, "q") == ["m2", "m3", "m6"]
    assert strategy_b(EVENTS, "q", depth=1, follow_seconds=0) == ["m3"]


def test_unknown_anchor_selects_nothing():
    assert (
        strategy_a(EVENTS, "nope") == strategy_b(EVENTS, "nope") == strategy_d(EVENTS, "nope") == []
    )
    assert strategy_c(EVENTS, "nope") == []


IMG = {"type": "image", "data": {"media_id": "x"}}


# The pre-2026-11-14 rules: the older tests below pin them without the background slice
# (`recent_chat`) and the quote neighbourhood, which have their own tests further down.
LEGACY_RULES = DEFAULT_RULES - {"recent_chat", "quote_context"}


def reasons(events, anchor="q", **kw):
    kw.setdefault("rules", LEGACY_RULES)
    return {p.message_id: (p.relation, p.method) for p in explain_c(events, anchor, **kw)}


def test_c_keeps_b_explicit_tree_with_relation_labels():
    got = reasons(EVENTS)
    assert got == {
        "m3": ("reply_to", "quote_chain"),
        "m2": ("answers", "quote_chain"),  # the bot message m3 answered m2
        "m6": ("asker_recent", "follow_window"),
    }
    assert strategy_c(EVENTS, "q", rules=LEGACY_RULES) == ["m2", "m3", "m6"]  # conversation order
    assert STRATEGIES["C"] is strategy_c


def test_c_labels_quoted_correction():
    events = [
        ev("a", "u_1", 100, at("u_bot"), text("这是谁")),
        ev("b", "u_bot", 110, reply("a"), text("是甲")),
        ev("q", "u_1", 200, reply("b"), at("u_bot"), text("不对吧，应该是乙")),
    ]
    assert reasons(events)["b"] == ("corrects", "quote_chain")


def test_c_continues_latest_bot_answer_to_the_asker_without_quote():
    events = [
        ev("img", "u_2", 40, IMG),
        ev("q1", "u_1", 50, reply("img"), at("u_bot"), text("笑点解析")),
        ev("b1", "u_bot", 60, reply("q1"), text("解释")),
        ev("o1", "u_3", 70, at("u_bot"), text("别的问题")),
        ev("b2", "u_bot", 80, reply("o1"), text("别的回答")),  # answer to someone else
        ev("q", "u_1", 200, at("u_bot"), text("所以呢")),
    ]
    got = reasons(events)
    assert got == {
        "b1": ("continues", "bot_answer_followup"),
        "q1": ("answers", "followup_chain"),
        "img": ("reply_to", "followup_chain"),
    }
    # Outside the window nothing is continued, and the answered request q1 is not
    # pulled back in as a split-question line either.
    assert reasons(events, continue_seconds=60, follow_seconds=200) == {}


def test_c_continues_skips_new_material_question_without_followup_wording():
    events = [
        ev("q1", "u_1", 50, at("u_bot"), text("问")),
        ev("b1", "u_bot", 60, reply("q1"), text("答")),
        ev("q", "u_1", 200, at("u_bot"), text("评价"), IMG),
    ]
    assert "b1" not in reasons(events)
    events[-1] = ev("q", "u_1", 200, at("u_bot"), text("详细说说"), IMG)
    assert reasons(events)["b1"] == ("continues", "bot_answer_followup")


def test_c_continues_needs_a_link_to_the_previous_exchange():
    # Incident 2026-11-09: a self-contained new request right after an answer must not
    # pull the previous question/answer in.
    events = [
        ev("q1", "u_1", 50, at("u_bot"), text("甲和乙谁更强")),
        ev("b1", "u_bot", 60, reply("q1"), text("甲更强，乙更稳")),
        ev("q", "u_1", 120, at("u_bot"), text("锐评某站用户丙丁戊")),
    ]
    assert reasons(events) == {}
    assert reasons(events, continue_link=False)["b1"] == ("continues", "bot_answer_followup")
    linked = [
        "那乙呢",  # back-reference
        "为啥",  # follow-up wording
        "我说的是另一个甲",  # restatement
        "你确定吗",  # correction wording
        "？",  # near-empty
        "甲和乙的出装",  # shares bigrams with the question
    ]
    for wording in linked:
        events[-1] = ev("q", "u_1", 120, at("u_bot"), text(wording))
        assert reasons(events).get("b1") == ("continues", "bot_answer_followup"), wording


def test_c_drops_the_askers_pending_request_from_recent_lines():
    # The asker's earlier @bot request without an answer yet is its own run.
    events = [
        ev("p", "u_1", 100, at("u_bot"), text("锐评甲")),
        ev("s", "u_1", 105, text("打个甲出个乙")),
        ev("q", "u_1", 110, at("u_bot"), text("锐评乙")),
    ]
    assert reasons(events) == {"s": ("asker_recent", "follow_window")}
    assert "p" in reasons(events, asker_pending=False)
    # A correction still builds on the asker's earlier statements.
    events[-1] = ev("q", "u_1", 110, reply("s"), at("u_bot"), text("不对，应该是丙"))
    assert reasons(events)["p"] == ("asker_recent", "follow_window")


def test_c_uses_send_mapping_for_unquoted_bot_answers():
    events = [
        ev("q1", "u_1", 50, at("u_bot"), text("问")),
        ev("b1", "u_bot", 60, text("答")),  # sent without quoting the question
        ev("q", "u_1", 200, at("u_bot"), text("为什么")),
    ]
    assert reasons(events) == {}
    assert reasons(events, answers={"b1": "q1"}) == {
        "b1": ("continues", "bot_answer_followup"),
        "q1": ("answers", "followup_chain"),
    }


def test_c_refers_to_the_askers_own_recent_media_only():
    events = [
        ev("mine", "u_1", 0, IMG),
        ev("theirs", "u_2", 190, IMG),
        ev("q", "u_1", 200, at("u_bot"), text("这是啥意思")),
    ]
    assert reasons(events) == {"mine": ("refers_to", "asker_latest_media")}
    events[-1] = ev("q", "u_1", 200, at("u_bot"))  # bare @ also points at own material
    assert reasons(events) == {"mine": ("refers_to", "asker_latest_media")}
    events[-1] = ev("q", "u_1", 200, at("u_bot"), text("今年冠军是谁"))
    assert reasons(events) == {}
    # Another member's image is never taken implicitly, even when it is the only one.
    assert reasons([events[1], ev("q", "u_1", 200, at("u_bot"), text("这是啥意思"))]) == {}


def test_c_author_burst_around_quoted_material():
    events = [
        ev("p0", "u_2", 80, text("更早的话")),
        ev("x", "u_3", 90, text("插话")),
        ev("p1", "u_2", 100, IMG),
        ev("p2", "u_2", 104, IMG),
        ev("p3", "u_2", 110, text("配文")),
        ev("p4", "u_2", 112, reply("x"), text("回别人")),  # its own branch
        ev("p5", "u_2", 150, text("很久以后")),
        ev("q", "u_1", 200, reply("p2"), at("u_bot"), text("锐评")),
    ]
    got = reasons(events)
    assert got["p2"] == ("reply_to", "quote_chain")
    assert got["p1"] == got["p3"] == ("continues", "author_burst")
    assert {"p0", "x", "p4", "p5"}.isdisjoint(got)
    assert "p1" not in reasons(events, rules=frozenset())


def test_c_asker_filter_drops_other_branches_but_keeps_split_question():
    events = [
        ev("o", "u_2", 100, text("别人的话")),
        ev("a1", "u_1", 110, reply("o"), text("回别人")),
        ev("a2", "u_1", 120, at("u_bot"), text("另一个问题")),
        ev("b2", "u_bot", 125, reply("a2"), text("答")),
        ev("a3", "u_1", 190, text("我想问的是")),
        ev("q", "u_1", 200, reply("b2"), at("u_bot"), text("那这个")),
    ]
    got = reasons(events)
    assert got["a3"] == ("asker_recent", "follow_window")
    assert "a1" not in got
    assert got["a2"] == ("answers", "quote_chain")  # explicit chain wins over the filter
    no_filter = reasons(events, rules=frozenset())
    assert no_filter["a1"] == ("asker_recent", "follow_window")


def test_c_correction_keeps_answered_earlier_request():
    events = [
        ev("q1", "u_1", 100, at("u_bot"), text("词是什么意思")),
        ev("b1", "u_bot", 104, reply("q1"), text("旧解释")),
        ev("f1", "u_1", 150, at("u_bot"), text("更正：新解释")),
        ev("b2", "u_bot", 153, reply("f1"), text("已更正")),
        ev("q", "u_1", 200, reply("b1"), at("u_bot"), text("这个解释不对")),
    ]
    assert reasons(events)["f1"] == ("asker_recent", "follow_window")
    events[-1] = ev("q", "u_1", 200, reply("b1"), at("u_bot"), text("再说一遍"))
    assert "f1" not in reasons(events)


def test_c_same_topic_by_text_overlap_is_capped():
    long = "雨夜擒蛟龙到底是哪场比赛的梗"
    events = [
        ev("t1", "u_2", 10, text("雨夜擒蛟龙是哪场比赛")),
        ev("t2", "u_3", 20, text("雨夜擒蛟龙那场比赛太经典")),
        ev("t3", "u_4", 30, text("雨夜擒蛟龙比赛的梗")),
        ev("u", "u_5", 40, text("中午吃什么")),
        ev("old", "u_2", -1000, text("雨夜擒蛟龙是哪场比赛")),
        ev("q", "u_1", 200, at("u_bot"), text(long)),
    ]
    got = reasons(events)
    assert len(got) == 2 and all(r == ("same_topic", "bigram_overlap>=0.5") for r in got.values())
    assert {"u", "old"}.isdisjoint(got)
    assert reasons(events, rules=frozenset({"continues"})) == {}


def test_c_budget_drops_low_priority_first():
    assert set(reasons(EVENTS, max_messages=2)) == {"m2", "m3"}  # asker_recent dropped
    heavy = [*EVENTS[:5], ev("m6", "u_1", 190, IMG), EVENTS[6]]
    assert "m6" in reasons(heavy)
    assert "m6" not in reasons(heavy, max_tokens=100)  # one image ~300 tokens


def test_c_is_pure_and_same_group_only():
    snapshot = repr(EVENTS)
    first = explain_c(EVENTS, "q")
    assert explain_c(EVENTS, "q") == first and repr(EVENTS) == snapshot
    assert "m5" not in {p.message_id for p in first}  # other group
    assert all(p.evidence_id in {"q", "m3", "m2"} for p in first)


def case_with(legacy, reply_text=None):
    events = [ev("q", "u_1", 200, at("u_bot"), text("?"))]
    if reply_text is not None:
        events.append(ev("r", "u_bot", 205, text(reply_text)))
        legacy.setdefault("bot_reply", {"message_id": "r"})
    return {"events": events, "legacy_result": legacy}


def test_legacy_route_inference():
    assert score.legacy_route(case_with({"diagnostics": {"teaching": True}})) == "teaching"
    assert score.legacy_route(case_with({"diagnostics": {"search_count": 1}})) == "web"
    assert score.legacy_route(case_with({}, "图看不清，能重新发一下原图吗？")) == "clarify"
    assert score.legacy_route(case_with({}, "这是一个常见的梗。")) == "direct"
    assert score.legacy_route({"events": [], "legacy_result": {}}) is None


def test_score_context_counts_required_and_forbidden():
    case = {
        "case_id": "c1",
        "anchor": {"message_id": "q"},
        "events": EVENTS,
        "annotations": {
            "status": "draft",
            "should_reply": True,
            "required_context_ids": ["m2", "m3"],
            "forbidden_context_ids": ["m4", "m7"],
        },
    }
    metrics, per_case = score.score_context([case])
    assert metrics["B"]["required_recall_micro"] == 1.0
    assert metrics["D"]["irrelevant_share"] == round(1 / 5, 3)
    assert per_case["c1"]["A"] == {"required": 2, "hit": 1, "selected": 3, "forbidden": 0}
    assert metrics["C"]["required_recall_micro"] == 1.0


def test_score_c_relations_ablation_and_splits(tmp_path):
    case = {
        "case_id": "c1",
        "anchor": {"message_id": "q"},
        "events": EVENTS,
        "annotations": {
            "status": "reviewed",
            "should_reply": True,
            "required_context_ids": ["m2", "m3"],
            "forbidden_context_ids": ["m6"],
        },
    }
    other = {**case, "case_id": "c2", "annotations": {**case["annotations"], "status": "draft"}}
    relations = score.relation_contribution([case])
    assert relations["recent_chat/recent_window"] == {
        "selected": 2,
        "required": 0,
        "forbidden": 0,
        "unmarked": 2,
    }
    assert relations["answers/quote_chain"] == {
        "selected": 1,
        "required": 1,
        "forbidden": 0,
        "unmarked": 0,
    }
    assert relations["asker_recent/follow_window"]["forbidden"] == 1
    ablation = score.rule_ablation([case])
    assert ablation["C-recent_chat"]["irrelevant_share"] == round(1 / 3, 3)  # pre-2026-11-14 C
    assert ablation["C"]["selected"] == 5.0 and ablation["C-none"]["selected"] == 3.0
    assert ablation["C-asker_filter"]["required_recall_micro"] == 1.0
    (tmp_path / "c1.json").write_text("{}", encoding="utf-8")
    splits = score.split_cases([case, other], tmp_path)
    assert [c["case_id"] for c in splits["tune"]] == ["c1"]
    assert [c["case_id"] for c in splits["held_out"]] == ["c2"]
    assert [c["case_id"] for c in splits["reviewed"]] == ["c1"]
