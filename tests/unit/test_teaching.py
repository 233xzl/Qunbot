"""Teaching commands recognised by code (legacy intent() phrasings), D-09 and owner-only."""

import pytest

from group_agent.teaching import REPLY_TEXT, KnowledgeLedger, parse_teaching


@pytest.mark.parametrize(
    "text,scope,term,content",
    [
        ("记住，2000椅子是鸣潮累充的梗。", None, "2000椅子", "2000椅子是鸣潮累充的梗。"),
        ("记一下，这是绝区零的梗，不是派蒙", None, None, "这是绝区零的梗，不是派蒙"),
        ("记下来：「开黑」指周五组队", None, "开黑", "指周五组队"),
        ("我教你：「摸鱼王」是上班摸鱼的人", None, "摸鱼王", "是上班摸鱼的人"),
        ("请记住 本群：「早八」指早上八点的课", "group:g_1", "早八", "指早上八点的课"),
        ("记住 公共知识，「芋泥」是一种甜品", "public", "芋泥", "是一种甜品"),
        ("记一下（本群）「群宠」是那只猫", "group:g_1", "群宠", "是那只猫"),
        ("帮我记住「下班」指六点，记到公共", "public", "下班", "指六点"),
        ("把这个梗记住：「冲」就是开干", None, "冲", "就是开干"),
        ("以后「开黑」就是周五晚上组队", None, "开黑", "周五晚上组队"),
        ("以后你看到2000椅子就是指鸣潮的累充奖励", None, "2000椅子", "指鸣潮的累充奖励"),
        ("以后摸鱼王就是小明（本群）", "group:g_1", "摸鱼王", "小明"),
        ("教学 公共：「梗」指某事。", "public", "梗", "指某事。"),
    ],
)
def test_add_phrasings(text, scope, term, content):
    cmd = parse_teaching(text, "g_1")
    assert cmd is not None and cmd.op == "add"
    assert (cmd.scope, cmd.term, cmd.content) == (scope, term, content)


@pytest.mark.parametrize(
    "text",
    [
        "你记住了吗",
        "记住没？",
        "以后周末就是休息日吗？",
        "以后再说吧",
        "教学楼在哪",
        "这个梗你记得吗",
        "他说记住，2000椅子是梗",  # not at the start of the sender's own text
        "记住",
    ],
)
def test_non_commands(text):
    assert parse_teaching(text, "g_1") is None


def test_missing_scope_is_refused_for_owner_and_unauthorized_for_others():
    ledger = KnowledgeLedger()
    cmd = parse_teaching("记住，2000椅子是鸣潮累充的梗。", "g_1")
    kwargs = dict(owner_id="owner", group_id="g_1", now=0, new_id="kb1")
    other = ledger.apply(cmd, actor="someone", source_message="m1", **kwargs)
    assert other["outcome"] == "unauthorized" and not other["applied"]
    owner = ledger.apply(cmd, actor="owner", source_message="m2", **kwargs)
    assert owner["outcome"] == "missing_scope" and not owner["applied"] and not ledger.records
    assert "公共" in REPLY_TEXT[("add", "missing_scope")]
    scoped = parse_teaching("记住 本群：2000椅子是鸣潮累充的梗。", "g_1")
    assert ledger.apply(scoped, actor="owner", source_message="m3", **kwargs)["applied"]
    assert ledger.records["kb1"].scope == "group:g_1" and ledger.records["kb1"].term == "2000椅子"
