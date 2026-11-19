"""Owner teaching: command parsing and reply texts (T4.3, LEARN-01..03, D-09).

Only the start of the sender's own top-level text counts (legacy `intent()` rule); an
optional 请/麻烦你/请你 is skipped. ``<ID>`` is a knowledge id (``K001``, ``C0001``),
optionally followed by the expected revision (``K001 v2``):

- ``教学 公共：「术语」释义`` / ``教学 本群：...``  add with an explicit scope (D-09);
- ``记住…`` ``记一下…`` ``记下来…`` ``我教你…`` ``帮我记住…`` ``把这个梗记住…`` add;
  the scope is explicit only as a leading ``公共``/``本群`` (optionally ``…知识``), a
  bracketed ``（公共）``/``【本群】``, or ``记到/记为/记成 公共|本群``; otherwise the
  base refuses the add and asks for the scope;
- ``以后「术语」就是释义`` / ``以后 X 就是 Y`` (not a question) add with term X;
- ``更正 「术语」：释义`` / ``更正 <ID>：释义`` / ``把 <ID> 改成 释义``  correct;
- quote a bot answer + ``...更正为：释义``         correct the revision that answer used;
- ``停用|禁用|别再用 <ID>|「术语」``, ``恢复|启用|重新启用 <ID>|「术语」``  disable / restore;
- ``看看|查看|看下 <ID>|「术语」``                 view (anyone; entries visible here).

Authorization (LEARN-01, SEC-01) comes only from the real sender of the triggering
message and its own top-level text; quoted or historical "记住" is material. Commit rules
(idempotency, expected revision, scope) live in `knowledge_base.KnowledgeBase`;
`KnowledgeLedger` is that base in memory (deterministic replay).
"""

from dataclasses import dataclass
import re

from .knowledge_base import KnowledgeBase, KnowledgeRecord, scope_words

__all__ = ["KnowledgeLedger", "KnowledgeRecord", "TeachingCommand", "parse_teaching", "reply_for"]

_TERM = re.compile(r"「([^」]+)」")
_POLITE = re.compile(r"^(?:请你|请|麻烦你)\s*")
_SEP = r"[\s,，:：、]*"
_ADD = re.compile(
    r"^(?:教学(?=\s*(?:公共|本群)?\s*[:：])|记住(?![了没吗])|记一下|记下来|我教你|帮我记(?:住|一下|下来)?|把这(?:个|条)(?:解释|答案|梗|知识)?记住)"
    + _SEP
    + r"(?:(公共|本群)(?:知识)?"
    + _SEP
    + r")?(.+)$",
    re.S,
)
_LATER = re.compile(
    r"^以后\s*(?:你\s*)?(?:看到|听到|遇到|有人说)?\s*(.{1,24}?)\s*(?:就是|就指|指的是)\s*(.+)$",
    re.S,
)
_SCOPE_MARK = re.compile(r"[（(【\[]\s*(公共|本群)\s*[）)】\]]|记(?:到|为|成)\s*(公共|本群)")
_QUESTION_END = re.compile(r"(?:[?？]|吗|呢)\s*$")
# Leading "X 是/指 ..." gives the term when X is a short name, not a pronoun.
_LEAD_TERM = re.compile(r"^([^\s,，。:：!！?？「」]{1,16}?)(?:就是|是指|指的是|是|指)")
_PRONOUNS = ("这", "那", "它", "他", "她", "这个", "那个", "这张", "这条", "这图")
_CORRECT = re.compile(r"^更正\s*「([^」]+)」\s*[:：]?\s*(.+)$", re.S)
_QUOTE_CORRECT = re.compile(r"更正为\s*[:：]\s*(.+)$", re.S)
_SCOPES = {"公共": "public", "本群": "group"}
_ID = r"([KCkc]\d{1,6})(?![A-Za-z0-9])(?:\s*[vV](\d+))?"
_TARGET = rf"(?:{_ID}|「([^」]+)」)"
_VIEW = re.compile(rf"^(?:看看|查看|看下|看一下)\s*(?:知识|条目)?\s*{_TARGET}\s*[。.!！]*$")
_DISABLE = re.compile(rf"^(?:别再用|停用|禁用)\s*(?:知识|条目)?\s*{_TARGET}")
_RESTORE = re.compile(rf"^(?:恢复|重新启用|启用)\s*(?:知识|条目)?\s*{_TARGET}")
_ID_CORRECT = re.compile(
    rf"^(?:把\s*)?{_ID}\s*(?:的(?:解释|说明|内容|释义|意思))?\s*"
    r"(?:改成|改为|更正为|修改为|更新为)\s*[:：]?\s*(.*)$",
    re.S,
)
_CORRECT_ID = re.compile(
    rf"^(?:更正|修改|纠正|更新)\s*(?:一下)?\s*{_ID}\s*[:：,，]?\s*(?:改?为|改成)?\s*[:：]?\s*(.*)$",
    re.S,
)


def _scope(word: str | None, group_id: str) -> str | None:
    kind = _SCOPES.get(word or "")
    return None if kind is None else "public" if kind == "public" else f"group:{group_id}"


@dataclass(frozen=True)
class TeachingCommand:
    op: str  # add | correct | disable | restore | view
    scope: str | None  # public | group:<id> | None (missing)
    term: str | None
    content: str
    via_quote: bool = False
    target_id: str | None = None  # K001 / C0001
    expected: int | None = None  # expected revision written by the owner (``K001 v2``)


def _target(op: str, m: re.Match) -> TeachingCommand:
    kid, rev, term = m.group(1), m.group(2), m.group(3)
    return TeachingCommand(
        op,
        None,
        term.strip() if term else None,
        "",
        target_id=kid.upper() if kid else None,
        expected=int(rev) if rev else None,
    )


def parse_teaching(text: str, group_id: str, quotes_bot: bool = False) -> TeachingCommand | None:
    text = _POLITE.sub("", text.strip())
    for op, pattern in (("view", _VIEW), ("disable", _DISABLE), ("restore", _RESTORE)):
        if m := pattern.match(text):
            return _target(op, m)
    for pattern in (_ID_CORRECT, _CORRECT_ID):
        if m := pattern.match(text):
            return TeachingCommand(
                "correct",
                None,
                None,
                m.group(3).strip(),
                target_id=m.group(1).upper(),
                expected=int(m.group(2)) if m.group(2) else None,
            )
    if quotes_bot and (m := _QUOTE_CORRECT.search(text)):
        return TeachingCommand("correct", None, None, m.group(1).strip(), via_quote=True)
    if m := _CORRECT.match(text):
        return TeachingCommand("correct", None, m.group(1).strip(), m.group(2).strip())
    if (m := _ADD.match(text)) and not _QUESTION_END.search(text):
        body = m.group(2).strip()
        scope = _scope(m.group(1), group_id)
        if scope is None and (mark := _SCOPE_MARK.search(body)):
            scope = _scope(mark.group(1) or mark.group(2), group_id)
            body = (body[: mark.start()] + body[mark.end() :]).strip(" ，,:：")
        if not body:
            return None
        term = _TERM.search(body)
        content = body
        if term and body.startswith(term.group(0)):
            content = body[len(term.group(0)) :].lstrip("：: ").strip()
            return TeachingCommand("add", scope, term.group(1), content)
        lead = _LEAD_TERM.match(body)
        name = lead.group(1) if lead and lead.group(1) not in _PRONOUNS else None
        return TeachingCommand("add", scope, term.group(1) if term else name, content)
    if (m := _LATER.match(text)) and not _QUESTION_END.search(text):
        term = m.group(1).strip().strip("「」“”\"'")
        content = m.group(2).strip()
        scope = None
        if mark := _SCOPE_MARK.search(content):
            scope = _scope(mark.group(1) or mark.group(2), group_id)
            content = (content[: mark.start()] + content[mark.end() :]).strip(" ，,:：")
        if term and content:
            return TeachingCommand("add", scope, term, content)
    return None


class KnowledgeLedger(KnowledgeBase):
    """The knowledge base in memory: same commit rules, no files (deterministic replay)."""

    def __init__(self):
        super().__init__(":memory:")


UNAUTHORIZED_TEXT = "只有主人可以新增、更正或停用知识；你可以查询和解梗。"
NEED_ID_TEXT = "请带上唯一的知识编号（如 K001），引用这条教学消息重发。"
NEED_TEXT = "还缺明确的梗名或解释，请引用这条教学消息补充后重发。"
# Wording follows the legacy plugin (`已记住 K004：名称，所有群可用。`); placeholders are
# filled by `reply_for` from the log entry.
REPLY_TEXT = {
    ("add", "committed"): "已记住 {id}：{name}，{where}可用。",
    ("correct", "committed"): "已更正 {id}：{name}，{where}可用。",
    ("disable", "committed"): "已停用 {id}：{name}，{where}生效。",
    ("restore", "committed"): "已恢复 {id}：{name}，{where}生效。",
    ("add", "duplicate"): "这条已经记录过了（{id}），没有重复保存。",
    ("correct", "duplicate"): "{id} 当前内容已经是这样，没有改动。",
    ("disable", "duplicate"): "{id} 已经是停用状态。",
    ("restore", "duplicate"): "{id} 已经是启用状态。",
    ("add", "conflict"): "已有同名条目 {id}，如需修改请用“更正 {id}：新的解释”。",
    ("correct", "conflict"): (
        "{id} 在你依据的版本之后已更新（当前 v{revision}），请先查看最新内容，确认后再更正。"
    ),
    ("disable", "conflict"): "{id} 已更新到 v{revision}，和你写的版本不一致，请确认后重发。",
    ("restore", "conflict"): "{id} 已更新到 v{revision}，和你写的版本不一致，请确认后重发。",
    ("add", "missing_scope"): (
        "请说明这条记为公共知识还是本群知识，例如“记住 公共：「名称」解释”或“记住 本群：…”，"
        "补充后重发。"
    ),
    ("add", "missing_name"): NEED_TEXT,
    ("correct", "missing_name"): NEED_TEXT,
    ("add", "image_unavailable"): "教学图片未能取回，请引用原图和这条教学内容重新发送。",
    ("correct", "image_unavailable"): "教学图片未能取回，请引用原图和这条教学内容重新发送。",
    ("view", "not_found"): "没有找到这条知识。",
}
ASKS_USER = {
    "conflict",
    "missing_scope",
    "missing_name",
    "not_found",
    "ambiguous",
    "image_unavailable",
}


def reply_for(entry: dict, describe=None) -> str:
    """Reply text for one `KnowledgeBase.apply` log entry; `describe(id)` renders a viewed
    entry (`KnowledgeBase.describe`)."""
    op, outcome = entry["op"], entry["outcome"]
    if outcome == "unauthorized":
        return UNAUTHORIZED_TEXT
    if outcome == "viewed":
        text = describe(entry["id"]) if describe is not None else None
        return text or f"[{entry['id']} v{entry['revision']}] {entry.get('term') or ''}"
    if outcome == "ambiguous":
        ids = "、".join(entry.get("ids") or [])
        return f"有多条同名条目（{ids}），{NEED_ID_TEXT}"
    if outcome == "not_found" and op != "view":
        return NEED_ID_TEXT
    template = REPLY_TEXT.get((op, outcome))
    if template is None:
        return "收到。"
    return template.format(
        id=entry.get("id") or "",
        name=entry.get("term") or "",
        where=scope_words(entry.get("scope") or "public"),
        revision=entry.get("revision"),
    )
