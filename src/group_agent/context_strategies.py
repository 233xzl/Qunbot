"""Context selection strategies compared on replay cases (requirements section 9).

Each strategy takes the case's ordered OneBot events and the anchor message id and
returns the ids of prior messages it would inject. Only messages before the anchor are
eligible. Strategies are pure functions so they can be scored offline.

- A: legacy AstrBot behaviour. Its group record holds only incoming messages (the bot's
  own sends are never recorded), capped at the most recent `record_cap`; it injects up to
  5 recent messages per involved sender (the asker, @-mentioned users, the quoted
  message's sender), and the directly quoted message is attached separately.
- D: naive window, the last `k` group messages within `minutes`, bot messages included.
- B: explicit reply tree: the anchor's quote ancestry (depth-capped), plus the asker's own
  messages in the `follow_seconds` before the anchor (a question split over several lines).
- C: B plus implicit links (requirements 4.1, USER-04), each pick carrying its relation
  type and the rule that produced it; see `explain_c`.
"""

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
import re


def _dedup(events: list[dict]) -> list[dict]:
    seen, out = set(), []
    for event in events:
        if event.get("post_type") not in ("message", "message_sent"):
            continue
        key = str(event.get("message_id"))
        if key in seen:
            continue
        seen.add(key)
        out.append(event)
    return out


def _segments(event: dict) -> list[dict]:
    message = event.get("message")
    return [s for s in message if isinstance(s, dict)] if isinstance(message, list) else []


def _reply_target(event: dict) -> str | None:
    for seg in _segments(event):
        if seg.get("type") == "reply":
            target = (seg.get("data") or {}).get("id")
            return None if target is None else str(target)
    return None


def _sender_name(event: dict) -> str:
    sender = event.get("sender") or {}
    return str(sender.get("card") or sender.get("nickname") or "").strip().casefold()


def _split(events: list[dict], anchor_id: str) -> tuple[list[dict], dict | None]:
    events = _dedup(events)
    for index, event in enumerate(events):
        if str(event.get("message_id")) == anchor_id:
            group = event.get("group_id")
            prior = [e for e in events[:index] if e.get("group_id") == group]
            return prior, event
    return [], None


def _is_bot(event: dict) -> bool:
    return str(event.get("user_id")) == str(event.get("self_id"))


def strategy_a(events: list[dict], anchor_id: str, record_cap: int = 50, per_sender: int = 5):
    prior, anchor = _split(events, anchor_id)
    if anchor is None:
        return []
    by_id = {str(e.get("message_id")): e for e in prior}
    involved_ids = {str(anchor.get("user_id"))}
    involved_names = {_sender_name(anchor)} - {""}
    for seg in _segments(anchor):
        if seg.get("type") == "at":
            qq = str((seg.get("data") or {}).get("qq") or "")
            if qq and qq not in ("all", "*"):
                involved_ids.add(qq)
    quoted = _reply_target(anchor)
    if quoted and quoted in by_id:
        involved_ids.add(str(by_id[quoted].get("user_id")))
        involved_names |= {_sender_name(by_id[quoted])} - {""}
    records = [e for e in prior if not _is_bot(e)][-record_cap:]
    counts: dict[str, int] = defaultdict(int)
    selected = []
    for event in reversed(records):
        uid, name = str(event.get("user_id")), _sender_name(event)
        if uid not in involved_ids and name not in involved_names:
            continue
        key = uid or f"name:{name}"
        if counts[key] >= per_sender:
            continue
        counts[key] += 1
        selected.append(str(event.get("message_id")))
    if quoted and quoted in by_id and quoted not in selected:
        selected.append(quoted)
    order = {str(e.get("message_id")): i for i, e in enumerate(prior)}
    return sorted(selected, key=order.__getitem__)


def strategy_d(events: list[dict], anchor_id: str, k: int = 20, minutes: float = 30):
    prior, anchor = _split(events, anchor_id)
    if anchor is None:
        return []
    start = (anchor.get("time") or 0) - minutes * 60
    window = [e for e in prior if (e.get("time") or 0) >= start]
    return [str(e.get("message_id")) for e in window[-k:]]


def strategy_b(events: list[dict], anchor_id: str, depth: int = 16, follow_seconds: float = 120):
    prior, anchor = _split(events, anchor_id)
    if anchor is None:
        return []
    by_id = {str(e.get("message_id")): e for e in prior}
    selected: list[str] = []
    current = anchor
    for _ in range(depth):
        target = _reply_target(current)
        if not target or target not in by_id or target in selected:
            break
        selected.append(target)
        current = by_id[target]
    asker, t0 = str(anchor.get("user_id")), anchor.get("time") or 0
    for event in reversed(prior):
        if (event.get("time") or 0) < t0 - follow_seconds:
            break
        if str(event.get("user_id")) == asker:
            selected.append(str(event.get("message_id")))
    order = {str(e.get("message_id")): i for i, e in enumerate(prior)}
    return sorted(set(selected), key=order.__getitem__)


# --- strategy C -----------------------------------------------------------------------

MEDIA_TYPES = frozenset({"image", "video", "file", "redacted", "json", "xml", "forward", "share"})
# Asker disputes or corrects the quoted message (label only; the quote is selected anyway).
CORRECTION = re.compile(r"不对|错了|说错|没对上|对不上|应该是|不是这样|你确定|搞错|看错")
# Follow-up wording that keeps a conversation with the bot going even when the asker
# brings new material ("详细讲一下这篇..." after an answer about the same article).
CONTINUATION = re.compile(
    r"详细|展开|继续|接着|还有|然后呢|那么|所以|为什么|为啥|怎么说|意思是|真的吗|确定吗|再来|"
    r"再说|换个|多说|具体|呢[?？]?$|^那"
)
# Deictic requests that point at material instead of naming it.
DEICTIC = re.compile(
    r"这个|那个|这张|那张|这图|那图|这是|这啥|这什么|啥意思|什么意思|上面|上图|图里|图中|"
    r"这里|这句|这段|这篇|这条|这视频|解析|笑点|看看|看下|看一下|评价一下|锐评"
)
PUNCT = re.compile(r"[\s\W_]+", re.UNICODE)
# Pronouns, time words and restatements that point back at an earlier exchange ("那他呢",
# "你刚才说的", "我说的是...", "你不是叫...么", "真的假的").
BACK_REFERENCE = re.compile(
    r"他|她|它|这|那|上面|上一|刚才|刚刚|之前|前面|你说|你不是|不是说|我说的|我是说|我问的|"
    r"我指的|我的意思|真的|假的|是吗|你从哪|你怎么知道|你咋知道"
)
# Stricter back-reference for an answer older than `continue_near_seconds`: a bare 这/那/他
# is too common in self-contained requests to link a request to an answer minutes old.
STRONG_BACK_REFERENCE = re.compile(
    r"这个|那个|这样|那样|那他|那她|那它|他呢|她呢|它呢|上面|上一|刚才|刚刚|之前|前面|"
    r"你说|你不是|不是说|我说的|我是说|我问的|我指的|我的意思|真的吗|是吗|"
    r"你从哪|你怎么知道|你咋知道"
)
# A bare urging line ("@bot 回答我"): the asker repeats a request that has no answer yet
# (strategy C rule `urges`). Matched on the whole text after stripping punctuation.
NUDGE_PHRASE = re.compile(
    r"(?:快点?|赶紧|倒是)?(?:回答我?|回复我?|回我|回话|说话|说啊|吱声)|快说|人呢|在吗|在不在|"
    r"还在吗|有人吗|怎么不回|咋不回|不回答"
)
NUDGE_MAX_CHARS = 6

# Priority when the budget forces a cut: lower is kept first. `recent_chat` is the
# background slice and goes first.
RELATION_PRIORITY = {
    "reply_to": 0,
    "corrects": 0,
    "urges": 0,
    "answers": 1,
    "quote_context": 1.5,
    "continues": 2,
    "refers_to": 3,
    "asker_recent": 4,
    "author_burst": 5,
    "same_topic": 6,
    "recent_chat": 7,
}
# Every implicit rule of `explain_c`; `rules=` selects a subset (ablation).
DEFAULT_RULES = frozenset(
    {
        "continues",
        "refers_to",
        "author_burst",
        "same_topic",
        "asker_filter",
        "recent_chat",
        "quote_context",
        "urges",
    }
)


@dataclass(frozen=True)
class Pick:
    """One selected message and why: `relation` is the requirements 4.1 relation type
    (or `asker_recent`, B's split-question rule), `method` the rule that fired and
    `evidence_id` the message whose link justified it (the anchor or an earlier pick)."""

    message_id: str
    relation: str
    method: str
    evidence_id: str


def _text(event: dict) -> str:
    return "".join(
        str((s.get("data") or {}).get("text", ""))
        for s in _segments(event)
        if s.get("type") == "text"
    ).strip()


def _has_media(event: dict) -> bool:
    return any(s.get("type") in MEDIA_TYPES for s in _segments(event))


def _mentions(event: dict, qq: str) -> bool:
    return any(
        s.get("type") == "at" and str((s.get("data") or {}).get("qq")) == qq
        for s in _segments(event)
    )


def _bigrams(text: str) -> set[str]:
    clean = PUNCT.sub("", text.casefold())
    return {clean[i : i + 2] for i in range(len(clean) - 1)}


def estimate_tokens(event: dict, image_tokens: int = 300) -> int:
    """Rough prompt cost of one context message: CJK ~1 token per char, plus a fixed
    header and a flat cost per attached media item."""
    media = sum(1 for s in _segments(event) if s.get("type") in MEDIA_TYPES)
    return 8 + len(_text(event)) + media * image_tokens


def is_nudge_text(text: str) -> bool:
    """A bare urging line: nothing but a short phrase such as "回答我" / "人呢" or a lone
    question mark (at most `NUDGE_MAX_CHARS` characters once punctuation is stripped)."""
    raw = "".join(text.split())
    if not raw:
        return False
    clean = PUNCT.sub("", text)
    if not clean:
        return bool(re.search(r"[?？]", raw))
    return len(clean) <= NUDGE_MAX_CHARS and NUDGE_PHRASE.fullmatch(clean) is not None


def explain_c(
    events: list[dict],
    anchor_id: str,
    *,
    answers: dict[str, str] | None = None,
    depth: int = 16,
    follow_seconds: float = 120,
    continue_seconds: float = 900,
    continue_near_seconds: float = 300,
    refer_seconds: float = 300,
    burst_seconds: float = 15,
    burst_max: int = 3,
    quote_burst_seconds: float = 60,
    topic_seconds: float = 600,
    topic_overlap: float = 0.5,
    topic_min_shared: int = 4,
    topic_cap: int = 2,
    max_messages: int = 24,
    max_tokens: int = 8000,
    continue_link: bool = True,
    continue_short_chars: int = 3,
    continue_min_shared: int = 2,
    asker_pending: bool = True,
    pending_seconds: float = 120,
    recent_seconds: float = 180,
    recent_max: int = 8,
    quote_short_chars: int = 10,
    quote_window_seconds: float = 120,
    quote_before: int = 3,
    quote_after: int = 2,
    nudge_seconds: float = 300,
    rules: frozenset[str] | None = None,
) -> list[Pick]:
    """Strategy C with reasons, in conversation order.

    `answers` maps a bot message id to the question it answered (RUN-08 send mapping);
    without it, a bot message's own quote of the question is used. `rules` limits the
    implicit rules for ablation: any of `DEFAULT_RULES` (`continues`, `refers_to`,
    `author_burst`, `same_topic`, `asker_filter`, `recent_chat`, `quote_context`, `urges`;
    None = all).

    "In flight" below means an @bot request that has no answer yet and is at most
    `pending_seconds` old (the run deadline): it is its own run, answered by that run.

    Rules, in priority order:
    1. reply_to / answers / corrects: the anchor's quote ancestry (as B). A hop from a bot
       message to the question it answered is `answers`; the anchor's own quote target is
       `corrects` when the anchor uses correction wording.
    2. urges (nudge, 2026-11-14): the anchor has no quote, no media and is only a short
       urging phrase (`is_nudge_text`: "回答我", "人呢", "？"...) -> the asker's latest
       earlier @bot request within `nudge_seconds` that has no answer (nudges themselves do
       not count), plus its quote ancestry. `continues` and `refers_to` are skipped then;
       the run flow decides whether that request's own run answers it (merge) or this run
       does (`runflow.RunFlow._work`).
    3. continues (bot answer): the anchor has no quote, and the bot answered this asker
       within `continue_seconds` (900 s) -> that latest answer, its question and the
       question's ancestry. Skipped when the anchor brings its own media without follow-up
       wording (a new material question). With `continue_link` (incident 2026-11-09: a new,
       self-contained request right after an answer pulled the previous exchange in) the
       anchor must also show a link to that exchange: follow-up/correction wording, a
       back-reference (pronoun, "刚才", "你从哪"), a short text (<= `continue_short_chars`
       content characters, e.g. "为啥"; only for an answer at most `continue_near_seconds`
       old), or at least `continue_min_shared` character bigrams shared with the earlier
       question, answer or the asker's follow-ups into that answer. For an answer older than
       `continue_near_seconds` the back-reference must be a strong one
       (`STRONG_BACK_REFERENCE`: 这个/那个/刚才/你说/你从哪..., not a bare 这/那/他).
       Then the asker's own lines that reply into the selected chain within
       `continue_seconds` are added (`asker_followup`), except answered or in-flight @bot
       requests (those are other branches / their own runs).
    4. quote_context: the anchor's direct quote target (first hop only) is older than
       `follow_seconds` or has a short text (<= `quote_short_chars` content characters, no
       media) -> up to `quote_before` messages before and `quote_after` after it within
       `quote_window_seconds` of its time, any author (not in-flight requests). Priority
       just below reply_to/answers.
    5. refers_to: no quote, no own media, no continuation, and deictic wording (or a
       near-empty text) -> the asker's latest media message within `refer_seconds` and
       its ancestry. Other members' media is never picked without an explicit link
       (USER-02; the annotation guide: do not grab an unrelated member's latest image).
    6. asker_recent: the asker's lines within `follow_seconds` (as B), minus lines that
       belong to another branch: they quote a message outside the selection, or they are
       earlier @bot requests the bot already answered (kept when the anchor is a
       correction, which builds on the asker's earlier statements). With `asker_pending`
       an earlier @bot request that is in flight is dropped too: it is its own run.
       An older unanswered request is no longer in flight and may be included.
    7. author_burst (continues): for each selected non-bot material message, the same
       author's adjacent messages (gap <= `burst_seconds`, or `quote_burst_seconds` around
       the anchor's direct quote target; at most `burst_max` per side)
       - a post split over several messages (long screenshots, image then caption).
    8. same_topic: non-bot messages within `topic_seconds` whose text shares at least
       `topic_min_shared` character bigrams and `topic_overlap` of the shorter text's
       bigrams with the anchor text; at most `topic_cap`.
    9. recent_chat (background): the most recent `recent_max` messages within
       `recent_seconds` before the anchor that are not picked yet, bot messages included,
       lowest priority. In-flight @bot requests (anyone's, including the asker's) are left
       out: they are separate requests with their own runs, and showing them could make the
       model answer them. The material renders this slice under its own heading.
    The result is cut to `max_messages` and `max_tokens` by dropping the lowest-priority,
    then oldest, picks.
    """
    enabled = rules if rules is not None else DEFAULT_RULES
    prior, anchor = _split(events, anchor_id)
    if anchor is None:
        return []
    by_id = {str(e.get("message_id")): e for e in prior}
    order = {mid: i for i, mid in enumerate(by_id)}
    bot = str(anchor.get("self_id"))
    asker, t0 = str(anchor.get("user_id")), anchor.get("time") or 0
    picks: dict[str, Pick] = {}

    def when(event: dict) -> float:
        return event.get("time") or 0

    def add(mid: str | None, relation: str, method: str, evidence: str) -> bool:
        if not mid or mid not in by_id or mid in picks:
            return False
        picks[mid] = Pick(mid, relation, method, evidence)
        return True

    def question_of(mid: str) -> str | None:
        if answers and mid in answers:
            return answers[mid]
        return _reply_target(by_id[mid]) if mid in by_id else None

    def ancestry(start: dict, start_id: str, first_relation: str, method: str) -> None:
        current, current_id = start, start_id
        relation = first_relation
        for _ in range(depth):
            is_bot = str(current.get("user_id")) == bot and current is not anchor
            target = question_of(current_id) if is_bot else _reply_target(current)
            if not target or target not in by_id or target in picks:
                break
            add(target, "answers" if is_bot else relation, method, current_id)
            current, current_id, relation = by_id[target], target, "reply_to"

    answered = {question_of(m) for m, e in by_id.items() if str(e.get("user_id")) == bot}

    def in_flight(event: dict) -> bool:
        """An @bot request without an answer that may still be running."""
        return (
            str(event.get("user_id")) != bot
            and _mentions(event, bot)
            and str(event.get("message_id")) not in answered
            and 0 <= t0 - when(event) <= pending_seconds
        )

    def readable(event: dict) -> bool:
        return bool(_text(event)) or _has_media(event)

    def plain_text(event: dict) -> bool:
        """Text without attachments: background picks (`recent_chat`, `quote_context`
        neighbours) never carry other members' images or cards into the prompt (USER-02)."""
        return bool(_text(event)) and not _has_media(event)

    def replies_into(roots: set[str]) -> list[dict]:
        """The asker's lines within `continue_seconds` that reply into `roots`, directly
        or through each other, in conversation order."""
        chain, found = set(roots), []
        for event in prior:
            if str(event.get("user_id")) != asker or t0 - when(event) > continue_seconds:
                continue
            if _reply_target(event) in chain:
                chain.add(str(event.get("message_id")))
                found.append(event)
        return found

    # 1. explicit quote ancestry
    quoted = _reply_target(anchor)
    first = "corrects" if quoted and CORRECTION.search(_text(anchor)) else "reply_to"
    ancestry(anchor, anchor_id, first, "quote_chain")

    anchor_text = _text(anchor)
    own_media = _has_media(anchor)

    # 2. urges: a bare nudge repeats the asker's earlier request that has no answer yet
    urged = False
    if quoted is None and not own_media and "urges" in enabled and is_nudge_text(anchor_text):
        for event in reversed(prior):
            if t0 - when(event) > nudge_seconds:
                break
            mid = str(event.get("message_id"))
            if (
                str(event.get("user_id")) == asker
                and _mentions(event, bot)
                and mid not in answered
                and readable(event)
                and not is_nudge_text(_text(event))
            ):
                add(mid, "urges", "nudge_pending_request", anchor_id)
                ancestry(event, mid, "reply_to", "urged_chain")
                urged = True
                break

    # 3. continues: the asker's follow-up to the bot's latest answer to them
    cue = bool(CONTINUATION.search(anchor_text))
    anchor_grams = _bigrams(anchor_text)

    def linked(answer: dict, question: dict, followups: list[dict]) -> bool:
        if not continue_link or cue or CORRECTION.search(anchor_text):
            return True
        near = t0 - when(answer) <= continue_near_seconds
        if (BACK_REFERENCE if near else STRONG_BACK_REFERENCE).search(anchor_text):
            return True
        if near and len(PUNCT.sub("", anchor_text)) <= continue_short_chars:
            return True
        pool = _bigrams(_text(question)) | _bigrams(_text(answer))
        for event in followups:
            pool |= _bigrams(_text(event))
        return len(anchor_grams & pool) >= continue_min_shared

    continued = False
    if quoted is None and not urged and "continues" in enabled and (cue or not own_media):
        for event in reversed(prior):
            if t0 - when(event) > continue_seconds:
                break
            mid = str(event.get("message_id"))
            if str(event.get("user_id")) != bot:
                continue
            question = question_of(mid)
            if question in by_id and str(by_id[question].get("user_id")) == asker:
                # Only the latest answer to the asker is a candidate; an unlinked,
                # self-contained new request starts a new branch.
                if linked(event, by_id[question], replies_into({mid})):
                    add(mid, "continues", "bot_answer_followup", anchor_id)
                    ancestry(event, mid, "reply_to", "followup_chain")
                    continued = True
                break
    if "continues" in enabled and picks:
        # The asker's own lines that reply into the chain selected so far ("介绍一下..."
        # under an answer), without other branches (answered requests) or in-flight runs.
        for event in replies_into(set(picks)):
            mid = str(event.get("message_id"))
            if mid in picks or in_flight(event) or not readable(event):
                continue
            if first != "corrects" and _mentions(event, bot) and mid in answered:
                continue
            add(mid, "continues", "asker_followup", str(_reply_target(event)))

    # 4. quote_context: the neighbourhood of an old or very short quoted message
    if quoted in by_id and "quote_context" in enabled:
        target = by_id[quoted]
        # A short quoted bot answer is explained by the question it answers (rule 1), and
        # a quoted image is its own material: only a short member text lacks its topic.
        short = (
            str(target.get("user_id")) != bot
            and not _has_media(target)
            and len(PUNCT.sub("", _text(target))) <= quote_short_chars
        )
        if when(target) < t0 - follow_seconds or short:
            ids = list(by_id)
            for step, limit in ((-1, quote_before), (1, quote_after)):
                i, taken = order[quoted] + step, 0
                while 0 <= i < len(ids) and taken < limit:
                    other = by_id[ids[i]]
                    if abs(when(other) - when(target)) > quote_window_seconds:
                        break
                    if not in_flight(other) and plain_text(other):
                        taken += 1
                        add(
                            ids[i],
                            "quote_context",
                            f"quote_neighbour_{'before' if step < 0 else 'after'}",
                            quoted,
                        )
                    i += step

    # 5. refers_to: deictic or near-empty request pointing at recent material
    bare = len(PUNCT.sub("", anchor_text)) <= 2
    deictic = bool(DEICTIC.search(anchor_text))
    if (
        quoted is None
        and not own_media
        and not continued
        and not urged
        and "refers_to" in enabled
        and (deictic or bare)
    ):
        for event in reversed(prior):
            if t0 - when(event) > refer_seconds:
                break
            if str(event.get("user_id")) == asker and _has_media(event):
                mine = str(event.get("message_id"))
                add(mine, "refers_to", "asker_latest_media", anchor_id)
                ancestry(event, mine, "reply_to", "refers_chain")
                break

    # 6. asker's recent lines (B), without lines of other branches
    for event in reversed(prior):
        if when(event) < t0 - follow_seconds:
            break
        mid = str(event.get("message_id"))
        if str(event.get("user_id")) != asker or mid in picks:
            continue
        if "asker_filter" in enabled:
            target = _reply_target(event)
            if target is not None and target not in picks:
                continue
            # A correction builds on the asker's earlier statements (AC-10, 4.4): keep
            # their answered requests then.
            if (
                first != "corrects"
                and _mentions(event, bot)
                and (mid in answered or (asker_pending and in_flight(event)))
            ):
                continue
        add(mid, "asker_recent", "follow_window", anchor_id)

    # 7. author bursts around selected material
    if "author_burst" in enabled:
        ids = list(by_id)
        seeds = [m for m, p in picks.items() if p.relation not in ("asker_recent", "quote_context")]
        for mid in seeds:
            event = by_id[mid]
            author = str(event.get("user_id"))
            if author == bot or author == asker or not (_has_media(event) or _text(event)):
                continue
            index = order[mid]
            # The anchor's direct quote target is explicitly linked: its author's image
            # posted shortly before a caption ("image, then 这是…评论" 18 s later; live
            # 2026-11-14) belongs to it, so allow a wider gap for media next to it only.
            wide = mid == quoted
            for step in (-1, 1):
                last, i, taken = event, index + step, 0
                while 0 <= i < len(ids) and taken < burst_max:
                    other = by_id[ids[i]]
                    gap = abs(when(other) - when(last))
                    limit = quote_burst_seconds if wide and _has_media(other) else burst_seconds
                    if str(other.get("user_id")) != author or gap > limit:
                        break
                    if _reply_target(other) is None:
                        add(ids[i], "continues", "author_burst", mid)
                    last, i, taken = other, i + step, taken + 1

    # 8. same_topic by lexical overlap with the anchor text
    grams = _bigrams(anchor_text)
    if "same_topic" in enabled and len(grams) >= topic_min_shared:
        scored = []
        for event in prior:
            mid = str(event.get("message_id"))
            if mid in picks or str(event.get("user_id")) == bot or when(event) < t0 - topic_seconds:
                continue
            other = _bigrams(_text(event))
            shared = len(grams & other)
            if shared < topic_min_shared:
                continue
            ratio = shared / min(len(grams), len(other))
            if ratio >= topic_overlap:
                scored.append((ratio, order[mid], mid))
        for _, _, mid in sorted(scored, reverse=True)[:topic_cap]:
            add(mid, "same_topic", f"bigram_overlap>={topic_overlap}", anchor_id)

    # 9. recent_chat: a short slice of the group conversation, background only
    if "recent_chat" in enabled and recent_max > 0:
        taken = 0
        for event in reversed(prior):
            if taken >= recent_max or when(event) < t0 - recent_seconds:
                break
            mid = str(event.get("message_id"))
            if mid in picks or in_flight(event) or not plain_text(event):
                continue
            add(mid, "recent_chat", "recent_window", anchor_id)
            taken += 1

    # Budget: keep high-priority, then recent, picks.
    ranked = sorted(
        picks.values(), key=lambda p: (RELATION_PRIORITY[p.relation], -order[p.message_id])
    )
    kept, tokens = [], 0
    for pick in ranked:
        cost = estimate_tokens(by_id[pick.message_id])
        if len(kept) >= max_messages or tokens + cost > max_tokens:
            continue
        kept.append(pick)
        tokens += cost
    return sorted(kept, key=lambda p: order[p.message_id])


def strategy_c(events: list[dict], anchor_id: str, **params) -> list[str]:
    return [p.message_id for p in explain_c(events, anchor_id, **params)]


STRATEGIES: dict[str, Callable[[list[dict], str], list[str]]] = {
    "A": strategy_a,
    "D": strategy_d,
    "B": strategy_b,
    "C": strategy_c,
}
# Strategies that can explain their picks (USER-02: record selection reasons).
EXPLAINERS: dict[str, Callable[[list[dict], str], list[Pick]]] = {"C": explain_c}
