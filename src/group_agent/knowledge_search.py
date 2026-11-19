"""Lexical knowledge search (T4.2): exact name/alias, quoted examples, character bigrams.

Matching follows the legacy keyword fallback in spirit (exact name/alias first, quoted
examples next, then lexical overlap) but uses character bigrams instead of jieba so it
needs no tokenizer dependency. The index is built in memory from the live entries of the
knowledge base (`knowledge_base.KnowledgeBase.search` rebuilds it when the base changes);
about 1,500 entries index in milliseconds, so no FTS table is kept.

Scope (USER-03): an entry is visible in a group when it is public or scoped to that group.
Live entries (KB-03): active, and not superseded by an active owner correction.

`KnowledgeSnapshot` opens a knowledge file read-only (`mode=ro`): the new knowledge base
or, for old experiment outputs, a legacy plugin snapshot (all entries public). Vector
recall is not implemented (see docs/CUTOVER.md, "知识库与教学").
"""

from collections.abc import Iterable
from dataclasses import dataclass
import json
from pathlib import Path
import re
import sqlite3

_KEEP = re.compile(r"[\w\u4e00-\u9fff]+")
# Bigrams that appear in almost every question and carry no topic.
_STOP = frozenset(
    "什么 意思 这个 那个 一下 解释 是什 么意 思是 这是 那是 怎么 为什 么是 可以 知道 有没 没有"
    " 看看 帮我 图片 梗是 出处 来源 哪里".split()
)


def _norm(text: str) -> str:
    return "".join(_KEEP.findall(str(text).casefold()))


def bigrams(text: str) -> set[str]:
    s = _norm(text)
    return {s[i : i + 2] for i in range(len(s) - 1)} - _STOP


@dataclass(frozen=True)
class KnowledgeEntry:
    id: str
    name: str
    aliases: tuple[str, ...]
    revision: int
    kind: str
    category: str
    explanation: str
    source: str
    examples: tuple[str, ...]
    scope: str = "public"
    images: tuple[str, ...] = ()


@dataclass(frozen=True)
class Hit:
    entry: KnowledgeEntry
    score: float
    match: str  # exact | example | name | overlap


def entry_from_data(d: dict, scope: str = "public") -> KnowledgeEntry:
    return KnowledgeEntry(
        id=str(d["id"]),
        name=str(d.get("name") or ""),
        aliases=tuple(str(a) for a in d.get("aliases") or [] if a),
        revision=int(d.get("revision") or 1),
        kind=str(d.get("kind") or ""),
        category=str(d.get("category") or ""),
        explanation=str(d.get("explanation") or ""),
        source=str(d.get("source") or ""),
        examples=tuple(str(e) for e in d.get("examples") or [] if e),
        scope=scope,
        images=tuple(str(i) for i in d.get("images") or [] if i),
    )


def live_entries(records: Iterable[tuple[dict, str]]) -> list[KnowledgeEntry]:
    """Active `(data, scope)` records minus those an owner correction supersedes (legacy
    `live()` rule, KB-03)."""
    records = list(records)
    superseded = {
        str(t) for d, _ in records if d.get("kind") == "owner" for t in d.get("supersedes") or []
    }
    return [entry_from_data(d, scope) for d, scope in records if str(d["id"]) not in superseded]


def visible(entry: KnowledgeEntry, group_id: str | None) -> bool:
    return entry.scope == "public" or (group_id is not None and entry.scope == f"group:{group_id}")


class KnowledgeIndex:
    def __init__(self, entries: Iterable[KnowledgeEntry]):
        self.entries: list[KnowledgeEntry] = list(entries)
        self._by_id = {e.id: e for e in self.entries}
        self._index = [
            (e, [_norm(n) for n in (e.name, *e.aliases) if _norm(n)], bigrams(e.explanation))
            for e in self.entries
        ]

    def get(self, entry_id: str) -> KnowledgeEntry | None:
        return self._by_id.get(entry_id)

    def search(self, query: str, limit: int = 5, group_id: str | None = None) -> list[Hit]:
        """Public entries plus those scoped to `group_id` (None: public only)."""
        q = _norm(query)
        if not q:
            return []
        q_grams = bigrams(query)
        hits: list[Hit] = []
        for entry, names, expl_grams in self._index:
            if not visible(entry, group_id):
                continue
            exact = [n for n in names if (len(n) > 1 and n in q) or n == q]
            if exact:
                # Longer names are more specific ("原神启动" over "原神").
                hits.append(Hit(entry, 100 + max(map(len, exact)), "exact"))
                continue
            if any(len(_norm(x)) >= 4 and _norm(x) in q for x in entry.examples):
                hits.append(Hit(entry, 60, "example"))
                continue
            best = 0.0
            for n in names:
                grams = bigrams(n)
                if len(n) >= 3 and grams:
                    best = max(best, len(grams & q_grams) / len(grams))
            if best >= 0.67:
                hits.append(Hit(entry, 40 * best, "name"))
                continue
            common = expl_grams & q_grams
            if q_grams and len(common) >= 4:
                hits.append(Hit(entry, 10 * len(common) / len(q_grams), "overlap"))
        hits.sort(key=lambda h: (-h.score, h.entry.id))
        return hits[:limit]


class KnowledgeSnapshot(KnowledgeIndex):
    """Read-only index over a knowledge file (new base or legacy snapshot)."""

    def __init__(self, path: Path):
        con = sqlite3.connect(f"{Path(path).resolve().as_uri()}?mode=ro", uri=True)
        try:
            cols = {r[1] for r in con.execute("PRAGMA table_info(entries)")}
            scope = "scope" if "scope" in cols else "'public'"
            rows = con.execute(f"SELECT data, {scope} FROM entries WHERE active = 1").fetchall()
        finally:
            con.close()
        super().__init__(live_entries((json.loads(raw), str(s)) for raw, s in rows))
