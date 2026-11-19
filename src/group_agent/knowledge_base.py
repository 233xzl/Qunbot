"""The new framework's own knowledge base (T4.1-T4.3; KB-01..03, LEARN-01..03).

One SQLite file (default `data/knowledge/knowledge.sqlite3`, WAL) with its example images
beside it (`images/<sha256>.<ext>`, the relative path stored in the entry data). It is kept
apart from the message store: knowledge is permanent and authoritative while messages
rotate under retention, so the file can be backed up, restored and re-migrated on its own.

Tables:

- `entries`: current state per stable id (`C0001` CHIME base, `K001` owner teaching):
  kind, scope (`public` | `group:<id>`), name, revision, active and the full record
  `data` (legacy JSON fields: aliases, category, explanation, source, examples, images,
  supersedes, ...), `content_sha256` over canonical data, author and source message;
- `revisions`: every revision ever committed (op, actor, source message, scope, active);
- `sources`: one row per teaching message that changed the base (LEARN-03 idempotency);
- `media`: example images by content hash;
- `teaching_log`: every teaching outcome (also refusals) without knowledge text;
- `imports`: migration runs (`scripts/migrate_knowledge.py`).

Writes run in one short transaction (RUN-06): a failed write leaves the previous revision
readable (LEARN-02). Authorisation is the caller's event sender id only (SEC-01): `apply`
never looks at text for identity. `revision()` answers the pre-publish check (RUN-07):
the live revision, or None once an entry is disabled or superseded.

`KnowledgeBase(":memory:")` (= `teaching.KnowledgeLedger`) is the same code for the
deterministic replay, seeded from case mocks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import io
import json
from pathlib import Path
import re
import sqlite3
from typing import Any

from .knowledge_search import Hit, KnowledgeEntry, KnowledgeIndex, live_entries

SCHEMA_VERSION = 1
DDL = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS entries (
    id TEXT PRIMARY KEY, kind TEXT NOT NULL, scope TEXT NOT NULL, name TEXT NOT NULL,
    revision INTEGER NOT NULL, active INTEGER NOT NULL, data TEXT NOT NULL,
    content_sha256 TEXT NOT NULL, origin TEXT NOT NULL,
    author TEXT, source_message TEXT, created REAL NOT NULL, updated REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS entries_name ON entries(name);
CREATE TABLE IF NOT EXISTS revisions (
    id TEXT NOT NULL, revision INTEGER NOT NULL, scope TEXT NOT NULL, active INTEGER NOT NULL,
    data TEXT NOT NULL, content_sha256 TEXT NOT NULL, created REAL NOT NULL,
    op TEXT NOT NULL, actor TEXT, source_message TEXT,
    PRIMARY KEY (id, revision)
);
CREATE TABLE IF NOT EXISTS sources (
    source_message TEXT PRIMARY KEY, id TEXT NOT NULL, revision INTEGER NOT NULL,
    op TEXT NOT NULL, t REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS media (
    sha256 TEXT PRIMARY KEY, path TEXT NOT NULL, bytes INTEGER NOT NULL, created REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS teaching_log (
    seq INTEGER PRIMARY KEY AUTOINCREMENT, t REAL NOT NULL, op TEXT NOT NULL,
    outcome TEXT NOT NULL, actor TEXT, source_message TEXT, id TEXT, revision INTEGER,
    scope TEXT, applied INTEGER NOT NULL, detail TEXT
);
CREATE TABLE IF NOT EXISTS imports (
    snapshot_sha256 TEXT NOT NULL, imported REAL NOT NULL, report TEXT NOT NULL
);
"""
IMAGE_EXT = {"PNG": ".png", "JPEG": ".jpg", "WEBP": ".webp", "GIF": ".gif"}
MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_IMAGES = 2  # legacy: the first two images of a teaching request
# Fields of a CHIME base record that an owner correction does not copy.
_BASE_ONLY = ("fingerprint", "snapshot", "ordinal", "original", "doc_id")


def canonical(data: dict) -> str:
    return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def content_hash(data: dict) -> str:
    return hashlib.sha256(canonical(data).encode("utf-8")).hexdigest()


def scope_words(scope: str) -> str:
    return "所有群" if scope == "public" else "本群"


@dataclass
class KnowledgeRecord:
    id: str
    scope: str
    revision: int
    active: bool
    content: str
    term: str | None = None
    sources: set[str] = field(default_factory=set)
    kind: str = "owner"


@dataclass
class _Row:
    id: str
    kind: str
    scope: str
    name: str
    revision: int
    active: bool
    data: dict

    @classmethod
    def of(cls, row: sqlite3.Row) -> _Row:
        return cls(
            row["id"],
            row["kind"],
            row["scope"],
            row["name"],
            int(row["revision"]),
            bool(row["active"]),
            json.loads(row["data"]),
        )


class KnowledgeBase:
    def __init__(self, path: Path | str = ":memory:"):
        self.memory = str(path) == ":memory:"
        self.path = None if self.memory else Path(path)
        self.root = None if self.path is None else self.path.parent
        if self.root is not None:
            self.root.mkdir(parents=True, exist_ok=True)
        self.con = sqlite3.connect(str(path), timeout=5)
        self.con.row_factory = sqlite3.Row
        if not self.memory:
            self.con.execute("PRAGMA journal_mode=WAL")
        self.con.execute("PRAGMA busy_timeout=5000")
        with self.con:
            self.con.executescript(DDL)
            self.con.execute(
                "INSERT OR IGNORE INTO meta VALUES ('schema_version', ?)", (str(SCHEMA_VERSION),)
            )
        self.log: list[dict] = []
        self._writes = 0
        self._index: tuple[tuple[int, int], KnowledgeIndex] | None = None

    def close(self) -> None:
        self.con.close()

    # -- reads -------------------------------------------------------------------------
    def _row(self, entry_id: str) -> _Row | None:
        row = self.con.execute("SELECT * FROM entries WHERE id=?", (entry_id,)).fetchone()
        return _Row.of(row) if row else None

    def _superseding(self, entry_id: str) -> list[_Row]:
        """Active owner corrections that supersede `entry_id`."""
        rows = self.con.execute(
            "SELECT * FROM entries WHERE kind='owner' AND active=1 AND data LIKE ?",
            (f'%"{entry_id}"%',),
        ).fetchall()
        return [r for r in map(_Row.of, rows) if entry_id in (r.data.get("supersedes") or [])]

    def _live(self, row: _Row | None) -> bool:
        return row is not None and row.active and not self._superseding(row.id)

    def revision(self, knowledge_id: str) -> int | None:
        """RUN-07 / KB-03: the live revision, None when disabled, superseded or unknown."""
        row = self._row(knowledge_id)
        return row.revision if self._live(row) else None

    def entry(self, knowledge_id: str) -> KnowledgeEntry | None:
        """The live entry (current revision); for a superseded entry its single active
        owner correction; None when disabled or unknown."""
        row = self._row(knowledge_id)
        if row is None or not row.active:
            return None
        corrections = self._superseding(row.id)
        if len(corrections) == 1:
            return self.index().get(corrections[0].id)
        return None if corrections else self.index().get(knowledge_id)

    def index(self) -> KnowledgeIndex:
        version = self.con.execute("PRAGMA data_version").fetchone()[0]
        key = (self._writes, version)
        if self._index is None or self._index[0] != key:
            rows = self.con.execute("SELECT data, scope FROM entries WHERE active=1").fetchall()
            entries = live_entries((json.loads(d), s) for d, s in rows)
            self._index = (key, KnowledgeIndex(entries))
        return self._index[1]

    def search(self, query: str, limit: int = 5, group_id: str | None = None) -> list[Hit]:
        return self.index().search(query, limit=limit, group_id=group_id)

    @property
    def records(self) -> dict[str, KnowledgeRecord]:
        sources: dict[str, set[str]] = {}
        for s, kid in self.con.execute("SELECT source_message, id FROM sources"):
            sources.setdefault(kid, set()).add(s)
        out = {}
        for row in map(_Row.of, self.con.execute("SELECT * FROM entries ORDER BY id")):
            out[row.id] = KnowledgeRecord(
                row.id,
                row.scope,
                row.revision,
                row.active,
                str(row.data.get("explanation") or ""),
                row.name or None,
                sources.get(row.id, set()),
                row.kind,
            )
        return out

    def counts(self) -> dict[str, Any]:
        q = self.con.execute
        return {
            "entries": q("SELECT count(*) FROM entries").fetchone()[0],
            "by_kind": dict(q("SELECT kind, count(*) FROM entries GROUP BY kind").fetchall()),
            "active": q("SELECT count(*) FROM entries WHERE active=1").fetchone()[0],
            "revisions": q("SELECT count(*) FROM revisions").fetchone()[0],
            "media": q("SELECT count(*) FROM media").fetchone()[0],
        }

    # -- write primitives --------------------------------------------------------------
    def _write(
        self,
        row: _Row,
        *,
        op: str,
        origin: str,
        actor: str | None,
        source_message: str | None,
        now: float,
        new: bool,
    ) -> None:
        row.data.update(id=row.id, kind=row.kind, name=row.name)
        row.data.update(revision=row.revision, active=row.active)
        payload, digest = canonical(row.data), content_hash(row.data)
        with self.con:
            self.con.execute(
                "INSERT INTO revisions VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    row.id,
                    row.revision,
                    row.scope,
                    int(row.active),
                    payload,
                    digest,
                    now,
                    op,
                    actor,
                    source_message,
                ),
            )
            if new:
                self.con.execute(
                    "INSERT INTO entries VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        row.id,
                        row.kind,
                        row.scope,
                        row.name,
                        row.revision,
                        int(row.active),
                        payload,
                        digest,
                        origin,
                        actor,
                        source_message,
                        now,
                        now,
                    ),
                )
            else:
                self.con.execute(
                    "UPDATE entries SET name=?, scope=?, revision=?, active=?, data=?,"
                    " content_sha256=?, updated=? WHERE id=?",
                    (
                        row.name,
                        row.scope,
                        row.revision,
                        int(row.active),
                        payload,
                        digest,
                        now,
                        row.id,
                    ),
                )
            if source_message is not None:
                self.con.execute(
                    "INSERT OR IGNORE INTO sources VALUES (?,?,?,?,?)",
                    (source_message, row.id, row.revision, op, now),
                )
        self._writes += 1

    def seed(self, record: KnowledgeRecord) -> None:
        """Replay only: add a record unless present (ignored for a file-backed base, whose
        content comes from migration and teaching)."""
        if not self.memory or self._row(record.id) is not None:
            return
        data = {"explanation": record.content, "aliases": [], "examples": [], "images": []}
        row = _Row(
            record.id,
            record.kind,
            record.scope,
            record.term or "",
            record.revision,
            record.active,
            data,
        )
        self._write(
            row, op="seed", origin="seed", actor=None, source_message=None, now=0.0, new=True
        )
        for source in record.sources:
            with self.con:
                self.con.execute(
                    "INSERT OR IGNORE INTO sources VALUES (?,?,?,?,?)",
                    (source, record.id, record.revision, "seed", 0.0),
                )

    def external_update(self, knowledge_id: str, revision: int, content: str, now: float, source):
        """A change made by another run or maintenance path (replay fault)."""
        row = self._row(knowledge_id)
        if row is None:
            return
        row.revision, row.data["explanation"] = revision, content
        self._write(
            row,
            op="external_update",
            origin="seed",
            actor=None,
            source_message=source,
            now=now,
            new=False,
        )
        self._log(now, "external_update", "committed", None, source, row, applied=True)

    def next_id(self, prefix: str = "K") -> str:
        row = self.con.execute(
            "SELECT MAX(CAST(SUBSTR(id, 2) AS INTEGER)) FROM entries WHERE id GLOB ?",
            (prefix + "[0-9]*",),
        ).fetchone()
        return f"{prefix}{(row[0] or 0) + 1:0{4 if prefix == 'C' else 3}d}"

    def _save_images(self, images: list[Path | None], now: float) -> list[str] | None:
        """Copy teaching images into `images/` by content hash; None when images were
        attached but none could be read as PNG/JPEG/WEBP/GIF."""
        if not images:
            return []
        if self.root is None:
            return []  # replay: no files
        from PIL import Image

        saved = []
        for path in images[:MAX_IMAGES]:
            try:
                if path is None or not path.is_file() or path.stat().st_size > MAX_IMAGE_BYTES:
                    continue
                raw = path.read_bytes()
                with Image.open(io.BytesIO(raw)) as picture:
                    ext = IMAGE_EXT.get(picture.format or "")
                    picture.verify()
            except (OSError, ValueError, SyntaxError):
                continue
            if not ext:
                continue
            digest = hashlib.sha256(raw).hexdigest()
            relative = f"images/{digest}{ext}"
            target = self.root / relative
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                tmp = target.with_suffix(target.suffix + ".tmp")
                tmp.write_bytes(raw)
                tmp.replace(target)
            with self.con:
                self.con.execute(
                    "INSERT OR IGNORE INTO media VALUES (?,?,?,?)",
                    (digest, relative, len(raw), now),
                )
            saved.append(relative)
        return saved or None

    # -- teaching ----------------------------------------------------------------------
    def _log(
        self, now, op, outcome, actor, source, row: _Row | None = None, applied=False, **extra
    ):
        entry = {
            "t": now,
            "op": op,
            "outcome": outcome,
            "actor": actor,
            "source_message": source,
            "id": row.id if row else None,
            "revision": row.revision if row else None,
            "scope": row.scope if row else None,
            "term": (row.name or None) if row else None,
            "applied": applied,
            **extra,
        }
        self.log.append(entry)
        detail = {k: v for k, v in extra.items() if k in ("expected", "ids", "kind", "images")}
        try:
            with self.con:
                self.con.execute(
                    "INSERT INTO teaching_log (t, op, outcome, actor, source_message, id,"
                    " revision, scope, applied, detail) VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (
                        now,
                        op,
                        outcome,
                        actor,
                        source,
                        entry["id"],
                        entry["revision"],
                        entry["scope"],
                        int(applied),
                        json.dumps(detail),
                    ),
                )
        except sqlite3.Error:
            pass  # the audit row is diagnostics; the outcome is already decided
        return entry

    def _visible(self, row: _Row, group_id: str) -> bool:
        return row.scope in ("public", f"group:{group_id}")

    def _by_name(self, term: str, group_id: str, *, live: bool) -> list[_Row]:
        """Visible entries named (or aliased) `term`; live ones, or disabled ones (restore).
        The group-scoped entry wins over a public one of the same name."""
        key = re.sub(r"\s+", "", term).casefold()
        rows = self.con.execute("SELECT * FROM entries").fetchall()
        found = []
        for row in map(_Row.of, rows):
            names = [row.name, *(row.data.get("aliases") or [])]
            if not self._visible(row, group_id):
                continue
            if key not in {re.sub(r"\s+", "", str(n)).casefold() for n in names if n}:
                continue
            if self._live(row) if live else not row.active:
                found.append(row)
        group = [r for r in found if r.scope != "public"]
        return group or found

    def _resolve(self, command, group_id: str, target, *, live: bool):
        """(row, expected revision, outcome) for correct/disable/restore/view."""
        if target is not None:
            row = self._row(str(target[0]))
            if row is None or not self._visible(row, group_id):
                return None, None, "not_found"
            return row, int(target[1]), None
        if command.target_id:
            row = self._row(command.target_id)
            if row is None or not self._visible(row, group_id):
                return None, None, "not_found"
            return row, command.expected, None
        if command.term:
            rows = self._by_name(command.term, group_id, live=live)
            if len(rows) > 1:
                return rows, None, "ambiguous"
            if rows:
                return rows[0], command.expected, None
        return None, None, "not_found"

    def apply(
        self,
        command,
        *,
        actor: str,
        owner_id: str | None,
        source_message: str,
        group_id: str,
        now: float,
        new_id: str | None = None,
        target: tuple[str, int] | None = None,
        images: list[Path | None] | None = None,
    ) -> dict:
        """Return a log entry; `applied` is True only when the base changed.

        `actor` is the event's real sender id; only `owner_id` may write (view is open to
        everyone, limited to entries visible in `group_id`). `target` is `(id, revision)`
        the quoted answer used; `images` are the teaching message's image files (None for
        an image that could not be fetched)."""
        op = command.op
        if op == "view":
            row, _, outcome = self._resolve(command, group_id, None, live=True)
            if outcome is None and isinstance(row, _Row):
                return self._log(now, op, "viewed", actor, source_message, row)
            if outcome == "not_found" and command.term:  # a disabled entry by name
                rows = self._by_name(command.term, group_id, live=False)
                if len(rows) == 1:
                    return self._log(now, op, "viewed", actor, source_message, rows[0])
            ids = [r.id for r in row] if isinstance(row, list) else None
            return self._log(now, op, outcome, actor, source_message, ids=ids)
        if owner_id is None or actor != owner_id:
            # No lookup before this point: a refusal says nothing about the base.
            return self._log(now, op, "unauthorized", actor, source_message)
        done = self.con.execute(
            "SELECT id FROM sources WHERE source_message=?", (source_message,)
        ).fetchone()
        if done is not None:
            return self._log(now, op, "duplicate", actor, source_message, self._row(done[0]))
        if op == "add":
            return self._add(command, actor, source_message, group_id, now, new_id, images)
        live = op != "restore"
        row, expected, outcome = self._resolve(command, group_id, target, live=live)
        if outcome == "ambiguous":
            return self._log(now, op, outcome, actor, source_message, ids=[r.id for r in row])
        if outcome is not None:
            return self._log(now, op, outcome, actor, source_message)
        if op == "correct":
            return self._correct(
                command,
                row,
                expected,
                target is not None,
                actor,
                source_message,
                group_id,
                now,
                new_id,
                images,
            )
        # disable | restore
        want = op == "restore"
        if expected is not None and expected != row.revision:
            return self._log(now, op, "conflict", actor, source_message, row, expected=expected)
        if row.active == want:
            return self._log(now, op, "duplicate", actor, source_message, row)
        row.active, row.revision = want, row.revision + 1
        self._write(
            row,
            op=op,
            origin="taught",
            actor=actor,
            source_message=source_message,
            now=now,
            new=False,
        )
        return self._log(now, op, "committed", actor, source_message, row, applied=True)

    def _add(self, command, actor, source_message, group_id, now, new_id, images) -> dict:
        if command.scope is None:
            return self._log(now, "add", "missing_scope", actor, source_message)
        if not command.term or not command.content:
            return self._log(now, "add", "missing_name", actor, source_message)
        # Same name in the same scope: a re-sent teaching (same content) or a clash. CHIME
        # base entries of the same name do not block: names can carry different meanings.
        rows = self.con.execute(
            "SELECT * FROM entries WHERE kind!='base' AND active=1 AND scope=? AND name=?",
            (command.scope, command.term),
        ).fetchall()
        same = [r for r in map(_Row.of, rows) if self._live(r)]
        if same:
            outcome = (
                "duplicate" if same[0].data.get("explanation") == command.content else "conflict"
            )
            return self._log(now, "add", outcome, actor, source_message, same[0])
        saved = self._save_images(list(images or []), now)
        if saved is None:
            return self._log(now, "add", "image_unavailable", actor, source_message)
        data = {
            "aliases": [],
            "category": "其他",
            "explanation": command.content,
            "source": "",
            "examples": [],
            "images": saved,
            "supersedes": [],
        }
        row = _Row(new_id or self.next_id(), "owner", command.scope, command.term, 1, True, data)
        self._write(
            row,
            op="add",
            origin="taught",
            actor=actor,
            source_message=source_message,
            now=now,
            new=True,
        )
        return self._log(
            now, "add", "committed", actor, source_message, row, applied=True, images=len(saved)
        )

    def _correct(
        self,
        command,
        row: _Row,
        expected,
        quoted,
        actor,
        source_message,
        group_id,
        now,
        new_id,
        images,
    ) -> dict:
        if not command.content:
            return self._log(now, "correct", "missing_name", actor, source_message)
        if not row.active:
            return self._log(now, "correct", "not_found", actor, source_message)
        if row.kind == "base":
            corrections = self._superseding(row.id)
            if corrections and quoted:
                # The quoted answer used the base entry; it has been corrected since.
                current = corrections[0]
                return self._log(
                    now, "correct", "conflict", actor, source_message, current, expected=expected
                )
            if len(corrections) > 1:
                ids = [r.id for r in corrections]
                return self._log(now, "correct", "ambiguous", actor, source_message, ids=ids)
            if corrections:
                row, expected = corrections[0], None
        if expected is not None and expected != row.revision:
            return self._log(
                now, "correct", "conflict", actor, source_message, row, expected=expected
            )
        if row.data.get("explanation") == command.content and not images:
            return self._log(now, "correct", "duplicate", actor, source_message, row)
        saved = self._save_images(list(images or []), now)
        if saved is None:
            return self._log(now, "correct", "image_unavailable", actor, source_message)
        if row.kind == "base":
            # Legacy rule: a correction of a CHIME entry is a new owner entry that
            # supersedes exactly that entry (KB-03); the base record stays unchanged.
            data = {k: v for k, v in row.data.items() if k not in _BASE_ONLY}
            data.update(explanation=command.content, supersedes=[row.id], images=saved)
            base_id = row.id
            row = _Row(new_id or self.next_id(), "owner", row.scope, row.name, 1, True, data)
            self._write(
                row,
                op="correct",
                origin="taught",
                actor=actor,
                source_message=source_message,
                now=now,
                new=True,
            )
            return self._log(
                now,
                "correct",
                "committed",
                actor,
                source_message,
                row,
                applied=True,
                supersedes=base_id,
            )
        row.data["explanation"] = command.content
        row.data["images"] = list(row.data.get("images") or []) + [
            p for p in saved if p not in (row.data.get("images") or [])
        ]
        row.revision += 1
        self._write(
            row,
            op="correct",
            origin="taught",
            actor=actor,
            source_message=source_message,
            now=now,
            new=False,
        )
        return self._log(now, "correct", "committed", actor, source_message, row, applied=True)

    def describe(self, knowledge_id: str) -> str | None:
        """Legacy `knowledge_text` rendering plus the status line (view)."""
        row = self._row(knowledge_id)
        if row is None:
            return None
        d = row.data
        lines = [
            f"[{row.id} v{row.revision}] {row.name}",
            "别名：" + "、".join(d.get("aliases") or []),
            "分类：" + str(d.get("category") or "其他"),
            "说明：" + str(d.get("explanation") or ""),
            "技术原意：" + str(d.get("technical_meaning") or ""),
            "出处：" + (str(d.get("source") or "") or "未提供出处"),
            "例句：" + "；".join(d.get("examples") or []),
            "资料来源：" + ("CHIME 原始记录" if row.kind == "base" else "主人明确教学"),
            "适用范围：" + ("所有群" if row.scope == "public" else "本群"),
        ]
        live = self._live(row)
        status = "已启用" if live else "已停用" if not row.active else "已被更正条目替代"
        return "\n".join(lines) + f"\n状态：{status}"
