"""Case `mocks` parsing (docs/REPLAY-FORMAT.md section 14).

Naming: `<base>` or `<base>:<message_id>`; the suffix scopes an item to the run started
by that message and is recognised only when it equals a message id in the case. Unscoped
model/tool items and per-run faults belong to the anchor run. Same-name items are served
in list order. Every item is tracked so unused and unknown names can be reported.
"""

from dataclasses import dataclass, field
from typing import Any

MODEL_NAMES = frozenset(
    {
        "answer",
        "primary",
        "fallback",
        "vision:primary",
        "draft_answer",
        "moderation",
        "neutral_answer",
        "route",
    }
)
TOOL_NAMES = frozenset(
    {
        "web_search",
        "web_fetch",
        "headless_fetch",
        "short_link_expand",
        "bilibili_video_info",
        "knowledge_search",
        "group_info",
        "group_member_list",
    }
)
# Tool-section items that describe state rather than calls the agent makes.
STATE_NAMES = frozenset({"knowledge_store", "answer_evidence", "knowledge_commit"})
RUN_FAULTS = frozenset({"hang:model", "cancel", "slow_ack", "vision_timeout", "outbound_ack_lost"})
GLOBAL_FAULTS = frozenset(
    {
        "embedding_offline",
        "knowledge_update",
        "blacklist_update",
        "usage_ledger",
        "event_redelivery",
    }
)
KNOWN = {
    "model": MODEL_NAMES,
    "tools": TOOL_NAMES | STATE_NAMES,
    "faults": RUN_FAULTS | GLOBAL_FAULTS,
}


@dataclass
class MockItem:
    section: str
    index: int
    name: str
    base: str
    scope: str | None
    response: Any
    delay_ms: int
    error: str | None
    used: bool = False

    @property
    def delay(self) -> float:
        return self.delay_ms / 1000

    def ref(self) -> str:
        return f"{self.section}[{self.index}]:{self.name}"


@dataclass
class RunMocks:
    """Items visible to one run, in case order."""

    message_id: str
    model: list[MockItem] = field(default_factory=list)
    tools: list[MockItem] = field(default_factory=list)
    faults: list[MockItem] = field(default_factory=list)

    def take(self, base: str, section: str = "model") -> MockItem | None:
        for item in getattr(self, section):
            if item.base == base and not item.used:
                item.used = True
                return item
        return None

    def has(self, base: str, section: str = "model") -> bool:
        return any(item.base == base for item in getattr(self, section))

    def fault(self, base: str) -> MockItem | None:
        for item in self.faults:
            if item.base == base:
                item.used = True
                return item
        return None

    def tool_plan(self) -> list[MockItem]:
        return [item for item in self.tools if item.base in TOOL_NAMES]


class MockBook:
    def __init__(self, mocks: dict, message_ids: set[str], anchor_id: str):
        self.anchor_id = anchor_id
        self.items: list[MockItem] = []
        self.unknown: list[str] = []
        for section in ("model", "tools", "faults"):
            for index, raw in enumerate(mocks.get(section) or []):
                name = str(raw["name"])
                base, scope = name, None
                if ":" in name:
                    head, tail = name.rsplit(":", 1)
                    if tail in message_ids:
                        base, scope = head, tail
                item = MockItem(
                    section,
                    index,
                    name,
                    base,
                    scope,
                    raw.get("response"),
                    int(raw.get("delay_ms") or 0),
                    raw.get("error"),
                )
                if base not in KNOWN[section]:
                    self.unknown.append(item.ref())
                self.items.append(item)

    def for_run(self, message_id: str) -> RunMocks:
        run = RunMocks(message_id)
        is_anchor = message_id == self.anchor_id
        for item in self.items:
            if item.base in GLOBAL_FAULTS or item.base in STATE_NAMES:
                continue
            if item.scope == message_id or (item.scope is None and is_anchor):
                getattr(run, item.section).append(item)
        return run

    def global_faults(self) -> list[MockItem]:
        return [i for i in self.items if i.section == "faults" and i.base in GLOBAL_FAULTS]

    def state(self, base: str, scope: str | None = None, mark: bool = True) -> list[MockItem]:
        found = [i for i in self.items if i.base == base and (scope is None or i.scope == scope)]
        for item in found if mark else ():
            item.used = True
        return found

    def unused(self) -> list[str]:
        return [i.ref() for i in self.items if not i.used]
