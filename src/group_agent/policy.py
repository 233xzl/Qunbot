"""Runtime policy rules shared by the replay harness and the phase-3 pipeline.

Values follow requirements section 8 and decisions D-04, D-08, D-10, D-11, D-12. They are
plain functions/dataclasses so the deterministic replay and the production scheduler use
the same rules; config wiring (T0.3) will later construct `RuntimeLimits` from settings.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

DAY_SECONDS = 86400


@dataclass(frozen=True)
class RuntimeLimits:
    # RUN-04 concurrency and queue (section 8).
    global_active: int = 4
    user_active: int = 2
    global_queue: int = 32
    user_queue: int = 4
    queue_wait_seconds: float = 30
    busy_notice_window_seconds: float = 60  # D-08
    # Total request deadline, counted from arrival (queue + work + send).
    deadline_seconds: float = 120
    # WEB-03 / D-11: per-run tool budget and per-call timeout.
    web_searches: int = 3
    web_reads: int = 5
    tool_timeout_seconds: float = 25
    finish_reserve_seconds: float = 30  # final no-tool step (25 s) + send reserve
    # Context: history cap in messages; degraded mode keeps fewer (proposal, RUN-10).
    context_messages: int = 80
    degraded_context_messages: int = 10
    # RUN-09 / D-10 send rate limit.
    group_min_interval_seconds: float = 2
    group_per_minute: int = 10
    account_per_minute: int = 20
    # RUN-10 / D-04 daily cost cap ratios.
    degrade_ratio: float = 0.8
    cap_ratio: float = 1.0
    # SEC-03 (P1): conservative initial threshold (D-12: prefer a neutral short answer).
    moderation_threshold: float = 0.5
    # Retention hard limit for chat records (D-01: 7 days).
    retention_days: float = 7


COST_NORMAL, COST_DEGRADED, COST_REJECTED = "normal", "degraded", "rejected"


def cost_mode(ratio: float, is_owner: bool, limits: RuntimeLimits = RuntimeLimits()) -> str:
    """D-04: degrade at 80 %; at 100 % only the owner is served (degraded).

    A rejected request gets no notice at all (D-12).
    """
    if ratio >= limits.cap_ratio:
        return COST_DEGRADED if is_owner else COST_REJECTED
    if ratio >= limits.degrade_ratio:
        return COST_DEGRADED
    return COST_NORMAL


WEB_SEARCH_TOOLS = frozenset({"web_search"})
WEB_READ_TOOLS = frozenset({"web_fetch", "headless_fetch", "bilibili_video_info"})
WEB_TOOLS = WEB_SEARCH_TOOLS | WEB_READ_TOOLS | {"short_link_expand"}


class ToolBudget:
    """Per-run web budget (WEB-03): failures consume budget too."""

    def __init__(self, limits: RuntimeLimits = RuntimeLimits()):
        self.left = {"search": limits.web_searches, "read": limits.web_reads}

    def try_consume(self, tool: str) -> bool:
        kind = "search" if tool in WEB_SEARCH_TOOLS else "read" if tool in WEB_READ_TOOLS else None
        if tool == "short_link_expand":
            # Expansion is part of the ensuing video read, not a second read.
            return self.left["read"] > 0
        if kind is None:
            return True
        if self.left[kind] <= 0:
            return False
        self.left[kind] -= 1
        return True


def tool_timeout(remaining_seconds: float, limits: RuntimeLimits = RuntimeLimits()) -> float:
    """WEB-03: min(configured timeout, remaining deadline minus finish/send reserve)."""
    return max(
        0.0, min(limits.tool_timeout_seconds, remaining_seconds - limits.finish_reserve_seconds)
    )


def is_expired(
    platform_time: float | None, now: float, limits: RuntimeLimits = RuntimeLimits()
) -> bool:
    """Retention hard limit by message production time; unknown time is kept."""
    return platform_time is not None and now - platform_time > limits.retention_days * DAY_SECONDS


def knowledge_visible(record: Mapping[str, Any], group_id: str) -> bool:
    """USER-03 / KB-03: active public knowledge, or active knowledge scoped to this group."""
    scope = record.get("scope")
    return bool(record.get("active", True)) and (scope == "public" or scope == f"group:{group_id}")


def prepublish_check(
    requester: str,
    group_id: str,
    blocked_users: frozenset[str],
    blocked_groups: frozenset[str],
    used_knowledge: Mapping[str, int],
    current_revision: Callable[[str], int | None],
) -> str | None:
    """RUN-07: re-check permissions and knowledge versions right before publishing.

    Returns a reason code when the result must not be published as is.
    """
    if group_id in blocked_groups:
        return "group_blocked"
    if requester in blocked_users:
        return "requester_blocked"
    for knowledge_id, revision in used_knowledge.items():
        if current_revision(knowledge_id) != revision:
            return "knowledge_changed"
    return None


def moderation_blocks(result: Any, limits: RuntimeLimits = RuntimeLimits()) -> bool:
    """SEC-03 (P1): a `choice/probability` judgement above threshold forces a neutral reply."""
    if not isinstance(result, Mapping):
        return False
    probability = result.get("probability")
    return (
        result.get("choice") == "sensitive"
        and isinstance(probability, (int, float))
        and probability >= limits.moderation_threshold
    )


BUSY_NOTICE_TEXT = "现在排队的请求有点多，请稍后再问。"
