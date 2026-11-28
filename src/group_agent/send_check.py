"""Final check of every outgoing group text (`live_send.LiveSender`): no ids of the bot,
the owner, the group or the asker, no account-number wording next to a number, and no
key-like strings."""

from __future__ import annotations

import re

# Account-number wording next to a 5-11 digit number, and key-like strings.
_ACCOUNT_NUMBER = re.compile(
    r"(?:QQ|qq|Qq|扣扣|群号|账号|帐号|号码|uin|UIN)\s*[:：号是为码]{0,3}\s*\d{5,11}(?!\d)"
)
_SECRET = re.compile(
    r"(?<![A-Za-z0-9])sk-[A-Za-z0-9_-]{16,}"
    r"|Bearer\s+[A-Za-z0-9._~+/=-]{16,}"
    r"|(?i:api[_-]?key|access[_-]?token|secret[_-]?key|password)\s*[:=]"
    r"|(?<![A-Za-z0-9])[A-Fa-f0-9]{32,}(?![A-Za-z0-9])"
    r"|(?<![A-Za-z0-9+_-])[A-Za-z0-9+_-]{40,}(?![A-Za-z0-9+_-])"
)


def final_text_check(text: str, forbidden: set[str]) -> str | None:
    """Reason to refuse the outgoing text, or None. `forbidden`: ids that must not appear
    (bot, owner, group, asker)."""
    for value in forbidden:
        if value and len(value) >= 5 and re.search(rf"(?<!\d){re.escape(value)}(?!\d)", text):
            return "contains_id"
    if _ACCOUNT_NUMBER.search(text):
        return "contains_account_number"
    if _SECRET.search(text):
        return "contains_secret_like"
    return None
