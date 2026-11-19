"""LINK-01: turn QQ json card segments into material (title, summary, source, link)."""

import json
import re
from typing import Any
from urllib.parse import urlsplit

# ASCII only: chat text often glues full-width punctuation to links.
URL = re.compile(r"https?://[^\s\"'<>\u0080-\uffff]+")


def parse_card(segment: Any) -> dict | None:
    """Return card material, or None when the segment is not a readable card."""
    if not isinstance(segment, dict) or segment.get("type") != "json":
        return None
    raw = (segment.get("data") or {}).get("data")
    try:
        card = json.loads(raw) if isinstance(raw, str) else None
    except ValueError:
        return None
    if not isinstance(card, dict):
        return None
    meta = card.get("meta") if isinstance(card.get("meta"), dict) else {}
    app = str(card.get("app") or "")
    if isinstance(meta.get("detail_1"), dict):  # mini program (e.g. bilibili)
        d = meta["detail_1"]
        title, desc, url, source = d.get("desc"), None, d.get("qqdocurl"), d.get("title")
        kind = "miniapp"
    elif isinstance(meta.get("news"), dict):  # tuwen share
        d = meta["news"]
        title, desc, url, source = d.get("title"), d.get("desc"), d.get("jumpUrl"), d.get("tag")
        kind = "news"
    else:
        title, desc, url, source, kind = card.get("prompt"), None, None, None, "other"
    return {
        "app": app,
        "kind": kind,
        "title": str(title) if title else None,
        "desc": str(desc) if desc else None,
        "url": str(url) if url else None,
        "source": str(source) if source else None,
    }


def message_cards(segments) -> list[dict]:
    return [c for c in (parse_card(s) for s in segments) if c is not None]


def text_urls(text: str) -> list[str]:
    return [u.rstrip(".,;:!?)，。；：！？）】") for u in URL.findall(text)]


BVID = re.compile(r"(?<![0-9A-Za-z])(BV[0-9A-Za-z]{10})(?![0-9A-Za-z])")
AVID = re.compile(r"/video/av(\d{1,12})(?!\d)", re.I)
BILIBILI_HOSTS = ("bilibili.com", "b23.tv")
SHORT_LINK_HOSTS = frozenset({"b23.tv"})


def _host(url: str) -> str:
    try:
        return (urlsplit(url).hostname or "").lower()
    except ValueError:
        return ""


def is_bilibili(url: str) -> bool:
    host = _host(url)
    return any(host == h or host.endswith("." + h) for h in BILIBILI_HOSTS)


def is_short_link(url: str) -> bool:
    return _host(url) in SHORT_LINK_HOSTS


def bilibili_video_ref(url: str) -> tuple[str, str] | None:
    """("bvid", "BV...") or ("aid", "123") for a bilibili video URL, else None."""
    if not is_bilibili(url) or is_short_link(url):
        return None
    if m := BVID.search(url):
        return "bvid", m.group(1)
    if m := AVID.search(url):
        return "aid", m.group(1)
    return None
