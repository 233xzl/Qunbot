"""Expand selected merged messages into bounded, read-only material for one answer."""

import asyncio
from collections import Counter

from .gateway import GatewayError
from .media import MediaUnavailable


class ForwardMaterial:
    def __init__(self, fetch, image_reader, deadline, blocked_users, max_images=4):
        self.fetch, self.image_reader = fetch, image_reader
        self.deadline, self.blocked_users = deadline, blocked_users
        self.images_left, self.chars_left = max_images, 12000
        self.stats, self.gaps = Counter(), Counter()
        self.seen = set()

    def gap(self, reason):
        self.gaps[reason] += 1
        return {"type": "forward_gap", "data": {}}

    async def expand(self, event):
        result = []
        for part in event["message"]:
            if part.get("type") == "forward":
                await self.resource(part.get("data", {}).get("id"), result, 0)
            else:
                result.append(part)
        event["message"] = result

    async def resource(self, key, out, depth):
        if not isinstance(key, str) or not key:
            out.append(self.gap("missing_id"))
            return
        if key in self.seen or depth > 2 or len(self.seen) >= 4:
            out.append(self.gap("repeat_or_depth_limit"))
            return
        self.seen.add(key)
        try:
            async with asyncio.timeout_at(self.deadline):
                nodes = await self.fetch(key)
        except (GatewayError, TimeoutError) as exc:
            out.append(self.gap(getattr(exc, "kind", "timeout")))
            return
        self.stats["nodes_received"] += len(nodes)
        if not nodes:
            out.append(self.gap("empty"))
        for node in nodes:
            if self.stats["nodes_used"] >= 60 or not self.chars_left:
                out.append(self.gap("truncated"))
                break
            sender = node.get("sender") or {}
            if (
                not isinstance(sender, dict)
                or str(sender.get("user_id", node.get("user_id"))) in self.blocked_users
            ):
                continue
            parts = node.get("message")
            if not isinstance(parts, list):
                out.append(self.gap("invalid_node"))
                continue
            self.stats["nodes_used"] += 1
            name = str(sender.get("card") or sender.get("nickname") or "群友")[:40]
            out.append({"type": "forward_speaker", "data": {"text": name}})
            for part in parts[:64]:
                if not isinstance(part, dict) or not isinstance(data := part.get("data"), dict):
                    continue
                kind = part.get("type")
                if kind == "text":
                    text = str(data.get("text", ""))
                    clipped = text[: self.chars_left]
                    self.chars_left -= len(clipped)
                    out.append({"type": "forward_text", "data": {"text": clipped}})
                    if clipped != text:
                        out.append(self.gap("truncated"))
                elif kind == "forward":
                    await self.resource(data.get("id"), out, depth + 1)
                elif kind == "image":
                    await self.image(data, out)
                else:
                    out.append(
                        {
                            "type": "forward_text",
                            "data": {
                                "text": {
                                    "at": "[提及成员]",
                                    "reply": "[引用]",
                                    "face": "[表情]",
                                }.get(kind, "[其他消息]")
                            },
                        }
                    )
            if len(parts) > 64:
                out.append(self.gap("truncated"))

    async def image(self, data, out):
        ref = {"status": "omitted", "reason": "image_budget"}
        if self.image_reader and self.images_left:
            self.images_left -= 1
            try:
                async with asyncio.timeout_at(self.deadline):
                    content = await self.image_reader(data)
                ref = {"status": "available", "content": content}
            except (MediaUnavailable, TimeoutError) as exc:
                ref = {"status": "unavailable", "reason": getattr(exc, "reason", "timeout")}
        if ref["status"] != "available":
            self.gaps[ref["reason"]] += 1
        out.append({"type": "image", "data": {"media_ref": ref}})
