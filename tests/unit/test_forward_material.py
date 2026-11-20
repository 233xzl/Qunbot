"""Only synthetic member text and media are used by these forward regressions."""

import asyncio
import io

from PIL import Image

from group_agent.forward_material import ForwardMaterial
from group_agent.gateway import ActionFailed
from group_agent.live_agent import _load_image, render_material


async def test_nested_partial_forward_attaches_image_and_retains_readable_nodes(tmp_path):
    def segment(kind, **data):
        return {"type": kind, "data": data}

    def node(uid, *parts):
        return {"sender": {"user_id": uid}, "message": list(parts)}

    async def fetch(key):
        if key == "inner":
            return [node("A", segment("text", text="限定原话"), segment("image", file="photo"))]
        if key == "missing":
            raise ActionFailed("bad_response")
        return [
            node("blocked", segment("text", text="屏蔽内容")),
            node("B", segment("forward", id="inner"), segment("forward", id="missing")),
        ]

    async def read_image(data):
        assert data == {"file": "photo"}
        image = io.BytesIO()
        Image.new("RGB", (8, 8)).save(image, format="PNG")
        return image.getvalue()

    anchor = {"user_id": "U", "message_id": "anchor", "message": [segment("forward", id="outer")]}
    resolver = ForwardMaterial(
        fetch, read_image, asyncio.get_running_loop().time() + 2, {"blocked"}
    )
    await resolver.expand(anchor)
    text, images, _, _ = render_material([], anchor, "bot", tmp_path)
    assert "限定原话" in text and "屏蔽内容" not in text
    assert "转发部分内容未取得" in text
    assert resolver.gaps == {"bad_response": 1}
    assert len(images) == 1 and _load_image(images[0][1], 64)[0]
