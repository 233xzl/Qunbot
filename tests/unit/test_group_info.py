import asyncio
from datetime import datetime, timedelta, timezone
import json

import pytest

from group_agent.gateway import READ_ONLY_ACTIONS, ActionTimeout, OneBotApi, OneBotGateway
from group_agent.group_info import GroupInfoCache, GroupInfoTool

JOIN = int(datetime(2024, 3, 5, 12, tzinfo=timezone(timedelta(hours=8))).timestamp())


def member(uid, role="member", card="", nickname="nick", title="", **extra):
    return {
        "group_id": 777000111,
        "user_id": uid,
        "nickname": nickname,
        "card": card,
        "role": role,
        "title": title,
        "join_time": JOIN,
        "last_sent_time": JOIN + 10,
        "sex": "unknown",
        "age": 0,
        "level": "5",
        **extra,
    }


MEMBERS = [
    member(900000001, "owner", card="老王", nickname="wang", title="创始人"),
    member(900000002, "admin", card="", nickname="阿管"),
    member(900000003, "admin", card="<b>二管</b>", nickname="guan2"),
    member(900000004, card="小明", nickname="xiaoming"),
    member(900000005, card="小红", nickname="hong"),
    member(900000006, card="小刚", nickname="gang"),
    member(900000007, card="小美", nickname="mei"),
    member(900000008, card="", nickname="机器人", is_robot=True),
]
INFO = {"group_id": 777000111, "group_name": "测试群", "member_count": 8, "max_member_count": 500}
UIDS = [str(m["user_id"]) for m in MEMBERS] + ["777000111"]


class FakeApi:
    def __init__(self, error=None):
        self.calls, self.error = [], error

    async def get_group_info(self, group_id):
        self.calls.append(("get_group_info", group_id))
        if self.error:
            raise self.error
        await asyncio.sleep(0)
        return dict(INFO)

    async def get_group_member_list(self, group_id):
        self.calls.append(("get_group_member_list", group_id))
        return [dict(m) for m in MEMBERS]


def make(api=None, now=None, requester="900000004"):
    api = api or FakeApi()
    now = now if now is not None else [0.0]
    cache = GroupInfoCache(api, clock=lambda: now[0])
    return api, cache, GroupInfoTool(cache, "777000111", requester)


def assert_no_ids(result):
    text = json.dumps(result, ensure_ascii=False)
    assert not any(uid in text for uid in UIDS)
    assert "user_id" not in text


async def test_summary_and_role_counts():
    api, _, tool = make()
    result = await tool.call({"query": "summary"})
    assert result["status"] == "ok"
    assert result["result"] == {
        "group_name": "测试群",
        "member_count": 8,
        "max_member_count": 500,
        "owner": "老王",
        "admin_count": 2,
    }
    counts = await tool.call({"query": "role_counts"})
    assert counts["result"] == {"owner": 1, "admin": 2, "member": 5, "bots": 1, "total_listed": 8}
    assert "群主 1 人，管理员 2 人" in counts["material"]
    assert_no_ids(result)
    assert_no_ids(counts)
    assert api.calls == [("get_group_info", "777000111"), ("get_group_member_list", "777000111")]


async def test_admins_lists_owner_and_admins_with_titles_and_join_date():
    _, _, tool = make()
    result = await tool.call({"query": "admins"})
    data = result["result"]
    assert data["owner"]["name"] == "老王" and data["owner"]["title"] == "创始人"
    assert data["owner"]["join_date"] == "2024-03-05"
    assert [a["name"] for a in data["admins"]] == ["阿管", "<b>二管</b>"]
    assert (
        "群主：「老王」，群主，昵称「wang」，头衔「创始人」，2024-03-05 入群" in result["material"]
    )
    # Member-set names cannot open or close tags inside the material section.
    assert "<b>" not in result["material"] and "＜b＞二管＜/b＞" in result["material"]
    assert result["material"].startswith("<group_info>\n")
    assert result["material"].endswith("\n</group_info>")
    assert "不可信" in result["material"] and "不代表机器人主人" in result["material"]
    assert_no_ids(result)


async def test_summary_count_follows_fetched_member_list():
    # get_group_info.member_count can lag the member list (live: 360 vs 362).
    class Lagging(FakeApi):
        async def get_group_info(self, group_id):
            return {**INFO, "member_count": 6}

    _, _, tool = make(api=Lagging())
    result = await tool.call({"query": "summary"})
    assert result["result"]["member_count"] == 8
    assert "成员 8 人" in result["material"] and "6" not in result["material"].split("上限")[0]

    class Empty(FakeApi):
        async def get_group_member_list(self, group_id):
            return []

    _, _, tool = make(api=Empty())
    assert (await tool.call({"query": "summary"}))["result"]["member_count"] == 8


async def test_large_admin_roster_is_reduced_to_count():
    api, cache, _ = make()
    tool = GroupInfoTool(cache, "777000111", max_admins=1)
    result = await tool.call({"query": "admins"})
    assert result["result"]["admins"] is None and result["result"]["admin_count"] == 2
    assert "不逐一列出" in result["material"]


async def test_member_lookup_exact_substring_ambiguous_and_missing():
    _, _, tool = make()
    exact = await tool.call({"query": "member", "name": "小明"})
    assert exact["status"] == "ok" and exact["result"]["members"][0]["role"] == "member"
    by_nick = await tool.call({"query": "member", "name": "WANG"})
    assert by_nick["result"]["members"][0]["name"] == "老王"
    broad = await tool.call({"query": "member", "name": "小"})
    assert broad["status"] == "ambiguous" and broad["result"] == {"match_count": 4}
    for name in ("小明", "小红", "小刚", "小美"):
        assert name not in broad["material"]
    missing = await tool.call({"query": "member", "name": "不存在"})
    assert missing["status"] == "not_found"
    for result in (exact, by_nick, broad, missing):
        assert_no_ids(result)


async def test_requester_query_uses_run_context():
    _, _, tool = make(requester="900000004")
    result = await tool.call({"query": "requester"})
    assert result["result"]["member"]["name"] == "小明"
    assert "2024-03-05 入群" in result["material"]
    assert_no_ids(result)
    _, cache, _ = make()
    no_requester = GroupInfoTool(cache, "777000111")
    assert (await no_requester.call({"query": "requester"}))["error"] == "no_requester"


@pytest.mark.parametrize(
    "arguments,status,error",
    [
        ({"query": "summary", "group_id": "123456"}, "refused", "only_current_group"),
        ({"query": "admins", "group": 123456}, "refused", "only_current_group"),
        ({"query": "summary", "target_group_name": "x"}, "refused", "only_current_group"),
        ({"query": "member", "user_id": "900000004"}, "error", "unknown_argument"),
        ({"query": "list_all"}, "error", "unknown_query"),
        ({"query": "member"}, "error", "name_required"),
        ({"query": "member", "name": "x" * 40}, "error", "name_too_long"),
        ("summary", "error", "bad_arguments"),
    ],
)
async def test_bad_or_cross_group_arguments_make_no_platform_call(arguments, status, error):
    api, _, tool = make()
    result = await tool.call(arguments)
    assert (result["status"], result["error"]) == (status, error)
    assert api.calls == []
    if status == "refused":
        assert "只能查询本次提问所在的群" in result["material"]


async def test_cache_ttl_invalidation_and_single_flight():
    now = [0.0]
    api, cache, tool = make(now=now)
    await asyncio.gather(tool.call({"query": "summary"}), tool.call({"query": "role_counts"}))
    assert len(api.calls) == 2  # one info + one member-list fetch for both calls
    now[0] = 599
    await tool.call({"query": "summary"})
    assert len(api.calls) == 2
    now[0] = 601
    await tool.call({"query": "summary"})
    assert len(api.calls) == 4
    notice = {
        "post_type": "notice",
        "notice_type": "group_increase",
        "self_id": "BOT",
        "group_id": 777000111,
    }
    assert cache.observe({**notice, "group_id": 1}, "BOT")  # another group's entry
    assert not cache.observe({**notice, "notice_type": "group_upload"}, "BOT")
    await tool.call({"query": "summary"})
    assert len(api.calls) == 4
    assert cache.observe(notice, "BOT")
    await tool.call({"query": "summary"})
    assert len(api.calls) == 6
    assert cache.observe({**notice, "notice_type": "notify", "sub_type": "title"}, "BOT")


async def test_gateway_failure_is_reported_and_not_cached():
    api, _, tool = make(FakeApi(error=ActionTimeout("timeout")))
    result = await tool.call({"query": "summary"})
    assert result["status"] == "unavailable" and result["error"] == "timeout"
    assert "不要猜测" in result["material"]
    api.error = None
    assert (await tool.call({"query": "summary"}))["status"] == "ok"


async def test_tool_through_real_gateway_sends_only_bound_group_queries():
    sent = []

    class WS:
        def __init__(self):
            self.inbound = asyncio.Queue()

        def __aiter__(self):
            return self._iterate()

        async def _iterate(self):
            while (frame := await self.inbound.get()) is not None:
                yield frame

        async def send(self, frame):
            request = json.loads(frame)
            sent.append(request)
            data = INFO if request["action"] == "get_group_info" else MEMBERS
            self.inbound.put_nowait(
                json.dumps({"status": "ok", "retcode": 0, "data": data, "echo": request["echo"]})
            )

        async def close(self):
            self.inbound.put_nowait(None)

    ws = WS()

    async def connect(url, **_):
        return ws

    stop = asyncio.Event()
    gateway = OneBotGateway("ws://127.0.0.1:3001", "tok", allow_actions=True, connect=connect)
    task = asyncio.create_task(gateway.run(stop))
    assert await gateway.wait_connected(1)
    tool = GroupInfoTool(GroupInfoCache(OneBotApi(gateway)), "777000111")
    result = await tool.call({"query": "member", "name": "小红", "group_id": "1"})
    assert result["status"] == "refused"
    result = await tool.call({"query": "member", "name": "小红"})
    stop.set()
    await asyncio.wait_for(task, 2)
    assert result["status"] == "ok"
    assert [(r["action"], r["params"]["group_id"]) for r in sent] == [
        ("get_group_info", 777000111),
        ("get_group_member_list", 777000111),
    ]
    assert {r["action"] for r in sent} <= READ_ONLY_ACTIONS
    assert_no_ids(result)
