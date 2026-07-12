"""会话历史功能：用户会话索引、列表与消息接口。"""

from __future__ import annotations

import httpx
from httpx import ASGITransport

from app.agent.orchestrator import TravelOrchestrator
from app.core.memory.session_store import RedisSessionStore
from app.main import create_app
from tests.test_p0_session_and_stream import FakeLLM, FakeRedis, _completion, _user


async def test_persist_updates_user_session_index() -> None:
    fake_redis = FakeRedis()
    store = RedisSessionStore(fake_redis, ttl_seconds=60)
    llm = FakeLLM(completions=[_completion("好的。")] * 4)
    orch = TravelOrchestrator(llm=llm, session_store=store)  # type: ignore[arg-type]

    await orch.run_completion([_user("帮我规划北京出差")], session_id="s1", user_id="E001")
    await orch.run_completion([_user("查差标")], session_id="s2", user_id="E001")

    sessions = await store.list_sessions("E001")
    assert [s["session_id"] for s in sessions] == ["s2", "s1"], "最新会话应置顶"
    assert sessions[1]["title"] == "帮我规划北京出差"

    # 旧会话再次活跃：置顶但保留原标题
    await orch.run_completion([_user("继续刚才的行程")], session_id="s1", user_id="E001")
    sessions = await store.list_sessions("E001")
    assert sessions[0]["session_id"] == "s1"
    assert sessions[0]["title"] == "帮我规划北京出差"

    # 无 user_id 不入索引
    await orch.run_completion([_user("匿名提问")], session_id="s3")
    assert all(s["session_id"] != "s3" for s in await store.list_sessions("E001"))


async def test_sessions_api_endpoints() -> None:
    app = create_app()
    fake_redis = FakeRedis()
    store = RedisSessionStore(fake_redis, ttl_seconds=60)
    llm = FakeLLM(completions=[_completion("北京到上海高铁约5.5小时。")])
    app.state.session_store = store
    app.state.orchestrator = TravelOrchestrator(llm=llm, session_store=store)  # type: ignore[arg-type]

    async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        await c.post(
            "/api/v1/chat",
            json={
                "messages": [{"role": "user", "content": "北京到上海多久?"}],
                "session_id": "s1",
                "user_id": "E001",
            },
        )
        r = await c.get("/api/v1/sessions", params={"user_id": "E001"})
        sessions = r.json()["sessions"]
        assert r.status_code == 200
        assert sessions[0]["session_id"] == "s1"
        assert sessions[0]["title"].startswith("北京到上海")

        r2 = await c.get("/api/v1/sessions/s1/messages")
        msgs = r2.json()["messages"]
        assert [m["role"] for m in msgs] == ["user", "assistant"]
        assert "高铁" in msgs[1]["content"]

        r3 = await c.get("/api/v1/sessions/nonexistent/messages")
        assert r3.status_code == 200 and r3.json()["messages"] == []


async def test_store_delete_removes_messages_and_index() -> None:
    fake_redis = FakeRedis()
    store = RedisSessionStore(fake_redis, ttl_seconds=60)
    llm = FakeLLM(completions=[_completion("好的。")] * 2)
    orch = TravelOrchestrator(llm=llm, session_store=store)  # type: ignore[arg-type]
    await orch.run_completion([_user("会话一")], session_id="s1", user_id="E001")
    await orch.run_completion([_user("会话二")], session_id="s2", user_id="E001")

    await store.delete("s1", user_id="E001")
    assert await store.load("s1") == []
    assert [s["session_id"] for s in await store.list_sessions("E001")] == ["s2"]


async def test_delete_session_api() -> None:
    app = create_app()
    fake_redis = FakeRedis()
    store = RedisSessionStore(fake_redis, ttl_seconds=60)
    llm = FakeLLM(completions=[_completion("好的。")])
    app.state.session_store = store
    app.state.orchestrator = TravelOrchestrator(llm=llm, session_store=store)  # type: ignore[arg-type]

    async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        await c.post(
            "/api/v1/chat",
            json={
                "messages": [{"role": "user", "content": "待删除会话"}],
                "session_id": "s1",
                "user_id": "E001",
            },
        )
        r = await c.delete("/api/v1/sessions/s1", params={"user_id": "E001"})
        assert r.status_code == 200 and r.json()["deleted"] is True
        assert (await c.get("/api/v1/sessions/s1/messages")).json()["messages"] == []
        assert (await c.get("/api/v1/sessions", params={"user_id": "E001"})).json()[
            "sessions"
        ] == []


async def test_sessions_api_without_store() -> None:
    app = create_app()
    app.state.session_store = None
    async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.get("/api/v1/sessions", params={"user_id": "E001"})
        assert r.status_code == 200 and r.json()["sessions"] == []
