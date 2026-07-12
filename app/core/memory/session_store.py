"""Redis 会话历史存储：按 session_id 持久化多轮消息。"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from app.domain.schemas import ChatMessage


class RedisSessionStore:
    """以整段 JSON 保存每个会话的消息列表，带 TTL。

    存储的是"用户可见"的对话线程（user / assistant / 摘要 system），
    工具调用的中间消息不落库。
    """

    def __init__(self, redis: Any, *, ttl_seconds: int = 604800) -> None:
        self._redis = redis
        self._ttl = ttl_seconds

    @staticmethod
    def _key(session_id: str) -> str:
        return f"session:{session_id}:messages"

    async def load(self, session_id: str) -> list[ChatMessage]:
        raw: str | None = await self._redis.get(self._key(session_id))
        if not raw:
            return []
        return [ChatMessage.model_validate(m) for m in json.loads(raw)]

    async def replace(self, session_id: str, messages: list[ChatMessage]) -> None:
        payload = json.dumps(
            [m.model_dump(mode="json", exclude_none=True) for m in messages],
            ensure_ascii=False,
        )
        await self._redis.set(self._key(session_id), payload, ex=self._ttl)

    # ---- 按用户的会话索引（供前端历史列表） ----

    _INDEX_MAX = 50

    @staticmethod
    def _index_key(user_id: str) -> str:
        return f"user:{user_id}:sessions"

    async def touch_session(self, user_id: str, session_id: str, title: str) -> None:
        """把会话置顶到该用户的索引；已存在则保留原标题，只更新时间。"""
        raw = await self._redis.get(self._index_key(user_id))
        items: list[dict] = json.loads(raw) if raw else []
        existing = next((it for it in items if it.get("session_id") == session_id), None)
        if existing is not None:
            title = existing.get("title") or title
            items = [it for it in items if it.get("session_id") != session_id]
        items.insert(
            0,
            {
                "session_id": session_id,
                "title": title[:40],
                "updated_at": datetime.now(UTC).isoformat(),
            },
        )
        payload = json.dumps(items[: self._INDEX_MAX], ensure_ascii=False)
        await self._redis.set(self._index_key(user_id), payload, ex=self._ttl)

    async def list_sessions(self, user_id: str) -> list[dict]:
        raw = await self._redis.get(self._index_key(user_id))
        return json.loads(raw) if raw else []
