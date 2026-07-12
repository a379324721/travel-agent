"""Redis 会话历史存储：按 session_id 持久化多轮消息。"""

from __future__ import annotations

import json
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
