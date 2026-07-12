"""P0 验收：会话记忆（Redis 存取、跨请求带历史）与真流式（增量透传、工具调用块）。"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

from app.agent.orchestrator import TravelOrchestrator
from app.core.memory.session_store import RedisSessionStore
from app.domain.schemas import ChatMessage, MessageRole, StreamChunkType


class FakeRedis:
    def __init__(self) -> None:
        self.data: dict[str, str] = {}

    async def get(self, key: str) -> str | None:
        return self.data.get(key)

    async def set(self, key: str, value: str, ex: int | None = None) -> None:
        self.data[key] = value

    async def delete(self, key: str) -> None:
        self.data.pop(key, None)


def _completion(content: str) -> Any:
    return SimpleNamespace(
        id="resp-1",
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content, tool_calls=None),
                finish_reason="stop",
            )
        ],
        usage=None,
    )


def _stream_chunk(
    content: str | None = None,
    tool_calls: list[Any] | None = None,
    finish_reason: str | None = None,
) -> Any:
    delta = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta, finish_reason=finish_reason)])


def _tool_call_delta(index: int, tc_id: str | None, name: str | None, args: str | None) -> Any:
    return SimpleNamespace(
        index=index,
        id=tc_id,
        function=SimpleNamespace(name=name, arguments=args),
    )


class FakeLLM:
    """按预设轮次返回；记录每次收到的 messages 以便断言。"""

    model = "fake-model"

    def __init__(
        self,
        completions: list[Any] | None = None,
        stream_rounds: list[list[Any]] | None = None,
    ) -> None:
        self._completions = list(completions or [])
        self._stream_rounds = list(stream_rounds or [])
        self.seen_messages: list[list[dict[str, Any]]] = []

    async def chat_completion(self, messages: list[dict[str, Any]], **kwargs: Any) -> Any:
        self.seen_messages.append(messages)
        return self._completions.pop(0)

    async def chat_completion_stream(self, messages: list[dict[str, Any]], **kwargs: Any):
        self.seen_messages.append(messages)
        for chunk in self._stream_rounds.pop(0):
            yield chunk


def _user(text: str) -> ChatMessage:
    return ChatMessage(role=MessageRole.USER, content=text)


async def test_run_completion_persists_and_recalls_session() -> None:
    fake_redis = FakeRedis()
    store = RedisSessionStore(fake_redis, ttl_seconds=60)
    llm = FakeLLM(completions=[_completion("北京到上海高铁约5.5小时。"), _completion("经济舱。")])
    orch = TravelOrchestrator(llm=llm, session_store=store)  # type: ignore[arg-type]

    await orch.run_completion([_user("北京到上海多久?")], session_id="s1")

    stored = json.loads(fake_redis.data["session:s1:messages"])
    assert [m["role"] for m in stored] == ["user", "assistant"]
    assert stored[1]["content"] == "北京到上海高铁约5.5小时。"

    # 第二轮只传增量消息，历史应自动带上
    await orch.run_completion([_user("那我该坐什么舱位?")], session_id="s1")
    second_call = llm.seen_messages[1]
    contents = [m["content"] for m in second_call]
    assert any("北京到上海多久" in c for c in contents), "历史用户消息未带上"
    assert any("高铁约5.5小时" in c for c in contents), "历史助手回复未带上"
    assert len(json.loads(fake_redis.data["session:s1:messages"])) == 4


async def test_run_completion_without_session_is_stateless() -> None:
    fake_redis = FakeRedis()
    store = RedisSessionStore(fake_redis, ttl_seconds=60)
    llm = FakeLLM(completions=[_completion("好的。")])
    orch = TravelOrchestrator(llm=llm, session_store=store)  # type: ignore[arg-type]

    await orch.run_completion([_user("你好")])
    assert fake_redis.data == {}


async def test_stream_completion_passes_through_deltas() -> None:
    llm = FakeLLM(
        stream_rounds=[
            [
                _stream_chunk(content="你"),
                _stream_chunk(content="好"),
                _stream_chunk(finish_reason="stop"),
            ]
        ]
    )
    orch = TravelOrchestrator(llm=llm)  # type: ignore[arg-type]

    chunks = [c async for c in orch.stream_completion([_user("hi")])]
    assert [c.type for c in chunks] == [
        StreamChunkType.CONTENT,
        StreamChunkType.CONTENT,
        StreamChunkType.DONE,
    ]
    assert "".join(c.delta or "" for c in chunks[:2]) == "你好"
    assert chunks[-1].finish_reason == "stop"


async def test_stream_completion_tool_loop_and_persist() -> None:
    fake_redis = FakeRedis()
    store = RedisSessionStore(fake_redis, ttl_seconds=60)
    # 第一轮：工具调用参数分片下发；第二轮：最终回答
    llm = FakeLLM(
        stream_rounds=[
            [
                _stream_chunk(
                    tool_calls=[_tool_call_delta(0, "call-1", "unknown_tool", '{"a":')]
                ),
                _stream_chunk(tool_calls=[_tool_call_delta(0, None, None, "1}")]),
                _stream_chunk(finish_reason="tool_calls"),
            ],
            [
                _stream_chunk(content="完成"),
                _stream_chunk(finish_reason="stop"),
            ],
        ]
    )
    orch = TravelOrchestrator(llm=llm, session_store=store)  # type: ignore[arg-type]

    chunks = [c async for c in orch.stream_completion([_user("测试")], session_id="s2")]
    types = [c.type for c in chunks]
    assert types == [
        StreamChunkType.TOOL_CALL,
        StreamChunkType.CONTENT,
        StreamChunkType.DONE,
    ]
    tool_chunk = chunks[0]
    assert tool_chunk.tool_name == "unknown_tool"
    assert tool_chunk.tool_args == {"a": 1}, "分片的工具参数应拼接完整"

    # 第二轮 LLM 请求应包含工具结果消息
    second_round = llm.seen_messages[1]
    assert any(m.get("role") == "tool" for m in second_round)

    stored = json.loads(fake_redis.data["session:s2:messages"])
    assert stored[-1] == {"role": "assistant", "content": "完成"}
