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
        self.stream_kwargs: list[dict[str, Any]] = []

    async def chat_completion(self, messages: list[dict[str, Any]], **kwargs: Any) -> Any:
        self.seen_messages.append(messages)
        return self._completions.pop(0)

    async def chat_completion_stream(self, messages: list[dict[str, Any]], **kwargs: Any):
        self.seen_messages.append(messages)
        self.stream_kwargs.append(kwargs)
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


async def test_run_completion_returns_intermediate_round_text() -> None:
    tool_round = SimpleNamespace(
        id="resp-t",
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content="我先查一下。",
                    tool_calls=[
                        SimpleNamespace(
                            id="call-1",
                            function=SimpleNamespace(name="unknown_tool", arguments="{}"),
                        )
                    ],
                ),
                finish_reason="tool_calls",
            )
        ],
        usage=None,
    )
    llm = FakeLLM(completions=[tool_round, _completion("查到了。")])
    orch = TravelOrchestrator(llm=llm)  # type: ignore[arg-type]

    resp = await orch.run_completion([_user("测试")])
    assert resp["choices"][0]["message"]["content"] == "我先查一下。\n\n查到了。"


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


async def test_stream_completion_persists_full_thread() -> None:
    fake_redis = FakeRedis()
    store = RedisSessionStore(fake_redis, ttl_seconds=60)
    # 第一轮：先说话再调工具；第二轮：最终回答。工具轮完整入历史。
    llm = FakeLLM(
        stream_rounds=[
            [
                _stream_chunk(content="我先查一下。"),
                _stream_chunk(
                    tool_calls=[_tool_call_delta(0, "call-1", "unknown_tool", "{}")]
                ),
                _stream_chunk(finish_reason="tool_calls"),
            ],
            [
                _stream_chunk(content="查到了。"),
                _stream_chunk(finish_reason="stop"),
            ],
            [
                _stream_chunk(content="好的。"),
                _stream_chunk(finish_reason="stop"),
            ],
        ]
    )
    orch = TravelOrchestrator(llm=llm, session_store=store)  # type: ignore[arg-type]

    [c async for c in orch.stream_completion([_user("测试")], session_id="s3")]

    stored = json.loads(fake_redis.data["session:s3:messages"])
    assert [m["role"] for m in stored] == ["user", "assistant", "tool", "assistant"]
    assert stored[1]["content"] == "我先查一下。"
    assert stored[1]["tool_calls"][0]["function"]["name"] == "unknown_tool"
    assert stored[2]["tool_call_id"] == "call-1"
    assert stored[-1]["content"] == "查到了。"

    # 下一轮请求应带上历史中的工具调用与结果
    [c async for c in orch.stream_completion([_user("继续")], session_id="s3")]
    second_call = llm.seen_messages[2]
    assert any(m.get("role") == "tool" for m in second_call), "历史工具结果未带上"
    assert any(m.get("tool_calls") for m in second_call), "历史 tool_calls 未带上"


async def test_stream_last_round_forces_text_wrap_up(monkeypatch) -> None:
    from app.config import settings

    monkeypatch.setattr(settings, "max_react_iterations", 2)
    llm = FakeLLM(
        stream_rounds=[
            [
                _stream_chunk(
                    tool_calls=[_tool_call_delta(0, "call-1", "unknown_tool", "{}")]
                ),
                _stream_chunk(finish_reason="tool_calls"),
            ],
            [
                _stream_chunk(content="先说结论。"),
                _stream_chunk(finish_reason="stop"),
            ],
        ]
    )
    orch = TravelOrchestrator(llm=llm)  # type: ignore[arg-type]

    chunks = [c async for c in orch.stream_completion([_user("测试")])]
    assert chunks[-1].type == StreamChunkType.DONE, "末轮强制收尾后应正常结束而非 ERROR"
    assert llm.stream_kwargs[-1]["tool_choice"] == "none"
    assert any(
        m.get("role") == "system" and "轮次上限" in m.get("content", "")
        for m in llm.seen_messages[-1]
    )


def test_short_term_memory_drops_orphan_tool_turns() -> None:
    from app.core.memory.short_term import ChatTurn, ShortTermMemory

    memory = ShortTermMemory(max_tokens=8000, max_turns=2)
    memory.extend(
        [
            ChatTurn(role="user", content="订票"),
            ChatTurn(role="assistant", content="", tool_calls=[{"id": "c1"}]),
            ChatTurn(role="tool", content="结果", tool_call_id="c1"),
            ChatTurn(role="assistant", content="已订"),
        ]
    )
    # max_turns=2 会裁掉发起 tool_calls 的 assistant，队首孤儿 tool 应一并丢弃
    assert [t.role for t in memory.snapshot()] == ["assistant"]
