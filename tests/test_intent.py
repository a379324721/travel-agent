"""Unit tests for intent recognizer."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from app.core.intent.llm_classifier import LLMIntentClassifier
from app.core.intent.recognizer import IntentRecognizer, StructuredLLMBridge, TravelIntent


@pytest.mark.asyncio
async def test_recognize_flight_intent() -> None:
    r = IntentRecognizer()
    out = await r.recognize("帮我查一下明天北京到上海的机票")
    assert out.intent is TravelIntent.SEARCH_FLIGHT
    assert out.confidence >= 0.4


@pytest.mark.asyncio
async def test_recognize_policy_intent() -> None:
    r = IntentRecognizer()
    out = await r.recognize("公司的差标报销政策上限是多少")
    assert out.intent is TravelIntent.POLICY


@pytest.mark.asyncio
async def test_recognize_general_fallback() -> None:
    r = IntentRecognizer()
    out = await r.recognize("你好")
    assert out.intent is TravelIntent.GENERAL


_FINAL_JSON = json.dumps(
    {
        "intent_slug": "policy",
        "confidence": 0.9,
        "standalone_query": "飞机舱位的差旅标准是多少",
        "rationale": "续问差标",
    },
    ensure_ascii=False,
)


class _ToolLoopFakeClient:
    """OpenAI 兼容结构：可选首轮返回 fetch_history 调用，随后返回 JSON。"""

    def __init__(self, *, call_tool_first: bool) -> None:
        self.calls: list[dict[str, Any]] = []
        self._call_tool_first = call_tool_first

    async def chat_completion(self, messages: list[dict[str, Any]], **kwargs: Any) -> Any:
        self.calls.append({"messages": list(messages), **kwargs})
        if self._call_tool_first and len(self.calls) == 1:
            msg = SimpleNamespace(
                content=None,
                tool_calls=[
                    SimpleNamespace(
                        id="t1",
                        function=SimpleNamespace(
                            name="fetch_history", arguments='{"count": 5}'
                        ),
                    )
                ],
            )
        else:
            msg = SimpleNamespace(content=_FINAL_JSON, tool_calls=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


@pytest.mark.asyncio
async def test_slow_lane_backtracks_history_via_tool() -> None:
    client = _ToolLoopFakeClient(call_tool_first=True)
    clf = LLMIntentClassifier(StructuredLLMBridge(client))
    fetched: list[int] = []

    def fetch(n: int) -> str:
        fetched.append(n)
        return "user: 出差住宿差标是多少\nassistant: 上限800元/晚"

    out = await clf.classify(
        "那飞机呢?", recent="assistant: 以公司制度为准。", fetch_history=fetch
    )
    assert fetched == [5], "回溯条数应取自模型的工具参数"
    assert out.intent_slug == "policy"
    assert out.standalone_query == "飞机舱位的差旅标准是多少"
    second = client.calls[1]
    assert second["tool_choice"] == "none", "工具往返后应强制出结果"
    assert any(
        m.get("role") == "tool" and "住宿差标" in m.get("content", "")
        for m in second["messages"]
    )
    last_msg = second["messages"][-1]
    assert last_msg["role"] == "system" and "只输出" in last_msg["content"], (
        "末轮前应重申 JSON 格式约束"
    )


@pytest.mark.asyncio
async def test_slow_lane_skips_tool_when_recent_suffices() -> None:
    client = _ToolLoopFakeClient(call_tool_first=False)
    clf = LLMIntentClassifier(StructuredLLMBridge(client))
    fetched: list[int] = []

    out = await clf.classify(
        "那飞机呢?",
        recent="user: 住宿差标\nassistant: 800元/晚",
        fetch_history=lambda n: fetched.append(n) or "",
    )
    assert len(client.calls) == 1, "最近对话足够时不应产生第二次调用"
    assert fetched == []
    assert out.intent_slug == "policy"


@pytest.mark.asyncio
async def test_slow_lane_falls_back_without_tool_support() -> None:
    class _StructuredOnly:
        def __init__(self) -> None:
            self.called = False

        async def complete_structured(
            self, *, system_prompt: str, user_content: str, response_format: str
        ) -> str:
            self.called = True
            return _FINAL_JSON

    client = _StructuredOnly()
    clf = LLMIntentClassifier(client)
    out = await clf.classify("那飞机呢?", recent="", fetch_history=lambda n: "")
    assert client.called, "客户端不支持工具时应退回单次结构化调用"
    assert out.intent_slug == "policy"


@pytest.mark.asyncio
async def test_anaphoric_query_forces_slow_lane_review() -> None:
    """「那飞机呢?」快车道按“飞机”误判 search_flight(0.82)，指代形态应强制慢车道复核。"""
    client = _ToolLoopFakeClient(call_tool_first=False)
    r = IntentRecognizer(llm_classifier=LLMIntentClassifier(StructuredLLMBridge(client)))
    out = await r.recognize("那飞机呢?", recent="user: 出差住宿差标是多少")
    assert out.intent is TravelIntent.POLICY
    assert out.metadata["merged"] == "prefer_slow"


@pytest.mark.asyncio
async def test_anaphoric_query_without_slow_lane_keeps_fast() -> None:
    r = IntentRecognizer()
    out = await r.recognize("那飞机呢?")
    assert out.intent is TravelIntent.SEARCH_FLIGHT, "慢车道未配置时保持快车道结果"


@pytest.mark.asyncio
async def test_recognizer_propagates_standalone_query() -> None:
    client = _ToolLoopFakeClient(call_tool_first=False)
    r = IntentRecognizer(llm_classifier=LLMIntentClassifier(StructuredLLMBridge(client)))
    out = await r.recognize("那这个呢?", recent="user: 住宿差标", fetch_history=lambda n: "")
    assert out.intent is TravelIntent.POLICY
    assert out.standalone_query == "飞机舱位的差旅标准是多少"
