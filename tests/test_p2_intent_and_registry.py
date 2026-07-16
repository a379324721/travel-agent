"""P2 验收：工具注册中心接管调度、意图路由（闲聊无工具/政策强制检索）、mock 搜索订票工具。"""

from __future__ import annotations

import json
from datetime import date, timedelta
from types import SimpleNamespace
from typing import Any

from app.agent.orchestrator import TravelOrchestrator
from app.agent.tools import build_default_registry
from app.core.intent.recognizer import IntentRecognizer, TravelIntent
from app.domain.schemas import ChatMessage, MessageRole
from app.domain.travel.policy import default_corporate_policy

EXPECTED_TOOLS = {
    "plan_travel_itinerary",
    "check_travel_policy",
    "search_travel_policy_docs",
    "search_flights",
    "search_hotels",
    "search_trains",
    "create_booking",
    "submit_travel_approval",
    "check_approval",
    "submit_expense_report",
}


def _tool_call_completion(name: str, arguments: str) -> Any:
    return SimpleNamespace(
        id="resp-tool",
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=None,
                    tool_calls=[
                        SimpleNamespace(
                            id="call-1",
                            function=SimpleNamespace(name=name, arguments=arguments),
                        )
                    ],
                ),
                finish_reason="tool_calls",
            )
        ],
        usage=None,
    )


def _final_completion(content: str) -> Any:
    return SimpleNamespace(
        id="resp-final",
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content, tool_calls=None),
                finish_reason="stop",
            )
        ],
        usage=None,
    )


class FakeLLM:
    model = "fake-model"

    def __init__(self, completions: list[Any]) -> None:
        self._completions = list(completions)
        self.calls: list[dict[str, Any]] = []

    async def chat_completion(self, messages: list[dict[str, Any]], **kwargs: Any) -> Any:
        self.calls.append({"messages": messages, **kwargs})
        return self._completions.pop(0)


def _user(text: str) -> ChatMessage:
    return ChatMessage(role=MessageRole.USER, content=text)


def test_default_registry_contains_all_tools() -> None:
    registry = build_default_registry(default_corporate_policy())
    assert {t.name for t in registry.list_tools()} == EXPECTED_TOOLS
    # 每个工具都有可用的 JSON Schema
    for t in registry.list_tools():
        assert t.json_schema.get("type") == "object", t.name


async def test_general_intent_keeps_tools_available() -> None:
    """GENERAL=规则未识别≠闲聊：工具须保持可用，否则未覆盖表达会让模型编造结果。"""
    llm = FakeLLM([_final_completion("你好，我是差旅助手。")])
    orch = TravelOrchestrator(llm=llm)  # type: ignore[arg-type]
    await orch.run_completion([_user("你好")])
    assert llm.calls[0].get("tools") is not None
    assert llm.calls[0]["tool_choice"] == "auto"


async def test_reimbursement_intent_recognized() -> None:
    from app.core.intent.recognizer import IntentRecognizer, TravelIntent

    r = IntentRecognizer()
    out = await r.recognize("行程结束了，一键提交报销")
    assert out.intent is TravelIntent.APPLICATION


async def test_policy_intent_forces_rag_tool_first_round(monkeypatch) -> None:
    from app.config import settings

    monkeypatch.setattr(settings, "llm_force_tool_choice", True)
    llm = FakeLLM(
        [
            _tool_call_completion("search_travel_policy_docs", '{"query": "差标"}'),
            _final_completion("以公司制度为准。"),
        ]
    )
    orch = TravelOrchestrator(llm=llm)  # type: ignore[arg-type]
    await orch.run_completion([_user("出差住宿差标是多少?")])

    first = llm.calls[0]
    assert first["tools"] is not None
    assert first["tool_choice"] == {
        "type": "function",
        "function": {"name": "search_travel_policy_docs"},
    }
    # 第二轮恢复 auto，避免死循环强制调用
    assert llm.calls[1]["tool_choice"] == "auto"


async def test_flight_search_tool_executes_mock_source() -> None:
    llm = FakeLLM(
        [
            _tool_call_completion(
                "search_flights",
                json.dumps(
                    {
                        "origin": "北京",
                        "destination": "上海",
                        "depart_date": (date.today() + timedelta(days=14)).isoformat(),
                    }
                ),
            ),
            _final_completion("找到 CA1501。"),
        ]
    )
    orch = TravelOrchestrator(llm=llm)  # type: ignore[arg-type]
    await orch.run_completion([_user("帮我查下周六北京到上海的机票")])

    tool_msgs = [m for m in llm.calls[1]["messages"] if m.get("role") == "tool"]
    assert tool_msgs
    payload = json.loads(tool_msgs[0]["content"])
    assert payload["mode"] == "flight"
    assert payload["results"][0]["flight_no"] == "CA1501"


async def test_search_tool_rejects_past_date() -> None:
    llm = FakeLLM(
        [
            _tool_call_completion(
                "search_flights",
                '{"origin": "北京", "destination": "上海", "depart_date": "2020-01-01"}',
            ),
            _final_completion("好的。"),
        ]
    )
    orch = TravelOrchestrator(llm=llm)  # type: ignore[arg-type]
    await orch.run_completion([_user("查机票")])

    tool_msgs = [m for m in llm.calls[1]["messages"] if m.get("role") == "tool"]
    payload = json.loads(tool_msgs[0]["content"])
    assert "已是过去" in payload["error"]
    assert date.today().isoformat() in payload["error"]


async def test_create_booking_tool_returns_confirmation() -> None:
    llm = FakeLLM(
        [
            _tool_call_completion(
                "create_booking",
                json.dumps(
                    {
                        "employee_id": "E001",
                        "booking_type": "flight",
                        "inventory_id": "CA1501",
                        "traveler_name": "王宁",
                        "contact_phone": "13800000000",
                        "amount_cny": 1280,
                    }
                ),
            ),
            _final_completion("已出票。"),
        ]
    )
    orch = TravelOrchestrator(llm=llm)  # type: ignore[arg-type]
    await orch.run_completion([_user("帮我订这张机票")])

    tool_msgs = [m for m in llm.calls[1]["messages"] if m.get("role") == "tool"]
    payload = json.loads(tool_msgs[0]["content"])
    assert payload["ok"] is True
    assert payload["booking"]["status"] == "confirmed"
    assert payload["booking"]["confirmation_code"].startswith("FL-")


async def test_invalid_tool_args_return_error_instead_of_crash() -> None:
    llm = FakeLLM(
        [
            _tool_call_completion("search_flights", '{"origin": "北京"}'),
            _final_completion("请补充目的地和日期。"),
        ]
    )
    orch = TravelOrchestrator(llm=llm)  # type: ignore[arg-type]
    result = await orch.run_completion([_user("查机票")])
    assert result["choices"][0]["message"]["content"]
    tool_msgs = [m for m in llm.calls[1]["messages"] if m.get("role") == "tool"]
    assert "error" in json.loads(tool_msgs[0]["content"])


# ---------------------------------------------------------------------------
# 意图识别慢车道（LLM 分类）


async def test_slow_lane_triggers_when_rules_miss() -> None:
    """规则完全不命中时走 LLM 分类，采纳其结果。"""
    llm = FakeLLM(
        [
            _final_completion(
                json.dumps(
                    {
                        "intent_slug": "trip_planning",
                        "confidence": 0.9,
                        "rationale": "用户在咨询差旅安排",
                    },
                    ensure_ascii=False,
                )
            )
        ]
    )
    recognizer = IntentRecognizer.with_llm(llm)
    result = await recognizer.recognize("预算有点紧张，帮我想想怎么弄")
    assert result.intent is TravelIntent.TRIP_PLANNING
    assert result.metadata["merged"] == "slow_only"
    assert len(llm.calls) == 1
    # 提示词须携带每个 slug 的中文含义，而非裸列表
    prompt = llm.calls[0]["messages"][1]["content"]
    assert "application: 提交出差申请" in prompt
    assert "rag: 需要引用公司内部" in prompt


async def test_fast_lane_skips_llm_when_rules_confident() -> None:
    """规则命中（置信度 0.82 达阈值）时不触发 LLM 调用。"""
    llm = FakeLLM([])  # 一旦被调用会 IndexError
    recognizer = IntentRecognizer.with_llm(llm)
    result = await recognizer.recognize("帮我查明天去上海的机票")
    assert result.intent is TravelIntent.SEARCH_FLIGHT
    assert result.metadata["merged"] == "fast_only"
    assert llm.calls == []


async def test_slow_lane_parse_error_falls_back_to_general() -> None:
    """LLM 输出非 JSON 时降级为 general 低置信度，不抛异常。"""
    llm = FakeLLM([_final_completion("我觉得这是订票意图")])
    recognizer = IntentRecognizer.with_llm(llm)
    result = await recognizer.recognize("嗯嗯好的谢谢")
    assert result.intent is TravelIntent.GENERAL
    assert result.confidence < 0.5
