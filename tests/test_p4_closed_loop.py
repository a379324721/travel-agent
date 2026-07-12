"""P4 验收：mock 身份注入、审批秒批、订票落库、一键报销闭环。"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

from app.agent.orchestrator import TravelOrchestrator
from app.domain.schemas import ChatMessage, MessageRole
from app.services.approval import ApprovalService
from app.services.booking_store import InMemoryBookingStore
from app.services.employee_directory import EmployeeDirectory


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


def _tool_payload(call: dict[str, Any]) -> dict[str, Any]:
    tool_msgs = [m for m in call["messages"] if m.get("role") == "tool"]
    return json.loads(tool_msgs[0]["content"])


def test_employee_directory_mock() -> None:
    d = EmployeeDirectory()
    assert d.get("E001").name == "王宁"
    assert d.get("E001").grade.value == "manager"
    assert d.get("不存在") is None
    assert d.get(None) is None


def test_approval_mock_auto_approves() -> None:
    svc = ApprovalService()
    ticket = svc.create("E001", "北京→上海 客户拜访", 6800)
    assert ticket["status"] == "approved"
    assert ticket["approval_id"].startswith("APV-")
    assert svc.query(ticket["approval_id"])["amount_cny"] == 6800
    assert svc.query("APV-NOPE") is None


async def test_in_memory_store_reimbursement_flow() -> None:
    store = InMemoryBookingStore()
    await store.add_booking(
        booking_id="b1", employee_id="E001", booking_type="flight",
        inventory_id="CA1501", confirmation_code="FL-X", amount_cny=1280,
    )
    await store.add_booking(
        booking_id="b2", employee_id="E001", booking_type="hotel",
        inventory_id="H1", confirmation_code="HO-Y", amount_cny=560,
    )
    await store.add_booking(
        booking_id="b3", employee_id="E002", booking_type="train",
        inventory_id="G103", confirmation_code="TR-Z", amount_cny=553,
    )
    assert len(await store.list_unreimbursed("E001")) == 2

    result = await store.create_reimbursement("E001")
    assert result["ok"] and result["total_cny"] == 1840.0 and result["booking_count"] == 2
    assert await store.list_unreimbursed("E001") == []
    # 不影响其他员工；重复提交无可报销订单
    assert len(await store.list_unreimbursed("E002")) == 1
    assert (await store.create_reimbursement("E001"))["ok"] is False


async def test_identity_injected_when_user_id_given() -> None:
    llm = FakeLLM([_final_completion("好的。"), _final_completion("好的。")])
    orch = TravelOrchestrator(llm=llm)  # type: ignore[arg-type]

    await orch.run_completion([_user("你好")], user_id="E001")
    system_msgs = [m for m in llm.calls[0]["messages"] if m["role"] == "system"]
    assert any("王宁" in m["content"] and "manager" in m["content"] for m in system_msgs)

    await orch.run_completion([_user("你好")])
    system_msgs = [m for m in llm.calls[1]["messages"] if m["role"] == "system"]
    assert not any("王宁" in m["content"] for m in system_msgs)


async def test_full_closed_loop_approval_booking_expense() -> None:
    """审批 → 订票（落库）→ 一键报销，同一 orchestrator 内跨请求完成。"""
    store = InMemoryBookingStore()
    llm = FakeLLM(
        [
            # 请求1：超标审批
            _tool_call_completion(
                "submit_travel_approval",
                '{"employee_id": "E001", "reason": "北京→上海 客户拜访", '
                '"estimated_total_cny": 6800}',
            ),
            _final_completion("审批已通过。"),
            # 请求2：订票
            _tool_call_completion(
                "create_booking",
                '{"employee_id": "E001", "booking_type": "flight", "inventory_id": "CA1501", '
                '"traveler_name": "王宁", "contact_phone": "13800000001", "amount_cny": 1280}',
            ),
            _final_completion("已出票。"),
            # 请求3：一键报销
            _tool_call_completion("submit_expense_report", '{"employee_id": "E001"}'),
            _final_completion("报销单已提交。"),
        ]
    )
    orch = TravelOrchestrator(llm=llm, booking_store=store)  # type: ignore[arg-type]

    await orch.run_completion([_user("这趟出差预估6800元，帮我提交审批")], user_id="E001")
    approval = _tool_payload(llm.calls[1])
    assert approval["status"] == "approved"

    await orch.run_completion([_user("订CA1501这张机票")], user_id="E001")
    booking = _tool_payload(llm.calls[3])
    assert booking["ok"] is True
    assert booking["record"]["employee_id"] == "E001"
    assert booking["record"]["amount_cny"] == 1280.0

    await orch.run_completion([_user("行程结束，一键提交报销")], user_id="E001")
    expense = _tool_payload(llm.calls[5])
    assert expense["ok"] is True
    assert expense["total_cny"] == 1280.0
    assert expense["reimbursement_id"].startswith("RMB-")
    assert expense["bookings"][0]["confirmation_code"].startswith("FL-")
