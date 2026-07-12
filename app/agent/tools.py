"""主线业务工具集：全部注册进 ToolRegistry，由 orchestrator 统一取定义与调度。"""

from __future__ import annotations

import uuid
from datetime import date
from decimal import Decimal
from typing import Any

from app.core.rag.service import PolicyRAG
from app.core.tools.booking import BookingRequest, create_booking
from app.core.tools.registry import ToolRegistry
from app.core.tools.travel_search import (
    FlightSearchRequest,
    HotelSearchRequest,
    TrainSearchRequest,
    search_flights,
    search_hotels,
    search_trains,
)
from app.domain.travel.itinerary import build_draft_itinerary, summarize_itinerary_text
from app.domain.travel.models import (
    EmployeeGrade,
    TravelClass,
    TravelPolicy,
    TravelRequest,
    TripPurpose,
)
from app.domain.travel.policy import apply_policy_to_itinerary
from app.services.approval import ApprovalService
from app.services.booking_store import InMemoryBookingStore


def _parse_date(s: str) -> date:
    y, mo, d = (int(x) for x in s.split("-", 2))
    return date(y, mo, d)


def _plan_itinerary_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "employee_id": {"type": "string"},
            "grade": {"type": "string", "enum": [g.value for g in EmployeeGrade]},
            "origin_city": {"type": "string"},
            "destination_city": {"type": "string"},
            "departure_date": {"type": "string", "description": "YYYY-MM-DD"},
            "return_date": {"type": "string", "description": "YYYY-MM-DD，可选"},
            "purpose": {"type": "string", "enum": [p.value for p in TripPurpose]},
            "preferred_class": {"type": "string", "enum": [c.value for c in TravelClass]},
        },
        "required": [
            "employee_id",
            "grade",
            "origin_city",
            "destination_city",
            "departure_date",
            "purpose",
        ],
    }


def _check_policy_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "employee_id": {"type": "string"},
            "grade": {"type": "string", "enum": [g.value for g in EmployeeGrade]},
            "origin_city": {"type": "string"},
            "destination_city": {"type": "string"},
            "departure_date": {"type": "string"},
            "return_date": {"type": "string"},
            "estimated_total_cny": {"type": "number"},
            "preferred_class": {"type": "string", "enum": [c.value for c in TravelClass]},
        },
        "required": [
            "employee_id",
            "grade",
            "origin_city",
            "destination_city",
            "departure_date",
            "estimated_total_cny",
        ],
    }


def build_default_registry(
    policy: TravelPolicy,
    rag: PolicyRAG | None = None,
    *,
    approvals: ApprovalService | None = None,
    booking_store: Any | None = None,
) -> ToolRegistry:
    registry = ToolRegistry()
    approvals = approvals or ApprovalService()
    booking_store = booking_store or InMemoryBookingStore()

    async def plan_travel_itinerary(**args: Any) -> str:
        req = TravelRequest(
            request_id=str(uuid.uuid4()),
            employee_id=args["employee_id"],
            grade=EmployeeGrade(args["grade"]),
            origin_city=args["origin_city"],
            destination_city=args["destination_city"],
            departure_date=_parse_date(args["departure_date"]),
            return_date=_parse_date(args["return_date"]) if args.get("return_date") else None,
            purpose=TripPurpose(args["purpose"]),
            preferred_class=TravelClass(args["preferred_class"])
            if args.get("preferred_class")
            else None,
        )
        it = build_draft_itinerary(req)
        it = apply_policy_to_itinerary(policy, req, it, preferred_class=req.preferred_class)
        return summarize_itinerary_text(it)

    async def check_travel_policy(**args: Any) -> str:
        req = TravelRequest(
            request_id=str(uuid.uuid4()),
            employee_id=args["employee_id"],
            grade=EmployeeGrade(args["grade"]),
            origin_city=args["origin_city"],
            destination_city=args["destination_city"],
            departure_date=_parse_date(args["departure_date"]),
            return_date=_parse_date(args["return_date"]) if args.get("return_date") else None,
            purpose=TripPurpose.CLIENT,
        )
        dummy = build_draft_itinerary(req)
        total = Decimal(str(args["estimated_total_cny"]))
        dummy = dummy.model_copy(update={"total_estimated_cny": total})
        pc = TravelClass(args["preferred_class"]) if args.get("preferred_class") else None
        checked = apply_policy_to_itinerary(policy, req, dummy, preferred_class=pc)
        return "差标校验结果：\n" + "\n".join(f"- {w}" for w in checked.policy_warnings)

    async def search_travel_policy_docs(query: str = "") -> str:
        if rag is None:
            return (
                "（知识库未配置，无法检索制度文档，"
                "请基于通用差旅常识回答并注明以公司制度为准。）"
            )
        return await rag.search_context(str(query))

    async def search_flights_tool(**args: Any) -> dict[str, Any]:
        return await search_flights(FlightSearchRequest(**args))

    async def search_hotels_tool(**args: Any) -> dict[str, Any]:
        return await search_hotels(HotelSearchRequest(**args))

    async def search_trains_tool(**args: Any) -> dict[str, Any]:
        return await search_trains(TrainSearchRequest(**args))

    async def create_booking_tool(**args: Any) -> dict[str, Any]:
        employee_id = str(args.pop("employee_id"))
        amount_cny = float(args.pop("amount_cny", 0) or 0)
        result = await create_booking(BookingRequest(**args))
        booking = result["booking"]
        record = await booking_store.add_booking(
            booking_id=booking["booking_id"],
            employee_id=employee_id,
            booking_type=args["booking_type"],
            inventory_id=args["inventory_id"],
            confirmation_code=booking["confirmation_code"],
            amount_cny=amount_cny,
        )
        return {**result, "record": record}

    async def submit_travel_approval(
        employee_id: str, reason: str, estimated_total_cny: float
    ) -> dict[str, Any]:
        return approvals.create(employee_id, reason, float(estimated_total_cny))

    async def check_approval(approval_id: str) -> dict[str, Any]:
        ticket = approvals.query(approval_id)
        return ticket or {"error": f"审批单 {approval_id} 不存在"}

    async def submit_expense_report(employee_id: str) -> dict[str, Any]:
        return await booking_store.create_reimbursement(employee_id)

    registry.register(
        "plan_travel_itinerary",
        plan_travel_itinerary,
        description="根据结构化差旅需求生成草稿行程与费用预估。",
        json_schema=_plan_itinerary_schema(),
    )
    registry.register(
        "check_travel_policy",
        check_travel_policy,
        description="对已有行程草稿执行差标校验（舱位、预算、提前预订等）。",
        json_schema=_check_policy_schema(),
    )
    registry.register(
        "search_travel_policy_docs",
        search_travel_policy_docs,
        description="检索公司差旅制度文档，用于回答差标、审批、报销等政策问题。",
        json_schema={
            "type": "object",
            "properties": {"query": {"type": "string", "description": "要查询的政策问题"}},
            "required": ["query"],
        },
    )
    registry.register(
        "search_flights",
        search_flights_tool,
        description="搜索航班库存（当前为演示数据源）。",
        json_schema={
            "type": "object",
            "properties": {
                "origin": {"type": "string", "description": "出发城市或 IATA 代码"},
                "destination": {"type": "string"},
                "depart_date": {"type": "string", "description": "YYYY-MM-DD"},
                "return_date": {"type": "string", "description": "YYYY-MM-DD，可选"},
                "cabin": {"type": "string", "description": "economy/business 等"},
                "passengers": {"type": "integer"},
            },
            "required": ["origin", "destination", "depart_date"],
        },
    )
    registry.register(
        "search_hotels",
        search_hotels_tool,
        description="搜索酒店库存（当前为演示数据源）。",
        json_schema={
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "check_in": {"type": "string", "description": "YYYY-MM-DD"},
                "check_out": {"type": "string", "description": "YYYY-MM-DD"},
                "keyword": {"type": "string"},
            },
            "required": ["city", "check_in", "check_out"],
        },
    )
    registry.register(
        "search_trains",
        search_trains_tool,
        description="搜索火车/高铁车次（当前为演示数据源）。",
        json_schema={
            "type": "object",
            "properties": {
                "origin_station": {"type": "string"},
                "dest_station": {"type": "string"},
                "depart_date": {"type": "string", "description": "YYYY-MM-DD"},
                "prefer_gd": {"type": "boolean", "description": "是否优先高铁/动车"},
            },
            "required": ["origin_station", "dest_station", "depart_date"],
        },
    )
    registry.register(
        "create_booking",
        create_booking_tool,
        description="对已确认的搜索结果创建预订并落库（当前为演示数据源，返回确认号）。"
        "差标校验有超标警告或金额超审批线时，应先提交审批。",
        json_schema={
            "type": "object",
            "properties": {
                "employee_id": {"type": "string", "description": "出差员工工号"},
                "booking_type": {"type": "string", "enum": ["flight", "hotel", "train"]},
                "inventory_id": {"type": "string", "description": "上游搜索结果中的库存标识"},
                "traveler_name": {"type": "string"},
                "contact_phone": {"type": "string"},
                "amount_cny": {"type": "number", "description": "订单金额（元）"},
                "policy_case_id": {"type": "string", "description": "差标校验/审批单号，可选"},
            },
            "required": [
                "employee_id",
                "booking_type",
                "inventory_id",
                "traveler_name",
                "contact_phone",
                "amount_cny",
            ],
        },
    )
    registry.register(
        "submit_travel_approval",
        submit_travel_approval,
        description="金额超过事前审批线（或差标校验提示需审批）时，提交出差审批单。"
        "当前为模拟 OA，自动通过。",
        json_schema={
            "type": "object",
            "properties": {
                "employee_id": {"type": "string"},
                "reason": {"type": "string", "description": "出差事由与行程摘要"},
                "estimated_total_cny": {"type": "number"},
            },
            "required": ["employee_id", "reason", "estimated_total_cny"],
        },
    )
    registry.register(
        "check_approval",
        check_approval,
        description="查询出差审批单状态。",
        json_schema={
            "type": "object",
            "properties": {"approval_id": {"type": "string"}},
            "required": ["approval_id"],
        },
    )
    registry.register(
        "submit_expense_report",
        submit_expense_report,
        description="一键报销：汇总该员工所有未报销订单生成报销单并提交（模拟财务系统）。",
        json_schema={
            "type": "object",
            "properties": {"employee_id": {"type": "string"}},
            "required": ["employee_id"],
        },
    )
    return registry
