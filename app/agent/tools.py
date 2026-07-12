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
    policy: TravelPolicy, rag: PolicyRAG | None = None
) -> ToolRegistry:
    registry = ToolRegistry()

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
        return await create_booking(BookingRequest(**args))

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
        description="对已确认的搜索结果创建预订（当前为演示数据源，返回确认号）。",
        json_schema={
            "type": "object",
            "properties": {
                "booking_type": {"type": "string", "enum": ["flight", "hotel", "train"]},
                "inventory_id": {"type": "string", "description": "上游搜索结果中的库存标识"},
                "traveler_name": {"type": "string"},
                "contact_phone": {"type": "string"},
                "policy_case_id": {"type": "string", "description": "差标校验/审批单号，可选"},
            },
            "required": ["booking_type", "inventory_id", "traveler_name", "contact_phone"],
        },
    )
    return registry
