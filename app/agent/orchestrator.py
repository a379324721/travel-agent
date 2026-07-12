from __future__ import annotations

import json
import time
import uuid
from collections.abc import AsyncIterator
from datetime import date
from decimal import Decimal
from typing import Any, Dict, List, Optional

from app.config import settings
from app.core.memory.session_store import RedisSessionStore
from app.core.memory.short_term import ChatTurn, ShortTermMemory
from app.core.memory.summary import MemorySummarizer
from app.core.rag.service import PolicyRAG
from app.domain.schemas import ChatMessage, MessageRole, StreamChunk, StreamChunkType
from app.domain.travel.itinerary import build_draft_itinerary, summarize_itinerary_text
from app.domain.travel.models import EmployeeGrade, TripPurpose, TravelRequest, TravelClass
from app.domain.travel.policy import apply_policy_to_itinerary, default_corporate_policy
from app.services.llm import LLMService

SYSTEM_PROMPT = """你是差旅助手「travel-agent」，帮助员工规划行程、解释差标与审批要求。
回答简洁专业，涉及金额与政策时标注「以公司制度为准」。必要时调用工具生成行程或校验差标。
涉及公司差旅制度、差标额度、审批与报销政策的问题，先调用 search_travel_policy_docs 检索制度原文再作答。"""


def _travel_tools() -> List[Dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "plan_travel_itinerary",
                "description": "根据结构化差旅需求生成草稿行程与费用预估。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "employee_id": {"type": "string"},
                        "grade": {
                            "type": "string",
                            "enum": [g.value for g in EmployeeGrade],
                        },
                        "origin_city": {"type": "string"},
                        "destination_city": {"type": "string"},
                        "departure_date": {"type": "string", "description": "YYYY-MM-DD"},
                        "return_date": {"type": "string", "description": "YYYY-MM-DD，可选"},
                        "purpose": {
                            "type": "string",
                            "enum": [p.value for p in TripPurpose],
                        },
                        "preferred_class": {
                            "type": "string",
                            "enum": [c.value for c in TravelClass],
                        },
                    },
                    "required": [
                        "employee_id",
                        "grade",
                        "origin_city",
                        "destination_city",
                        "departure_date",
                        "purpose",
                    ],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "check_travel_policy",
                "description": "对已有行程草稿执行差标校验（舱位、预算、提前预订等）。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "employee_id": {"type": "string"},
                        "grade": {
                            "type": "string",
                            "enum": [g.value for g in EmployeeGrade],
                        },
                        "origin_city": {"type": "string"},
                        "destination_city": {"type": "string"},
                        "departure_date": {"type": "string"},
                        "return_date": {"type": "string"},
                        "estimated_total_cny": {"type": "number"},
                        "preferred_class": {
                            "type": "string",
                            "enum": [c.value for c in TravelClass],
                        },
                    },
                    "required": [
                        "employee_id",
                        "grade",
                        "origin_city",
                        "destination_city",
                        "departure_date",
                        "estimated_total_cny",
                    ],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search_travel_policy_docs",
                "description": "检索公司差旅制度文档，用于回答差标、审批、报销等政策问题。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "要查询的政策问题"},
                    },
                    "required": ["query"],
                },
            },
        },
    ]


def _to_openai_messages(messages: List[ChatMessage]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for m in messages:
        d: dict[str, Any] = {"role": m.role.value, "content": m.content}
        if m.name:
            d["name"] = m.name
        if m.tool_call_id:
            d["tool_call_id"] = m.tool_call_id
        out.append(d)
    return out


def _parse_date(s: str) -> date:
    y, mo, d = (int(x) for x in s.split("-", 2))
    return date(y, mo, d)


def _safe_json_args(arguments: str) -> Dict[str, Any]:
    try:
        parsed = json.loads(arguments) if arguments else {}
        return parsed if isinstance(parsed, dict) else {"value": parsed}
    except json.JSONDecodeError:
        return {"raw": arguments}


class TravelOrchestrator:
    def __init__(
        self,
        llm: Optional[LLMService] = None,
        session_store: Optional[RedisSessionStore] = None,
        policy_rag: Optional[PolicyRAG] = None,
    ) -> None:
        self._llm = llm or LLMService()
        self._policy = default_corporate_policy()
        self._sessions = session_store
        self._rag = policy_rag
        self._summarizer = MemorySummarizer(
            self._llm,
            token_threshold=settings.memory_summary_token_threshold,
        )

    async def _prepare_thread(
        self, messages: list[ChatMessage], session_id: Optional[str]
    ) -> list[ChatMessage]:
        """历史加载 → 拼接本轮消息 → token 预算裁剪 → 超阈值时摘要压缩。"""
        history: list[ChatMessage] = []
        if self._sessions is not None and session_id:
            history = await self._sessions.load(session_id)
        memory = ShortTermMemory(
            max_tokens=settings.memory_max_tokens,
            max_turns=settings.memory_window_size,
        )
        memory.extend(
            [ChatTurn(role=m.role.value, content=m.content) for m in [*history, *messages]]
        )
        await self._summarizer.maybe_compress(memory)
        return [ChatMessage(role=MessageRole(t.role), content=t.content) for t in memory.snapshot()]

    async def _persist_thread(
        self, session_id: Optional[str], thread: list[ChatMessage]
    ) -> None:
        if self._sessions is not None and session_id:
            await self._sessions.replace(session_id, thread)

    async def _execute_tool(self, name: str, arguments: str) -> str:
        args: dict[str, Any] = json.loads(arguments) if arguments else {}

        if name == "plan_travel_itinerary":
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
            pc = req.preferred_class
            it = apply_policy_to_itinerary(self._policy, req, it, preferred_class=pc)
            return summarize_itinerary_text(it)

        if name == "check_travel_policy":
            dep = _parse_date(args["departure_date"])
            ret = _parse_date(args["return_date"]) if args.get("return_date") else None
            req = TravelRequest(
                request_id=str(uuid.uuid4()),
                employee_id=args["employee_id"],
                grade=EmployeeGrade(args["grade"]),
                origin_city=args["origin_city"],
                destination_city=args["destination_city"],
                departure_date=dep,
                return_date=ret,
                purpose=TripPurpose.CLIENT,
            )
            dummy = build_draft_itinerary(req)
            total = Decimal(str(args["estimated_total_cny"]))
            dummy = dummy.model_copy(update={"total_estimated_cny": total})
            pc = TravelClass(args["preferred_class"]) if args.get("preferred_class") else None
            checked = apply_policy_to_itinerary(self._policy, req, dummy, preferred_class=pc)
            return "差标校验结果：\n" + "\n".join(f"- {w}" for w in checked.policy_warnings)

        if name == "search_travel_policy_docs":
            if self._rag is None:
                return "（知识库未配置，无法检索制度文档，请基于通用差旅常识回答并注明以公司制度为准。）"
            return await self._rag.search_context(str(args.get("query", "")))

        return json.dumps({"error": f"unknown tool {name}"}, ensure_ascii=False)

    async def run_completion(
        self, messages: list[ChatMessage], session_id: Optional[str] = None
    ) -> dict[str, Any]:
        msgs = await self._prepare_thread(messages, session_id)
        openai_msgs: List[Dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
        openai_msgs.extend(_to_openai_messages(msgs))

        tools = _travel_tools()
        for _ in range(settings.max_react_iterations):
            resp = await self._llm.chat_completion(openai_msgs, tools=tools, tool_choice="auto")
            choice = resp.choices[0]
            msg = choice.message

            if msg.tool_calls:
                assistant_msg: Dict[str, Any] = {
                    "role": "assistant",
                    "content": msg.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments or "{}",
                            },
                        }
                        for tc in msg.tool_calls
                    ],
                }
                openai_msgs.append(assistant_msg)
                for tc in msg.tool_calls:
                    out = await self._execute_tool(tc.function.name, tc.function.arguments)
                    openai_msgs.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": out,
                        }
                    )
                continue

            content = msg.content or ""
            await self._persist_thread(
                session_id,
                [*msgs, ChatMessage(role=MessageRole.ASSISTANT, content=content)],
            )
            return {
                "id": getattr(resp, "id", str(uuid.uuid4())),
                "created": int(time.time()),
                "model": self._llm.model,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": content}}],
                "usage": {
                    "prompt_tokens": resp.usage.prompt_tokens if resp.usage else 0,
                    "completion_tokens": resp.usage.completion_tokens if resp.usage else 0,
                    "total_tokens": resp.usage.total_tokens if resp.usage else 0,
                },
            }

        return {
            "id": str(uuid.uuid4()),
            "created": int(time.time()),
            "model": self._llm.model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "已达到最大推理轮次，请简化问题后重试。",
                    },
                }
            ],
            "usage": None,
        }

    async def stream_completion(
        self, messages: list[ChatMessage], session_id: Optional[str] = None
    ) -> AsyncIterator[StreamChunk]:
        """真流式：透传 LLM 增量输出；工具调用轮次发 TOOL_CALL 状态块。"""
        msgs = await self._prepare_thread(messages, session_id)
        openai_msgs: List[Dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
        openai_msgs.extend(_to_openai_messages(msgs))

        tools = _travel_tools()
        idx = 0
        for _ in range(settings.max_react_iterations):
            content_parts: list[str] = []
            calls: dict[int, dict[str, str]] = {}
            finish_reason: Optional[str] = None

            async for chunk in self._llm.chat_completion_stream(
                openai_msgs, tools=tools, tool_choice="auto"
            ):
                if not getattr(chunk, "choices", None):
                    continue
                choice = chunk.choices[0]
                delta = choice.delta
                if delta is None:
                    continue
                if delta.content:
                    content_parts.append(delta.content)
                    yield StreamChunk(
                        type=StreamChunkType.CONTENT, index=idx, delta=delta.content
                    )
                    idx += 1
                for tc in delta.tool_calls or []:
                    acc = calls.setdefault(tc.index, {"id": "", "name": "", "arguments": ""})
                    if tc.id:
                        acc["id"] = tc.id
                    fn = tc.function
                    if fn is not None and fn.name:
                        acc["name"] = fn.name
                    if fn is not None and fn.arguments:
                        acc["arguments"] += fn.arguments
                if choice.finish_reason:
                    finish_reason = choice.finish_reason

            if calls:
                ordered = [calls[i] for i in sorted(calls)]
                openai_msgs.append(
                    {
                        "role": "assistant",
                        "content": "".join(content_parts) or None,
                        "tool_calls": [
                            {
                                "id": c["id"],
                                "type": "function",
                                "function": {
                                    "name": c["name"],
                                    "arguments": c["arguments"] or "{}",
                                },
                            }
                            for c in ordered
                        ],
                    }
                )
                for c in ordered:
                    yield StreamChunk(
                        type=StreamChunkType.TOOL_CALL,
                        index=idx,
                        tool_name=c["name"],
                        tool_args=_safe_json_args(c["arguments"]),
                    )
                    idx += 1
                    out = await self._execute_tool(c["name"], c["arguments"])
                    openai_msgs.append(
                        {"role": "tool", "tool_call_id": c["id"], "content": out}
                    )
                continue

            final = "".join(content_parts)
            await self._persist_thread(
                session_id,
                [*msgs, ChatMessage(role=MessageRole.ASSISTANT, content=final)],
            )
            yield StreamChunk(
                type=StreamChunkType.DONE,
                index=idx,
                finish_reason=finish_reason or "stop",
            )
            return

        yield StreamChunk(
            type=StreamChunkType.ERROR,
            index=idx,
            error="已达到最大推理轮次，请简化问题后重试。",
        )
