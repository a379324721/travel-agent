from __future__ import annotations

import json
import time
import uuid
from datetime import date
from collections.abc import AsyncIterator
from typing import Any

from app.agent.tools import build_default_registry
from app.config import settings
from app.core.intent.recognizer import IntentRecognizer, TravelIntent
from app.core.logging import get_logger
from app.core.memory.session_store import RedisSessionStore
from app.core.memory.short_term import ChatTurn, ShortTermMemory
from app.core.memory.summary import MemorySummarizer
from app.core.rag.service import PolicyRAG
from app.core.tools.registry import ToolRegistry, invoke
from app.domain.schemas import ChatMessage, MessageRole, StreamChunk, StreamChunkType
from app.domain.travel.policy import default_corporate_policy
from app.infrastructure.observability.metrics import get_metrics
from app.services.approval import ApprovalService
from app.services.employee_directory import EmployeeDirectory
from app.services.llm import LLMService

logger = get_logger(__name__)

SYSTEM_PROMPT = """你是差旅助手「travel-agent」，帮助员工完成差旅全流程：
规划行程 → 差标校验 →（金额超审批线时先用 submit_travel_approval 提交审批）→
create_booking 订票 → 行程结束后 submit_expense_report 一键报销。
回答简洁专业，涉及金额与政策时标注「以公司制度为准」。必要时调用工具生成行程或校验差标。
涉及公司差旅制度、差标额度、审批与报销政策的问题，
先调用 search_travel_policy_docs 检索制度原文再作答。
系统提供「当前用户」信息时，直接使用其工号与职级调用工具，不要再向用户询问。"""

_WEEKDAY_CN = "一二三四五六日"


def _system_message() -> dict[str, Any]:
    today = date.today()
    return {
        "role": "system",
        "content": (
            f"{SYSTEM_PROMPT}\n"
            f"今天是 {today.isoformat()}，星期{_WEEKDAY_CN[today.weekday()]}。"
            "解析「明天」「下周三」等相对日期时以此为准。"
        ),
    }


def _to_openai_messages(messages: list[ChatMessage]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for m in messages:
        d: dict[str, Any] = {"role": m.role.value, "content": m.content}
        if m.name:
            d["name"] = m.name
        if m.tool_call_id:
            d["tool_call_id"] = m.tool_call_id
        out.append(d)
    return out


def _identity_message(profile: Any) -> dict[str, Any]:
    return {
        "role": "system",
        "content": (
            f"当前用户：{profile.name}（工号 {profile.employee_id}），"
            f"职级 {profile.grade.value}，部门 {profile.department}，"
            f"联系电话 {profile.phone}。"
        ),
    }


def _safe_json_args(arguments: str) -> dict[str, Any]:
    try:
        parsed = json.loads(arguments) if arguments else {}
        return parsed if isinstance(parsed, dict) else {"value": parsed}
    except json.JSONDecodeError:
        return {"raw": arguments}


class TravelOrchestrator:
    def __init__(
        self,
        llm: LLMService | None = None,
        session_store: RedisSessionStore | None = None,
        policy_rag: PolicyRAG | None = None,
        registry: ToolRegistry | None = None,
        intent_recognizer: IntentRecognizer | None = None,
        directory: EmployeeDirectory | None = None,
        approvals: ApprovalService | None = None,
        booking_store: Any | None = None,
    ) -> None:
        self._llm = llm or LLMService()
        self._policy = default_corporate_policy()
        self._sessions = session_store
        self._registry = registry or build_default_registry(
            self._policy, policy_rag, approvals=approvals, booking_store=booking_store
        )
        if intent_recognizer is not None:
            self._recognizer = intent_recognizer
        elif settings.intent_slow_lane_enabled:
            self._recognizer = IntentRecognizer.with_llm(
                self._llm, slow_lane_threshold=settings.intent_slow_lane_threshold
            )
        else:
            self._recognizer = IntentRecognizer()
        self._directory = directory or EmployeeDirectory()
        self._summarizer = MemorySummarizer(
            self._llm,
            token_threshold=settings.memory_summary_token_threshold,
        )

    async def _prepare_thread(
        self, messages: list[ChatMessage], session_id: str | None
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
        self,
        session_id: str | None,
        thread: list[ChatMessage],
        user_id: str | None = None,
    ) -> None:
        if self._sessions is None or not session_id:
            return
        await self._sessions.replace(session_id, thread)
        if user_id:
            title = next(
                (m.content for m in thread if m.role == MessageRole.USER), "新会话"
            )
            await self._sessions.touch_session(user_id, session_id, title)

    def _tool_defs(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.json_schema or {"type": "object", "properties": {}},
                },
            }
            for t in self._registry.list_tools()
        ]

    async def _route(
        self, messages: list[ChatMessage]
    ) -> tuple[list[dict[str, Any]] | None, Any]:
        """意图路由：政策类问题首轮强制检索制度文档。

        GENERAL 表示"规则未识别"而非闲聊，工具保持可用（tool_choice=auto），
        否则未覆盖的表达（如报销）会让模型无工具可调而编造结果。
        """
        last_user = next(
            (m.content for m in reversed(messages) if m.role == MessageRole.USER), ""
        )
        result = await self._recognizer.recognize(last_user)
        logger.info(
            "intent.recognized",
            intent=result.intent.value,
            confidence=result.confidence,
            lane=result.metadata.get("merged"),
        )
        is_policy = result.intent in (TravelIntent.POLICY, TravelIntent.RAG)
        if is_policy and settings.llm_force_tool_choice:
            forced = {"type": "function", "function": {"name": "search_travel_policy_docs"}}
            return self._tool_defs(), forced
        return self._tool_defs(), "auto"

    async def _execute_tool(self, name: str, arguments: str) -> str:
        metrics = get_metrics()
        if not self._registry.has(name):
            metrics.record_success("tool", ok=False)
            return json.dumps({"error": f"unknown tool {name}"}, ensure_ascii=False)
        try:
            with metrics.time_block(f"tool_{name}"):
                result = await invoke(self._registry, name, _safe_json_args(arguments))
        except Exception as exc:
            metrics.record_success("tool", ok=False)
            return json.dumps({"error": str(exc)}, ensure_ascii=False)
        metrics.record_success("tool", ok=True)
        if isinstance(result, str):
            return result
        return json.dumps(result, ensure_ascii=False, default=str)

    async def run_completion(
        self,
        messages: list[ChatMessage],
        session_id: str | None = None,
        user_id: str | None = None,
    ) -> dict[str, Any]:
        msgs = await self._prepare_thread(messages, session_id)
        openai_msgs: list[dict[str, Any]] = [_system_message()]
        profile = self._directory.get(user_id)
        if profile is not None:
            openai_msgs.append(_identity_message(profile))
        openai_msgs.extend(_to_openai_messages(msgs))

        metrics = get_metrics()
        metrics.increment_counter("chat_requests")
        tools, tool_choice = await self._route(msgs)
        for _ in range(settings.max_react_iterations):
            round_tool_choice = tool_choice
            tool_choice = "auto"
            with metrics.time_block("llm_completion"):
                resp = await self._llm.chat_completion(
                    openai_msgs, tools=tools, tool_choice=round_tool_choice
                )
            if getattr(resp, "usage", None):
                metrics.record_tokens(resp.usage.prompt_tokens, resp.usage.completion_tokens)
            choice = resp.choices[0]
            msg = choice.message

            if msg.tool_calls:
                assistant_msg: dict[str, Any] = {
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
                user_id=user_id,
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
        self,
        messages: list[ChatMessage],
        session_id: str | None = None,
        user_id: str | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """真流式：透传 LLM 增量输出；工具调用轮次发 TOOL_CALL 状态块。"""
        msgs = await self._prepare_thread(messages, session_id)
        openai_msgs: list[dict[str, Any]] = [_system_message()]
        profile = self._directory.get(user_id)
        if profile is not None:
            openai_msgs.append(_identity_message(profile))
        openai_msgs.extend(_to_openai_messages(msgs))

        tools, tool_choice = await self._route(msgs)
        idx = 0
        round_texts: list[str] = []
        for _ in range(settings.max_react_iterations):
            round_tool_choice = tool_choice
            tool_choice = "auto"
            content_parts: list[str] = []
            calls: dict[int, dict[str, str]] = {}
            finish_reason: str | None = None

            async for chunk in self._llm.chat_completion_stream(
                openai_msgs, tools=tools, tool_choice=round_tool_choice
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

            round_text = "".join(content_parts)
            if round_text:
                round_texts.append(round_text)

            if calls:
                ordered = [calls[i] for i in sorted(calls)]
                openai_msgs.append(
                    {
                        "role": "assistant",
                        "content": round_text or None,
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

            final = "\n\n".join(round_texts)
            await self._persist_thread(
                session_id,
                [*msgs, ChatMessage(role=MessageRole.ASSISTANT, content=final)],
                user_id=user_id,
            )
            yield StreamChunk(
                type=StreamChunkType.DONE,
                index=idx,
                finish_reason=finish_reason or "stop",
            )
            return

        error_text = "已达到最大推理轮次，请简化问题后重试。"
        await self._persist_thread(
            session_id,
            [
                *msgs,
                ChatMessage(
                    role=MessageRole.ASSISTANT,
                    content="\n\n".join([*round_texts, f"（{error_text}）"]),
                ),
            ],
            user_id=user_id,
        )
        yield StreamChunk(
            type=StreamChunkType.ERROR,
            index=idx,
            error=error_text,
        )
