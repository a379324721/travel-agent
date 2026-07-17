"""
LLM 意图分类器（慢车道）- 对复杂表述进行结构化分类。
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from app.core.intent.intents import TravelIntent
from app.core.logging import get_logger

logger = get_logger(__name__)


@runtime_checkable
class StructuredLLMClient(Protocol):
    """可注入的异步 LLM，返回结构化 JSON 字符串或对象。"""

    async def complete_structured(
        self,
        *,
        system_prompt: str,
        user_content: str,
        response_format: str,
    ) -> str:
        """返回 JSON 字符串，包含 intent_slug 与 confidence。"""


_JSON_BLOCK = re.compile(r"\{[\s\S]*\}")

# 只给分类器用的私有工具，不进主 ToolRegistry
_FETCH_HISTORY_TOOL = {
    "type": "function",
    "function": {
        "name": "fetch_history",
        "description": "回溯比已给出的「最近对话」更早的聊天记录。"
        "仅当最近对话不足以判断意图时调用，自行决定回溯条数。",
        "parameters": {
            "type": "object",
            "properties": {
                "count": {"type": "integer", "description": "要回溯的条数，1-10"}
            },
            "required": ["count"],
        },
    },
}

_MAX_FETCH_COUNT = 10


@dataclass
class LLMClassification:
    """LLM 分类输出。"""

    intent_slug: str
    confidence: float
    rationale: str = ""
    standalone_query: str = ""


class LLMIntentClassifier:
    """
    使用 LLM 进行意图分类，要求模型输出固定 JSON 结构以便解析与校验。
    """

    def __init__(self, llm: StructuredLLMClient) -> None:
        self._llm = llm

    async def classify(
        self,
        text: str,
        *,
        recent: str = "",
        fetch_history: Callable[[int], str] | None = None,
    ) -> LLMClassification:
        """对文本进行异步分类；解析失败时回退为 general 低置信度。

        `recent` 为最近几条对话记录（消歧用）；`fetch_history` 允许模型在
        最近对话仍不足以判断时自行回溯更早的记录。
        """
        system = (
            "你是商旅助手意图分类器。判断「待分类的用户输入」的意图，"
            "从给定意图中选一个最匹配的；「最近对话」仅用于消歧。"
            "必须只输出一个 JSON 对象，字段：intent_slug (string), confidence (0-1 小数), "
            "standalone_query (string，把用户输入改写为不依赖上下文的完整表述；"
            "输入本身已完整时原样返回), rationale (简短中文理由)。不要输出其它文字。"
        )
        catalog = "\n".join(f"- {i.value}: {i.description}" for i in TravelIntent)
        context = f"最近对话：\n{recent or '（无）'}\n\n"
        user = f"{context}待分类的用户输入：\n{text}\n\n可选意图（slug: 含义）：\n{catalog}"
        supports_tools = hasattr(self._llm, "complete_with_tools")
        if fetch_history is not None and supports_tools:
            raw = await self._llm.complete_with_tools(
                system_prompt=system,
                user_content=user,
                tools=[_FETCH_HISTORY_TOOL],
                tool_executor=self._make_executor(fetch_history),
            )
        else:
            raw = await self._llm.complete_structured(
                system_prompt=system,
                user_content=user,
                response_format="json",
            )
        return self._parse_response(raw)

    @staticmethod
    def _make_executor(fetch_history: Callable[[int], str]) -> Callable[[str, str], str]:
        def executor(name: str, arguments: str) -> str:
            if name != "fetch_history":
                return f"未知工具 {name}"
            try:
                count = int(json.loads(arguments or "{}").get("count", 3))
            except (json.JSONDecodeError, TypeError, ValueError):
                count = 3
            return fetch_history(max(1, min(count, _MAX_FETCH_COUNT)))

        return executor

    def _parse_response(self, raw: str) -> LLMClassification:
        try:
            payload = self._extract_json(raw)
            slug = str(payload.get("intent_slug", "general")).strip()
            conf = float(payload.get("confidence", 0.5))
            rationale = str(payload.get("rationale", ""))
            standalone = str(payload.get("standalone_query", ""))
        except (json.JSONDecodeError, TypeError, ValueError):
            logger.warning("intent.slow_lane.parse_error", raw=raw[:500])
            return LLMClassification("general", 0.35, "parse_error")

        conf = max(0.0, min(1.0, conf))
        try:
            TravelIntent(slug)
        except ValueError:
            return LLMClassification(
                "general", conf * 0.6, rationale or "unknown_slug", standalone
            )
        return LLMClassification(slug, conf, rationale, standalone)

    def _extract_json(self, raw: str) -> dict[str, Any]:
        raw = raw.strip()
        try:
            data = json.loads(raw)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass
        match = _JSON_BLOCK.search(raw)
        if not match:
            raise json.JSONDecodeError("no json object", raw, 0)
        parsed = json.loads(match.group(0))
        if not isinstance(parsed, dict):
            raise TypeError("expected object")
        return parsed
