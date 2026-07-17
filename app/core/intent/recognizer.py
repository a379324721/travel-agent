"""快慢车道意图识别：规则引擎（快）+ LLM 分类（慢），合并置信度。"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from app.core.intent.intents import TravelIntent
from app.core.intent.llm_classifier import (
    LLMClassification,
    LLMIntentClassifier,
)
from app.core.intent.rule_engine import RuleEngine

BusinessIntent = TravelIntent

# 指代/省略形态：意图寄生在上文里，关键词命中也可能是误判
# （如「那飞机呢?」按“飞机”被判成机票搜索），须由慢车道结合上下文复核
_CONTEXT_DEPENDENT = re.compile(
    r"^(那|这|继续|接着|然后)|(那个|这个|刚才|之前|上面|它)|呢[?？]?$"
)


def _needs_context(text: str) -> bool:
    return bool(_CONTEXT_DEPENDENT.search(text.strip()))


@dataclass(slots=True)
class IntentResult:
    """意图识别结果，含合并后的置信度与元数据。"""

    intent: TravelIntent
    confidence: float
    slots: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    standalone_query: str = ""


class StructuredLLMBridge:
    """将主线 `LLMService` 适配为 `StructuredLLMClient`（依赖提示词约束 JSON 输出）。"""

    def __init__(self, client: Any, *, model: str | None = None) -> None:
        if not hasattr(client, "chat_completion"):
            raise TypeError("client must provide chat_completion()")
        self._client = client
        self._model = model  # 预留：LLMService 当前使用自身配置的模型

    async def complete_structured(
        self,
        *,
        system_prompt: str,
        user_content: str,
        response_format: str,
    ) -> str:
        _ = response_format
        resp = await self._client.chat_completion(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=0.0,
            max_tokens=1024,
        )
        return resp.choices[0].message.content or ""

    async def complete_with_tools(
        self,
        *,
        system_prompt: str,
        user_content: str,
        tools: list[dict[str, Any]],
        tool_executor: Callable[[str, str], str],
        max_rounds: int = 2,
    ) -> str:
        """带工具的 mini 循环：最多 `max_rounds` 次 LLM 调用，
        末轮 tool_choice=none 强制出结果。"""
        msgs: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        content = ""
        for i in range(max_rounds):
            is_last = i == max_rounds - 1
            if is_last and i > 0:
                # 工具往返后模型对格式约束的遵从率下降，末轮前重申只输出 JSON
                msgs.append(
                    {
                        "role": "system",
                        "content": "请现在只输出要求的 JSON 对象，不要输出其它文字。",
                    }
                )
            resp = await self._client.chat_completion(
                msgs,
                tools=tools,
                tool_choice="none" if is_last else "auto",
                temperature=0.0,
                max_tokens=1024,
            )
            msg = resp.choices[0].message
            content = msg.content or ""
            if not getattr(msg, "tool_calls", None):
                return content
            msgs.append(
                {
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
            )
            for tc in msg.tool_calls:
                msgs.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": tool_executor(
                            tc.function.name, tc.function.arguments or "{}"
                        ),
                    }
                )
        # 末轮仍返回 tool_calls（供应商忽略 none）时，退回已有文本而非崩溃
        return content


class IntentRecognizer:
    """
    快慢车道意图识别器。

    - 快车道：`RuleEngine` 固定模式匹配。
    - 慢车道：可选 `LLMIntentClassifier`，在规则置信不足或需消歧时调用。
    - 合并策略：两路一致时抬升置信度；冲突时取置信更高一路，并记录 `metadata`。
    """

    def __init__(
        self,
        *,
        rule_engine: RuleEngine | None = None,
        llm_classifier: LLMIntentClassifier | None = None,
        slow_lane_threshold: float = 0.82,
    ) -> None:
        self._rules = rule_engine or RuleEngine()
        self._llm = llm_classifier
        self._slow_threshold = slow_lane_threshold

    @classmethod
    def with_llm(
        cls,
        llm_client: Any,
        *,
        slow_lane_threshold: float = 0.82,
        model: str | None = None,
    ) -> IntentRecognizer:
        """使用内置 `StructuredLLMBridge` 与默认慢车道分类器构造识别器。"""
        bridge = StructuredLLMBridge(llm_client, model=model)
        classifier = LLMIntentClassifier(bridge)
        return cls(llm_classifier=classifier, slow_lane_threshold=slow_lane_threshold)

    def _slug_to_intent(self, slug: str) -> TravelIntent:
        try:
            return TravelIntent(slug)
        except ValueError:
            return TravelIntent.GENERAL

    def _merge(
        self,
        fast: tuple[str, float] | None,
        slow: LLMClassification | None,
    ) -> IntentResult:
        if slow is None and fast is None:
            return IntentResult(
                TravelIntent.GENERAL,
                0.4,
                metadata={"fast_lane": None, "slow_lane": None},
            )
        if slow is None and fast is not None:
            intent = self._slug_to_intent(fast[0])
            return IntentResult(
                intent,
                fast[1],
                metadata={"fast_lane": fast, "slow_lane": None, "merged": "fast_only"},
            )
        if fast is None and slow is not None:
            intent = self._slug_to_intent(slow.intent_slug)
            return IntentResult(
                intent,
                slow.confidence,
                metadata={"fast_lane": None, "slow_lane": slow, "merged": "slow_only"},
                standalone_query=slow.standalone_query,
            )
        assert fast is not None and slow is not None
        i_fast = self._slug_to_intent(fast[0])
        i_slow = self._slug_to_intent(slow.intent_slug)
        if i_fast == i_slow:
            conf = min(1.0, (fast[1] + slow.confidence) / 2 + 0.05)
            return IntentResult(
                i_fast,
                conf,
                metadata={
                    "fast_lane": fast,
                    "slow_lane": slow,
                    "merged": "agree",
                },
                standalone_query=slow.standalone_query,
            )
        if fast[1] >= slow.confidence + 0.05:
            return IntentResult(
                i_fast,
                fast[1],
                metadata={"fast_lane": fast, "slow_lane": slow, "merged": "prefer_fast"},
            )
        return IntentResult(
            i_slow,
            slow.confidence,
            metadata={"fast_lane": fast, "slow_lane": slow, "merged": "prefer_slow"},
            standalone_query=slow.standalone_query,
        )

    async def recognize(
        self,
        text: str,
        *,
        recent: str = "",
        fetch_history: Callable[[int], str] | None = None,
    ) -> IntentResult:
        """异步识别用户文本，返回意图与合并置信度。

        `recent`/`fetch_history` 仅供慢车道消歧；快车道规则始终只看 `text`。
        """
        fast = self._rules.classify(text)
        need_slow = self._llm is not None and (
            fast is None or fast[1] < self._slow_threshold or _needs_context(text)
        )
        slow: LLMClassification | None = None
        if need_slow:
            slow = await self._llm.classify(
                text, recent=recent, fetch_history=fetch_history
            )
        if not need_slow and fast is not None:
            intent = self._slug_to_intent(fast[0])
            return IntentResult(
                intent,
                fast[1],
                metadata={"fast_lane": fast, "slow_lane": None, "merged": "fast_only"},
            )
        return self._merge(fast, slow)
