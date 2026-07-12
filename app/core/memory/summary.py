"""Summarizes long chat history when token budget is exceeded."""

from __future__ import annotations

from dataclasses import dataclass

from app.core.memory.short_term import ChatTurn, ShortTermMemory
from app.services.llm import LLMService


@dataclass(slots=True)
class SummaryState:
    summary_text: str
    covered_turns: int


class MemorySummarizer:
    """Compresses older turns into a rolling summary when memory grows too large."""

    def __init__(
        self,
        llm: LLMService,
        *,
        token_threshold: int = 6000,
        summary_max_tokens: int = 512,
    ) -> None:
        self._llm = llm
        self._token_threshold = token_threshold
        self._summary_max_tokens = summary_max_tokens

    async def maybe_compress(self, memory: ShortTermMemory) -> SummaryState | None:
        if memory.total_tokens() <= self._token_threshold:
            return None
        turns = memory.snapshot()
        if len(turns) < 4:
            return None
        pivot = max(2, len(turns) // 2)
        older, recent = turns[:pivot], turns[pivot:]
        transcript = "\n".join(f"{t.role}: {t.content}" for t in older)
        prompt = (
            "请将以下对话压缩为简洁要点摘要，保留用户目标、约束与已确认事实；"
            "使用中文要点列表。\n\n"
            f"{transcript}"
        )
        resp = await self._llm.chat_completion(
            [
                {"role": "system", "content": "你是对话摘要助手。"},
                {"role": "user", "content": prompt},
            ],
            max_tokens=self._summary_max_tokens,
            temperature=0.1,
        )
        text = resp.choices[0].message.content or ""
        memory.clear()
        memory.extend([ChatTurn(role="system", content=f"[历史摘要]\n{text}")])
        memory.extend(recent)
        return SummaryState(summary_text=text, covered_turns=len(older))
