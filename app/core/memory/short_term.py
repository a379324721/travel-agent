"""Short-term conversation memory with token budgeting."""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass
from typing import Any, Literal

try:
    import tiktoken
except ImportError:
    tiktoken = None

Role = Literal["system", "user", "assistant", "tool"]


def _approx_token_count(text: str) -> int:
    """无 tiktoken 时的近似计数：CJK 每字约 1 token，其余每 4 字符约 1 token。"""
    cjk = sum(1 for ch in text if "一" <= ch <= "鿿")
    return cjk + (len(text) - cjk) // 4 + 1


@dataclass(slots=True)
class ChatTurn:
    role: Role
    content: str
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[dict[str, Any]] | None = None


class ShortTermMemory:
    """Holds conversation turns; trim_to_budget() 是兜底保险丝，摘要压缩才是主力。"""

    def __init__(
        self,
        *,
        model_encoding: str = "cl100k_base",
        max_tokens: int = 12000,
    ) -> None:
        self._enc = tiktoken.get_encoding(model_encoding) if tiktoken is not None else None
        self._max_tokens = max_tokens
        self._turns: deque[ChatTurn] = deque()

    def _count_tokens(self, text: str) -> int:
        if self._enc is not None:
            return len(self._enc.encode(text))
        return _approx_token_count(text)

    def _turn_tokens(self, turn: ChatTurn) -> int:
        tokens = self._count_tokens(turn.content)
        if turn.tool_calls:
            tokens += self._count_tokens(json.dumps(turn.tool_calls, ensure_ascii=False))
        return tokens

    def total_tokens(self) -> int:
        return sum(self._turn_tokens(t) for t in self._turns)

    def extend(self, turns: list[ChatTurn]) -> None:
        self._turns.extend(turns)

    def trim_to_budget(self) -> None:
        """兜底硬裁：摘要压缩后仍超预算时，从队首丢最旧的。"""
        total = self.total_tokens()
        while self._turns and total > self._max_tokens:
            total -= self._turn_tokens(self._turns.popleft())
        self._drop_orphan_tool_turns()

    def _drop_orphan_tool_turns(self) -> None:
        """裁剪可能砍掉发起 tool_calls 的 assistant 消息，
        留在队首的 tool 结果就成了孤儿（LLM API 会拒绝），一并丢弃。"""
        while self._turns and self._turns[0].role == "tool":
            self._turns.popleft()

    def clear(self) -> None:
        self._turns.clear()

    def snapshot(self) -> list[ChatTurn]:
        return list(self._turns)
