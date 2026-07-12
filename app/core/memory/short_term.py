"""Short-term conversation memory with sliding window and token budgeting."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Literal

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


class ShortTermMemory:
    """Keeps recent turns under a token ceiling using a sliding window."""

    def __init__(
        self,
        *,
        model_encoding: str = "cl100k_base",
        max_tokens: int = 8000,
        max_turns: int = 40,
    ) -> None:
        self._enc = tiktoken.get_encoding(model_encoding) if tiktoken is not None else None
        self._max_tokens = max_tokens
        self._max_turns = max_turns
        self._turns: deque[ChatTurn] = deque()

    def _count_tokens(self, text: str) -> int:
        if self._enc is not None:
            return len(self._enc.encode(text))
        return _approx_token_count(text)

    def total_tokens(self) -> int:
        return sum(self._count_tokens(t.content) for t in self._turns)

    def append(self, turn: ChatTurn) -> None:
        self._turns.append(turn)
        self._trim_by_turns()
        self._trim_by_tokens()

    def extend(self, turns: list[ChatTurn]) -> None:
        for t in turns:
            self._turns.append(t)
        self._trim_by_turns()
        self._trim_by_tokens()

    def _trim_by_turns(self) -> None:
        while len(self._turns) > self._max_turns:
            self._turns.popleft()

    def _trim_by_tokens(self) -> None:
        while self._turns and self.total_tokens() > self._max_tokens:
            self._turns.popleft()

    def as_messages(self) -> list[dict[str, str]]:
        return [{"role": t.role, "content": t.content} for t in self._turns]

    def clear(self) -> None:
        self._turns.clear()

    def snapshot(self) -> list[ChatTurn]:
        return list(self._turns)
