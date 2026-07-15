from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from openai import AsyncOpenAI

from app.config import settings
from app.core.circuit_breaker import CircuitBreaker


class LLMService:
    def __init__(self) -> None:
        self._client = AsyncOpenAI(
            api_key=settings.openai_api_key or "dummy",
            base_url=settings.openai_base_url,
        )
        self._model = settings.openai_model
        self._breaker = CircuitBreaker(
            failure_threshold=settings.circuit_breaker_failure_threshold,
            recovery_timeout=float(settings.circuit_breaker_recovery_timeout),
            half_open_max_calls=settings.circuit_breaker_half_open_max_calls,
        )

    @property
    def model(self) -> str:
        return self._model

    async def chat_completion(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = "auto",
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> Any:
        async def _call() -> Any:
            kwargs: dict[str, Any] = {
                "model": self._model,
                "messages": messages,
                "temperature": temperature,
            }
            if max_tokens is not None:
                kwargs["max_tokens"] = max_tokens
            if tools:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = tool_choice
            if settings.llm_enable_thinking is not None:
                kwargs["extra_body"] = {"enable_thinking": settings.llm_enable_thinking}
            return await self._client.chat.completions.create(**kwargs)

        return await self._breaker.call(_call)

    async def chat_completion_stream(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = "auto",
        temperature: float = 0.2,
    ) -> AsyncIterator[Any]:
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
            "stream": True,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice
        if settings.llm_enable_thinking is not None:
            kwargs["extra_body"] = {"enable_thinking": settings.llm_enable_thinking}

        stream = await self._client.chat.completions.create(**kwargs)
        async for chunk in stream:
            yield chunk
