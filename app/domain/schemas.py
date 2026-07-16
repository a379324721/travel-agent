from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class MessageRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ChatMessage(BaseModel):
    role: MessageRole
    content: str
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[dict[str, Any]] | None = None


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(..., min_length=1, description="对话消息列表")
    stream: bool = False
    session_id: str | None = Field(None, description="会话 ID，用于记忆与审计")
    user_id: str | None = Field(None, description="企业用户标识")
    locale: str = Field("zh-CN", description="语言区域")


class ToolCallInfo(BaseModel):
    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ChatResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[dict[str, Any]]
    usage: dict[str, int] | None = None
    session_id: str | None = None
    tool_calls: list[ToolCallInfo] = Field(default_factory=list)


class StreamChunkType(str, Enum):
    CONTENT = "content"
    TOOL_CALL = "tool_call"
    DONE = "done"
    ERROR = "error"


class StreamChunk(BaseModel):
    type: StreamChunkType
    index: int = 0
    delta: str | None = None
    tool_name: str | None = None
    tool_args: dict[str, Any] | None = None
    finish_reason: str | None = None
    error: str | None = None


class DocumentIngestRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=512)
    content: str = Field(..., min_length=1, description="纯文本正文，用于向量检索")
    doc_type: Literal["policy", "sop", "city_guide", "other"] = "policy"
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentIngestResponse(BaseModel):
    doc_id: str
    collection: str
    inserted_at: datetime
    vector_dim: int | None = None
    chunks: int | None = None


class HealthStatus(BaseModel):
    status: Literal["ok", "degraded", "unhealthy"]
    version: str = "0.1.0"
    checks: dict[str, bool] = Field(default_factory=dict)
    detail: str | None = None
