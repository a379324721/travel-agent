"""会话历史接口：按用户列出会话、拉取单个会话的消息。"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Query, Request

router = APIRouter(tags=["sessions"])


def _store(request: Request) -> Any:
    return getattr(request.app.state, "session_store", None)


@router.get("/sessions")
async def list_sessions(request: Request, user_id: str = Query(..., min_length=1)) -> dict:
    store = _store(request)
    if store is None:
        return {"user_id": user_id, "sessions": []}
    return {"user_id": user_id, "sessions": await store.list_sessions(user_id)}


@router.get("/sessions/{session_id}/messages")
async def session_messages(session_id: str, request: Request) -> dict:
    store = _store(request)
    if store is None:
        return {"session_id": session_id, "messages": []}
    messages = await store.load(session_id)
    return {
        "session_id": session_id,
        "messages": [m.model_dump(mode="json", exclude_none=True) for m in messages],
    }


@router.delete("/sessions/{session_id}")
async def delete_session(
    session_id: str, request: Request, user_id: str | None = Query(None)
) -> dict:
    store = _store(request)
    if store is None:
        return {"session_id": session_id, "deleted": False}
    await store.delete(session_id, user_id=user_id)
    return {"session_id": session_id, "deleted": True}
