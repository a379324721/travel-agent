from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from app.agent.orchestrator import TravelOrchestrator
from app.domain.schemas import ChatRequest, ChatResponse, StreamChunk, StreamChunkType

router = APIRouter(tags=["chat"])


def get_orchestrator(request: Request) -> TravelOrchestrator:
    return request.app.state.orchestrator


@router.post("/chat", response_model=None)
async def chat(
    body: ChatRequest,
    request: Request,
    orchestrator: TravelOrchestrator = Depends(get_orchestrator),
) -> ChatResponse | StreamingResponse:
    if body.stream:

        async def event_gen() -> AsyncIterator[str]:
            idx = 0
            try:
                async for chunk in orchestrator.stream_completion(
                    body.messages, session_id=body.session_id
                ):
                    payload = chunk.model_dump(mode="json")
                    yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                    idx += 1
            except Exception as exc:
                err = StreamChunk(
                    type=StreamChunkType.ERROR,
                    index=idx,
                    error=str(exc),
                )
                yield f"data: {json.dumps(err.model_dump(mode='json'), ensure_ascii=False)}\n\n"

        return StreamingResponse(event_gen(), media_type="text/event-stream")

    raw: dict[str, Any] = await orchestrator.run_completion(
        body.messages, session_id=body.session_id
    )
    return ChatResponse(
        id=raw["id"],
        created=raw["created"],
        model=raw["model"],
        choices=raw["choices"],
        usage=raw.get("usage"),
        session_id=body.session_id,
    )
