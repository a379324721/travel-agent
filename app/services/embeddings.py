from __future__ import annotations

from openai import AsyncOpenAI

from app.config import settings


class EmbeddingService:
    def __init__(self, model: str = "text-embedding-3-small") -> None:
        self._client = AsyncOpenAI(
            api_key=settings.openai_api_key or "dummy",
            base_url=settings.openai_base_url,
        )
        self._model = model

    async def embed_text(self, text: str) -> list[float]:
        resp = await self._client.embeddings.create(model=self._model, input=text[:8000])
        return list(resp.data[0].embedding)

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        resp = await self._client.embeddings.create(
            model=self._model, input=[t[:8000] for t in texts]
        )
        return [list(d.embedding) for d in resp.data]
