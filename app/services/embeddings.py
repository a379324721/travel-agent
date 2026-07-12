from __future__ import annotations

from openai import AsyncOpenAI

from app.config import settings

# DashScope 等兼容端点对批量嵌入有条数限制，保守按 10 条一批
_BATCH_SIZE = 10


class EmbeddingService:
    def __init__(self, model: str | None = None) -> None:
        self._client = AsyncOpenAI(
            api_key=settings.openai_api_key or "dummy",
            base_url=settings.openai_base_url,
        )
        self._model = model or settings.openai_embedding_model
        self._dimensions = settings.embedding_dimensions

    async def embed_text(self, text: str) -> list[float]:
        resp = await self._client.embeddings.create(
            model=self._model, input=text[:8000], dimensions=self._dimensions
        )
        return list(resp.data[0].embedding)

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), _BATCH_SIZE):
            batch = [t[:8000] for t in texts[i : i + _BATCH_SIZE]]
            resp = await self._client.embeddings.create(
                model=self._model, input=batch, dimensions=self._dimensions
            )
            out.extend(list(d.embedding) for d in resp.data)
        return out
