"""Document ingestion: parse → chunk → embed → Milvus."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from openai import AsyncOpenAI

from app.infrastructure.vector.milvus_client import MilvusVectorClient


@dataclass(slots=True)
class IngestionConfig:
    chunk_size: int = 800
    chunk_overlap: int = 120
    embedding_model: str = "text-embedding-3-small"
    default_intent: str = "general"


def parse_plain_text(raw: str) -> str:
    text = raw.replace("\r\n", "\n").strip()
    return re.sub(r"\n{3,}", "\n\n", text)


def chunk_text(text: str, *, chunk_size: int, overlap: int) -> list[str]:
    if not text:
        return []
    chunks: list[str] = []
    start, n = 0, len(text)
    while start < n:
        end = min(start + chunk_size, n)
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= n:
            break
        start = max(end - overlap, start + 1)
    return chunks


def stable_chunk_id(doc_id: str, index: int, content: str) -> str:
    h = hashlib.sha256(f"{doc_id}:{index}:{content}".encode()).hexdigest()[:20]
    return f"chk_{doc_id}_{h}"


async def embed_openai(client: AsyncOpenAI, texts: list[str], *, model: str) -> list[list[float]]:
    resp = await client.embeddings.create(model=model, input=texts)
    return [list(d.embedding) for d in resp.data]


class DocumentIngestionPipeline:
    """End-to-end ingestion into Milvus."""

    def __init__(
        self,
        milvus: MilvusVectorClient,
        embedder: AsyncOpenAI,
        *,
        config: IngestionConfig | None = None,
    ) -> None:
        self._milvus = milvus
        self._embedder = embedder
        self._cfg = config or IngestionConfig()

    async def run(self, doc_id: str, raw_text: str, *, intent: str | None = None) -> int:
        text = parse_plain_text(raw_text)
        pieces = chunk_text(text, chunk_size=self._cfg.chunk_size, chunk_overlap=self._cfg.chunk_overlap)
        if not pieces:
            return 0
        intent_val = intent or self._cfg.default_intent
        embeddings = await embed_openai(self._embedder, pieces, model=self._cfg.embedding_model)
        chunk_ids = [stable_chunk_id(doc_id, i, p) for i, p in enumerate(pieces)]
        await self._milvus.insert_vectors(
            embeddings=embeddings,
            texts=pieces,
            chunk_ids=chunk_ids,
            intents=[intent_val] * len(pieces),
        )
        return len(pieces)
