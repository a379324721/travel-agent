"""Document ingestion: parse → chunk → embed → Milvus."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from app.services.embeddings import EmbeddingService
from app.services.milvus_store import MilvusDocumentStore


@dataclass(slots=True)
class IngestionConfig:
    chunk_size: int = 800
    chunk_overlap: int = 120


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


class DocumentIngestionPipeline:
    """End-to-end ingestion into Milvus."""

    def __init__(
        self,
        store: MilvusDocumentStore,
        embedder: EmbeddingService,
        *,
        config: IngestionConfig | None = None,
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._cfg = config or IngestionConfig()

    async def run(
        self,
        doc_id: str,
        raw_text: str,
        *,
        title: str,
        doc_type: str = "policy",
    ) -> int:
        text = parse_plain_text(raw_text)
        pieces = chunk_text(text, chunk_size=self._cfg.chunk_size, overlap=self._cfg.chunk_overlap)
        if not pieces:
            return 0
        embeddings = await self._embedder.embed_texts(pieces)
        rows = [
            {
                "id": stable_chunk_id(doc_id, i, piece),
                "title": title,
                "doc_type": doc_type,
                "content": piece,
                "vector": vector,
            }
            for i, (piece, vector) in enumerate(zip(pieces, embeddings, strict=True))
        ]
        self._store.insert_chunks(rows)
        return len(rows)
