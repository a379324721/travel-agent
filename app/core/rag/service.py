"""政策知识库检索服务：embed 查询 → 多路召回 → 格式化为 LLM 上下文。"""

from __future__ import annotations

import asyncio
from typing import Any

from app.core.rag.retriever import MultiChannelRetriever
from app.services.embeddings import EmbeddingService
from app.services.milvus_store import MilvusDocumentStore


class _MilvusStoreVectorBackend:
    """把主线 MilvusDocumentStore（同步接口）适配成 retriever 的向量通道。"""

    def __init__(self, store: MilvusDocumentStore) -> None:
        self._store = store

    async def search_vectors(
        self, embedding: list[float], *, top_k: int, intent_filter: str | None = None
    ) -> list[dict[str, Any]]:
        rows = await asyncio.to_thread(self._store.search, embedding, top_k)
        return [
            {
                "id": r.get("id") or "",
                "text": r.get("content") or "",
                "score": float(r.get("score") or 0.0),
                "title": r.get("title"),
                "doc_type": r.get("doc_type"),
            }
            for r in rows
        ]


class _EmptyKeywordBackend:
    """关键词通道占位：暂无关键词索引，返回空。"""

    async def search(self, query: str, *, top_k: int) -> list[dict[str, Any]]:
        return []


class _EmptyIntentBackend:
    """意图通道占位：P2 接入意图识别后启用。"""

    async def search_by_intent(
        self, query: str, intent: str | None, *, top_k: int
    ) -> list[dict[str, Any]]:
        return []


class PolicyRAG:
    """chat 内政策问答的检索入口，知识库不可用时优雅降级。"""

    def __init__(
        self,
        store: MilvusDocumentStore,
        embedder: EmbeddingService | None = None,
        *,
        top_k: int = 5,
    ) -> None:
        self._store = store
        self._embedder = embedder or EmbeddingService()
        self._retriever = MultiChannelRetriever(
            milvus=_MilvusStoreVectorBackend(store),
            keyword_backend=_EmptyKeywordBackend(),
            intent_backend=_EmptyIntentBackend(),
        )
        self._top_k = top_k

    @property
    def available(self) -> bool:
        return bool(getattr(self._store, "connected", False))

    async def search_context(self, query: str) -> str:
        if not query.strip():
            return "（查询为空，无法检索制度文档。）"
        if not self.available:
            return "（知识库暂不可用，请基于通用差旅常识回答，并提醒用户以公司制度为准。）"
        embedding = await self._embedder.embed_text(query)
        chunks = await self._retriever.retrieve(query, embedding, top_k=self._top_k)
        if not chunks:
            return "（知识库中未找到相关制度条款。）"
        lines = [
            f"[{i + 1}]（{c.metadata.get('title') or '制度文档'}）{c.text}"
            for i, c in enumerate(chunks)
        ]
        header = "以下为公司差旅制度相关条款，请依据条款回答并注明「以公司制度为准」：\n"
        return header + "\n\n".join(lines)
