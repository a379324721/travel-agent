from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Request

from app.config import settings
from app.domain.schemas import DocumentIngestRequest, DocumentIngestResponse
from app.etl.pipeline import DocumentIngestionPipeline, IngestionConfig
from app.services.embeddings import EmbeddingService
from app.services.milvus_store import new_doc_id, utc_now

router = APIRouter(tags=["documents"])


@router.post("/documents/ingest", response_model=DocumentIngestResponse)
async def ingest_document(body: DocumentIngestRequest, request: Request) -> DocumentIngestResponse:
    milvus = getattr(request.app.state, "milvus", None)
    if milvus is None or not milvus.connected:
        raise HTTPException(status_code=503, detail="Milvus 未连接，无法写入知识库")

    pipeline = DocumentIngestionPipeline(
        milvus,
        EmbeddingService(),
        config=IngestionConfig(
            chunk_size=settings.rag_chunk_size,
            chunk_overlap=settings.rag_chunk_overlap,
        ),
    )
    doc_id = new_doc_id()
    try:
        chunks = await pipeline.run(doc_id, body.content, title=body.title, doc_type=body.doc_type)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"向量库写入失败: {exc}") from exc

    return DocumentIngestResponse(
        doc_id=doc_id,
        collection=getattr(milvus, "collection_name", "travel_knowledge"),
        inserted_at=utc_now(),
        chunks=chunks,
    )


@router.get("/documents/search")
async def search_documents(
    request: Request,
    q: str = Query(..., min_length=1, description="查询文本"),
    top_k: int = Query(5, ge=1, le=20),
) -> dict:
    milvus = getattr(request.app.state, "milvus", None)
    if milvus is None or not milvus.connected:
        raise HTTPException(status_code=503, detail="Milvus 未连接，无法检索")

    embedder = EmbeddingService()
    vector = await embedder.embed_text(q)
    try:
        hits = milvus.search(vector, top_k=top_k)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"检索失败: {exc}") from exc

    return {"query": q, "results": hits}
