"""把 data/policies/ 下的制度文档分块入库（需 Milvus 与 OpenAI 密钥）。

用法: uv run python scripts/seed_policies.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings
from app.etl.pipeline import DocumentIngestionPipeline, IngestionConfig
from app.services.embeddings import EmbeddingService
from app.services.milvus_store import get_milvus_store, new_doc_id

POLICY_DIR = Path(__file__).resolve().parent.parent / "data" / "policies"


async def main() -> None:
    store = get_milvus_store()
    if not store.connect():
        print("Milvus 未连接，无法入库。请先启动依赖: docker compose up -d milvus-standalone")
        raise SystemExit(1)

    pipeline = DocumentIngestionPipeline(
        store,
        EmbeddingService(),
        config=IngestionConfig(
            chunk_size=settings.rag_chunk_size,
            chunk_overlap=settings.rag_chunk_overlap,
        ),
    )
    files = sorted(POLICY_DIR.glob("*.md"))
    if not files:
        print(f"{POLICY_DIR} 下没有 .md 文档")
        raise SystemExit(1)
    for path in files:
        chunks = await pipeline.run(
            new_doc_id(),
            path.read_text(encoding="utf-8"),
            title=path.stem,
            doc_type="policy",
        )
        print(f"{path.name}: {chunks} chunks 已入库")


if __name__ == "__main__":
    asyncio.run(main())
