from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List

from app.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

COLLECTION_NAME = "travel_knowledge"
_EMBED_DIM = 1536


def _load_pymilvus() -> tuple[Any, ...]:
    from pymilvus import (
        Collection,
        CollectionSchema,
        DataType,
        FieldSchema,
        connections,
        utility,
    )

    return (Collection, CollectionSchema, DataType, FieldSchema, connections, utility)


@dataclass
class MilvusDocumentStore:
    host: str
    port: int
    collection_name: str = COLLECTION_NAME
    _collection: Any = None
    _connected: bool = field(default=False, init=False)

    @property
    def connected(self) -> bool:
        return self._connected

    def connect(self) -> bool:
        if self._connected and self._collection is not None:
            return True
        try:
            (
                MilvusCollection,
                MilvusCollectionSchema,
                DataType,
                FieldSchema,
                connections,
                utility,
            ) = _load_pymilvus()
            alias = "default"
            connections.connect(alias=alias, host=self.host, port=str(self.port))
            if not utility.has_collection(self.collection_name):
                fields = [
                    FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=64, is_primary=True),
                    FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=512),
                    FieldSchema(name="doc_type", dtype=DataType.VARCHAR, max_length=32),
                    FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
                    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=_EMBED_DIM),
                ]
                schema = MilvusCollectionSchema(fields, description="Business travel knowledge base")
                col = MilvusCollection(name=self.collection_name, schema=schema)
                index = {
                    "index_type": "IVF_FLAT",
                    "metric_type": "COSINE",
                    "params": {"nlist": 128},
                }
                col.create_index(field_name="embedding", index_params=index)
            self._collection = MilvusCollection(self.collection_name)
            self._collection.load()
            self._connected = True
            return True
        except Exception as exc:
            logger.warning("milvus.connect_failed", error=str(exc))
            self._connected = False
            self._collection = None
            return False

    def insert_vector(
        self,
        doc_id: str,
        title: str,
        doc_type: str,
        content: str,
        vector: List[float],
    ) -> None:
        if not self._collection:
            raise RuntimeError("Milvus not connected")
        self._collection.insert(
            [
                [doc_id],
                [title[:512]],
                [doc_type[:32]],
                [content[:65530]],
                [vector],
            ]
        )
        self._collection.flush()

    def insert_chunks(self, rows: List[Dict[str, Any]]) -> None:
        """批量写入分块。每行需含 id / title / doc_type / content / vector。"""
        if not self._collection:
            raise RuntimeError("Milvus not connected")
        self._collection.insert(
            [
                [r["id"] for r in rows],
                [r["title"][:512] for r in rows],
                [r["doc_type"][:32] for r in rows],
                [r["content"][:65530] for r in rows],
                [r["vector"] for r in rows],
            ]
        )
        self._collection.flush()

    def search(self, vector: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        if not self._collection:
            raise RuntimeError("Milvus not connected")
        self._collection.load()
        res = self._collection.search(
            data=[vector],
            anns_field="embedding",
            param={"metric_type": "COSINE", "params": {"nprobe": 16}},
            limit=top_k,
            output_fields=["title", "doc_type", "content"],
        )
        hits: List[Dict[str, Any]] = []
        for hit in res[0]:
            ent: Dict[str, Any] = {}
            raw = getattr(hit, "entity", None)
            if raw is not None:
                if hasattr(raw, "to_dict"):
                    ent = raw.to_dict()
                elif isinstance(raw, dict):
                    ent = raw
                else:
                    try:
                        ent = dict(raw)
                    except Exception:
                        ent = {}
            dist = getattr(hit, "distance", None)
            hits.append(
                {
                    "id": getattr(hit, "id", None),
                    "score": float(dist) if dist is not None else 0.0,
                    "title": ent.get("title"),
                    "doc_type": ent.get("doc_type"),
                    "content": str(ent.get("content") or "")[:2000],
                }
            )
        return hits


def get_milvus_store() -> MilvusDocumentStore:
    return MilvusDocumentStore(host=settings.milvus_host, port=settings.milvus_port)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def new_doc_id() -> str:
    return str(uuid.uuid4())
