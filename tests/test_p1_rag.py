"""P1 验收：ETL 分块入库、PolicyRAG 检索上下文、chat 内政策问答工具。"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from app.agent.orchestrator import TravelOrchestrator
from app.core.rag.service import PolicyRAG
from app.domain.schemas import ChatMessage, MessageRole
from app.etl.pipeline import DocumentIngestionPipeline, IngestionConfig, chunk_text


class FakeEmbedder:
    async def embed_text(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3]

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3] for _ in texts]


class FakeStore:
    collection_name = "travel_knowledge"

    def __init__(self, connected: bool = True, hits: list[dict[str, Any]] | None = None) -> None:
        self.connected = connected
        self.inserted: list[dict[str, Any]] = []
        self._hits = hits or []

    def insert_chunks(self, rows: list[dict[str, Any]]) -> None:
        self.inserted.extend(rows)

    def search(self, vector: list[float], top_k: int = 5) -> list[dict[str, Any]]:
        return self._hits[:top_k]


def test_chunk_text_overlap() -> None:
    text = "a" * 1000
    chunks = chunk_text(text, chunk_size=400, overlap=100)
    assert len(chunks) == 3
    assert len(chunks[0]) == 400
    # 相邻块应有重叠：第二块从 300 开始
    assert chunks[1] == text[300:700]


async def test_pipeline_chunks_and_inserts() -> None:
    store = FakeStore()
    pipeline = DocumentIngestionPipeline(
        store,  # type: ignore[arg-type]
        FakeEmbedder(),  # type: ignore[arg-type]
        config=IngestionConfig(chunk_size=100, chunk_overlap=20),
    )
    n = await pipeline.run("doc1", "差旅制度" * 100, title="差旅管理制度", doc_type="policy")
    assert n == len(store.inserted) > 1
    row = store.inserted[0]
    assert row["id"].startswith("chk_doc1_")
    assert row["title"] == "差旅管理制度"
    assert row["doc_type"] == "policy"
    assert row["vector"] == [0.1, 0.2, 0.3]
    # chunk id 稳定：同样输入两次入库 id 一致
    ids_first = [r["id"] for r in store.inserted]
    store.inserted.clear()
    await pipeline.run("doc1", "差旅制度" * 100, title="差旅管理制度")
    assert [r["id"] for r in store.inserted] == ids_first


async def test_policy_rag_formats_context() -> None:
    store = FakeStore(
        hits=[
            {"id": "1", "content": "一线城市每晚不超过800元。", "score": 0.9,
             "title": "差旅管理制度"},
            {"id": "2", "content": "STAFF 限经济舱。", "score": 0.8, "title": "差旅管理制度"},
        ]
    )
    rag = PolicyRAG(store, embedder=FakeEmbedder(), top_k=5)  # type: ignore[arg-type]
    ctx = await rag.search_context("酒店标准是多少")
    assert "800元" in ctx
    assert "差旅管理制度" in ctx
    assert "以公司制度为准" in ctx


async def test_policy_rag_degrades_when_unavailable() -> None:
    rag = PolicyRAG(FakeStore(connected=False), embedder=FakeEmbedder())  # type: ignore[arg-type]
    ctx = await rag.search_context("酒店标准")
    assert "知识库暂不可用" in ctx


def _tool_call_completion(name: str, arguments: str) -> Any:
    return SimpleNamespace(
        id="resp-tool",
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=None,
                    tool_calls=[
                        SimpleNamespace(
                            id="call-1",
                            function=SimpleNamespace(name=name, arguments=arguments),
                        )
                    ],
                ),
                finish_reason="tool_calls",
            )
        ],
        usage=None,
    )


def _final_completion(content: str) -> Any:
    return SimpleNamespace(
        id="resp-final",
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content, tool_calls=None),
                finish_reason="stop",
            )
        ],
        usage=None,
    )


class FakeLLM:
    model = "fake-model"

    def __init__(self, completions: list[Any]) -> None:
        self._completions = list(completions)
        self.seen_messages: list[list[dict[str, Any]]] = []

    async def chat_completion(self, messages: list[dict[str, Any]], **kwargs: Any) -> Any:
        self.seen_messages.append(messages)
        return self._completions.pop(0)


async def test_chat_policy_tool_returns_rag_context() -> None:
    store = FakeStore(
        hits=[{"id": "1", "content": "单次超5000元需OA事前审批。", "score": 0.9,
               "title": "差旅管理制度"}]
    )
    rag = PolicyRAG(store, embedder=FakeEmbedder())  # type: ignore[arg-type]
    llm = FakeLLM(
        [
            _tool_call_completion("search_travel_policy_docs", '{"query": "审批线是多少"}'),
            _final_completion("超过5000元需要OA事前审批（以公司制度为准）。"),
        ]
    )
    orch = TravelOrchestrator(llm=llm, policy_rag=rag)  # type: ignore[arg-type]

    result = await orch.run_completion(
        [ChatMessage(role=MessageRole.USER, content="多少钱要走审批?")]
    )
    assert "5000" in result["choices"][0]["message"]["content"]
    # 第二次 LLM 调用应携带工具返回的制度条款
    second = llm.seen_messages[1]
    tool_msgs = [m for m in second if m.get("role") == "tool"]
    assert tool_msgs and "OA事前审批" in tool_msgs[0]["content"]


async def test_chat_policy_tool_without_rag_configured() -> None:
    llm = FakeLLM(
        [
            _tool_call_completion("search_travel_policy_docs", '{"query": "差标"}'),
            _final_completion("建议咨询行政，以公司制度为准。"),
        ]
    )
    orch = TravelOrchestrator(llm=llm)  # type: ignore[arg-type]
    result = await orch.run_completion([ChatMessage(role=MessageRole.USER, content="差标?")])
    assert result["choices"][0]["message"]["content"]
    tool_msgs = [m for m in llm.seen_messages[1] if m.get("role") == "tool"]
    assert tool_msgs and "知识库未配置" in tool_msgs[0]["content"]
