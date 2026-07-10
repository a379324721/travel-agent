# travel-agent

基于 FastAPI 的差旅 AI Agent 服务，覆盖对话编排、工具调用、RAG 向量检索、长对话记忆等核心能力。

## 功能

- **对话与工具调用**：ReAct 风格多轮推理，内置行程草稿、差标校验等工具
- **RAG**：文本嵌入（OpenAI Embeddings）写入 Milvus，支持相似度检索与重排
- **对话记忆**：短期消息窗口 + 长对话摘要
- **工程化**：健康检查（Redis / PostgreSQL / Milvus）、熔断器、结构化日志、OpenTelemetry

## 架构

```
┌─────────────┐     ┌──────────────────┐     ┌────────────────────┐
│  Client/UI  │────▶│  FastAPI         │────▶│ TravelOrchestrator │
└─────────────┘     │  /api/v1/chat    │     │  + LLM + Tools     │
                    │  /health         │     └────────────────────┘
                    │  /documents/*    │            │
                    └────────┬─────────┘            ▼
                             │               ┌───────────────┐
                    ┌────────┴──────────┐    │ itinerary /   │
                    │ Redis │ PG │ Milvus│    │ policy 领域   │
                    └───────────────────┘    └───────────────┘
```

- **应用层**：`app/main.py` 注册路由与生命周期（连接池、向量库）
- **编排层**：`app/agent/orchestrator.py` 汇总消息窗口、摘要长对话、调用 LLM 与工具
- **Agent 核心**：`app/core/agent/`（ReAct、planner、reflection）、`app/core/tools/`、`app/core/memory/`、`app/core/rag/`
- **领域层**：`app/domain/travel/` 行程构建、差标规则与校验
- **基础设施**：`app/infrastructure/`（数据库、缓存、LLM 客户端、可观测性）、`app/services/`

## 环境要求

- Python 3.11+，[uv](https://docs.astral.sh/uv/)
- PostgreSQL、Redis、Milvus 2.x（可选；未启动时健康检查为 degraded，文档接口可能返回 503）
- 兼容 OpenAI API 的密钥与 `base_url`

## 安装与运行

```bash
uv sync                     # 创建 .venv 并按 uv.lock 安装依赖
cp .env.example .env        # 填 OPENAI_API_KEY 等
uv run uvicorn app.main:app --reload --port 8000
```

- Swagger UI：<http://127.0.0.1:8000/docs>
- ReDoc：<http://127.0.0.1:8000/redoc>

依赖服务可用 Docker 一键起：`docker compose up -d postgres redis etcd minio milvus-standalone`

## API

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/` | 服务名与文档链接 |
| GET | `/api/v1/health` | 依赖健康状态 |
| POST | `/api/v1/chat` | 对话；`stream: true` 时返回 SSE |
| POST | `/api/v1/documents/ingest` | 文档入库（需 Milvus） |
| GET | `/api/v1/documents/search` | 向量检索 |

### POST `/api/v1/chat`

```json
{
  "messages": [
    { "role": "user", "content": "下周从北京去上海出差一天，帮我估费用并看差标。" }
  ],
  "stream": false,
  "session_id": "optional-session-id"
}
```

- `stream: false`：返回 JSON，结构与 OpenAI Chat Completions 类似（`choices[0].message.content`）
- `stream: true`：`text/event-stream`，每行 `data: {JSON}`，含 `StreamChunk`（`content` / `done` / `error`）

## 配置

见 `.env.example`：`OPENAI_*`、`DATABASE_URL`、`REDIS_URL`、`MILVUS_*`、`LOG_LEVEL`。Agent 相关阈值（窗口、摘要、熔断）在 `app/config.py`。

## 测试

```bash
uv run pytest
```
