# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 常用命令

```bash
uv sync                                          # 安装依赖（uv 管理 .venv）
uv run uvicorn app.main:app --reload --port 8000 # 启动服务
uv run pytest                                    # 全部测试
uv run pytest tests/test_p4_closed_loop.py::test_name  # 单个测试
uvx ruff check .                                 # lint（ruff 未装入 venv，用 uvx）
docker compose up -d postgres redis etcd minio milvus-standalone  # 依赖服务
uv run python scripts/seed_policies.py           # 差旅制度文档入库（需 Milvus + API key）
```

pytest 配置了 `asyncio_mode = "auto"`，async 测试不需要标记。ruff 规则在 pyproject.toml（line-length 100，E/F/I/UP）。

## 架构

FastAPI 差旅 AI Agent：对话编排 + 工具调用 + RAG + 长对话记忆。核心是 `app/agent/orchestrator.py` 的 `TravelOrchestrator`，请求流程：

1. `_prepare_thread`：从 Redis 加载会话历史 → 拼本轮消息 → 超阈值先 LLM 摘要压缩（`MemorySummarizer`）→ token 预算硬裁兜底（`ShortTermMemory`）
2. `_route` 意图路由：快车道规则引擎（`app/core/intent/rule_engine.py`）置信度不足时走慢车道 LLM 分类（`llm_classifier.py`）；POLICY/RAG 意图首轮强制 `tool_choice=search_travel_policy_docs`，并把慢车道改写出的检索词作为 system hint 追加
3. ReAct 工具循环（上限 `max_react_iterations`）：末轮 `tool_choice="none"` + 追加收尾指令，强制文字答复而非报错
4. `_persist_thread`：完整落库回 Redis（含 tool_calls 与 tool 结果）

**双路径必须保持行为一致**：`run_completion`（非流式）和 `stream_completion`（SSE）是并行实现，改一边要同步另一边（历史上多个 commit 在对齐两者）。

**缓存安全约定**：动态 system 消息（路由 hint、收尾指令）只追加到消息尾部，不改前缀，保住提示词缓存。

### 分层

- `app/main.py`：lifespan 中组装所有依赖。Redis/PG/Milvus 均可缺席优雅降级（session 持久化关闭 / booking 回退内存 / 健康检查 degraded），本地无 Docker 也能跑
- `app/agent/tools.py`：`build_default_registry` 注册全部 10 个业务工具到 `ToolRegistry`（`app/core/tools/registry.py`）
- `app/domain/travel/`：行程构建、差标规则与校验（纯业务逻辑）
- `app/services/`：LLM、嵌入、Milvus，以及 mock 适配层（`approval.py` OA 审批、`booking_store.py` 订票、`employee_directory.py` 员工目录）——接口设计为可替换真实系统
- `app/core/memory/`：`session_store.py`（Redis）、`short_term.py`（token 窗口）、`summary.py`（摘要压缩）
- `app/core/rag/` + `app/etl/`：文档分块 → 嵌入 → Milvus，多路召回检索
- `app/config.py`：pydantic-settings，从 `.env` 加载

### LLM 端点兼容性

实际联调用的是 DashScope 的 OpenAI 兼容端点。两个关联开关（`app/config.py`）：DashScope 思考模式不支持 `tool_choice` 对象，所以 `llm_enable_thinking=false` 才能用 `llm_force_tool_choice=true`（政策问题强制检索依赖它）。

## 测试约定

- 测试不依赖真实服务：每个测试文件自带 `FakeLLM`（按队列吐预设应答）、FakeStore 等
- `tests/conftest.py` 有 autouse fixture 全局关闭意图慢车道——否则慢车道会消耗 FakeLLM 的应答队列。新测试若要测慢车道，参考 `test_p2_intent_and_registry.py` 单独构造分类器
