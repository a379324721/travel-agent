from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """应用与 Agent 运行时配置（自环境变量与 `.env` 加载）。"""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "travel-agent"
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4o"
    openai_embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536  # 需与 Milvus 集合维度一致
    database_url: str = "postgresql+asyncpg://user:pass@localhost:5432/travel_agent"
    redis_url: str = "redis://localhost:6379/0"
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    log_level: str = "INFO"

    # Agent config
    # 部分兼容端点（如 DashScope 思考模式）不支持强制指定 tool_choice 对象，可关掉退化为 auto
    llm_force_tool_choice: bool = True
    # 意图识别慢车道：规则置信不足时调 LLM 分类，每次触发多一轮 LLM 调用
    intent_slow_lane_enabled: bool = True
    intent_slow_lane_threshold: float = 0.82
    max_react_iterations: int = 10
    memory_window_size: int = 20
    memory_max_tokens: int = 8000
    memory_summary_token_threshold: int = 6000
    session_ttl_seconds: int = 604800

    # RAG config
    rag_top_k: int = 5
    rag_chunk_size: int = 800
    rag_chunk_overlap: int = 120

    # Circuit breaker config
    circuit_breaker_failure_threshold: int = 5
    circuit_breaker_recovery_timeout: int = 30
    circuit_breaker_half_open_max_calls: int = 3


settings = Settings()
