from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """应用与 Agent 运行时配置（自环境变量与 `.env` 加载）。"""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "travel-agent"
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4o"
    database_url: str = "postgresql+asyncpg://user:pass@localhost:5432/travel_agent"
    redis_url: str = "redis://localhost:6379/0"
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    log_level: str = "INFO"

    # Agent config
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
