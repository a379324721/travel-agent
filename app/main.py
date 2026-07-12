from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

import redis.asyncio as redis
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from sqlalchemy.ext.asyncio import AsyncEngine

from app.agent.orchestrator import TravelOrchestrator
from app.api.routes import chat as chat_routes
from app.api.routes import documents as documents_routes
from app.api.routes import health as health_routes
from app.api.routes import sessions as sessions_routes
from app.config import settings
from app.core.logging import configure_logging, get_logger
from app.core.memory.session_store import RedisSessionStore
from app.core.rag.service import PolicyRAG
from app.infrastructure.database.models import Base
from app.infrastructure.database.session import create_async_pg_engine, create_session_factory
from app.services.booking_store import PgBookingStore
from app.services.milvus_store import get_milvus_store

logger = get_logger(__name__)


def _setup_tracing() -> None:
    provider = TracerProvider(resource=Resource.create({"service.name": settings.app_name}))
    trace.set_tracer_provider(provider)


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging(settings.log_level)
    _setup_tracing()
    trace.get_tracer(__name__)

    app.state.redis = None
    try:
        app.state.redis = redis.from_url(settings.redis_url, decode_responses=True)
        await app.state.redis.ping()
        logger.info("redis.connected")
    except Exception as exc:
        logger.warning("redis.unavailable", error=str(exc))
        app.state.redis = None

    session_store = None
    if app.state.redis is not None:
        session_store = RedisSessionStore(
            app.state.redis, ttl_seconds=settings.session_ttl_seconds
        )
    else:
        logger.warning("session_store.disabled", reason="redis unavailable")
    app.state.session_store = session_store

    store = get_milvus_store()
    store.connect()
    app.state.milvus = store
    if store.connected:
        logger.info("milvus.connected")
    else:
        logger.warning("milvus.unavailable")

    app.state.db_engine: AsyncEngine | None = None
    app.state.db_sessionmaker = None
    booking_store = None
    try:
        app.state.db_engine = create_async_pg_engine(
            settings.database_url,
            pool_size=5,
            max_overflow=10,
        )
        app.state.db_sessionmaker = create_session_factory(app.state.db_engine)
        async with app.state.db_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        booking_store = PgBookingStore(app.state.db_sessionmaker)
        logger.info("database.ready", booking_store="postgres")
    except Exception as exc:
        logger.warning("database.unavailable_fallback_memory", error=str(exc))

    app.state.orchestrator = TravelOrchestrator(
        session_store=session_store,
        policy_rag=PolicyRAG(store, top_k=settings.rag_top_k),
        booking_store=booking_store,
    )

    yield

    if app.state.redis is not None:
        await app.state.redis.close()
    eng = getattr(app.state, "db_engine", None)
    if eng is not None:
        await eng.dispose()


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health_routes.router, prefix="/api/v1")
    app.include_router(chat_routes.router, prefix="/api/v1")
    app.include_router(documents_routes.router, prefix="/api/v1")
    app.include_router(sessions_routes.router, prefix="/api/v1")

    web_index = Path(__file__).resolve().parent.parent / "web" / "index.html"

    @app.get("/", include_in_schema=False, response_model=None)
    async def root() -> FileResponse | dict[str, str]:
        if web_index.exists():
            return FileResponse(web_index, media_type="text/html")
        return {"service": settings.app_name, "docs": "/docs"}

    return app


app = create_app()
