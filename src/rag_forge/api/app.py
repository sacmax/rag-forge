from contextlib import asynccontextmanager
from fastapi import FastAPI
from rag_forge.api.routes import health, ingest, query
from rag_forge.api.middleware import add_middleware
from rag_forge.api.dependencies import get_pipeline
import structlog

logger = structlog.get_logger()

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("building pipeline...")
    app.state.pipeline = get_pipeline()
    logger.info("pipeline ready")
    yield

def create_app() -> FastAPI:
    app = FastAPI(
        title="RAG Forge API",
        version="0.5.0",
        lifespan=lifespan
    )
    add_middleware(app)
    app.include_router(health.router, tags=["health"])
    app.include_router(ingest.router, tags=["ingest"])
    app.include_router(query.router, tags=["query"])
    return app

app = create_app()


