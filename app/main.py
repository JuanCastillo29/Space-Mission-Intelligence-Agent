from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import api_settings
from app.routes import documents, evaluate, health, ingest, query

logging.basicConfig(level=api_settings.LOG_LEVEL)
log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    log.info("Loading embedding model…")
    from ingestion.embedder import SentenceTransformerEmbedder

    embedder = SentenceTransformerEmbedder()
    app.state.embedder = embedder

    log.info("Loading reranker model…")
    from scripts.retrival.reranker import BGEReranker

    reranker = BGEReranker()

    from scripts.retrival.pipeline import RetrievalPipeline

    app.state.retrieval_pipeline = RetrievalPipeline(
        embedder=embedder, reranker=reranker
    )

    from scripts.generation.pipeline import GenerationPipeline

    app.state.generation_pipeline = GenerationPipeline()

    log.info("API ready.")
    yield
    log.info("Shutting down.")


def create_app() -> FastAPI:
    app = FastAPI(
        title=api_settings.API_TITLE,
        version=api_settings.API_VERSION,
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=api_settings.API_CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    prefix = "/api/v1"
    app.include_router(query.router, prefix=prefix, tags=["Query"])
    app.include_router(ingest.router, prefix=prefix, tags=["Ingestion"])
    app.include_router(documents.router, prefix=prefix, tags=["Documents"])
    app.include_router(health.router, prefix=prefix, tags=["Health"])
    app.include_router(evaluate.router, prefix=prefix, tags=["Evaluation"])

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=api_settings.API_HOST,
        port=api_settings.API_PORT,
        reload=True,
    )
