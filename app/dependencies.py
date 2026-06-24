from __future__ import annotations

from collections.abc import AsyncGenerator

from fastapi import Request
from sqlalchemy.ext.asyncio import AsyncSession

from db.session import get_session as _db_get_session
from ingestion.embedder import Embedder
from scripts.generation.pipeline import GenerationPipeline
from scripts.retrival.pipeline import RetrievalPipeline


async def get_session() -> AsyncGenerator[AsyncSession]:
    async for session in _db_get_session():
        yield session


def get_generation_pipeline(request: Request) -> GenerationPipeline:
    return request.app.state.generation_pipeline  # type: ignore[no-any-return]


def get_retrieval_pipeline(request: Request) -> RetrievalPipeline:
    return request.app.state.retrieval_pipeline  # type: ignore[no-any-return]


def get_embedder(request: Request) -> Embedder:
    return request.app.state.embedder  # type: ignore[no-any-return]
