from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import api_settings
from app.dependencies import get_session
from app.schemas.health import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check(
    request: Request,
    session: AsyncSession = Depends(get_session),
) -> HealthResponse:
    db_ok = False
    try:
        await session.execute(text("SELECT 1"))
        db_ok = True
    except Exception:
        pass

    state = request.app.state
    embedder_loaded = getattr(state, "embedder", None) is not None
    reranker_loaded = (
        getattr(state, "retrieval_pipeline", None) is not None
        and getattr(state.retrieval_pipeline, "_reranker", None) is not None
    )

    status = (
        "healthy" if (db_ok and embedder_loaded and reranker_loaded) else "degraded"
    )

    return HealthResponse(
        status=status,
        database=db_ok,
        embedder_loaded=embedder_loaded,
        reranker_loaded=reranker_loaded,
        version=api_settings.API_VERSION,
    )
