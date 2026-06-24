from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import get_generation_pipeline, get_session
from app.schemas.query import CitationOut, QueryRequest, QueryResponse
from scripts.generation.pipeline import GenerationPipeline

log = logging.getLogger(__name__)

router = APIRouter()


@router.post("/query", response_model=QueryResponse)
async def submit_query(
    body: QueryRequest,
    session: AsyncSession = Depends(get_session),
    pipeline: GenerationPipeline = Depends(get_generation_pipeline),
) -> QueryResponse:
    try:
        result = await pipeline.run(body.question, session)
    except Exception:
        log.exception("Generation pipeline failed for query: %s", body.question)
        raise HTTPException(status_code=502, detail="LLM generation failed")

    return QueryResponse(
        answer=result.answer,
        citations=[CitationOut(**c.model_dump()) for c in result.citations],
        sources_section=result.sources_section,
        query_type=result.query_type.value,
        model_name=result.model_name,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        latency_ms=result.latency_ms,
    )
