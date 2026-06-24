from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import get_session
from app.schemas.documents import DocumentListResponse, DocumentOut
from db.models import Chunk, Document

router = APIRouter()


@router.get("/documents", response_model=DocumentListResponse)
async def list_documents(
    session: AsyncSession = Depends(get_session),
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> DocumentListResponse:
    total = await session.scalar(select(func.count()).select_from(Document)) or 0

    stmt = (
        select(Document, func.count(Chunk.id).label("chunk_count"))
        .outerjoin(Chunk)
        .group_by(Document.id)
        .order_by(Document.ingested_at.desc())
        .offset(offset)
        .limit(limit)
    )
    rows = (await session.execute(stmt)).all()

    docs = [
        DocumentOut(
            id=doc.id,
            title=doc.title,
            source_type=doc.source_type.value,
            source_url=doc.source_url,
            mission_name=doc.mission_name,
            chunk_count=chunk_count,
            ingested_at=doc.ingested_at,
            metadata=doc.metadata_,
        )
        for doc, chunk_count in rows
    ]

    return DocumentListResponse(documents=docs, total=total)
