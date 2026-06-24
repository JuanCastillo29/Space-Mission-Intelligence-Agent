from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import get_embedder, get_session
from app.schemas.ingest import IngestRequest, IngestResponse
from ingestion.embedder import Embedder
from ingestion.pipeline import ingest_pdf

log = logging.getLogger(__name__)

router = APIRouter()


@router.post("/ingest", response_model=IngestResponse, status_code=201)
async def ingest_document(
    body: IngestRequest,
    session: AsyncSession = Depends(get_session),
    embedder: Embedder = Depends(get_embedder),
) -> IngestResponse:
    path = Path(body.file_path)

    if not path.exists():
        raise HTTPException(status_code=400, detail="File does not exist")
    if path.suffix.lower() != ".pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    try:
        ingested = await ingest_pdf(path, session, embedder)
    except Exception:
        log.exception("Ingestion failed for %s", path)
        raise HTTPException(status_code=500, detail="Ingestion failed")

    if not ingested:
        return IngestResponse(
            success=True,
            message="Skipped (already ingested)",
            document_title=path.stem,
        )

    return IngestResponse(
        success=True,
        message="Ingested successfully",
        document_title=path.stem,
    )
