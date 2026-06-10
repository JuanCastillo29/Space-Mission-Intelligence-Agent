from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from db import Document, Chunk, SourceType, async_session_factory
from db.models import EMBEDDING_DIM
from ingestion.checksum import compute_file_checksum
from ingestion.embedder import Embedder, SentenceTransformerEmbedder
from scripts.ingestion.chunker import chunk_markdown
from scripts.ingestion.pdf_extractor import extract_pdf

log = logging.getLogger(__name__)


@dataclass
class IngestReport:
    total_files: int = 0
    ingested: int = 0
    skipped: int = 0
    failed: int = 0
    errors: list[str] = field(default_factory=list)


async def ingest_pdf(
    pdf_path: Path,
    session: AsyncSession,
    embedder: Embedder,
    *,
    max_tokens: int = 1024,
    overlap_tokens: int = 128,
    embed_batch_size: int = 64,
) -> bool:
    """Ingest a single PDF. Returns True if ingested, False if skipped."""
    checksum = await asyncio.to_thread(compute_file_checksum, pdf_path)

    exists = await session.scalar(
        select(Document.id).where(Document.checksum == checksum)
    )
    if exists:
        log.info("Skipping %s (already ingested)", pdf_path.name)
        return False

    log.info("Extracting %s", pdf_path.name)
    markdown, report = await asyncio.to_thread(extract_pdf, str(pdf_path))

    log.info("Chunking (%d pages, quality %.2f)", report.total_pages, report.avg_quality)
    chunks = await asyncio.to_thread(
        chunk_markdown,
        markdown,
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
    )
    log.info("Generated %d chunks", len(chunks))

    texts = [c.content for c in chunks]
    all_vectors: list[list[float]] = []
    for i in range(0, len(texts), embed_batch_size):
        batch = texts[i : i + embed_batch_size]
        vectors = await asyncio.to_thread(embedder.embed, batch)
        all_vectors.extend(vectors)
        log.debug("Embedded batch %d/%d", i // embed_batch_size + 1, -(-len(texts) // embed_batch_size))

    doc = Document(
        title=pdf_path.stem,
        source_type=SourceType.PDF,
        checksum=checksum,
        metadata_={
            "total_pages": report.total_pages,
            "avg_quality": round(report.avg_quality, 3),
            "source_file": pdf_path.name,
        },
    )
    session.add(doc)
    await session.flush()

    for chunk_result, vector in zip(chunks, all_vectors):
        chunk = Chunk(
            document_id=doc.id,
            content=chunk_result.content,
            embedding=vector,
            chunk_index=chunk_result.chunk_index,
            section_path=chunk_result.section_path,
            token_count=chunk_result.token_count,
            metadata_=chunk_result.metadata,
        )
        session.add(chunk)

    await session.commit()
    log.info("Stored %s: %d chunks with embeddings", pdf_path.name, len(chunks))
    return True


async def ingest_directory(
    directory: Path,
    embedder: Embedder | None = None,
    *,
    max_tokens: int = 1024,
    overlap_tokens: int = 128,
    embed_batch_size: int = 64,
) -> IngestReport:
    if embedder is None:
        log.info("Loading embedding model (first run downloads ~1.3GB)...")
        embedder = SentenceTransformerEmbedder()

    if embedder.dim != EMBEDDING_DIM:
        raise ValueError(
            f"Embedder dimension {embedder.dim} != expected {EMBEDDING_DIM}"
        )

    pdfs = sorted(directory.glob("*.pdf"))
    if not pdfs:
        log.warning("No PDF files found in %s", directory)
        return IngestReport()

    report = IngestReport(total_files=len(pdfs))
    log.info("Found %d PDF(s) in %s", len(pdfs), directory)

    async with async_session_factory() as session:
        for pdf_path in pdfs:
            try:
                ingested = await ingest_pdf(
                    pdf_path,
                    session,
                    embedder,
                    max_tokens=max_tokens,
                    overlap_tokens=overlap_tokens,
                    embed_batch_size=embed_batch_size,
                )
                if ingested:
                    report.ingested += 1
                else:
                    report.skipped += 1
            except Exception as exc:
                report.failed += 1
                msg = f"{pdf_path.name}: {exc}"
                report.errors.append(msg)
                log.error("Failed to ingest %s: %s", pdf_path.name, exc)
                await session.rollback()

    return report
