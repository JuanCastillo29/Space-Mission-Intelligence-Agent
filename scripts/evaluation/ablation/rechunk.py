"""Re-chunk the ingested corpus at alternative chunk sizes for the chunk-size
ablation.

For each requested ``chunk_size`` the source PDFs are re-extracted and
re-chunked with ``max_tokens=chunk_size``, then stored as *additional* chunk
rows attached to the existing ``Document`` (matched by title == ``pdf.stem``),
tagged with ``metadata_["eval_chunk_size"] = chunk_size``.

The ``ConfigurableRetrievalPipeline`` isolates each variant with a
``Chunk.metadata_["eval_chunk_size"].as_integer() == chunk_size`` filter, so
the tagged variants coexist with the natively-ingested (untagged) corpus
without polluting the ``baseline`` config.

Idempotent: a (document, chunk_size) pair that already has tagged chunks is
skipped, so the script is safe to re-run.

Usage:
    python -m scripts.evaluation.ablation.rechunk --pdf-dir data/pdfs 512 2048
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from db import Chunk, Document, async_session_factory
from db.models import EMBEDDING_DIM
from ingestion.embedder import Embedder, SentenceTransformerEmbedder
from scripts.ingestion.chunker import chunk_markdown
from scripts.ingestion.pdf_extractor import extract_pdf

log = logging.getLogger(__name__)

EVAL_CHUNK_SIZE_KEY = "eval_chunk_size"


async def _variant_exists(
    session: AsyncSession, document_id, chunk_size: int
) -> bool:
    stmt = (
        select(func.count())
        .select_from(Chunk)
        .where(
            Chunk.document_id == document_id,
            Chunk.metadata_[EVAL_CHUNK_SIZE_KEY].as_integer() == chunk_size,
        )
    )
    count = await session.scalar(stmt)
    return bool(count)


async def rechunk_document(
    document: Document,
    pdf_path: Path,
    chunk_size: int,
    session: AsyncSession,
    embedder: Embedder,
    *,
    overlap_tokens: int = 128,
    embed_batch_size: int = 64,
) -> int:
    """Re-chunk one document's PDF at ``chunk_size`` and store tagged chunks.

    Returns the number of chunks created (0 if the variant already existed).
    """
    if await _variant_exists(session, document.id, chunk_size):
        log.info(
            "Skipping %s @ %d (variant already present)", pdf_path.name, chunk_size
        )
        return 0

    markdown, _report = await asyncio.to_thread(extract_pdf, str(pdf_path))
    chunks = await asyncio.to_thread(
        chunk_markdown,
        markdown,
        max_tokens=chunk_size,
        overlap_tokens=overlap_tokens,
        metadata={EVAL_CHUNK_SIZE_KEY: chunk_size},
    )
    if not chunks:
        log.warning("No chunks produced for %s @ %d", pdf_path.name, chunk_size)
        return 0

    texts = [c.content for c in chunks]
    vectors: list[list[float]] = []
    for i in range(0, len(texts), embed_batch_size):
        batch = texts[i : i + embed_batch_size]
        vectors.extend(await asyncio.to_thread(embedder.embed, batch))

    for chunk_result, vector in zip(chunks, vectors):
        session.add(
            Chunk(
                document_id=document.id,
                content=chunk_result.content,
                embedding=vector,
                chunk_index=chunk_result.chunk_index,
                section_path=chunk_result.section_path,
                token_count=chunk_result.token_count,
                metadata_=chunk_result.metadata,
            )
        )

    await session.commit()
    log.info("Stored %s @ %d: %d chunks", pdf_path.name, chunk_size, len(chunks))
    return len(chunks)


async def rechunk_corpus(
    pdf_dir: Path,
    chunk_sizes: list[int],
    *,
    embedder: Embedder | None = None,
    overlap_tokens: int = 128,
) -> None:
    if embedder is None:
        log.info("Loading embedding model...")
        embedder = SentenceTransformerEmbedder()

    if embedder.dim != EMBEDDING_DIM:
        raise ValueError(
            f"Embedder dimension {embedder.dim} != expected {EMBEDDING_DIM}"
        )

    async with async_session_factory() as session:
        documents = (await session.execute(select(Document))).scalars().all()
        if not documents:
            log.warning("No documents in the database; run ingestion first.")
            return

        by_title = {doc.title: doc for doc in documents}

        for chunk_size in chunk_sizes:
            for pdf_path in sorted(pdf_dir.glob("*.pdf")):
                document = by_title.get(pdf_path.stem)
                if document is None:
                    log.warning(
                        "No ingested document matches %s (title %r); skipping",
                        pdf_path.name,
                        pdf_path.stem,
                    )
                    continue
                try:
                    await rechunk_document(
                        document,
                        pdf_path,
                        chunk_size,
                        session,
                        embedder,
                        overlap_tokens=overlap_tokens,
                    )
                except Exception as exc:
                    log.error(
                        "Failed to rechunk %s @ %d: %s", pdf_path.name, chunk_size, exc
                    )
                    await session.rollback()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    parser = argparse.ArgumentParser(
        description="Re-chunk the corpus at alternative chunk sizes for ablation."
    )
    parser.add_argument(
        "chunk_sizes",
        type=int,
        nargs="+",
        help="Chunk sizes (max_tokens) to produce, e.g. 512 2048",
    )
    parser.add_argument(
        "--pdf-dir",
        default="data/pdfs",
        help="Directory of source PDFs (default: data/pdfs)",
    )
    parser.add_argument("--overlap", type=int, default=128)
    args = parser.parse_args()

    asyncio.run(
        rechunk_corpus(
            Path(args.pdf_dir),
            args.chunk_sizes,
            overlap_tokens=args.overlap,
        )
    )


if __name__ == "__main__":
    main()
