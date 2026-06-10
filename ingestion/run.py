from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from ingestion.pipeline import ingest_directory, ingest_pdf
from ingestion.embedder import SentenceTransformerEmbedder
from db import async_session_factory
from db.models import EMBEDDING_DIM


log = logging.getLogger("ingestion")


async def _ingest_single(
    path: Path,
    *,
    max_tokens: int,
    overlap_tokens: int,
    embed_batch_size: int,
) -> None:
    log.info("Loading embedding model (first run downloads ~1.3GB)...")
    embedder = SentenceTransformerEmbedder()
    if embedder.dim != EMBEDDING_DIM:
        sys.exit(f"Embedder dimension {embedder.dim} != expected {EMBEDDING_DIM}")

    async with async_session_factory() as session:
        ingested = await ingest_pdf(
            path,
            session,
            embedder,
            max_tokens=max_tokens,
            overlap_tokens=overlap_tokens,
            embed_batch_size=embed_batch_size,
        )
    if ingested:
        log.info("Done.")
    else:
        log.info("Skipped (already ingested).")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest PDF documents into the RAG database",
    )
    parser.add_argument(
        "path",
        nargs="?",
        default="data/pdfs",
        help="PDF file or directory of PDFs (default: data/pdfs)",
    )
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--overlap", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)-5s  %(message)s",
    )

    target = Path(args.path)

    if target.is_file() and target.suffix.lower() == ".pdf":
        asyncio.run(
            _ingest_single(
                target,
                max_tokens=args.max_tokens,
                overlap_tokens=args.overlap,
                embed_batch_size=args.batch_size,
            )
        )
    elif target.is_dir():
        report = asyncio.run(
            ingest_directory(
                target,
                max_tokens=args.max_tokens,
                overlap_tokens=args.overlap,
                embed_batch_size=args.batch_size,
            )
        )
        log.info(
            "Summary: %d ingested, %d skipped, %d failed out of %d",
            report.ingested,
            report.skipped,
            report.failed,
            report.total_files,
        )
        if report.errors:
            for err in report.errors:
                log.error("  %s", err)
    else:
        sys.exit(f"Error: {target} is not a PDF file or directory")


if __name__ == "__main__":
    main()
