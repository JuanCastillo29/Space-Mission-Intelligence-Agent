"""Diagnose where ground-truth chunks are lost in the retrieval pipeline."""
from __future__ import annotations

import asyncio
import json
from uuid import UUID

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from db import Chunk, async_session_factory
from ingestion.embedder import SentenceTransformerEmbedder
from scripts.evaluation.dataset import load_dataset, resolve_ground_truth_chunk_ids
from scripts.retrival.reranker import BGEReranker
from scripts.retrival.schemas import ScoredChunk
from scripts.retrival.search import (
    hybrid_search,
    keyword_search,
    reciprocal_rank_fusion,
    semantic_search,
)
from scripts.retrival.mmr import mmr_diversity_filter

DATASET_PATH = "scripts/evaluation/data/golden_qa.json"


def find_chunk_rank(chunk_id: UUID, results: list[ScoredChunk]) -> int | None:
    for i, sc in enumerate(results):
        if sc.chunk_id == chunk_id:
            return i + 1
    return None


async def diagnose():
    dataset = load_dataset(DATASET_PATH)
    embedder = SentenceTransformerEmbedder()
    reranker = BGEReranker()

    async with async_session_factory() as session:
        gt_map = await resolve_ground_truth_chunk_ids(dataset.pairs, session)

        for pair in dataset.pairs:
            gt_ids = gt_map.get(pair.query, [])
            if not gt_ids:
                continue

            print("=" * 90)
            print(f"QUERY: {pair.query}")
            print(f"GT chunks: {len(gt_ids)}")

            # Fetch GT chunk content previews
            for gt_id in gt_ids:
                row = await session.execute(
                    select(Chunk.content).where(Chunk.id == gt_id)
                )
                content = row.scalar()
                preview = (content or "")[:120].replace("\n", " ")
                print(f"  GT {gt_id}: {preview}...")

            query_embedding = embedder.embed([pair.query])[0]

            # Stage 1: Semantic search (top 100)
            sem_results = await semantic_search(
                query_embedding, session, top_k=100
            )
            # Stage 2: Keyword search (top 100)
            kw_results = await keyword_search(
                pair.query, session, top_k=100
            )
            # Stage 3: RRF fusion (top 20)
            fused = reciprocal_rank_fusion(
                sem_results, kw_results, top_k=20
            )
            # Stage 4: Reranker (top 10)
            reranked = reranker.rerank(pair.query, fused, top_k=10)
            # Stage 5: MMR (top 5)
            diverse = mmr_diversity_filter(reranked, top_k=5, lambda_param=0.7)

            print()
            for gt_id in gt_ids:
                sem_rank = find_chunk_rank(gt_id, sem_results)
                kw_rank = find_chunk_rank(gt_id, kw_results)
                fused_rank = find_chunk_rank(gt_id, fused)
                rerank_rank = find_chunk_rank(gt_id, reranked)
                mmr_rank = find_chunk_rank(gt_id, diverse)

                # Cosine similarity
                row = await session.execute(
                    select(Chunk.embedding).where(Chunk.id == gt_id)
                )
                gt_emb = row.scalar()
                if gt_emb and query_embedding:
                    import numpy as np
                    q = np.array(query_embedding)
                    g = np.array(gt_emb)
                    cos_sim = float(np.dot(q, g) / (np.linalg.norm(q) * np.linalg.norm(g)))
                else:
                    cos_sim = None

                status = "FOUND" if mmr_rank else "MISSED"
                print(f"  [{status}] {gt_id}")
                print(f"    cosine_sim={cos_sim:.4f}" if cos_sim else "    cosine_sim=N/A")
                print(f"    semantic={sem_rank or '>100'}/100  "
                      f"keyword={kw_rank or '>100'}/100  "
                      f"fused={fused_rank or '>20'}/20  "
                      f"reranked={rerank_rank or '>10'}/10  "
                      f"mmr={mmr_rank or '>5'}/5")

            # Show what IS in the top-5 for context
            print()
            print("  Top-5 actually retrieved:")
            for i, sc in enumerate(diverse, 1):
                is_gt = "GT" if sc.chunk_id in gt_ids else "  "
                preview = sc.content[:100].replace("\n", " ")
                print(f"    {i}. [{is_gt}] {sc.chunk_id} (score={sc.score:.4f})")
                print(f"         {preview}...")

            print()


if __name__ == "__main__":
    asyncio.run(diagnose())
