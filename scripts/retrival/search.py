from __future__ import annotations

from collections.abc import Sequence
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.orm import selectinload

from db import Chunk
from scripts.retrival.schemas import ScoredChunk

RRF_K = 60


def _chunk_to_scored(chunk: Chunk, score: float) -> ScoredChunk:
    return ScoredChunk(
        chunk_id=chunk.id,
        document_id=chunk.document_id,
        content=chunk.content,
        score=score,
        chunk_index=chunk.chunk_index,
        section_path=chunk.section_path,
        token_count=chunk.token_count,
        metadata_=chunk.metadata_ or {},
        embedding=chunk.embedding,
    )


async def semantic_search(
    query_embedding: list[float],
    session,
    *,
    top_k: int = 20,
) -> list[ScoredChunk]:
    distance = Chunk.embedding.cosine_distance(query_embedding)
    score = 1 - distance

    stmt = (
        select(Chunk, score.label("score"))
        .options(selectinload(Chunk.document))
        .where(Chunk.embedding.is_not(None))
        .order_by(distance.asc())
        .limit(top_k)
    )

    result = await session.execute(stmt)

    return [_chunk_to_scored(chunk, float(sc)) for chunk, sc in result.all()]


async def keyword_search(
    query: str,
    session,
    *,
    top_k: int = 20,
) -> list[ScoredChunk]:
    ts_query = func.websearch_to_tsquery("english", query)

    rank = func.ts_rank_cd(Chunk.search_vector, ts_query)

    stmt = (
        select(Chunk, rank.label("score"))
        .options(selectinload(Chunk.document))
        .where(Chunk.search_vector.op("@@")(ts_query))
        .order_by(rank.desc())
        .limit(top_k)
    )

    result = await session.execute(stmt)

    return [_chunk_to_scored(chunk, float(sc)) for chunk, sc in result.all()]


def reciprocal_rank_fusion(
    semantic_results: Sequence[ScoredChunk],
    keyword_results: Sequence[ScoredChunk],
    *,
    k: int = RRF_K,
    top_k: int = 20,
) -> list[ScoredChunk]:
    scores: dict[UUID, float] = {}
    best: dict[UUID, ScoredChunk] = {}

    for rank, sc in enumerate(semantic_results, start=1):
        scores[sc.chunk_id] = scores.get(sc.chunk_id, 0) + 1 / (k + rank)
        best[sc.chunk_id] = sc

    for rank, sc in enumerate(keyword_results, start=1):
        scores[sc.chunk_id] = scores.get(sc.chunk_id, 0) + 1 / (k + rank)
        if sc.chunk_id not in best or best[sc.chunk_id].embedding is None:
            best[sc.chunk_id] = sc

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)

    return [
        best[chunk_id].model_copy(update={"score": rrf_score})
        for chunk_id, rrf_score in ranked[:top_k]
    ]


async def hybrid_search(
    query: str,
    query_embedding: list[float],
    session,
    *,
    top_k: int = 20,
) -> list[ScoredChunk]:
    semantic_results = await semantic_search(query_embedding, session, top_k=100)
    keyword_results = await keyword_search(query, session, top_k=100)

    return reciprocal_rank_fusion(
        semantic_results,
        keyword_results,
        top_k=top_k,
    )
