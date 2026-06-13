from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from db import Document
from ingestion.embedder import Embedder, SentenceTransformerEmbedder
from scripts.retrival.mmr import assemble_context, mmr_diversity_filter
from scripts.retrival.reranker import BGEReranker
from scripts.retrival.schemas import RetrievalResult, ScoredChunk
from scripts.retrival.search import hybrid_search

SEARCH_TOP_K = 20
RERANK_TOP_K = 10
FINAL_TOP_K = 5
MMR_LAMBDA = 0.7


class RetrievalPipeline:
    def __init__(
        self,
        embedder: Embedder | None = None,
        reranker: BGEReranker | None = None,
    ):
        self._embedder = embedder
        self._reranker = reranker

    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = SentenceTransformerEmbedder()
        return self._embedder

    @property
    def reranker(self) -> BGEReranker:
        if self._reranker is None:
            self._reranker = BGEReranker()
        return self._reranker

    def _embed_query(self, query: str) -> list[float]:
        return self.embedder.embed([query])[0]

    async def _fetch_source_titles(
        self, session: AsyncSession, chunks: list[ScoredChunk]
    ) -> dict[str, str]:
        doc_ids = {c.document_id for c in chunks}
        if not doc_ids:
            return {}

        stmt = select(Document.id, Document.title).where(Document.id.in_(doc_ids))
        rows = await session.execute(stmt)
        return {str(doc_id): title for doc_id, title in rows.all()}

    async def run(
        self,
        query: str,
        session: AsyncSession,
        *,
        search_top_k: int = SEARCH_TOP_K,
        rerank_top_k: int = RERANK_TOP_K,
        final_top_k: int = FINAL_TOP_K,
        mmr_lambda: float = MMR_LAMBDA,
    ) -> RetrievalResult:
        query_embedding = self._embed_query(query)

        fused = await hybrid_search(query, query_embedding, session, top_k=search_top_k)

        reranked = self.reranker.rerank(query, fused, top_k=rerank_top_k)

        diverse = mmr_diversity_filter(
            reranked, top_k=final_top_k, lambda_param=mmr_lambda
        )

        source_titles = await self._fetch_source_titles(session, diverse)

        return assemble_context(
            query,
            diverse,
            source_titles,
            semantic_count=search_top_k,
            keyword_count=search_top_k,
            fused_count=len(fused),
            reranked_count=len(reranked),
        )
