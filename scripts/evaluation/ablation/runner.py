from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from db import Chunk
from ingestion.embedder import Embedder
from scripts.evaluation.ablation.configs import AblationConfig
from scripts.generation.pipeline import GenerationPipeline
from scripts.generation.schemas import GenerationResult
from scripts.retrival.mmr import assemble_context, mmr_diversity_filter
from scripts.retrival.pipeline import RetrievalPipeline
from scripts.retrival.reranker import BGEReranker
from scripts.retrival.schemas import RetrievalResult, ScoredChunk
from scripts.retrival.search import hybrid_search, keyword_search, semantic_search


class NoOpReranker:
    """Pass-through reranker used for the ``no_reranking`` ablation.

    Mirrors the ``BGEReranker.rerank`` signature so it is a drop-in
    replacement, but keeps the fused ordering and simply truncates to
    ``top_k`` (matching the ``passthrough_rerank`` mock used in the tests).
    """

    def rerank(
        self,
        query: str,
        chunks: list[ScoredChunk],
        *,
        top_k: int = 5,
        batch_size: int = 16,
    ) -> list[ScoredChunk]:
        return list(chunks[:top_k])


class ConfigurableRetrievalPipeline(RetrievalPipeline):
    """A ``RetrievalPipeline`` whose search/rerank/filter behaviour is driven
    by an :class:`AblationConfig`.

    It overrides :meth:`run` to dispatch on ``search_mode`` and to inject
    metadata / chunk-size filters, while reusing the base class's embedding,
    title-fetch, and context-assembly helpers unchanged.
    """

    def __init__(
        self,
        config: AblationConfig,
        embedder: Embedder | None = None,
        reranker: BGEReranker | None = None,
    ) -> None:
        effective_reranker = reranker
        if not config.reranking_enabled:
            effective_reranker = NoOpReranker()  # type: ignore[assignment]
        super().__init__(embedder=embedder, reranker=effective_reranker)
        self.config = config

    def _build_filters(
        self, metadata_filter: dict[str, Any] | None
    ) -> list[Any]:
        filters: list[Any] = []

        if self.config.chunk_size_filter is not None:
            filters.append(
                Chunk.metadata_["eval_chunk_size"].as_integer()
                == self.config.chunk_size_filter
            )

        if self.config.metadata_filtering and metadata_filter:
            for key, value in metadata_filter.items():
                filters.append(Chunk.metadata_[key].as_string() == str(value))

        return filters

    async def run(  # type: ignore[override]
        self,
        query: str,
        session: AsyncSession,
        *,
        search_top_k: int | None = None,
        rerank_top_k: int | None = None,
        final_top_k: int | None = None,
        mmr_lambda: float | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> RetrievalResult:
        cfg = self.config
        search_top_k = search_top_k or cfg.search_top_k
        rerank_top_k = rerank_top_k or cfg.rerank_top_k
        final_top_k = final_top_k or cfg.final_top_k
        mmr_lambda = cfg.mmr_lambda if mmr_lambda is None else mmr_lambda

        query_embedding = self._embed_query(query)
        filters = self._build_filters(metadata_filter)

        if cfg.search_mode == "semantic_only":
            fused = await semantic_search(
                query_embedding, session, top_k=search_top_k, filters=filters
            )
        elif cfg.search_mode == "keyword_only":
            fused = await keyword_search(
                query, session, top_k=search_top_k, filters=filters
            )
        else:
            fused = await hybrid_search(
                query, query_embedding, session, top_k=search_top_k, filters=filters
            )

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


@dataclass
class QueryExecution:
    """Raw execution output for a single query under one ablation config.

    Metric computation (retrieval metrics, RAGAS, custom generation metrics)
    is intentionally left to the orchestrator so that RAGAS can be batched
    across the whole dataset.
    """

    query: str
    retrieval_result: RetrievalResult
    generation_result: GenerationResult | None

    @property
    def retrieved_chunk_ids(self) -> list[str]:
        return [str(block.chunk_id) for block in self.retrieval_result.blocks]

    @property
    def context_texts(self) -> list[str]:
        return [block.content for block in self.retrieval_result.blocks]


class AblationRunner:
    """Builds the configured pipelines for one :class:`AblationConfig` and
    executes queries against them.

    The retrieval pipeline is always configured. The generation pipeline is
    shared across configs (it is not part of the ablation matrix) and can be
    injected for testing or skipped entirely for retrieval-only runs.
    """

    def __init__(
        self,
        config: AblationConfig,
        *,
        embedder: Embedder | None = None,
        reranker: BGEReranker | None = None,
        generation_pipeline: GenerationPipeline | None = None,
    ) -> None:
        self.config = config
        self.retrieval_pipeline = ConfigurableRetrievalPipeline(
            config, embedder=embedder, reranker=reranker
        )
        self._generation_pipeline = generation_pipeline

    @property
    def generation_pipeline(self) -> GenerationPipeline:
        if self._generation_pipeline is None:
            self._generation_pipeline = GenerationPipeline()
        return self._generation_pipeline

    async def retrieve(
        self,
        query: str,
        session: AsyncSession,
        *,
        metadata_filter: dict[str, Any] | None = None,
    ) -> RetrievalResult:
        return await self.retrieval_pipeline.run(
            query, session, metadata_filter=metadata_filter
        )

    async def execute_query(
        self,
        query: str,
        session: AsyncSession,
        *,
        metadata_filter: dict[str, Any] | None = None,
        generate: bool = True,
    ) -> QueryExecution:
        retrieval_result = await self.retrieve(
            query, session, metadata_filter=metadata_filter
        )

        generation_result: GenerationResult | None = None
        if generate:
            generation_result = await self.generation_pipeline.generate(
                retrieval_result
            )

        return QueryExecution(
            query=query,
            retrieval_result=retrieval_result,
            generation_result=generation_result,
        )
