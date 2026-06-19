"""Tests for the generation pipeline — unit tests with mocked dependencies."""

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from scripts.generation.pipeline import GenerationPipeline
from scripts.generation.schemas import GenerationResult, QueryType
from scripts.retrival.schemas import ContextBlock, RetrievalResult


def _make_block(ref_index: int, title: str = "Doc") -> ContextBlock:
    return ContextBlock(
        ref_index=ref_index,
        content="chunk text",
        source_title=title,
        section_path=None,
        document_id=uuid.uuid4(),
        chunk_id=uuid.uuid4(),
        score=0.9,
    )


def _make_retrieval_result(
    query: str = "test query",
    blocks: list[ContextBlock] | None = None,
) -> RetrievalResult:
    blocks = blocks or [_make_block(1, "Source A"), _make_block(2, "Source B")]
    parts = [f"[{b.ref_index}] {b.source_title}\n{b.content}" for b in blocks]
    return RetrievalResult(
        context_text="\n\n".join(parts),
        blocks=blocks,
        query=query,
        semantic_count=20,
        keyword_count=20,
        fused_count=10,
        reranked_count=5,
        final_count=len(blocks),
    )


def _mock_client(response: str = "Answer [1] and [2].") -> AsyncMock:
    client = AsyncMock()
    client.complete.return_value = (
        response,
        {"prompt_tokens": 100, "completion_tokens": 50},
    )
    return client


def _mock_router(query_type: QueryType = QueryType.RETRIEVAL) -> AsyncMock:
    router = AsyncMock()
    router.classify.return_value = MagicMock(query_type=query_type)
    return router


class TestGenerationPipelineInit:
    def test_accepts_custom_dependencies(self):
        client = _mock_client()
        router = _mock_router()
        p = GenerationPipeline(client=client, router=router)
        assert p.client is client
        assert p.router is router

    def test_lazy_init(self):
        p = GenerationPipeline()
        assert p._client is None
        assert p._router is None


class TestGenerate:
    @pytest.mark.asyncio
    async def test_returns_generation_result(self):
        client = _mock_client("Fact from [1] and [2].")
        pipeline = GenerationPipeline(client=client)
        retrieval = _make_retrieval_result()

        result = await pipeline.generate(retrieval)

        assert isinstance(result, GenerationResult)
        assert result.query == "test query"
        assert result.query_type == QueryType.RETRIEVAL
        assert "[1]" in result.answer
        assert "[2]" in result.answer
        assert len(result.citations) == 2

    @pytest.mark.asyncio
    async def test_validates_citations(self):
        client = _mock_client("Fact [1] and hallucinated [9].")
        pipeline = GenerationPipeline(client=client)
        retrieval = _make_retrieval_result()

        result = await pipeline.generate(retrieval)

        assert "[1]" in result.answer
        assert "[9]" not in result.answer

    @pytest.mark.asyncio
    async def test_records_token_usage(self):
        client = _mock_client("Answer [1].")
        pipeline = GenerationPipeline(client=client)
        retrieval = _make_retrieval_result()

        result = await pipeline.generate(retrieval)

        assert result.prompt_tokens == 100
        assert result.completion_tokens == 50

    @pytest.mark.asyncio
    async def test_records_latency(self):
        client = _mock_client("Answer [1].")
        pipeline = GenerationPipeline(client=client)
        retrieval = _make_retrieval_result()

        result = await pipeline.generate(retrieval)

        assert result.latency_ms is not None
        assert result.latency_ms >= 0

    @pytest.mark.asyncio
    async def test_preserves_raw_llm_output(self):
        raw = "Raw [1] with ## Sources\n- Fake source"
        client = _mock_client(raw)
        pipeline = GenerationPipeline(client=client)
        retrieval = _make_retrieval_result()

        result = await pipeline.generate(retrieval)

        assert result.raw_llm_output == raw
        assert "Fake source" not in result.answer

    @pytest.mark.asyncio
    async def test_empty_context(self):
        client = _mock_client("I don't have sufficient information.")
        pipeline = GenerationPipeline(client=client)
        retrieval = RetrievalResult(
            context_text="",
            blocks=[],
            query="unknown topic",
            semantic_count=0,
            keyword_count=0,
            fused_count=0,
            reranked_count=0,
            final_count=0,
        )

        result = await pipeline.generate(retrieval)

        assert result.citations == []
        assert "sufficient information" in result.answer.lower()


class TestRun:
    @pytest.mark.asyncio
    async def test_routes_and_generates(self):
        client = _mock_client("Answer [1].")
        router = _mock_router(QueryType.RETRIEVAL)
        pipeline = GenerationPipeline(client=client, router=router)

        retrieval_result = _make_retrieval_result("space mission query")
        mock_retrieval_pipeline = AsyncMock()
        mock_retrieval_pipeline.run.return_value = retrieval_result

        session = AsyncMock()
        result = await pipeline.run(
            "space mission query",
            session,
            retrieval_pipeline=mock_retrieval_pipeline,
        )

        assert isinstance(result, GenerationResult)
        assert result.query_type == QueryType.RETRIEVAL
        router.classify.assert_called_once_with("space mission query")
        mock_retrieval_pipeline.run.assert_called_once()

    @pytest.mark.asyncio
    async def test_passes_query_type_from_router(self):
        client = _mock_client("Answer [1].")
        router = _mock_router(QueryType.HYBRID)
        pipeline = GenerationPipeline(client=client, router=router)

        retrieval_result = _make_retrieval_result()
        mock_retrieval_pipeline = AsyncMock()
        mock_retrieval_pipeline.run.return_value = retrieval_result

        session = AsyncMock()
        result = await pipeline.run(
            "hybrid query",
            session,
            retrieval_pipeline=mock_retrieval_pipeline,
        )

        assert result.query_type == QueryType.HYBRID
