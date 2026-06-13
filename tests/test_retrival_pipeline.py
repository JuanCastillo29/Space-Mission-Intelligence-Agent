"""Tests for the full retrieval pipeline — unit tests with mocked dependencies."""

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from scripts.retrival.pipeline import RetrievalPipeline
from scripts.retrival.schemas import RetrievalResult, ScoredChunk


def _make_scored(
    *,
    chunk_id: uuid.UUID | None = None,
    document_id: uuid.UUID | None = None,
    score: float = 0.5,
    content: str = "text",
    embedding: list[float] | None = None,
) -> ScoredChunk:
    return ScoredChunk(
        chunk_id=chunk_id or uuid.uuid4(),
        document_id=document_id or uuid.uuid4(),
        content=content,
        score=score,
        chunk_index=0,
        token_count=len(content.split()),
        embedding=embedding or [0.1] * 10,
    )


@pytest.fixture()
def mock_embedder():
    embedder = MagicMock()
    embedder.embed.return_value = [[0.1] * 10]
    return embedder


@pytest.fixture()
def mock_reranker():
    reranker = MagicMock()

    def passthrough_rerank(query, chunks, *, top_k=5, batch_size=16):
        return chunks[:top_k]

    reranker.rerank.side_effect = passthrough_rerank
    return reranker


@pytest.fixture()
def pipeline(mock_embedder, mock_reranker):
    return RetrievalPipeline(embedder=mock_embedder, reranker=mock_reranker)


class TestRetrievalPipelineInit:
    def test_accepts_custom_dependencies(self, mock_embedder, mock_reranker):
        p = RetrievalPipeline(embedder=mock_embedder, reranker=mock_reranker)
        assert p.embedder is mock_embedder
        assert p.reranker is mock_reranker

    def test_lazy_init_embedder(self):
        p = RetrievalPipeline()
        assert p._embedder is None

    def test_lazy_init_reranker(self):
        p = RetrievalPipeline()
        assert p._reranker is None


class TestEmbedQuery:
    def test_delegates_to_embedder(self, pipeline, mock_embedder):
        result = pipeline._embed_query("test query")
        mock_embedder.embed.assert_called_once_with(["test query"])
        assert result == [0.1] * 10


class TestPipelineRun:
    @pytest.mark.asyncio
    async def test_returns_retrieval_result(self, pipeline):
        doc_id = uuid.uuid4()
        chunks = [
            _make_scored(document_id=doc_id, score=0.9, content="chunk a"),
            _make_scored(document_id=doc_id, score=0.7, content="chunk b"),
        ]
        session = AsyncMock()

        with patch(
            "scripts.retrival.pipeline.hybrid_search", new_callable=AsyncMock
        ) as mock_search:
            mock_search.return_value = chunks
            mock_rows = MagicMock()
            mock_rows.all.return_value = [(doc_id, "Test Doc")]
            session.execute.return_value = mock_rows

            result = await pipeline.run("test query", session)

        assert isinstance(result, RetrievalResult)
        assert result.query == "test query"
        assert result.final_count > 0

    @pytest.mark.asyncio
    async def test_calls_hybrid_search_with_correct_args(self, pipeline):
        session = AsyncMock()
        mock_rows = MagicMock()
        mock_rows.all.return_value = []
        session.execute.return_value = mock_rows

        with patch(
            "scripts.retrival.pipeline.hybrid_search", new_callable=AsyncMock
        ) as mock_search:
            mock_search.return_value = []
            await pipeline.run("space mission", session, search_top_k=15)

            mock_search.assert_called_once()
            call_kwargs = mock_search.call_args
            assert call_kwargs[0][0] == "space mission"
            assert call_kwargs[1]["top_k"] == 15

    @pytest.mark.asyncio
    async def test_calls_reranker(self, pipeline, mock_reranker):
        doc_id = uuid.uuid4()
        chunks = [_make_scored(document_id=doc_id, content="data")]
        session = AsyncMock()
        mock_rows = MagicMock()
        mock_rows.all.return_value = [(doc_id, "Doc")]
        session.execute.return_value = mock_rows

        with patch(
            "scripts.retrival.pipeline.hybrid_search", new_callable=AsyncMock
        ) as mock_search:
            mock_search.return_value = chunks
            await pipeline.run("query", session, rerank_top_k=8)

        mock_reranker.rerank.assert_called_once()
        assert mock_reranker.rerank.call_args[1]["top_k"] == 8

    @pytest.mark.asyncio
    async def test_empty_search_results(self, pipeline):
        session = AsyncMock()
        mock_rows = MagicMock()
        mock_rows.all.return_value = []
        session.execute.return_value = mock_rows

        with patch(
            "scripts.retrival.pipeline.hybrid_search", new_callable=AsyncMock
        ) as mock_search:
            mock_search.return_value = []
            result = await pipeline.run("nothing here", session)

        assert result.final_count == 0
        assert result.context_text == ""
        assert result.blocks == []

    @pytest.mark.asyncio
    async def test_custom_top_k_params_flow_through(self, pipeline):
        session = AsyncMock()
        mock_rows = MagicMock()
        mock_rows.all.return_value = []
        session.execute.return_value = mock_rows

        with patch(
            "scripts.retrival.pipeline.hybrid_search", new_callable=AsyncMock
        ) as mock_search:
            mock_search.return_value = []
            result = await pipeline.run(
                "q", session, search_top_k=30, rerank_top_k=15, final_top_k=7
            )

        assert result.semantic_count == 30
        assert result.keyword_count == 30

    @pytest.mark.asyncio
    async def test_source_titles_resolved(self, pipeline):
        doc_id = uuid.uuid4()
        chunks = [_make_scored(document_id=doc_id, content="propulsion data")]
        session = AsyncMock()
        mock_rows = MagicMock()
        mock_rows.all.return_value = [(doc_id, "Propulsion Manual")]
        session.execute.return_value = mock_rows

        with patch(
            "scripts.retrival.pipeline.hybrid_search", new_callable=AsyncMock
        ) as mock_search:
            mock_search.return_value = chunks
            result = await pipeline.run("propulsion", session)

        assert "Propulsion Manual" in result.context_text
