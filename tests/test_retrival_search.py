"""Tests for semantic search, keyword search, and hybrid search — requires PostgreSQL with pgvector."""

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

from db.config import db_settings
from db.models import Chunk, Document, SourceType, EMBEDDING_DIM
from scripts.retrival.schemas import ScoredChunk
from scripts.retrival.search import (
    semantic_search,
    keyword_search,
    hybrid_search,
    _chunk_to_scored,
)


def _embedding(value: float = 0.0, index: int | None = None) -> list[float]:
    vec = [value] * EMBEDDING_DIM
    if index is not None:
        vec[index] = 1.0
    return vec


def _norm(vec: list[float]) -> list[float]:
    mag = sum(x * x for x in vec) ** 0.5
    if mag == 0:
        return vec
    return [x / mag for x in vec]


def _make_doc(session, checksum: str, title: str = "Test Doc") -> Document:
    doc = Document(title=title, source_type=SourceType.PDF, checksum=checksum)
    session.add(doc)
    session.flush()
    return doc


def _make_chunk(
    session,
    doc: Document,
    content: str,
    chunk_index: int,
    embedding: list[float] | None = None,
) -> Chunk:
    chunk = Chunk(
        document_id=doc.id,
        content=content,
        embedding=embedding,
        chunk_index=chunk_index,
        token_count=len(content.split()),
    )
    session.add(chunk)
    session.flush()
    session.expire(chunk)
    return chunk


@pytest_asyncio.fixture()
async def async_session():
    engine = create_async_engine(db_settings.async_url)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as session:
        async with session.begin():
            yield session
    await engine.dispose()


class TestChunkToScored:
    def test_maps_all_fields(self, session):
        doc = _make_doc(session, "cts1")
        emb = _norm(_embedding(0.1))
        chunk = _make_chunk(session, doc, "test content", 0, emb)

        sc = _chunk_to_scored(chunk, 0.95)

        assert sc.chunk_id == chunk.id
        assert sc.document_id == doc.id
        assert sc.content == "test content"
        assert sc.score == 0.95
        assert sc.chunk_index == 0
        assert sc.embedding is not None
        assert isinstance(sc, ScoredChunk)

    def test_handles_none_metadata(self, session):
        doc = _make_doc(session, "cts2")
        chunk = _make_chunk(session, doc, "no meta", 0)
        chunk.metadata_ = None
        sc = _chunk_to_scored(chunk, 0.5)
        assert sc.metadata_ == {}


async def _seed_doc_and_chunks(async_session, checksum, chunks_data):
    doc = Document(title="Test Doc", source_type=SourceType.PDF, checksum=checksum)
    async_session.add(doc)
    await async_session.flush()

    created = []
    for content, idx, emb in chunks_data:
        chunk = Chunk(
            document_id=doc.id,
            content=content,
            embedding=emb,
            chunk_index=idx,
            token_count=len(content.split()),
        )
        async_session.add(chunk)
        created.append(chunk)

    await async_session.flush()
    await async_session.execute(
        text("""
            UPDATE chunks SET search_vector = to_tsvector('english', content)
            WHERE search_vector IS NULL
        """)
    )
    await async_session.flush()
    return doc, created


class TestSemanticSearch:
    @pytest.mark.asyncio
    async def test_returns_results_ordered_by_similarity(self, async_session):
        close_emb = _norm(_embedding(0.0, index=0))
        far_emb = _norm(_embedding(0.0, index=1))
        await _seed_doc_and_chunks(async_session, "sem1", [
            ("close chunk", 0, close_emb),
            ("far chunk", 1, far_emb),
        ])

        query_emb = _norm(_embedding(0.0, index=0))
        results = await semantic_search(query_emb, async_session, top_k=10)
        assert len(results) >= 2
        assert results[0].score >= results[1].score

    @pytest.mark.asyncio
    async def test_top_k_limits_results(self, async_session):
        await _seed_doc_and_chunks(async_session, "sem2", [
            (f"chunk {i}", i, _norm(_embedding(0.1))) for i in range(5)
        ])

        results = await semantic_search(
            _norm(_embedding(0.1)), async_session, top_k=2
        )
        assert len(results) <= 2

    @pytest.mark.asyncio
    async def test_scores_are_positive(self, async_session):
        await _seed_doc_and_chunks(async_session, "sem3", [
            ("positive score", 0, _norm(_embedding(0.5))),
        ])

        results = await semantic_search(
            _norm(_embedding(0.5)), async_session, top_k=10
        )
        for r in results:
            assert r.score >= 0

    @pytest.mark.asyncio
    async def test_carries_embedding_through(self, async_session):
        emb = _norm(_embedding(0.2))
        await _seed_doc_and_chunks(async_session, "sem4", [
            ("with embedding", 0, emb),
        ])

        results = await semantic_search(
            _norm(_embedding(0.2)), async_session, top_k=10
        )
        assert len(results) >= 1
        assert results[0].embedding is not None


class TestKeywordSearch:
    @pytest.mark.asyncio
    async def test_finds_matching_chunks(self, async_session):
        await _seed_doc_and_chunks(async_session, "kw1", [
            ("thermal analysis of spacecraft propulsion system", 0, None),
            ("orbital mechanics calculation results", 1, None),
        ])

        results = await keyword_search("thermal spacecraft", async_session, top_k=10)
        assert len(results) >= 1
        assert any("thermal" in r.content for r in results)

    @pytest.mark.asyncio
    async def test_no_match_returns_empty(self, async_session):
        await _seed_doc_and_chunks(async_session, "kw2", [
            ("completely unrelated content about biology", 0, None),
        ])

        results = await keyword_search("xylophone quaternion", async_session, top_k=10)
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_top_k_limits_results(self, async_session):
        await _seed_doc_and_chunks(async_session, "kw3", [
            (f"spacecraft mission report number {i}", i, None) for i in range(5)
        ])

        results = await keyword_search("spacecraft mission", async_session, top_k=2)
        assert len(results) <= 2

    @pytest.mark.asyncio
    async def test_ranking_order(self, async_session):
        await _seed_doc_and_chunks(async_session, "kw4", [
            ("thermal thermal thermal thermal analysis", 0, None),
            ("one mention of thermal", 1, None),
        ])

        results = await keyword_search("thermal", async_session, top_k=10)
        if len(results) >= 2:
            assert results[0].score >= results[1].score


class TestHybridSearch:
    @pytest.mark.asyncio
    async def test_combines_semantic_and_keyword(self, async_session):
        emb = _norm(_embedding(0.0, index=0))
        await _seed_doc_and_chunks(async_session, "hyb1", [
            ("spacecraft thermal control subsystem design requirements", 0, emb),
            ("orbital debris mitigation strategy document", 1, _norm(_embedding(0.0, index=1))),
        ])

        query_emb = _norm(_embedding(0.0, index=0))
        results = await hybrid_search(
            "spacecraft thermal", query_emb, async_session, top_k=10
        )
        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_returns_scored_chunks(self, async_session):
        await _seed_doc_and_chunks(async_session, "hyb2", [
            ("propulsion system performance analysis", 0, _norm(_embedding(0.3))),
        ])

        results = await hybrid_search(
            "propulsion", _norm(_embedding(0.3)), async_session, top_k=5
        )
        for r in results:
            assert isinstance(r, ScoredChunk)
            assert r.score > 0

    @pytest.mark.asyncio
    async def test_top_k_respected(self, async_session):
        await _seed_doc_and_chunks(async_session, "hyb3", [
            (f"spacecraft telemetry data point {i}", i, _norm(_embedding(0.1)))
            for i in range(10)
        ])

        results = await hybrid_search(
            "spacecraft telemetry", _norm(_embedding(0.1)), async_session, top_k=3
        )
        assert len(results) <= 3
