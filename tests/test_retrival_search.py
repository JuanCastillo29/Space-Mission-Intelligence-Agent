"""Tests for semantic search, keyword search, and hybrid search — requires PostgreSQL with pgvector."""

import asyncio

import pytest
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


def _run_async(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


async def _async_session():
    engine = create_async_engine(db_settings.async_url)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as session:
        yield session
    await engine.dispose()


async def _with_async_session(fn):
    engine = create_async_engine(db_settings.async_url)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as session:
        result = await fn(session)
    await engine.dispose()
    return result


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


class TestSemanticSearch:
    def test_returns_results_ordered_by_similarity(self, session):
        doc = _make_doc(session, "sem1")
        close_emb = _norm(_embedding(0.0, index=0))
        far_emb = _norm(_embedding(0.0, index=1))
        _make_chunk(session, doc, "close chunk", 0, close_emb)
        _make_chunk(session, doc, "far chunk", 1, far_emb)
        session.commit()

        query_emb = _norm(_embedding(0.0, index=0))

        async def run(s):
            return await semantic_search(query_emb, s, top_k=10)

        results = _run_async(_with_async_session(run))
        assert len(results) >= 2
        assert results[0].score >= results[1].score

    def test_top_k_limits_results(self, session):
        doc = _make_doc(session, "sem2")
        for i in range(5):
            _make_chunk(session, doc, f"chunk {i}", i, _norm(_embedding(0.1)))
        session.commit()

        async def run(s):
            return await semantic_search(_norm(_embedding(0.1)), s, top_k=2)

        results = _run_async(_with_async_session(run))
        assert len(results) <= 2

    def test_scores_are_positive(self, session):
        doc = _make_doc(session, "sem3")
        _make_chunk(session, doc, "positive score", 0, _norm(_embedding(0.5)))
        session.commit()

        async def run(s):
            return await semantic_search(_norm(_embedding(0.5)), s, top_k=10)

        results = _run_async(_with_async_session(run))
        for r in results:
            assert r.score >= 0

    def test_carries_embedding_through(self, session):
        doc = _make_doc(session, "sem4")
        emb = _norm(_embedding(0.2))
        _make_chunk(session, doc, "with embedding", 0, emb)
        session.commit()

        async def run(s):
            return await semantic_search(_norm(_embedding(0.2)), s, top_k=10)

        results = _run_async(_with_async_session(run))
        assert len(results) >= 1
        assert results[0].embedding is not None


class TestKeywordSearch:
    def test_finds_matching_chunks(self, session):
        doc = _make_doc(session, "kw1")
        _make_chunk(session, doc, "thermal analysis of spacecraft propulsion system", 0)
        _make_chunk(session, doc, "orbital mechanics calculation results", 1)
        session.commit()

        async def run(s):
            return await keyword_search("thermal spacecraft", s, top_k=10)

        results = _run_async(_with_async_session(run))
        assert len(results) >= 1
        assert any("thermal" in r.content for r in results)

    def test_no_match_returns_empty(self, session):
        doc = _make_doc(session, "kw2")
        _make_chunk(session, doc, "completely unrelated content about biology", 0)
        session.commit()

        async def run(s):
            return await keyword_search("xylophone quaternion", s, top_k=10)

        results = _run_async(_with_async_session(run))
        assert len(results) == 0

    def test_top_k_limits_results(self, session):
        doc = _make_doc(session, "kw3")
        for i in range(5):
            _make_chunk(session, doc, f"spacecraft mission report number {i}", i)
        session.commit()

        async def run(s):
            return await keyword_search("spacecraft mission", s, top_k=2)

        results = _run_async(_with_async_session(run))
        assert len(results) <= 2

    def test_ranking_order(self, session):
        doc = _make_doc(session, "kw4")
        _make_chunk(session, doc, "thermal thermal thermal thermal analysis", 0)
        _make_chunk(session, doc, "one mention of thermal", 1)
        session.commit()

        async def run(s):
            return await keyword_search("thermal", s, top_k=10)

        results = _run_async(_with_async_session(run))
        if len(results) >= 2:
            assert results[0].score >= results[1].score


class TestHybridSearch:
    def test_combines_semantic_and_keyword(self, session):
        doc = _make_doc(session, "hyb1")
        emb = _norm(_embedding(0.0, index=0))
        _make_chunk(
            session, doc,
            "spacecraft thermal control subsystem design requirements", 0, emb,
        )
        _make_chunk(
            session, doc,
            "orbital debris mitigation strategy document", 1,
            _norm(_embedding(0.0, index=1)),
        )
        session.commit()

        query_emb = _norm(_embedding(0.0, index=0))

        async def run(s):
            return await hybrid_search("spacecraft thermal", query_emb, s, top_k=10)

        results = _run_async(_with_async_session(run))
        assert len(results) >= 1

    def test_returns_scored_chunks(self, session):
        doc = _make_doc(session, "hyb2")
        _make_chunk(
            session, doc,
            "propulsion system performance analysis", 0,
            _norm(_embedding(0.3)),
        )
        session.commit()

        async def run(s):
            return await hybrid_search(
                "propulsion", _norm(_embedding(0.3)), s, top_k=5,
            )

        results = _run_async(_with_async_session(run))
        for r in results:
            assert isinstance(r, ScoredChunk)
            assert r.score > 0

    def test_top_k_respected(self, session):
        doc = _make_doc(session, "hyb3")
        for i in range(10):
            _make_chunk(
                session, doc,
                f"spacecraft telemetry data point {i}", i,
                _norm(_embedding(0.1)),
            )
        session.commit()

        async def run(s):
            return await hybrid_search(
                "spacecraft telemetry", _norm(_embedding(0.1)), s, top_k=3,
            )

        results = _run_async(_with_async_session(run))
        assert len(results) <= 3
