"""Tests for reciprocal rank fusion and schemas — no database required."""

import uuid

from scripts.retrival.schemas import (
    ContextBlock,
    RetrievalResult,
    ScoredChunk,
)
from scripts.retrival.search import reciprocal_rank_fusion, RRF_K


def _make_scored(
    *,
    chunk_id: uuid.UUID | None = None,
    score: float = 0.5,
    content: str = "text",
    embedding: list[float] | None = None,
) -> ScoredChunk:
    return ScoredChunk(
        chunk_id=chunk_id or uuid.uuid4(),
        document_id=uuid.uuid4(),
        content=content,
        score=score,
        chunk_index=0,
        token_count=1,
        embedding=embedding,
    )


class TestScoredChunk:
    def test_frozen(self):
        sc = _make_scored()
        import pytest

        with pytest.raises(Exception):
            sc.score = 0.9  # type: ignore[misc]

    def test_model_copy_updates_score(self):
        sc = _make_scored(score=0.5)
        updated = sc.model_copy(update={"score": 0.9})
        assert updated.score == 0.9
        assert sc.score == 0.5


class TestContextBlock:
    def test_creation(self):
        block = ContextBlock(
            ref_index=1,
            content="test",
            source_title="Doc",
            section_path=None,
            document_id=uuid.uuid4(),
            chunk_id=uuid.uuid4(),
            score=0.8,
        )
        assert block.ref_index == 1


class TestRetrievalResult:
    def test_creation(self):
        result = RetrievalResult(
            context_text="[1] test",
            blocks=[],
            query="test query",
            semantic_count=5,
            keyword_count=3,
            fused_count=6,
            reranked_count=5,
            final_count=5,
        )
        assert result.final_count == 5


class TestReciprocalRankFusion:
    def test_single_list_semantic_only(self):
        chunks = [_make_scored(score=0.9), _make_scored(score=0.7)]
        fused = reciprocal_rank_fusion(chunks, [], top_k=10)
        assert len(fused) == 2
        assert fused[0].score > fused[1].score

    def test_single_list_keyword_only(self):
        chunks = [_make_scored(score=0.9), _make_scored(score=0.7)]
        fused = reciprocal_rank_fusion([], chunks, top_k=10)
        assert len(fused) == 2

    def test_overlapping_chunks_get_boosted(self):
        shared_id = uuid.uuid4()
        sem = [_make_scored(chunk_id=shared_id, score=0.9)]
        kw = [_make_scored(chunk_id=shared_id, score=0.8)]
        only_sem_id = uuid.uuid4()
        sem.append(_make_scored(chunk_id=only_sem_id, score=0.8))

        fused = reciprocal_rank_fusion(sem, kw, top_k=10)

        scores = {sc.chunk_id: sc.score for sc in fused}
        assert scores[shared_id] > scores[only_sem_id]

    def test_rrf_score_formula(self):
        cid = uuid.uuid4()
        sem = [_make_scored(chunk_id=cid)]
        kw = [_make_scored(chunk_id=cid)]

        fused = reciprocal_rank_fusion(sem, kw, k=RRF_K, top_k=10)

        expected = 2 * (1 / (RRF_K + 1))
        assert abs(fused[0].score - expected) < 1e-9

    def test_top_k_truncation(self):
        chunks = [_make_scored() for _ in range(10)]
        fused = reciprocal_rank_fusion(chunks, [], top_k=3)
        assert len(fused) == 3

    def test_empty_inputs(self):
        fused = reciprocal_rank_fusion([], [], top_k=10)
        assert fused == []

    def test_prefers_version_with_embedding(self):
        cid = uuid.uuid4()
        doc_id = uuid.uuid4()
        sem = [
            ScoredChunk(
                chunk_id=cid,
                document_id=doc_id,
                content="text",
                score=0.9,
                chunk_index=0,
                token_count=1,
                embedding=[0.1] * 10,
            )
        ]
        kw = [
            ScoredChunk(
                chunk_id=cid,
                document_id=doc_id,
                content="text",
                score=0.8,
                chunk_index=0,
                token_count=1,
                embedding=None,
            )
        ]

        fused = reciprocal_rank_fusion(sem, kw, top_k=10)
        assert fused[0].embedding is not None

    def test_preserves_metadata(self):
        sc = _make_scored(content="specific content")
        fused = reciprocal_rank_fusion([sc], [], top_k=10)
        assert fused[0].content == "specific content"
        assert fused[0].chunk_id == sc.chunk_id

    def test_ordering_by_rrf_score(self):
        shared_id = uuid.uuid4()
        solo_id = uuid.uuid4()
        sem = [
            _make_scored(chunk_id=solo_id),
            _make_scored(chunk_id=shared_id),
        ]
        kw = [_make_scored(chunk_id=shared_id)]

        fused = reciprocal_rank_fusion(sem, kw, top_k=10)
        assert fused[0].chunk_id == shared_id
