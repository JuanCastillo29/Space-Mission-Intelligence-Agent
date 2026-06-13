"""Tests for MMR diversity filter and context assembly — no database required."""

import uuid

from scripts.retrival.mmr import (
    _cosine_similarity,
    assemble_context,
    mmr_diversity_filter,
)
from scripts.retrival.schemas import ScoredChunk


def _make_scored(
    *,
    chunk_id: uuid.UUID | None = None,
    document_id: uuid.UUID | None = None,
    score: float = 0.5,
    content: str = "text",
    embedding: list[float] | None = None,
    chunk_index: int = 0,
    section_path: str | None = None,
) -> ScoredChunk:
    return ScoredChunk(
        chunk_id=chunk_id or uuid.uuid4(),
        document_id=document_id or uuid.uuid4(),
        content=content,
        score=score,
        chunk_index=chunk_index,
        section_path=section_path,
        token_count=len(content.split()),
        embedding=embedding,
    )


class TestCosineSimilarity:
    def test_identical_vectors(self):
        v = [1.0, 0.0, 0.0]
        assert abs(_cosine_similarity(v, v) - 1.0) < 1e-6

    def test_orthogonal_vectors(self):
        a = [1.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0]
        assert abs(_cosine_similarity(a, b)) < 1e-6

    def test_opposite_vectors(self):
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert abs(_cosine_similarity(a, b) - (-1.0)) < 1e-6

    def test_zero_vector_returns_zero(self):
        assert _cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0

    def test_both_zero_vectors(self):
        assert _cosine_similarity([0.0, 0.0], [0.0, 0.0]) == 0.0


class TestMMRDiversityFilter:
    def test_empty_input(self):
        assert mmr_diversity_filter([]) == []

    def test_single_chunk(self):
        c = _make_scored(embedding=[1.0, 0.0])
        result = mmr_diversity_filter([c], top_k=5)
        assert len(result) == 1
        assert result[0].chunk_id == c.chunk_id

    def test_top_k_limits_output(self):
        chunks = [
            _make_scored(score=0.9 - i * 0.1, embedding=[float(i), 1.0])
            for i in range(6)
        ]
        result = mmr_diversity_filter(chunks, top_k=3)
        assert len(result) == 3

    def test_diverse_chunks_preferred_over_similar(self):
        similar_a = _make_scored(score=0.9, embedding=[1.0, 0.0, 0.0])
        similar_b = _make_scored(score=0.85, embedding=[0.99, 0.01, 0.0])
        diverse = _make_scored(score=0.8, embedding=[0.0, 0.0, 1.0])

        result = mmr_diversity_filter(
            [similar_a, similar_b, diverse], top_k=2, lambda_param=0.5
        )
        ids = {r.chunk_id for r in result}
        assert similar_a.chunk_id in ids
        assert diverse.chunk_id in ids

    def test_high_lambda_favors_relevance(self):
        high = _make_scored(score=0.95, embedding=[1.0, 0.0])
        mid = _make_scored(score=0.90, embedding=[0.99, 0.01])
        low = _make_scored(score=0.3, embedding=[0.0, 1.0])

        result = mmr_diversity_filter([high, mid, low], top_k=2, lambda_param=0.99)
        ids = [r.chunk_id for r in result]
        assert ids[0] == high.chunk_id
        assert ids[1] == mid.chunk_id

    def test_chunks_without_embedding_handled(self):
        with_emb = _make_scored(score=0.9, embedding=[1.0, 0.0])
        without_emb = _make_scored(score=0.8, embedding=None)

        result = mmr_diversity_filter([with_emb, without_emb], top_k=5)
        assert len(result) >= 1

    def test_all_chunks_without_embedding(self):
        chunks = [_make_scored(score=0.9 - i * 0.1) for i in range(4)]
        result = mmr_diversity_filter(chunks, top_k=2)
        assert len(result) == 2

    def test_first_selected_is_highest_scored(self):
        low = _make_scored(score=0.3, embedding=[1.0, 0.0])
        high = _make_scored(score=0.9, embedding=[0.0, 1.0])
        mid = _make_scored(score=0.6, embedding=[0.5, 0.5])

        result = mmr_diversity_filter([low, high, mid], top_k=3)
        assert result[0].chunk_id == high.chunk_id


class TestAssembleContext:
    def test_empty_chunks(self):
        result = assemble_context("query", [], {})
        assert result.context_text == ""
        assert result.blocks == []
        assert result.final_count == 0
        assert result.query == "query"

    def test_single_chunk(self):
        doc_id = uuid.uuid4()
        chunk = _make_scored(
            document_id=doc_id, content="Launch vehicle specs", section_path="Section 3"
        )
        titles = {str(doc_id): "Mission Report"}

        result = assemble_context(
            "launch specs",
            [chunk],
            titles,
            semantic_count=10,
            keyword_count=10,
            fused_count=8,
            reranked_count=5,
        )

        assert result.final_count == 1
        assert result.semantic_count == 10
        assert result.keyword_count == 10
        assert result.fused_count == 8
        assert result.reranked_count == 5
        assert "[1] Mission Report — Section 3" in result.context_text
        assert "Launch vehicle specs" in result.context_text
        assert result.blocks[0].source_title == "Mission Report"
        assert result.blocks[0].ref_index == 1
        assert result.blocks[0].section_path == "Section 3"

    def test_multiple_chunks_numbered_sequentially(self):
        doc_id = uuid.uuid4()
        chunks = [
            _make_scored(document_id=doc_id, content=f"Content {i}") for i in range(3)
        ]
        titles = {str(doc_id): "Doc"}

        result = assemble_context("q", chunks, titles)
        assert result.final_count == 3
        assert "[1] Doc" in result.context_text
        assert "[2] Doc" in result.context_text
        assert "[3] Doc" in result.context_text

    def test_missing_title_falls_back_to_unknown(self):
        chunk = _make_scored(content="orphan chunk")
        result = assemble_context("q", [chunk], {})
        assert "Unknown Source" in result.context_text
        assert result.blocks[0].source_title == "Unknown Source"

    def test_no_section_path_omits_dash(self):
        doc_id = uuid.uuid4()
        chunk = _make_scored(document_id=doc_id, content="text", section_path=None)
        titles = {str(doc_id): "Report"}

        result = assemble_context("q", [chunk], titles)
        assert "—" not in result.context_text
        assert "[1] Report" in result.context_text

    def test_blocks_carry_correct_ids(self):
        doc_id = uuid.uuid4()
        chunk = _make_scored(document_id=doc_id, score=0.77, content="data")
        titles = {str(doc_id): "T"}

        result = assemble_context("q", [chunk], titles)
        block = result.blocks[0]
        assert block.chunk_id == chunk.chunk_id
        assert block.document_id == doc_id
        assert abs(block.score - 0.77) < 1e-9
