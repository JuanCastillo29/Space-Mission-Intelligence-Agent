"""Tests for citation extraction, validation, and post-processing."""

import uuid

from scripts.generation.citations import (
    build_sources_section,
    extract_citation_refs,
    postprocess_response,
    validate_citations,
)
from scripts.retrival.schemas import ContextBlock


def _make_block(
    ref_index: int,
    title: str = "Doc",
    section: str | None = None,
) -> ContextBlock:
    return ContextBlock(
        ref_index=ref_index,
        content="chunk text",
        source_title=title,
        section_path=section,
        document_id=uuid.uuid4(),
        chunk_id=uuid.uuid4(),
        score=0.9,
    )


class TestExtractCitationRefs:
    def test_basic(self):
        assert extract_citation_refs("claim [1] and [2]") == {1, 2}

    def test_adjacent(self):
        assert extract_citation_refs("claim [1][2]") == {1, 2}

    def test_no_citations(self):
        assert extract_citation_refs("no citations here") == set()

    def test_duplicate_refs(self):
        assert extract_citation_refs("[1] and again [1]") == {1}

    def test_multi_digit(self):
        assert extract_citation_refs("ref [12] noted") == {12}


class TestValidateCitations:
    def test_keeps_valid_refs(self):
        blocks = [_make_block(1), _make_block(2)]
        text = "Fact [1] and detail [2]."
        result = validate_citations(text, blocks)
        assert "[1]" in result
        assert "[2]" in result

    def test_strips_hallucinated_refs(self):
        blocks = [_make_block(1), _make_block(2)]
        text = "Fact [1] and fake [5]."
        result = validate_citations(text, blocks)
        assert "[1]" in result
        assert "[5]" not in result

    def test_strips_all_invalid(self):
        blocks = [_make_block(1)]
        text = "Fake [3] and [7]."
        result = validate_citations(text, blocks)
        assert "[3]" not in result
        assert "[7]" not in result

    def test_empty_text(self):
        assert validate_citations("", [_make_block(1)]) == ""

    def test_no_blocks(self):
        result = validate_citations("claim [1].", [])
        assert "[1]" not in result


class TestBuildSourcesSection:
    def test_builds_from_used_refs(self):
        blocks = [_make_block(1, "Rosetta Report"), _make_block(2, "ECSS Standard")]
        text = "Data from [1] and [2]."
        sources, citations = build_sources_section(text, blocks)

        assert "## Sources" in sources
        assert "Rosetta Report" in sources
        assert "ECSS Standard" in sources
        assert len(citations) == 2

    def test_includes_section_path(self):
        blocks = [_make_block(1, "Report", "Section 3.2")]
        text = "Detail [1]."
        sources, _ = build_sources_section(text, blocks)
        assert "Section 3.2" in sources

    def test_skips_unused_refs(self):
        blocks = [_make_block(1, "Used"), _make_block(2, "Unused")]
        text = "Only [1] here."
        sources, citations = build_sources_section(text, blocks)
        assert "Used" in sources
        assert "Unused" not in sources
        assert len(citations) == 1

    def test_no_citations_in_text(self):
        blocks = [_make_block(1)]
        sources, citations = build_sources_section("no refs", blocks)
        assert citations == []


class TestPostprocessResponse:
    def test_strips_llm_sources_section(self):
        blocks = [_make_block(1, "Real Source")]
        raw = "Answer [1].\n\n## Sources\n- Hallucinated source"
        answer, citations, _ = postprocess_response(raw, blocks)
        assert "Hallucinated source" not in answer
        assert "Real Source" in answer
        assert len(citations) == 1

    def test_validates_and_rebuilds(self):
        blocks = [_make_block(1, "Doc A"), _make_block(2, "Doc B")]
        raw = "Fact [1], fake [9], detail [2]."
        answer, citations, sources = postprocess_response(raw, blocks)
        assert "[9]" not in answer
        assert "[1]" in answer
        assert "[2]" in answer
        assert len(citations) == 2

    def test_empty_response(self):
        answer, citations, sources = postprocess_response("", [_make_block(1)])
        assert citations == []

    def test_no_blocks(self):
        answer, citations, _ = postprocess_response("Some text [1].", [])
        assert citations == []
        assert "[1]" not in answer
