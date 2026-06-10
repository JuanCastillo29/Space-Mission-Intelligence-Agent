"""Tests for the markdown chunker."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.ingestion.chunker import (
    chunk_markdown,
    count_tokens,
    _get_encoder,
    _strip_extraction_metadata,
    _parse_sections,
    _extract_blocks,
    _split_sentences,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_markdown(body: str) -> str:
    return f"# Extracted: test.pdf\n\n---\n*Page 1 [single] (quality: 1.00)*\n\n{body}"


def _long_text(token_target: int, seed: str = "satellite") -> str:
    enc = _get_encoder()
    text = (seed + " ") * (token_target * 2)
    tokens = enc.encode(text)
    return enc.decode(tokens[:token_target]).strip()


# ---------------------------------------------------------------------------
# TestTokenCounting
# ---------------------------------------------------------------------------


class TestTokenCounting:
    def test_count_tokens_basic(self):
        assert count_tokens("hello world") > 0

    def test_count_tokens_empty(self):
        assert count_tokens("") == 0

    def test_count_tokens_consistent(self):
        text = "The spacecraft entered orbit around Mars."
        assert count_tokens(text) == count_tokens(text)


# ---------------------------------------------------------------------------
# TestPreprocessing
# ---------------------------------------------------------------------------


class TestPreprocessing:
    def test_strips_page_separators(self):
        md = (
            "# Extracted: test.pdf\n\n"
            "---\n*Page 1 [single] (quality: 1.00)*\n\n"
            "Hello world.\n\n"
            "---\n*Page 2 [dual] (quality: 0.95)*\n\n"
            "Second page."
        )
        result = _strip_extraction_metadata(md)
        assert "*Page 1" not in result
        assert "*Page 2" not in result
        assert "Hello world." in result
        assert "Second page." in result

    def test_strips_extracted_header(self):
        md = "# Extracted: foo.pdf\n\nSome content."
        result = _strip_extraction_metadata(md)
        assert "# Extracted" not in result
        assert "Some content." in result

    def test_handles_empty_input(self):
        assert chunk_markdown("") == []
        assert chunk_markdown("   ") == []

    def test_handles_only_metadata(self):
        md = "# Extracted: test.pdf\n\n---\n*Page 1 [single] (quality: 1.00)*\n"
        assert chunk_markdown(md) == []


# ---------------------------------------------------------------------------
# TestSectionParsing
# ---------------------------------------------------------------------------


class TestSectionParsing:
    def test_builds_section_path_from_nested_headings(self):
        text = "## Introduction\n\nParagraph one.\n\n### Background\n\nParagraph two."
        sections = _parse_sections(text)
        paths = [s.path for s in sections]
        assert "Introduction" in paths
        assert "Introduction > Background" in paths

    def test_heading_level_reset(self):
        text = "## A\n\nText A.\n\n### B\n\nText B.\n\n## C\n\nText C."
        sections = _parse_sections(text)
        paths = [s.path for s in sections]
        assert "A" in paths
        assert "A > B" in paths
        assert "C" in paths

    def test_content_before_first_heading(self):
        text = "Some preamble text.\n\n## Section One\n\nBody."
        sections = _parse_sections(text)
        assert sections[0].path is None
        assert sections[0].blocks

    def test_heading_only_sections(self):
        text = "## A\n## B\n\nContent of B."
        sections = _parse_sections(text)
        titles = [s.title for s in sections]
        assert "A" in titles
        assert "B" in titles
        b_section = [s for s in sections if s.title == "B"][0]
        assert len(b_section.blocks) > 0


# ---------------------------------------------------------------------------
# TestAtomicBlocks
# ---------------------------------------------------------------------------


class TestAtomicBlocks:
    def test_table_detected_as_single_block(self):
        text = "| A | B |\n| --- | --- |\n| 1 | 2 |\n| 3 | 4 |"
        blocks = _extract_blocks(text)
        assert len(blocks) == 1
        assert blocks[0].kind == "table"
        assert "| 3 | 4 |" in blocks[0].text

    def test_figure_caption_atomic(self):
        text = "Some paragraph.\n\n[Figure: Diagram of the satellite orbit.]"
        blocks = _extract_blocks(text)
        kinds = [b.kind for b in blocks]
        assert "figure" in kinds
        fig = [b for b in blocks if b.kind == "figure"][0]
        assert "Diagram of the satellite" in fig.text

    def test_paragraphs_split_on_blank_lines(self):
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        blocks = _extract_blocks(text)
        assert len(blocks) == 3
        assert all(b.kind == "paragraph" for b in blocks)

    def test_mixed_content(self):
        text = (
            "Intro text.\n\n"
            "| Col1 | Col2 |\n| --- | --- |\n| v1 | v2 |\n\n"
            "[Figure: Fig 1.]\n\n"
            "Closing paragraph."
        )
        blocks = _extract_blocks(text)
        kinds = [b.kind for b in blocks]
        assert kinds == ["paragraph", "table", "figure", "paragraph"]


# ---------------------------------------------------------------------------
# TestSentenceSplitting
# ---------------------------------------------------------------------------


class TestSentenceSplitting:
    def test_splits_on_period(self):
        text = "First sentence. Second sentence. Third sentence."
        parts = _split_sentences(text)
        assert len(parts) == 3

    def test_preserves_abbreviations(self):
        text = "See Fig. 3 for details. The results show improvement."
        parts = _split_sentences(text)
        assert len(parts) == 2
        assert "Fig. 3" in parts[0]

    def test_single_sentence(self):
        text = "Just one sentence here."
        parts = _split_sentences(text)
        assert len(parts) == 1

    def test_eg_abbreviation(self):
        text = "Some materials (e.g. Kevlar) are used. They perform well."
        parts = _split_sentences(text)
        assert any("e.g." in p or "e.g" in p for p in parts)


# ---------------------------------------------------------------------------
# TestChunkAssembly
# ---------------------------------------------------------------------------


class TestChunkAssembly:
    def test_basic_chunking_respects_token_limit(self):
        body = _long_text(200)
        md = _make_markdown(f"## Section\n\n{body}")
        chunks = chunk_markdown(md, max_tokens=100, overlap_tokens=0)
        assert len(chunks) > 1
        for c in chunks:
            if not c.metadata.get("is_oversized"):
                assert c.token_count <= 100

    def test_overlap_present(self):
        body = _long_text(300)
        md = _make_markdown(f"## Section\n\n{body}")
        chunks = chunk_markdown(md, max_tokens=100, overlap_tokens=20)
        assert len(chunks) >= 2
        for i in range(1, len(chunks)):
            prev_content = chunks[i - 1].content
            curr_content = chunks[i].content
            overlap_found = any(
                word in curr_content for word in prev_content.split()[-10:]
            )
            assert overlap_found, f"Chunk {i} should contain overlap from chunk {i - 1}"

    def test_no_overlap_across_sections(self):
        text_a = _long_text(150, seed="spacecraft")
        text_b = _long_text(150, seed="telescope")
        section_a = f"## Section A\n\n{text_a}"
        section_b = f"## Section B\n\n{text_b}"
        md = _make_markdown(f"{section_a}\n\n{section_b}")
        chunks = chunk_markdown(md, max_tokens=80, overlap_tokens=20)
        a_chunks = [
            c for c in chunks if c.section_path and "Section A" in c.section_path
        ]
        b_chunks = [
            c for c in chunks if c.section_path and "Section B" in c.section_path
        ]
        assert len(a_chunks) > 0
        assert len(b_chunks) > 0
        first_b = b_chunks[0]
        assert "spacecraft" not in first_b.content

    def test_table_not_split(self):
        table = (
            "| Config | Temp |\n| --- | --- |\n| A | 100 |\n| B | 200 |\n| C | 300 |"
        )
        md = _make_markdown(f"## Data\n\nSome intro.\n\n{table}\n\nConclusion.")
        chunks = chunk_markdown(md, max_tokens=1024, overlap_tokens=0)
        table_in_single_chunk = any(
            "| A | 100 |" in c.content and "| C | 300 |" in c.content for c in chunks
        )
        assert table_in_single_chunk

    def test_oversized_table(self):
        rows = ["| Col1 | Col2 |", "| --- | --- |"]
        for i in range(200):
            rows.append(f"| value_{i} | data_{i} |")
        table = "\n".join(rows)
        md = _make_markdown(f"## Big Table\n\n{table}")
        chunks = chunk_markdown(md, max_tokens=50, overlap_tokens=0)
        oversized = [c for c in chunks if c.metadata.get("is_oversized")]
        assert len(oversized) >= 1
        assert oversized[0].metadata.get("has_table")

    def test_small_document_single_chunk(self):
        md = _make_markdown("## Title\n\nShort document.")
        chunks = chunk_markdown(md, max_tokens=1024, overlap_tokens=128)
        assert len(chunks) == 1

    def test_chunk_indices_sequential(self):
        body = _long_text(500)
        md = _make_markdown(f"## S1\n\n{body}\n\n## S2\n\n{body}")
        chunks = chunk_markdown(md, max_tokens=100, overlap_tokens=0)
        indices = [c.chunk_index for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_section_path_propagated(self):
        md = _make_markdown("## Introduction\n\nText.\n\n### Methods\n\nMore text.")
        chunks = chunk_markdown(md, max_tokens=1024, overlap_tokens=0)
        paths = [c.section_path for c in chunks]
        assert "Introduction" in paths
        assert "Introduction > Methods" in paths


# ---------------------------------------------------------------------------
# TestMetadata
# ---------------------------------------------------------------------------


class TestMetadata:
    def test_caller_metadata_merged(self):
        md = _make_markdown("## S\n\nText.")
        chunks = chunk_markdown(
            md, metadata={"source_id": "abc", "mission": "TechSat-1"}
        )
        assert len(chunks) >= 1
        for c in chunks:
            assert c.metadata["source_id"] == "abc"
            assert c.metadata["mission"] == "TechSat-1"

    def test_has_table_flag(self):
        table = "| A | B |\n| --- | --- |\n| 1 | 2 |"
        md = _make_markdown(f"## Data\n\n{table}")
        chunks = chunk_markdown(md)
        assert any(c.metadata.get("has_table") for c in chunks)

    def test_has_figure_flag(self):
        md = _make_markdown("## Results\n\n[Figure: Temperature distribution map.]")
        chunks = chunk_markdown(md)
        assert any(c.metadata.get("has_figure") for c in chunks)

    def test_no_false_flags(self):
        md = _make_markdown("## Text\n\nJust a paragraph.")
        chunks = chunk_markdown(md)
        for c in chunks:
            assert "has_table" not in c.metadata
            assert "has_figure" not in c.metadata


# ---------------------------------------------------------------------------
# TestGoldenSample
# ---------------------------------------------------------------------------


class TestGoldenSample:
    @pytest.fixture()
    def golden_md(self) -> str:
        return (FIXTURES_DIR / "golden_sample_expected.md").read_text(encoding="utf-8")

    def test_produces_chunks(self, golden_md):
        chunks = chunk_markdown(golden_md)
        assert len(chunks) > 0

    def test_all_content_preserved(self, golden_md):
        chunks = chunk_markdown(golden_md, overlap_tokens=0)
        combined = " ".join(c.content for c in chunks)
        assert "CubeSat" in combined
        assert "thermal" in combined.lower()
        assert "TechSat-1" in combined
        assert "110.2" in combined

    def test_table_intact(self, golden_md):
        chunks = chunk_markdown(golden_md)
        table_in_single = any(
            "Body-mounted" in c.content and "Double-deploy" in c.content for c in chunks
        )
        assert table_in_single

    def test_section_paths(self, golden_md):
        chunks = chunk_markdown(golden_md)
        paths = {c.section_path for c in chunks}
        assert any("Results" in p for p in paths if p)
        assert any("Introduction" in p for p in paths if p)
        assert any("Abstract" in p for p in paths if p)
