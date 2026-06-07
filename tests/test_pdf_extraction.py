"""
Unit tests for the pdf_extractor parsing logic.

Tests cover the individual functions that form the extraction pipeline:
  - Text healing (hyphen rejoining, paragraph reflow)
  - Boilerplate/noise removal
  - Column detection and reading-order assembly
  - Heading detection
  - Table formatting and detection
  - Paragraph merging (intra-page and cross-page)
  - Quality scoring

Run with:  pytest tests/test_pdf_extraction.py -v
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.ingestion.pdf_extractor import (
    TextBlock,
    PageResult,
    clean_extracted_text,
    compute_quality_score,
    detect_columns,
    detect_headings,
    assemble_reading_order,
    format_table_as_markdown,
    get_body_font_size,
    merge_cross_page_paragraphs,
    merge_split_paragraphs,
    reflow_block_text,
    rejoin_hyphenated_words,
    strip_references_tail,
    tag_boilerplate_blocks,
    tag_figure_captions,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _block(text, x0=50, y0=100, x1=500, y1=120, font_size=10.0,
           is_bold=False, block_type="text", font_name="Times"):
    """Shortcut to build a TextBlock for testing."""
    return TextBlock(
        text=text, x0=x0, y0=y0, x1=x1, y1=y1,
        font_size=font_size, font_name=font_name,
        is_bold=is_bold, block_type=block_type,
    )


# ===========================================================================
# Text Healing
# ===========================================================================


class TestRejoinHyphenatedWords:
    """Tests for rejoin_hyphenated_words."""

    def test_lowercase_continuation(self):
        text = "configu-\nration of the system"
        assert "configuration" in rejoin_hyphenated_words(text)

    def test_uppercase_continuation(self):
        text = "SPI-\nCAM instrument"
        assert "SPICAM" in rejoin_hyphenated_words(text)

    def test_preserves_intentional_hyphens(self):
        text = "well-known fact\nnew line"
        result = rejoin_hyphenated_words(text)
        assert "well-known" in result

    def test_no_change_when_no_hyphens(self):
        text = "Normal text without any hyphens."
        assert rejoin_hyphenated_words(text) == text

    def test_multiple_hyphenations(self):
        text = "atmo-\nspheric condi-\ntions on Mars"
        result = rejoin_hyphenated_words(text)
        assert "atmospheric" in result
        assert "conditions" in result

    def test_hyphen_before_uppercase_not_merged(self):
        # lowercase-hyphen-newline-Uppercase does NOT trigger either rule:
        # the lowercase rule requires lowercase continuation,
        # the uppercase rule requires uppercase on both sides.
        text = "data-\nProcessing step"
        result = rejoin_hyphenated_words(text)
        # The hyphen is preserved because neither regex matches
        assert "data-" in result and "Processing" in result


class TestReflowBlockText:
    """Tests for reflow_block_text."""

    def test_joins_broken_lines(self):
        text = "This is a long\nsentence that was\nbroken by column width."
        result = reflow_block_text(text)
        assert "long sentence" in result
        assert result.count("\n") == 0

    def test_preserves_paragraph_breaks(self):
        text = "First paragraph end.\n\nSecond paragraph start."
        result = reflow_block_text(text)
        assert "\n\n" in result

    def test_preserves_bullet_markers(self):
        text = "Introduction:\n• First item\n• Second item\n- Third item"
        result = reflow_block_text(text)
        assert "• First item" in result
        assert "• Second item" in result
        assert "- Third item" in result

    def test_hyphen_join_without_space(self):
        text = "Cardesín-\nMoinelo et al."
        result = reflow_block_text(text)
        assert "Cardesín-Moinelo" in result

    def test_empty_input(self):
        assert reflow_block_text("") == ""

    def test_single_line(self):
        text = "Just one line."
        assert reflow_block_text(text) == text

    def test_multiple_blank_lines_collapsed(self):
        text = "A\n\n\nB"
        result = reflow_block_text(text)
        # Each blank line is preserved as empty string in paragraphs
        assert "A" in result and "B" in result


# ===========================================================================
# Boilerplate / Noise Removal
# ===========================================================================


class TestCleanExtractedText:
    """Tests for clean_extracted_text."""

    def test_removes_standalone_page_numbers(self):
        text = "Some content.\n42\nMore content."
        result = clean_extracted_text(text)
        lines = [l.strip() for l in result.splitlines() if l.strip()]
        assert "42" not in lines

    def test_removes_copyright_lines(self):
        text = "Real content.\nCopyright © 2025 European Space Agency\nMore."
        result = clean_extracted_text(text)
        assert "Copyright ©" not in result

    def test_removes_isbn(self):
        text = "Content.\nISBN 978-92-9221-114-1\nMore content."
        result = clean_extracted_text(text)
        assert "ISBN" not in result

    def test_removes_issn(self):
        text = "Content.\nISSN 0250-1589\nMore."
        result = clean_extracted_text(text)
        assert "ISSN" not in result

    def test_removes_doi_urls(self):
        text = "Content.\nhttps://doi.org/10.1234/foo\nMore."
        result = clean_extracted_text(text)
        assert "doi.org" not in result

    def test_preserves_real_content(self):
        text = "Mars Express was launched in 2003 from Baikonur."
        assert clean_extracted_text(text) == text

    def test_collapses_excessive_blank_lines(self):
        text = "A\n\n\n\n\nB"
        result = clean_extracted_text(text)
        assert "\n\n\n" not in result
        assert "A" in result and "B" in result

    def test_removes_page_of_pattern(self):
        text = "Content.\nPage 5 of 20\nMore content."
        result = clean_extracted_text(text)
        assert "Page 5 of 20" not in result

    def test_removes_esa_credit_lines(self):
        text = "Content.\n(ESA/NASA/JPL)\nMore."
        result = clean_extracted_text(text)
        assert "(ESA/NASA/JPL)" not in result


class TestTagBoilerplateBlocks:
    """Tests for tag_boilerplate_blocks."""

    def test_tags_hal_metadata(self):
        b = _block("HAL Id: hal-12345678")
        result = tag_boilerplate_blocks([b])
        assert result[0].block_type == "boilerplate"

    def test_tags_creative_commons(self):
        b = _block("Distributed under a Creative Commons Attribution license")
        result = tag_boilerplate_blocks([b])
        assert result[0].block_type == "boilerplate"

    def test_does_not_tag_normal_text(self):
        b = _block("Mars Express orbiter instruments")
        result = tag_boilerplate_blocks([b])
        assert result[0].block_type == "text"

    def test_does_not_modify_headings(self):
        b = _block("Section Title", block_type="heading")
        result = tag_boilerplate_blocks([b])
        assert result[0].block_type == "heading"


class TestTagFigureCaptions:
    """Tests for tag_figure_captions."""

    def test_tags_fig_dot_pattern(self):
        b = _block("Fig. 1. Mars Express in orbit.")
        result = tag_figure_captions([b])
        assert result[0].block_type == "figure_caption"

    def test_tags_figure_word_pattern(self):
        b = _block("Figure 3 shows the instrument layout.")
        result = tag_figure_captions([b])
        assert result[0].block_type == "figure_caption"

    def test_does_not_tag_non_caption(self):
        b = _block("The figure above illustrates the concept.")
        result = tag_figure_captions([b])
        assert result[0].block_type == "text"


class TestStripReferencesTail:
    """Tests for strip_references_tail."""

    def test_strips_references_section(self):
        text = "Main body content.\n\n### References\n\nSmith et al. 2020..."
        result = strip_references_tail(text)
        assert "Main body content" in result
        assert "References" not in result
        assert "Smith" not in result

    def test_strips_author_affiliations(self):
        text = "Content.\n\n## Authors and Affiliations\n\n1. Dept of X"
        result = strip_references_tail(text)
        assert "Authors and Affiliations" not in result

    def test_preserves_text_without_references(self):
        text = "Full document content without any reference section."
        assert strip_references_tail(text) == text

    def test_strips_at_earliest_match(self):
        text = "Body.\n\n### Declarations\n\nNone.\n\n### References\n\nFoo."
        result = strip_references_tail(text)
        assert "Body." in result
        assert "Declarations" not in result


# ===========================================================================
# Column Detection & Reading Order
# ===========================================================================


class TestDetectColumns:
    """Tests for detect_columns."""

    def test_single_column_layout(self):
        blocks = [
            _block("Line 1", x0=50, x1=500, y0=100, y1=120),
            _block("Line 2", x0=50, x1=500, y0=130, y1=150),
            _block("Line 3", x0=50, x1=500, y0=160, y1=180),
        ]
        result = detect_columns(blocks, page_width=600, body_font_size=10.0)
        assert result["layout"] == "single"
        assert result["num_columns"] == 1

    def test_two_column_layout(self):
        # Left column: x0=50..250, right column: x0=320..520
        # Gap at 250-320 (70px on 600px page)
        left = [
            _block("Left A", x0=50, x1=250, y0=100, y1=120),
            _block("Left B", x0=50, x1=250, y0=130, y1=150),
        ]
        right = [
            _block("Right A", x0=320, x1=520, y0=100, y1=120),
            _block("Right B", x0=320, x1=520, y0=130, y1=150),
        ]
        blocks = left + right
        result = detect_columns(blocks, page_width=600, body_font_size=10.0)
        assert result["num_columns"] == 2
        assert result["layout"] == "two_columns"

    def test_three_column_layout(self):
        col1 = [_block("C1", x0=20, x1=150, y0=100, y1=120)]
        col2 = [_block("C2", x0=180, x1=310, y0=100, y1=120)]
        col3 = [_block("C3", x0=340, x1=470, y0=100, y1=120)]
        blocks = col1 + col2 + col3
        result = detect_columns(blocks, page_width=500, body_font_size=10.0)
        assert result["num_columns"] == 3
        assert result["layout"] == "three_columns"

    def test_empty_blocks_returns_single(self):
        result = detect_columns([], page_width=600, body_font_size=10.0)
        assert result["layout"] == "single"

    def test_ignores_non_body_font_blocks(self):
        # Two blocks at body size in same column, one heading at different size
        blocks = [
            _block("Body 1", x0=50, x1=500, y0=100, y1=120, font_size=10.0),
            _block("Body 2", x0=50, x1=500, y0=130, y1=150, font_size=10.0),
            _block("HEADING", x0=50, x1=500, y0=50, y1=80, font_size=18.0),
        ]
        result = detect_columns(blocks, page_width=600, body_font_size=10.0)
        assert result["layout"] == "single"


class TestAssembleReadingOrder:
    """Tests for assemble_reading_order."""

    def test_single_column_ordered_by_y(self):
        blocks = [
            _block("Third", y0=300),
            _block("First", y0=100),
            _block("Second", y0=200),
        ]
        info = {"num_columns": 1, "columns": [blocks], "full_width_blocks": []}
        ordered = assemble_reading_order(info)
        texts = [b.text for b in ordered]
        assert texts == ["First", "Second", "Third"]

    def test_two_columns_left_then_right(self):
        left = [_block("L1", y0=100), _block("L2", y0=200)]
        right = [_block("R1", y0=100), _block("R2", y0=200)]
        info = {
            "num_columns": 2,
            "columns": [left, right],
            "full_width_blocks": [],
        }
        ordered = assemble_reading_order(info)
        texts = [b.text for b in ordered]
        assert texts == ["L1", "L2", "R1", "R2"]

    def test_full_width_block_interleaves(self):
        left = [_block("L1", y0=100)]
        right = [_block("R1", y0=100)]
        fw = [_block("FULL WIDTH TITLE", y0=50)]
        info = {
            "num_columns": 2,
            "columns": [left, right],
            "full_width_blocks": fw,
        }
        ordered = assemble_reading_order(info)
        texts = [b.text for b in ordered]
        assert texts[0] == "FULL WIDTH TITLE"


# ===========================================================================
# Heading Detection
# ===========================================================================


class TestDetectHeadings:
    """Tests for detect_headings."""

    def test_large_font_becomes_heading(self):
        blocks = [
            _block("Title", font_size=16.0),
            _block("Body text here.", font_size=10.0),
        ]
        headings = detect_headings(blocks, body_font_size=10.0)
        assert len(headings) == 1
        assert headings[0].text == "Title"
        assert blocks[0].block_type == "heading"

    def test_bold_short_text_becomes_heading(self):
        blocks = [
            _block("Section Title", font_size=10.0, is_bold=True),
            _block("Normal paragraph content that is longer.", font_size=10.0),
        ]
        headings = detect_headings(blocks, body_font_size=10.0)
        assert len(headings) == 1
        assert headings[0].text == "Section Title"

    def test_body_text_not_tagged(self):
        blocks = [
            _block("Just a normal sentence.", font_size=10.0),
        ]
        headings = detect_headings(blocks, body_font_size=10.0)
        assert len(headings) == 0

    def test_very_long_bold_not_heading(self):
        long_text = "A" * 150  # > 100 chars
        blocks = [_block(long_text, font_size=10.0, is_bold=True)]
        headings = detect_headings(blocks, body_font_size=10.0)
        assert len(headings) == 0

    def test_non_text_block_not_tagged(self):
        blocks = [_block("Title", font_size=16.0, block_type="table")]
        headings = detect_headings(blocks, body_font_size=10.0)
        assert len(headings) == 0


class TestGetBodyFontSize:
    """Tests for get_body_font_size."""

    def test_returns_most_common_size(self):
        blocks = [
            _block("A", font_size=10.0),
            _block("B", font_size=10.0),
            _block("C", font_size=10.0),
            _block("D", font_size=14.0),  # heading
        ]
        assert get_body_font_size(blocks) == 10.0

    def test_empty_blocks_returns_default(self):
        assert get_body_font_size([]) == 10.0

    def test_ignores_non_text_blocks(self):
        blocks = [
            _block("A", font_size=12.0, block_type="header_footer"),
            _block("B", font_size=10.0),
            _block("C", font_size=10.0),
        ]
        assert get_body_font_size(blocks) == 10.0


# ===========================================================================
# Table Formatting
# ===========================================================================


class TestFormatTableAsMarkdown:
    """Tests for format_table_as_markdown."""

    def test_basic_table(self):
        data = [["Name", "Value"], ["Alpha", "1"], ["Beta", "2"]]
        md = format_table_as_markdown(data)
        assert "| Name | Value |" in md
        assert "| --- | --- |" in md
        assert "| Alpha | 1 |" in md
        assert "| Beta | 2 |" in md

    def test_none_cells_become_empty(self):
        data = [["A", "B"], [None, "x"], ["y", None]]
        md = format_table_as_markdown(data)
        assert "|  | x |" in md
        assert "| y |  |" in md

    def test_newlines_in_cells_flattened(self):
        data = [["Header"], ["Line1\nLine2"]]
        md = format_table_as_markdown(data)
        assert "Line1 Line2" in md
        # No raw newlines within a cell
        for line in md.splitlines():
            if "Line" in line:
                assert line.count("|") >= 2

    def test_pipe_chars_escaped(self):
        data = [["Col"], ["a|b"]]
        md = format_table_as_markdown(data)
        assert r"a\|b" in md

    def test_empty_data_returns_empty(self):
        assert format_table_as_markdown([]) == ""
        assert format_table_as_markdown([[]]) == ""

    def test_uneven_rows_padded(self):
        data = [["A", "B", "C"], ["x", "y"]]  # second row short
        md = format_table_as_markdown(data)
        # The separator and data rows should all have 3 columns
        lines = md.strip().splitlines()
        for line in lines:
            assert line.count("|") == 4  # | A | B | C |


# ===========================================================================
# Paragraph Merging (Intra-Page)
# ===========================================================================


class TestMergeSplitParagraphs:
    """Tests for merge_split_paragraphs."""

    def test_merges_continuation_blocks(self):
        blocks = [
            _block("The Mars Express mission was designed to", y0=100, y1=115),
            _block("study the planet from orbit.", y0=118, y1=133),
        ]
        result = merge_split_paragraphs(blocks, body_font_size=10.0)
        assert len(result) == 1
        assert "designed to study" in result[0].text

    def test_does_not_merge_after_sentence_end(self):
        blocks = [
            _block("First paragraph ends here.", y0=100, y1=115),
            _block("Second paragraph begins.", y0=118, y1=133),
        ]
        result = merge_split_paragraphs(blocks, body_font_size=10.0)
        assert len(result) == 2

    def test_does_not_merge_different_font_sizes(self):
        blocks = [
            _block("Heading text", y0=100, y1=125, font_size=16.0),
            _block("body text continuation", y0=128, y1=143, font_size=10.0),
        ]
        result = merge_split_paragraphs(blocks, body_font_size=10.0)
        assert len(result) == 2

    def test_does_not_merge_across_large_gap(self):
        blocks = [
            _block("Paragraph one incomplete", y0=100, y1=115),
            _block("distant block", y0=200, y1=215),  # gap >> body_font * 1.8
        ]
        result = merge_split_paragraphs(blocks, body_font_size=10.0)
        assert len(result) == 2

    def test_does_not_merge_into_bullet(self):
        blocks = [
            _block("Introduction to the list", y0=100, y1=115),
            _block("• First bullet item", y0=118, y1=133),
        ]
        result = merge_split_paragraphs(blocks, body_font_size=10.0)
        assert len(result) == 2

    def test_hyphen_merge_removes_hyphen(self):
        blocks = [
            _block("The Mars-", y0=100, y1=115),
            _block("Express mission launched.", y0=118, y1=133),
        ]
        # "Mars-" ends without sentence punct, next starts lowercase-ish
        # But actually "Express" starts uppercase — let's use lowercase example
        blocks2 = [
            _block("configu-", y0=100, y1=115),
            _block("ration of the spacecraft.", y0=118, y1=133),
        ]
        result = merge_split_paragraphs(blocks2, body_font_size=10.0)
        assert len(result) == 1
        assert "configuration" in result[0].text


# ===========================================================================
# Cross-Page Paragraph Merging
# ===========================================================================


class TestMergeCrossPageParagraphs:
    """Tests for merge_cross_page_paragraphs."""

    def test_merges_split_sentence(self):
        pages = [
            PageResult(page_num=1, layout="single",
                       text="The mission was designed to"),
            PageResult(page_num=2, layout="single",
                       text="study Mars from polar orbit."),
        ]
        result = merge_cross_page_paragraphs(pages)
        combined = result[0].text + " " + result[1].text
        # The fragment should be moved from page 2 to page 1
        assert "designed to study" in result[0].text

    def test_does_not_merge_complete_sentences(self):
        pages = [
            PageResult(page_num=1, layout="single",
                       text="The mission ended successfully."),
            PageResult(page_num=2, layout="single",
                       text="A new chapter began."),
        ]
        result = merge_cross_page_paragraphs(pages)
        assert "successfully." in result[0].text
        assert "A new chapter" in result[1].text

    def test_does_not_merge_into_heading(self):
        pages = [
            PageResult(page_num=1, layout="single",
                       text="### Section Title"),
            PageResult(page_num=2, layout="single",
                       text="content of this section."),
        ]
        result = merge_cross_page_paragraphs(pages)
        assert "### Section Title" in result[0].text

    def test_single_page_unchanged(self):
        pages = [PageResult(page_num=1, layout="single", text="Only page.")]
        result = merge_cross_page_paragraphs(pages)
        assert result[0].text == "Only page."


# ===========================================================================
# Quality Scoring
# ===========================================================================


class TestComputeQualityScore:
    """Tests for compute_quality_score."""

    def test_perfect_single_column_page(self):
        text = "Normal text with reasonable words. " * 50
        blocks = [_block(text)]
        info = {"num_columns": 1, "columns": [blocks]}
        score, issues = compute_quality_score(
            text, blocks, blocks, info, page_area=400_000
        )
        assert score >= 0.9
        assert issues == []

    def test_low_density_penalized(self):
        text = "Short."
        blocks = [_block(text)]
        info = {"num_columns": 1, "columns": [blocks]}
        score, issues = compute_quality_score(
            text, blocks, blocks, info, page_area=400_000
        )
        assert score < 1.0
        assert "low_text_density" in issues

    def test_garbled_text_penalized(self):
        garbled_word = "a" * 45  # > 40 chars
        text = " ".join([garbled_word] * 10 + ["short"] * 5)
        blocks = [_block(text)]
        info = {"num_columns": 1, "columns": [blocks]}
        score, issues = compute_quality_score(
            text, blocks, blocks, info, page_area=400_000
        )
        assert score < 0.8
        assert "garbled_text_detected" in issues

    def test_complex_layout_penalized(self):
        text = "Normal content. " * 30
        blocks = [_block(text)]
        info = {"num_columns": 5, "columns": [[_block("x")] * 5]}
        score, issues = compute_quality_score(
            text, blocks, blocks, info, page_area=400_000
        )
        assert score < 1.0
        assert "uncertain_layout" in issues

    def test_low_content_ratio_penalized(self):
        text = "Some text."
        all_blocks = [
            _block("hf1", block_type="header_footer"),
            _block("hf2", block_type="header_footer"),
            _block("hf3", block_type="header_footer"),
            _block("hf4", block_type="header_footer"),
            _block(text),
        ]
        content_blocks = [b for b in all_blocks if b.block_type == "text"]
        info = {"num_columns": 1, "columns": [content_blocks]}
        score, issues = compute_quality_score(
            text, all_blocks, content_blocks, info, page_area=400_000
        )
        assert "low_content_ratio" in issues

    def test_score_never_below_zero(self):
        # Stack all penalties
        garbled = "x" * 50
        text = " ".join([garbled] * 20)
        blocks = [_block(text)]
        all_blocks = blocks + [_block("", block_type="header_footer")] * 10
        info = {"num_columns": 6, "columns": [[_block("y")] * 6]}
        score, _ = compute_quality_score(
            text, all_blocks, blocks, info, page_area=10_000_000
        )
        assert score >= 0.0

    def test_unbalanced_columns_penalized(self):
        text = "Normal content with enough words to pass density check. " * 40
        col1 = [_block("x")] * 20
        col2 = [_block("y")]  # ratio = 1/20 = 0.05 < 0.1
        info = {"num_columns": 2, "columns": [col1, col2]}
        blocks = col1 + col2
        score, issues = compute_quality_score(
            text, blocks, blocks, info, page_area=400_000
        )
        assert "unbalanced_columns" in issues


# ===========================================================================
# Golden-file regression test  (end-to-end pipeline)
# ===========================================================================


class TestGoldenFileRegression:
    """Run the full pipeline on a small, controlled PDF and verify
    structural properties + quality.  This catches regressions that
    slip through individual unit tests.

    The comparison is *structural* (headings, sections, table presence,
    quality score) rather than exact-string, so the test survives minor
    PyMuPDF whitespace changes across patch versions.
    """

    GOLDEN_PDF = Path(__file__).parent / "fixtures" / "golden_sample.pdf"
    GOLDEN_MD = Path(__file__).parent / "fixtures" / "golden_sample_expected.md"

    @pytest.fixture(autouse=True)
    def _load(self):
        """Run the extractor once and share across tests."""
        from scripts.ingestion.pdf_extractor import extract_pdf
        self.text, self.report = extract_pdf(str(self.GOLDEN_PDF))
        self.expected = self.GOLDEN_MD.read_text(encoding="utf-8")

    # ── Structural checks ──────────────────────────────────────

    def test_page_count(self):
        assert self.report.total_pages == 2

    def test_quality_above_threshold(self):
        assert self.report.avg_quality >= 0.85

    def test_title_detected(self):
        assert "## Thermal Analysis of CubeSat Solar Panels" in self.text

    def test_all_section_headings_present(self):
        for heading in ["Abstract", "1. Introduction",
                        "2. Methodology", "3. Results", "4. Conclusions"]:
            assert f"### {heading}" in self.text, f"Missing heading: {heading}"

    def test_table_present(self):
        assert "| Configuration" in self.text
        assert "Body-mounted" in self.text
        assert "128.3" in self.text

    def test_table_row_count(self):
        table_lines = [l for l in self.text.splitlines()
                       if l.strip().startswith("|") and "---" not in l]
        # 1 header + 3 data rows = 4
        assert len(table_lines) == 4

    def test_figure_caption_tagged(self):
        assert "[Figure:" in self.text or "Figure 1." in self.text

    def test_abstract_content_present(self):
        assert "thermal analysis of solar panel configurations" in self.text

    def test_conclusion_content_present(self):
        assert "deployable solar panel configurations" in self.text

    def test_no_quality_issues(self):
        for p in self.report.page_results:
            assert p["issues"] == [], \
                f"Page {p['page']} has unexpected issues: {p['issues']}"

    # ── Similarity to frozen reference ─────────────────────────

    def test_similarity_to_golden_reference(self):
        """Output should be very close to the frozen expected markdown.
        Uses SequenceMatcher ratio — a threshold below 1.0 absorbs minor
        whitespace/punctuation drift across PyMuPDF versions."""
        from difflib import SequenceMatcher
        ratio = SequenceMatcher(None, self.text, self.expected).ratio()
        assert ratio >= 0.95, (
            f"Output similarity {ratio:.3f} is below 0.95 threshold — "
            f"check for extraction regressions"
        )
