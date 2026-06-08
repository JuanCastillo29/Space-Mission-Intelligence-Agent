"""
Column-Aware PDF extractor for RAG system
-------------------------------------------

The script uses PyMuPDF to extract text from PDFs with correct reading order,
handling single-column, multi-column (2, 3, …) and mixed layouts.

Includes:
 - Automatic N-column detection per page (gap analysis on merged x-ranges)
 - Correct reading order (columns left-to-right, interleaved at full-width blocks)
 - Heading detection and hierarchy
 - Quality scoring per page
 - Text healing (hyphen rejoining, paragraph reflow, noise removal)
 - Cross-page paragraph merging
 - Boilerplate stripping (headers/footers, HAL metadata, page numbers)
 - Figure caption tagging and reference-list exclusion
 - Table detection (bordered via find_tables; borderless via word-position analysis)
 - Structured markdown ready for RAG chunking

Usage:
    python pdf_extractor.py input.pdf [--output output.md] [--report]
"""

from __future__ import annotations

import re
import fitz
import json
import statistics
import argparse
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TextBlock:
    text: str
    x0: int
    y0: int
    x1: int
    y1: int
    font_size: float = 0.0
    font_name: str = ""
    is_bold: bool = False
    block_type: str = "text"

    @property
    def center_x(self):
        return (self.x1 + self.x0) / 2

    @property
    def center_y(self):
        return (self.y1 + self.y0) / 2

    @property
    def width(self):
        return abs(self.x1 - self.x0)

    @property
    def height(self):
        return abs(self.y1 - self.y0)


@dataclass
class PageResult:
    page_num: int
    layout: str
    text: str
    heading: list = field(default_factory=list)
    quality_score: float = 1.0
    issues: list = field(default_factory=list)
    block_count: int = 0
    char_count: int = 0


@dataclass
class ExtractionReport:
    total_pages: int = 0
    pages_single_col: int = 0
    pages_multi_col: int = 0
    pages_other: int = 0
    avg_quality: float = 0.0
    total_issues: int = 0
    issue_summary: dict = field(default_factory=dict)
    page_results: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Text-healing helpers
# ---------------------------------------------------------------------------

_HYPHEN_LINE_LC = re.compile(r'(\w)-\n\s*([a-z])')
_HYPHEN_LINE_UC = re.compile(r'([A-Z])-\n\s*([A-Z])')


def rejoin_hyphenated_words(text: str) -> str:
    """Rejoin words split with a hyphen at line breaks.

    Handles both lowercase continuations (``configu-\\nration``)
    and all-caps acronym splits (``SPI-\\nCAM``).
    """
    text = _HYPHEN_LINE_LC.sub(r'\1\2', text)
    text = _HYPHEN_LINE_UC.sub(r'\1\2', text)
    return text


_BULLET_RE = re.compile(r'^[•–\-\*\+]\s')


def fix_heading_spaces(page: fitz.Page, block: TextBlock) -> str:
    """Re-extract heading text using character-level positions to restore
    missing inter-word spaces dropped by PyMuPDF in large-font headings.
    """
    clip = fitz.Rect(block.x0 - 1, block.y0 - 1, block.x1 + 1, block.y1 + 1)
    rawdict = page.get_text("rawdict", clip=clip,
                            flags=fitz.TEXT_PRESERVE_WHITESPACE)
    lines_out = []
    for blk in rawdict.get("blocks", []):
        if blk.get("type") != 0:
            continue
        for line in blk.get("lines", []):
            if not _line_matches_heading(line, block.font_size, block.is_bold):
                continue
            line_chars = []
            for span in line.get("spans", []):
                font_size = span.get("size", 10)
                space_threshold = font_size * 0.1
                for ch in span.get("chars", []):
                    line_chars.append((ch["bbox"], ch["c"], space_threshold))
            if not line_chars:
                continue
            result = line_chars[0][1]
            for i in range(1, len(line_chars)):
                gap = line_chars[i][0][0] - line_chars[i - 1][0][2]
                prev_char = line_chars[i - 1][1]
                curr_char = line_chars[i][1]
                if (gap > line_chars[i - 1][2]
                        and prev_char != ' ' and curr_char != ' '):
                    result += ' '
                result += curr_char
            lines_out.append(result.strip())
    if not lines_out:
        return block.text
    return '\n'.join(lines_out)


def _line_matches_heading(line: dict, heading_size: float,
                          heading_bold: bool) -> bool:
    """Check if a raw-dict line matches the expected heading font."""
    spans = line.get("spans", [])
    if not spans:
        return False
    for span in spans:
        has_content = (
            span.get("text", "").strip()
            or any(c.get("c", "").strip() for c in span.get("chars", []))
        )
        if not has_content:
            continue
        span_bold = (
            "bold" in span.get("font", "").lower()
            or span.get("flags", 0) & 2**4
        )
        span_size = span.get("size", 0)
        size_close = abs(span_size - heading_size) < heading_size * 0.2
        if heading_bold and not span_bold:
            return False
        if not size_close:
            return False
    return True


def _join_reflow(parts: list[str]) -> str:
    """Join reflowed line fragments, collapsing trailing hyphens."""
    if not parts:
        return ''
    result = parts[0]
    for p in parts[1:]:
        if result.endswith('-'):
            result += p
        else:
            result += ' ' + p
    return result


def reflow_block_text(text: str) -> str:
    """Join hard-wrapped lines within a single PDF block into paragraphs.

    Blank lines and bullet markers are preserved as paragraph separators.
    Non-blank consecutive lines are joined with a space so that PDF
    column-width line breaks don't leak into the output.  When the
    previous fragment ends with ``-``, the join is done without a space
    (e.g. ``Cardesín-`` + ``Moinelo`` → ``Cardesín-Moinelo``).
    """
    lines = text.split('\n')
    paragraphs: list[str] = []
    current: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if current:
                paragraphs.append(_join_reflow(current))
                current = []
            paragraphs.append('')
        elif _BULLET_RE.match(stripped):
            if current:
                paragraphs.append(_join_reflow(current))
                current = []
            current.append(stripped)
        else:
            current.append(stripped)
    if current:
        paragraphs.append(_join_reflow(current))
    return '\n'.join(paragraphs)


# ---------------------------------------------------------------------------
# Block tagging – boilerplate / captions
# ---------------------------------------------------------------------------

_FIGURE_CAPTION_RE = re.compile(
    r"^Fig\.?\s*\d+|^Figure\s+\d+", re.IGNORECASE
)

_TABLE_CAPTION_RE = re.compile(
    r"^Table\s+\d+", re.IGNORECASE
)

_HAL_MARKERS = [
    re.compile(r"To\s+cite\s+this\s+version", re.IGNORECASE),
    re.compile(r"HAL\s+Id\s*:", re.IGNORECASE),
    re.compile(r"hal\.science", re.IGNORECASE),
    re.compile(r"HAL\s+is\s+a\s+multi.disciplinary", re.IGNORECASE),
    re.compile(r"archive\s+ouverte\s+pluridisciplinaire", re.IGNORECASE),
    re.compile(r"Submitted\s+on\s+\d+", re.IGNORECASE),
    re.compile(r"Distributed\s+under\s+a\s+Creative\s+Commons", re.IGNORECASE),
    re.compile(r"tin[ée]e?\s+au\s+d[eé]p[oô]t", re.IGNORECASE),
    re.compile(r"⟨hal-\d+", re.IGNORECASE),
    re.compile(r"⟨10\.\d{4,}", re.IGNORECASE),
]


def tag_boilerplate_blocks(blocks: list[TextBlock]) -> list[TextBlock]:
    """Tag blocks containing HAL metadata or other repository boilerplate."""
    for block in blocks:
        if block.block_type not in ("text", "heading"):
            continue
        for marker in _HAL_MARKERS:
            if marker.search(block.text):
                block.block_type = "boilerplate"
                break
    return blocks


def tag_figure_captions(blocks: list[TextBlock]) -> list[TextBlock]:
    """Tag blocks that are figure captions."""
    for block in blocks:
        stripped = block.text.strip()
        if _FIGURE_CAPTION_RE.match(stripped):
            block.block_type = "figure_caption"
    return blocks


# ---------------------------------------------------------------------------
# Borderless-table detection via word-position analysis
# ---------------------------------------------------------------------------

def _find_table_extent(blocks: list[TextBlock], caption_idx: int,
                       body_font_size: float) -> int:
    """Return the index (exclusive) of the last block belonging to the table
    that starts at *caption_idx*.

    A block is still "table data" when it is short, doesn't end with sentence
    punctuation, or has a font size close to the caption.  The run ends at the
    first block that looks like a real body paragraph or a heading.
    """
    end = caption_idx + 1
    for j in range(caption_idx + 1, len(blocks)):
        b = blocks[j]
        if b.block_type in ("heading", "figure_caption", "boilerplate"):
            break
        text = b.text.strip()
        if not text:
            break
        # Detect headings by font size / bold before detect_headings runs
        if b.font_size > body_font_size * 1.15 and len(text) < 200:
            break
        if b.is_bold and len(text) < 100:
            break
        if len(text) > 120 and text[-1] in '.!?':
            break
        end = j + 1
    return end


def _build_table_from_words(page: fitz.Page, y0: float, y1: float,
                            x0: float, x1: float) -> str | None:
    """Build a markdown table from the word positions in a page region.

    1. Collect all words whose centre falls inside the clip region.
    2. Find column gaps (significant horizontal whitespace).
    3. Cluster words into rows by y-position.
    4. Assign each word to a column.
    5. Format as markdown.
    """
    words = page.get_text("words")
    region_words = []
    for w in words:
        cy = (w[1] + w[3]) / 2
        cx = (w[0] + w[2]) / 2
        if y0 <= cy <= y1 and x0 - 5 <= cx <= x1 + 5:
            region_words.append(w)
    if len(region_words) < 4:
        return None

    # --- detect column boundaries ----------------------------------------
    x_ranges = sorted([(w[0], w[2]) for w in region_words])
    merged = [list(x_ranges[0])]
    for s, e in x_ranges[1:]:
        if s <= merged[-1][1] + 2:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])

    min_gap = max(12.0, (x1 - x0) * 0.03)
    gaps = []
    for i in range(len(merged) - 1):
        gap = merged[i + 1][0] - merged[i][1]
        if gap >= min_gap:
            gaps.append((merged[i][1] + merged[i + 1][0]) / 2)
    if not gaps:
        return None
    ncols = len(gaps) + 1

    # --- cluster words into rows -----------------------------------------
    row_tol = max(6.0, (y1 - y0) / max(1, len(region_words)) * 2)
    sorted_by_y = sorted(region_words, key=lambda w: w[1])
    rows: list[list] = []
    current_row: list = [sorted_by_y[0]]
    for w in sorted_by_y[1:]:
        if w[1] - current_row[-1][1] > row_tol:
            rows.append(current_row)
            current_row = [w]
        else:
            current_row.append(w)
    if current_row:
        rows.append(current_row)
    if len(rows) < 2:
        return None

    # --- assign words to columns and build cell text ---------------------
    def col_idx(cx):
        for i, g in enumerate(gaps):
            if cx < g:
                return i
        return ncols - 1

    table: list[list[str]] = []
    for row_words in rows:
        cells = [""] * ncols
        for w in sorted(row_words, key=lambda w: w[0]):
            cx = (w[0] + w[2]) / 2
            ci = col_idx(cx)
            if cells[ci]:
                cells[ci] += " " + w[4]
            else:
                cells[ci] = w[4]
        if any(c.strip() for c in cells):
            table.append(cells)
    if len(table) < 2:
        return None

    # Merge continuation rows (first-column cell empty) into the row above
    merged_table: list[list[str]] = [table[0]]
    for row in table[1:]:
        if not row[0].strip() and merged_table:
            for c in range(ncols):
                if row[c].strip():
                    if merged_table[-1][c]:
                        merged_table[-1][c] += " " + row[c]
                    else:
                        merged_table[-1][c] = row[c]
        else:
            merged_table.append(row)
    if len(merged_table) < 2:
        return None

    return format_table_as_markdown(merged_table)


def tag_and_extract_tables(blocks: list[TextBlock], page: fitz.Page,
                           body_font_size: float) -> list[TextBlock]:
    """Detect table captions, find their extent, attempt structured extraction,
    and replace the caption+data blocks with a single table block.

    Falls back to wrapping the raw text in ``[Table: ...]`` when word-level
    extraction fails.
    """
    result: list[TextBlock] = []
    i = 0
    while i < len(blocks):
        b = blocks[i]
        if not _TABLE_CAPTION_RE.match(b.text.strip()):
            result.append(b)
            i += 1
            continue

        end = _find_table_extent(blocks, i, body_font_size)
        table_blocks = blocks[i:end]

        # Bounding box of the whole table region
        ty0 = min(tb.y0 for tb in table_blocks)
        ty1 = max(tb.y1 for tb in table_blocks)
        tx0 = min(tb.x0 for tb in table_blocks)
        tx1 = max(tb.x1 for tb in table_blocks)

        # Try bordered detection first (clip region)
        md = None
        try:
            clip = fitz.Rect(tx0 - 5, ty0 - 5, tx1 + 5, ty1 + 5)
            finder = page.find_tables(clip=clip)
            for t in finder.tables:
                data = t.extract()
                if data and len(data) >= 2:
                    total = sum(len(r) for r in data)
                    filled = sum(1 for r in data for c in r if c and c.strip())
                    if total and filled / total > 0.15:
                        md = format_table_as_markdown(data)
                        break
        except Exception:
            pass

        # Try word-position extraction
        if not md:
            md = _build_table_from_words(page, ty0, ty1, tx0, tx1)

        caption_text = table_blocks[0].text.strip()
        # Extract just the "Table N <title>" part before data begins
        caption_line = caption_text.split('\n')[0] if '\n' in caption_text else caption_text

        if md:
            text = f"[Table: {caption_line}]\n\n{md}"
        else:
            raw = '\n'.join(tb.text.strip() for tb in table_blocks)
            text = f"[Table: {raw}]"

        result.append(TextBlock(
            text=text,
            x0=tx0, y0=ty0, x1=tx1, y1=ty1,
            font_size=table_blocks[0].font_size,
            font_name=table_blocks[0].font_name,
            block_type="table",
        ))
        i = end

    return result


# ---------------------------------------------------------------------------
# Bordered-table extraction (find_tables with default strategy)
# ---------------------------------------------------------------------------

def format_table_as_markdown(data: list[list[str | None]]) -> str:
    """Format extracted table data as a markdown table."""
    if not data or not data[0]:
        return ""

    ncols = max(len(row) for row in data)

    def clean_cell(cell):
        if cell is None:
            return ""
        return cell.strip().replace("\n", " ").replace("|", "\\|")

    rows = []
    for row in data:
        cells = [clean_cell(c) for c in row]
        while len(cells) < ncols:
            cells.append("")
        rows.append(cells)

    lines = []
    lines.append("| " + " | ".join(rows[0]) + " |")
    lines.append("| " + " | ".join(["---"] * ncols) + " |")
    for row in rows[1:]:
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


def extract_page_tables(page: fitz.Page) -> list[TextBlock]:
    """Extract bordered tables from a page and return as TextBlock objects."""
    try:
        finder = page.find_tables()
    except (AttributeError, Exception):
        return []

    page_area = page.rect.width * page.rect.height
    table_blocks = []

    for table in finder.tables:
        data = table.extract()
        if not data or len(data) < 2:
            continue

        bbox = table.bbox
        table_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        area_ratio = table_area / page_area if page_area > 0 else 0
        num_rows = len(data)

        total_cells = sum(len(row) for row in data)
        non_empty = sum(1 for row in data for c in row if c and c.strip())
        fill_ratio = non_empty / total_cells if total_cells > 0 else 0

        if area_ratio > 0.50:
            continue

        if fill_ratio < 0.2:
            continue

        if area_ratio > 0.30 and (fill_ratio < 0.4 or num_rows < 3):
            continue

        md = format_table_as_markdown(data)
        if not md:
            continue

        table_blocks.append(TextBlock(
            text=md,
            x0=bbox[0], y0=bbox[1], x1=bbox[2], y1=bbox[3],
            block_type="table"
        ))

    return table_blocks


def _blocks_overlap(block: TextBlock, table: TextBlock, margin: float = 5.0) -> bool:
    return not (block.x1 < table.x0 - margin or
                block.x0 > table.x1 + margin or
                block.y1 < table.y0 - margin or
                block.y0 > table.y1 + margin)


# ---------------------------------------------------------------------------
# Block extraction
# ---------------------------------------------------------------------------

def get_text_blocks(page: fitz.Page, vmargin_pct: float = 0.05) -> list[TextBlock]:
    page_height = page.rect.height
    top_margin = page_height * vmargin_pct
    bottom_margin = page_height * (1 - vmargin_pct)

    word_bboxes = {}
    for w in page.get_text("words"):
        bno = w[5]
        if bno not in word_bboxes:
            word_bboxes[bno] = [w[0], w[1], w[2], w[3]]
        else:
            word_bboxes[bno][0] = min(word_bboxes[bno][0], w[0])
            word_bboxes[bno][1] = min(word_bboxes[bno][1], w[1])
            word_bboxes[bno][2] = max(word_bboxes[bno][2], w[2])
            word_bboxes[bno][3] = max(word_bboxes[bno][3], w[3])

    blocks = []
    raw_dict = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)

    for block_idx, block in enumerate(raw_dict.get("blocks", [])):
        if block.get("type") != 0:
            continue

        if block_idx in word_bboxes:
            x0, y0, x1, y1 = word_bboxes[block_idx]
        else:
            x0, y0, x1, y1 = block["bbox"]

        is_hf = y0 < top_margin or y1 > bottom_margin
        block_type = "header_footer" if is_hf else "text"

        line_groups = _split_block_lines(block.get("lines", []))
        for group in line_groups:
            full_text = ""
            font_sizes = []
            font_names = []
            bold_count = 0
            total_spans = 0
            g_y0 = g_y1 = None

            for line in group:
                lbbox = line.get("bbox", (x0, y0, x1, y1))
                if g_y0 is None:
                    g_y0 = lbbox[1]
                g_y1 = lbbox[3]

                line_text = ""
                for span in line.get("spans", []):
                    span_text = span.get("text", "")
                    line_text += span_text
                    font_sizes.append(span.get("size", 0))
                    font_names.append(span.get("font", ""))
                    total_spans += 1
                    span_font = span.get("font", "").lower()
                    span_flags = span.get("flags", 0)
                    if "bold" in span_font or span_flags & 2**4:
                        bold_count += 1

                full_text += line_text + "\n"

            full_text = full_text.strip()
            if not full_text:
                continue

            full_text = rejoin_hyphenated_words(full_text)
            full_text = reflow_block_text(full_text)

            avg_font_size = (
                statistics.mean(font_sizes) if font_sizes else 0.0
            )
            dominant_font = (
                max(set(font_names), key=font_names.count)
                if font_names else None
            )
            is_bold = (
                bold_count > total_spans / 2 if total_spans > 0
                else False
            )

            blocks.append(TextBlock(
                text=full_text,
                x0=x0, y0=g_y0 or y0, x1=x1, y1=g_y1 or y1,
                font_size=avg_font_size,
                font_name=dominant_font,
                is_bold=is_bold,
                block_type=block_type,
            ))
    return blocks


def _split_block_lines(lines: list[dict]) -> list[list[dict]]:
    """Split a PyMuPDF block's lines into groups at font-change boundaries.

    When a single PyMuPDF block contains lines with different font
    characteristics (e.g. a bold heading followed by body text), this splits
    them so each group can become its own TextBlock with accurate metadata.
    """
    if len(lines) <= 1:
        return [lines] if lines else []

    groups: list[list[dict]] = []
    current: list[dict] = [lines[0]]
    prev_bold, prev_size = _line_font_info(lines[0])

    for line in lines[1:]:
        cur_bold, cur_size = _line_font_info(line)
        bold_changed = cur_bold != prev_bold
        size_changed = prev_size > 0 and abs(cur_size - prev_size) / prev_size > 0.15
        if bold_changed or size_changed:
            groups.append(current)
            current = [line]
        else:
            current.append(line)
        prev_bold = cur_bold
        prev_size = cur_size

    if current:
        groups.append(current)
    return groups


def _line_font_info(line: dict) -> tuple[bool, float]:
    """Return (is_bold, avg_font_size) for a single line dict."""
    spans = line.get("spans", [])
    if not spans:
        return False, 0.0
    bold_count = 0
    sizes = []
    for span in spans:
        sizes.append(span.get("size", 0))
        if "bold" in span.get("font", "").lower() or span.get("flags", 0) & 2**4:
            bold_count += 1
    is_bold = bold_count > len(spans) / 2
    avg_size = statistics.mean(sizes) if sizes else 0.0
    return is_bold, avg_size


# ---------------------------------------------------------------------------
# Column detection / reading order
# ---------------------------------------------------------------------------

def _find_column_gaps(blocks: list[TextBlock], page_width: float,
                      min_gap_pct: float = 0.01,
                      max_gap_abs: float = 20.0) -> list[dict]:
    if not blocks:
        return []

    x_ranges = sorted([(b.x0, b.x1) for b in blocks], key=lambda r: r[0])
    merged = [list(x_ranges[0])]
    for start, end in x_ranges[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])

    min_gap = max(6.0, min(page_width * min_gap_pct, max_gap_abs))

    gaps = []
    for i in range(len(merged) - 1):
        gap_start = merged[i][1]
        gap_end = merged[i + 1][0]
        gap_size = gap_end - gap_start
        if gap_size >= min_gap:
            gaps.append({
                "start": gap_start,
                "end": gap_end,
                "midpoint": (gap_start + gap_end) / 2,
                "size": gap_size,
            })
    return gaps


def detect_columns(blocks: list[TextBlock], page_width: float,
                   body_font_size: float,
                   min_gap_pct: float = 0.01,
                   full_width_pct: float = 0.6,
                   font_tolerance: float = 0.15) -> dict:
    content_blocks = [b for b in blocks if b.block_type == "text"]
    body_blocks = [
        b for b in content_blocks
        if abs(b.font_size - body_font_size) / body_font_size < font_tolerance
    ]

    def _single(blks: list[TextBlock]) -> dict:
        return {
            "layout": "single",
            "num_columns": 1,
            "column_boundaries": [],
            "columns": [sorted(blks, key=lambda b: b.y0)],
            "full_width_blocks": [],
        }

    if len(body_blocks) < 2:
        return _single(content_blocks)

    gaps = _find_column_gaps(body_blocks, page_width, min_gap_pct)

    if not gaps:
        return _single(content_blocks)

    boundaries = sorted(g["midpoint"] for g in gaps)
    num_columns = len(boundaries) + 1

    columns: list[list[TextBlock]] = [[] for _ in range(num_columns)]
    full_width_blocks: list[TextBlock] = []

    for block in content_blocks:
        start_col = 0
        end_col = 0
        for i, boundary in enumerate(boundaries):
            if block.x0 >= boundary:
                start_col = i + 1
            if block.x1 >= boundary:
                end_col = i + 1
        if start_col == end_col:
            columns[start_col].append(block)
        else:
            full_width_blocks.append(block)

    for col in columns:
        col.sort(key=lambda b: b.y0)
    full_width_blocks.sort(key=lambda b: b.y0)

    non_empty = [(i, col) for i, col in enumerate(columns) if col]
    if len(non_empty) <= 1:
        return _single(content_blocks)
    if len(non_empty) < num_columns:
        columns = [col for _, col in non_empty]
        kept_indices = {i for i, _ in non_empty}
        boundaries = [
            b for idx, b in enumerate(boundaries)
            if idx in kept_indices or (idx + 1) in kept_indices
        ][:len(columns) - 1]
        num_columns = len(columns)

    _COL_NAMES = {1: "single", 2: "two_columns", 3: "three_columns"}
    layout = _COL_NAMES.get(num_columns, f"{num_columns}_columns")

    return {
        "layout": layout,
        "num_columns": num_columns,
        "column_boundaries": boundaries,
        "columns": columns,
        "full_width_blocks": full_width_blocks,
    }


def assemble_reading_order(column_info: dict) -> list[TextBlock]:
    columns = column_info["columns"]

    if column_info["num_columns"] == 1:
        return sorted(columns[0], key=lambda b: b.y0)

    full_width_blocks = column_info.get("full_width_blocks", [])
    ordered: list[TextBlock] = []

    breakpoints = sorted([(b.y0, i, b) for i, b in enumerate(full_width_blocks)])

    if not breakpoints:
        for col in columns:
            ordered.extend(col)
        return ordered

    prev_y = 0
    for bp_y, _, fw_block in breakpoints:
        for col in columns:
            section = [b for b in col if prev_y < b.y0 < bp_y]
            ordered.extend(section)
        ordered.append(fw_block)
        prev_y = bp_y

    for col in columns:
        remaining = [b for b in col if b.y0 >= prev_y]
        ordered.extend(remaining)

    return ordered


# ---------------------------------------------------------------------------
# Paragraph merging (intra-page)
# ---------------------------------------------------------------------------

def merge_split_paragraphs(blocks: list[TextBlock],
                           body_font_size: float,
                           max_y_gap_factor: float = 1.8,
                           x_overlap_pct: float = 0.5) -> list[TextBlock]:
    if not blocks:
        return []

    merged = [blocks[0]]
    for block in blocks[1:]:
        prev = merged[-1]

        font_similar = (
            abs(prev.font_size - block.font_size) < body_font_size * 0.15
        )

        overlap_start = max(prev.x0, block.x0)
        overlap_end = min(prev.x1, block.x1)
        overlap = max(0, overlap_end - overlap_start)
        min_width = min(prev.width, block.width) or 1
        x_overlaps = overlap / min_width > x_overlap_pct

        y_gap = block.y0 - prev.y1
        close_y = 0 <= y_gap < body_font_size * max_y_gap_factor

        prev_text = prev.text.rstrip()
        ends_mid_sentence = (
            prev_text and prev_text[-1] not in ".!?:;\n" and not prev.is_bold
        )

        # Don't merge into a block that starts with a bullet marker
        next_starts_bullet = _BULLET_RE.match(block.text.lstrip())

        if (font_similar and x_overlaps and close_y
                and ends_mid_sentence and not next_starts_bullet
                and not block.is_bold):
            prev_stripped = prev.text.rstrip()
            if prev_stripped.endswith('-'):
                merged_text = prev_stripped[:-1] + block.text.lstrip()
            else:
                merged_text = prev_stripped + " " + block.text.lstrip()
            merged[-1] = TextBlock(
                text=merged_text,
                x0=min(prev.x0, block.x0),
                y0=prev.y0,
                x1=max(prev.x1, block.x1),
                y1=block.y1,
                font_size=prev.font_size,
                font_name=prev.font_name,
                is_bold=prev.is_bold,
                block_type=prev.block_type,
            )
        else:
            merged.append(block)

    return merged


# ---------------------------------------------------------------------------
# Heading detection
# ---------------------------------------------------------------------------

def detect_headings(blocks: list[TextBlock], body_font_size: float) -> list[TextBlock]:
    heading_blocks = []
    for block in blocks:
        is_heading = False

        if block.font_size > body_font_size * 1.2:
            is_heading = True
        if block.is_bold and 5 < len(block.text) < 100:
            is_heading = True

        if is_heading and block.block_type == "text":
            block.block_type = "heading"
            heading_blocks.append(block)

    return heading_blocks


def get_body_font_size(blocks: list[TextBlock]) -> float:
    sizes = [b.font_size for b in blocks if b.block_type == "text" and b.font_size > 0]
    if not sizes:
        return 10.0
    size_counts: dict[float, int] = {}
    for s in sizes:
        rounded = round(s, 1)
        size_counts[rounded] = size_counts.get(rounded, 0) + 1
    return max(size_counts, key=size_counts.get)


# ---------------------------------------------------------------------------
# Noise / boilerplate line-level cleaning
# ---------------------------------------------------------------------------

# Case-insensitive patterns
_NOISE_LINE_PATTERNS = re.compile(
    r"(?i)"
    r"^copyright\s*©.*|"
    r"^©\s*.+|"
    r"^ISBN\s*.*|"
    r"^ISSN\s*.*|"
    r"^[\d\-\s]{5,20}$|"
    r"^(author|designer|editor|production\s*editor|cover\s*image)\s*[\t ]*.*|"
    r"^an\s+esa\s+production$|"
    r"^(credit|image|photo|source|spacecraft)\s*:.*|"
    r"^\(?(ESA|NASA|JPL|ATG\s*medialab)[/&,;\s\w()]*\)?\s*$|"
    r"^[\w\s]+\((ESA|NASA|JPL)[/&,;\s\w()]*\)(,\s*[\w\s]+\([/&,;\s\w()]*\))*\s*$|"
    r"^.{3,60}[\t ]{2,}\d{1,3}\s*$|"
    r"^\d{1,4}\s*$|"
    r"^Page\s+\d+(\s+of\s+\d+)?\s*$|"
    r"^\d{1,4}\s+Page\s+\d+\s+of\s+\d+\s*$|"
    r"^Publisher.s\s+Note\s+.*|"
    r"^Open\s+Access\s+This\s+article\s+is\s+licensed\s+under.*|"
    r"^(article.s\s+)?Creative\s+Commons\s+licen[sc]e.*|"
    r"^(To\s+view\s+a\s+copy\s+of\s+this\s+licen[sc]e|you\s+will\s+need\s+to\s+obtain).*|"
    r"^(not\s+included\s+in\s+the\s+article.s|regulation\s+or\s+exceeds).*|"
    r"^Extended\s+author\s+information\s+available.*|"
    r"^https?://doi\.org/.*"
)

# Case-SENSITIVE pattern for document codes like BR-174, SP/1240
_DOC_CODE_RE = re.compile(r"^[A-Z]{2,4}[-/]\d{2,4}")

_ROLE_LABEL = re.compile(
    r"(?i)^(author|designer|editor|production\s*editor|cover\s*image)\s*$"
)


def clean_extracted_text(text: str) -> str:
    """Remove metadata, image credits, and boilerplate via line-level filtering."""
    lines = text.splitlines()
    cleaned = []
    skip_next_value = False

    for line in lines:
        stripped = line.strip()

        if not stripped:
            skip_next_value = False
            cleaned.append(line)
            continue

        if _NOISE_LINE_PATTERNS.match(stripped) or _DOC_CODE_RE.match(stripped):
            if _ROLE_LABEL.match(stripped):
                skip_next_value = True
            continue

        if skip_next_value:
            skip_next_value = False
            if len(stripped) < 60:
                continue

        skip_next_value = False
        cleaned.append(line)

    result = re.sub(r"\n{3,}", "\n\n", "\n".join(cleaned))
    return result.strip()


# ---------------------------------------------------------------------------
# Quality scoring
# ---------------------------------------------------------------------------

def compute_quality_score(
    text: str,
    blocks: list[TextBlock],
    content_blocks: list[TextBlock],
    column_info: dict,
    page_area: float,
) -> tuple[float, list[str]]:
    """Compute a 0.0-1.0 quality score for a page extraction."""
    score = 1.0
    issues: list[str] = []

    char_count = len(text)
    if page_area > 0:
        density = char_count / page_area
        if density < 0.0005:
            score -= 0.3
            issues.append("low_text_density")
        elif density < 0.001:
            score -= 0.1
            issues.append("sparse_text")

    total_blocks = len(blocks)
    content_count = len(content_blocks)
    if total_blocks > 0 and content_count / total_blocks < 0.3:
        score -= 0.2
        issues.append("low_content_ratio")

    words = text.split()
    if words:
        long_words = sum(1 for w in words if len(w) > 40)
        long_word_ratio = long_words / len(words)
        if long_word_ratio > 0.05:
            score -= 0.3
            issues.append("garbled_text_detected")
        elif long_words > 0:
            score -= 0.1
            issues.append("possible_garbled_text")

    num_cols = column_info.get("num_columns", 1)
    if num_cols >= 5:
        score -= 0.2
        issues.append("uncertain_layout")
    elif num_cols >= 4:
        score -= 0.1
        issues.append("complex_layout")

    if num_cols >= 2:
        col_sizes = [len(col) for col in column_info.get("columns", [])]
        if col_sizes and max(col_sizes) > 0:
            balance = min(col_sizes) / max(col_sizes)
            if balance < 0.1:
                score -= 0.15
                issues.append("unbalanced_columns")

    return max(0.0, round(score, 2)), issues


# ---------------------------------------------------------------------------
# Single-page extraction
# ---------------------------------------------------------------------------

def extract_page(page: fitz.Page, page_num: int) -> PageResult:
    blocks = get_text_blocks(page)

    if not blocks:
        return PageResult(
            page_num=page_num, layout="empty", text="",
            quality_score=0.0, issues=["empty_page"],
        )

    tag_boilerplate_blocks(blocks)
    tag_figure_captions(blocks)

    # Extract bordered tables and replace overlapping text blocks
    bordered_tables = extract_page_tables(page)
    if bordered_tables:
        blocks = [
            b for b in blocks
            if not any(_blocks_overlap(b, tb) for tb in bordered_tables)
        ]
        blocks.extend(bordered_tables)

    body_size = get_body_font_size(blocks)

    # Detect borderless tables by caption and replace caption+data blocks
    blocks = tag_and_extract_tables(blocks, page, body_size)

    content_blocks = [
        b for b in blocks
        if b.block_type not in ("header_footer", "boilerplate")
    ]
    column_info = detect_columns(content_blocks, page.rect.width, body_size)

    ordered_blocks = assemble_reading_order(column_info)

    # Re-insert table / caption blocks at correct y-position
    extra = [b for b in content_blocks
             if b.block_type in ("table", "figure_caption")
             and b not in ordered_blocks]
    if extra:
        ordered_blocks.extend(extra)
        ordered_blocks.sort(key=lambda b: b.y0)

    ordered_blocks = merge_split_paragraphs(ordered_blocks, body_size)
    headings = detect_headings(ordered_blocks, body_size)

    lines = []
    for block in ordered_blocks:
        if block.block_type == "heading":
            size_ratio = block.font_size / body_size if body_size > 0 else 1.0
            if size_ratio > 1.8:
                prefix = "# "
            elif size_ratio > 1.4:
                prefix = "## "
            else:
                prefix = "### "
            heading_text = fix_heading_spaces(page, block)
            lines.append(f"\n{prefix}{heading_text.strip()}\n")
        elif block.block_type == "table":
            lines.append(f"\n{block.text}\n")
        elif block.block_type == "figure_caption":
            lines.append(f"\n[Figure: {block.text.strip()}]\n")
        else:
            lines.append(block.text.strip())

    raw_text = "\n".join(lines)
    raw_text = clean_extracted_text(raw_text)

    page_area = page.rect.width * page.rect.height
    quality, quality_issues = compute_quality_score(
        raw_text, blocks, content_blocks, column_info, page_area,
    )

    return PageResult(
        page_num=page_num,
        layout=column_info["layout"],
        text=raw_text,
        heading=[h.text for h in headings],
        quality_score=quality,
        issues=quality_issues,
        block_count=len(blocks),
        char_count=len(raw_text),
    )


# ---------------------------------------------------------------------------
# Cross-page paragraph merging
# ---------------------------------------------------------------------------

_TAIL_SECTION_RES = [
    re.compile(r"^#{1,3}\s*References\s*$", re.MULTILINE),
    re.compile(r"^#{1,3}\s*Authors?\s+and\s+Affiliations?\s*$", re.MULTILINE),
    re.compile(r"^#{1,3}\s*Declarations?\s*$", re.MULTILINE),
]


def strip_references_tail(text: str) -> str:
    """Remove References, Author Affiliations, and Declarations sections."""
    earliest = len(text)
    for pattern in _TAIL_SECTION_RES:
        m = pattern.search(text)
        if m and m.start() < earliest:
            earliest = m.start()
    if earliest < len(text):
        text = text[:earliest].rstrip()
    return text


def _try_merge_at(results: list[PageResult], src: int,
                   search_indices: list[int]) -> int:
    """Try to merge one split sentence from *search_indices* into
    *results[src]*.  Returns the target page index on success, -1 on failure.
    """
    prev_text = results[src].text.rstrip()
    if not prev_text:
        return -1

    prev_lines = prev_text.split('\n')
    last_idx = -1
    for k in range(len(prev_lines) - 1, -1, -1):
        if prev_lines[k].strip():
            last_idx = k
            break
    if last_idx < 0:
        return -1
    last_line = prev_lines[last_idx].strip()

    if last_line.startswith('#') or last_line.startswith('['):
        return -1
    if last_line[-1] in '.!?:;)':
        return -1

    # Build search list from the candidate pages
    search_texts = []
    for idx in search_indices:
        t = results[idx].text.strip()
        if t:
            search_texts.append((idx, t))

    first_idx = -1
    target_page_idx = -1
    next_lines: list[str] = []
    for page_idx, st in search_texts:
        candidate_lines = st.split('\n')
        in_block = False
        for k, line in enumerate(candidate_lines):
            stripped = line.strip()
            if stripped.startswith('[Table:') or stripped.startswith('[Figure:'):
                in_block = True
                continue
            if in_block:
                if not stripped:
                    in_block = False
                continue
            if not stripped or stripped.startswith('#') or stripped.startswith('['):
                continue
            first_idx = k
            target_page_idx = page_idx
            next_lines = candidate_lines
            break
        if first_idx >= 0:
            break
    if first_idx < 0:
        return -1
    first_line = next_lines[first_idx].strip()

    # Don't merge into a bullet list item
    if _BULLET_RE.match(first_line):
        return -1

    last_word = last_line.split()[-1] if last_line.split() else ''
    continues = (first_line[0].islower() or
                 (last_word and last_word[-1].islower()) or
                 last_line.endswith('-'))
    if not continues:
        return -1

    # Collect ONLY the sentence fragment that was split at the page
    # boundary.  After reflow each paragraph is one very long line, so
    # we must also truncate *within* the first line at the first
    # sentence-ending punctuation.
    _SENT_END = re.compile(r'[.!?:;)]\s')
    m_sent = _SENT_END.search(first_line)
    if m_sent:
        take = first_line[:m_sent.start() + 1]
        leftover = first_line[m_sent.start() + 1:].lstrip()
    else:
        take = first_line
        leftover = ''

    if last_line.endswith('-'):
        prev_lines[last_idx] = last_line[:-1] + take
    else:
        prev_lines[last_idx] = last_line + ' ' + take

    results[src].text = '\n'.join(prev_lines)

    if leftover:
        next_lines[first_idx] = leftover
    else:
        next_lines.pop(first_idx)
    results[target_page_idx].text = '\n'.join(next_lines)
    return target_page_idx


def merge_cross_page_paragraphs(results: list[PageResult]) -> list[PageResult]:
    """Merge paragraphs that span page boundaries.

    Runs multiple passes (one paragraph per boundary per pass) so that
    a paragraph spanning 3+ pages is reassembled without swallowing
    entire pages.  A final stitching pass handles gaps left by emptied
    intermediate pages.
    """
    if len(results) < 2:
        return results

    for _pass in range(4):
        changed = False
        # Track both sources and targets so a page that just had text
        # removed (target) doesn't immediately become a source — this
        # prevents forward cascades that drain entire pages.
        touched: set[int] = set()
        for i in range(len(results) - 1):
            if i in touched:
                continue
            if not results[i].text.strip():
                continue
            candidates = list(range(i + 1, min(i + 4, len(results))))
            target = _try_merge_at(results, i, candidates)
            if target >= 0:
                touched.add(i)
                touched.add(target)
                changed = True
        if not changed:
            break

    # Final stitching pass: skip over emptied pages to find the next
    # non-empty page and attempt one merge.  Uses the same anti-cascade
    # guard as the main loop.
    non_empty = [i for i in range(len(results)) if results[i].text.strip()]
    touched: set[int] = set()
    for idx in range(len(non_empty) - 1):
        src = non_empty[idx]
        if src in touched:
            continue
        candidates = [non_empty[j] for j in range(idx + 1, min(idx + 4, len(non_empty)))]
        target = _try_merge_at(results, src, candidates)
        if target >= 0:
            touched.add(src)
            touched.add(target)

    return results


# ---------------------------------------------------------------------------
# Full-PDF extraction
# ---------------------------------------------------------------------------

def extract_pdf(pdf_path: str, verbose: bool = False) -> tuple[str, ExtractionReport]:
    doc = fitz.open(pdf_path)
    report = ExtractionReport(total_pages=len(doc))

    page_results = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        result = extract_page(page, page_num + 1)

        if result.layout == "single":
            report.pages_single_col += 1
        elif "columns" in result.layout:
            report.pages_multi_col += 1
        else:
            report.pages_other += 1

        report.total_issues += len(result.issues)
        for issue in result.issues:
            issue_type = issue.split(":")[0]
            report.issue_summary[issue_type] = report.issue_summary.get(issue_type, 0) + 1

        report.page_results.append({
            "page": result.page_num,
            "layout": result.layout,
            "quality": round(result.quality_score, 2),
            "issues": result.issues,
            "headings": result.heading,
            "char": result.char_count,
        })

        page_results.append(result)

        if verbose:
            print(f" Page {result.page_num:3d}: {result.layout:<12s} ")

    doc.close()

    page_results = merge_cross_page_paragraphs(page_results)

    all_text = [f"# Extracted: {Path(pdf_path).name}\n"]
    for result in page_results:
        if result.text.strip():
            all_text.append(
                f"\n---\n*Page {result.page_num} [{result.layout}]"
                f" (quality: {result.quality_score:.2f})*\n"
            )
            all_text.append(result.text)

    qualities = [p["quality"] for p in report.page_results]
    report.avg_quality = statistics.mean(qualities) if qualities else 0.0

    full_text = "\n".join(all_text)
    full_text = strip_references_tail(full_text)

    return full_text, report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Column-aware PDF extractor for RAG systems"
    )
    parser.add_argument("pdf_path", help="Path to PDF file")
    parser.add_argument("--output", "-o", help="Output markdown file path")
    parser.add_argument("--report", "-r", action="store_true",
                        help="Print detailed quality report")
    parser.add_argument("--report-json", help="Save report as JSON")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print per-page progress")
    args = parser.parse_args()

    pdf_path = args.pdf_path
    if not Path(pdf_path).exists():
        print(f"Error: File not found: {pdf_path}")
        return

    print(f"Extracting: {pdf_path}")
    full_text, report = extract_pdf(pdf_path, verbose=args.verbose)

    output_path = args.output or str(Path(pdf_path).with_suffix(".extracted.md"))
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(full_text)
    print(f"Output saved to: {output_path}")

    if args.report or args.verbose:
        print(f"\n{'=' * 60}")
        print(f"EXTRACTION REPORT")
        print(f"{'=' * 60}")
        print(f"Total pages:      {report.total_pages}")
        print(f"Single-column:    {report.pages_single_col}")
        print(f"Multi-column:     {report.pages_multi_col}")
        print(f"Other/empty:      {report.pages_other}")
        print(f"Average quality:  {report.avg_quality:.2f}")
        print(f"Total issues:     {report.total_issues}")
        if report.issue_summary:
            print(f"\nIssue breakdown:")
            for issue, count in sorted(report.issue_summary.items(),
                                       key=lambda x: -x[1]):
                print(f"  {issue}: {count}")

        low_quality = [p for p in report.page_results if p["quality"] < 0.7]
        if low_quality:
            print(f"\nPages needing review ({len(low_quality)}):")
            for p in low_quality:
                print(f"  Page {p['page']}: quality={p['quality']:.2f} "
                      f"issues={p['issues']}")

    if args.report_json:
        with open(args.report_json, "w") as f:
            json.dump({
                "total_pages": report.total_pages,
                "layout_stats": {
                    "single_column": report.pages_single_col,
                    "multi_column": report.pages_multi_col,
                    "other_or_empty": report.pages_other,
                },
                "avg_quality": round(report.avg_quality, 3),
                "total_issues": report.total_issues,
                "issue_summary": report.issue_summary,
                "pages": report.page_results,
            }, f, indent=2)
        print(f"Report saved to: {args.report_json}")


if __name__ == "__main__":
    main()
