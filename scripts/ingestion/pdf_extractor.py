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
 - Text healing (merged words, garbled headings, noise removal)
 - Structured markdown ready for RAG chunking

Usage:
    python pdf_extractor.py input.pdf [--output output.md] [--report]
"""

import re
import fitz
import json
import statistics
import argparse
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class TextBlock:
    """
    Textblock dataclass with spatial and font metadata
    """
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
        return (self.x1 + self.x0)/2

    @property
    def center_y(self):
        return (self.y1 + self.y0)/2

    @property
    def width(self):
        return abs(self.x1 - self.x0)

    @property
    def height(self):
        return abs(self.y1 - self.y0)

@dataclass
class PageResult:
    """
    Extraction result for a single page.
    """
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
    """
    Overall quality report
    """
    total_pages: int = 0
    pages_single_col: int = 0
    pages_multi_col: int = 0
    pages_other: int = 0
    avg_quality: float = 0.0
    total_issues: int = 0
    issue_summary: dict = field(default_factory=dict)
    page_results: list = field(default_factory=list)

def get_text_blocks(page: fitz.Page, vmargin_pct: float = 0.05) -> list[TextBlock]:
    page_height = page.rect.height
    top_margin = page_height * vmargin_pct
    bottom_margin = page_height * (1 - vmargin_pct)

    # Build tight bboxes from word-level glyph positions
    word_bboxes = {}
    for w in page.get_text("words"):
        # w = (x0, y0, x1, y1, text, block_no, line_no, word_no)
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

        # Use word-level tight bbox if available, fall back to block bbox
        if block_idx in word_bboxes:
            x0, y0, x1, y1 = word_bboxes[block_idx]
        else:
            x0, y0, x1, y1 = block["bbox"]

        is_hf = y0 < top_margin or y1 > bottom_margin

        full_text = ""
        font_sizes = []
        font_names = []
        bold_count = 0
        total_spans = 0

        for line in block.get("lines", []):
            line_text = ""
            for span in line.get("spans", []):
                span_text = span.get("text", "")
                line_text += span_text
                font_sizes.append(span.get("size", 0))
                font_names.append(span.get("font", ""))
                total_spans += 1
                if "bold" in span.get("font", "").lower() or span.get("flags", 0) & 2**4:
                    bold_count += 1

            full_text += line_text + "\n"
            if not full_text:
                continue
        full_text = full_text.strip()
        if not full_text:
            continue

        avg_font_size = statistics.mean(font_sizes) if font_sizes else 0.0
        dominant_font = max(set(font_names), key=font_names.count) if font_names else None
        is_bold = bold_count > total_spans / 2 if total_spans > 0 else False

        block_type = "header_footer" if is_hf else "text"

        blocks.append(TextBlock(
            text=full_text,
            x0=x0, y0=y0, x1=x1, y1=y1,
            font_size=avg_font_size,
            font_name=dominant_font,
            is_bold=is_bold,
            block_type=block_type
        ))
    return blocks

def _find_column_gaps(blocks: list[TextBlock], page_width: float,
                      min_gap_pct: float = 0.03) -> list[dict]:
    """
    Find all significant vertical whitespace gaps that may separate columns.

    Merges overlapping block x-ranges first, then returns every gap whose
    width exceeds *min_gap_pct* of the page width.
    """
    if not blocks:
        return []

    # Sort x-ranges and merge overlapping / touching spans
    x_ranges = sorted([(b.x0, b.x1) for b in blocks], key=lambda r: r[0])
    merged = [list(x_ranges[0])]
    for start, end in x_ranges[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])

    min_gap = page_width * min_gap_pct
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
                   body_font_size : float,
                   min_gap_pct: float = 0.03,
                   full_width_pct: float = 0.6,
                   font_tolerance: float = 0.15) -> dict:
    """
    Detect N-column layout by finding all significant vertical gaps.

    Returns
    -------
    dict with keys:
        layout           : str   – "single", "two_columns", "three_columns", …
        num_columns       : int
        column_boundaries : list[float]  – gap midpoints separating columns
        columns           : list[list[TextBlock]]  – blocks per column (sorted by y0)
        full_width_blocks : list[TextBlock]  – blocks spanning most of the page
    """
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

    # Each gap becomes a column boundary
    boundaries = sorted(g["midpoint"] for g in gaps)
    num_columns = len(boundaries) + 1

    # Assign narrow blocks to columns based on center_x
    columns: list[list[TextBlock]] = [[] for _ in range(num_columns)]
    full_width_blocks: list[TextBlock] = []

    for block in content_blocks:
        start_col = 0
        end_col = 0
        for i, boundary in enumerate(boundaries):
            if block.x0 >= boundary:
                start_col = i+1
            if block.x1 >= boundary:
                end_col = i+1
        if start_col == end_col:
            columns[start_col].append(block)
        else:
            full_width_blocks.append(block)

    # Sort each column top-to-bottom
    for col in columns:
        col.sort(key=lambda b: b.y0)
    full_width_blocks.sort(key=lambda b: b.y0)

    # Drop empty columns that could arise from noisy gaps
    non_empty = [(i, col) for i, col in enumerate(columns) if col]
    if len(non_empty) <= 1:
        return _single(content_blocks)
    # Rebuild columns & boundaries after pruning
    if len(non_empty) < num_columns:
        columns = [col for _, col in non_empty]
        kept_indices = {i for i, _ in non_empty}
        # Keep only boundaries that separate two kept columns
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
    """
    Assemble blocks in correct reading order for N-column layouts.

    Within each vertical section (delimited by full-width blocks) the
    columns are read left-to-right, each top-to-bottom.
    """
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

    # Remaining blocks after the last full-width block
    for col in columns:
        remaining = [b for b in col if b.y0 >= prev_y]
        ordered.extend(remaining)

    return ordered

def detect_headings(blocks: list[TextBlock], body_font_size: float) -> list[TextBlock]:
    """
    Classify blocks as headings based on font size and style.
    Modifies blocks in place and return heading blocks
    """
    heading_blocks = []
    for block in blocks:
        is_heading = False

        if block.font_size > body_font_size * 1.2:
            is_heading = True
        if block.is_bold and 5 < len(block.text) < 100:
            is_heading = True

        if is_heading:
            block.block_type = "heading"
            heading_blocks.append(block)

    return heading_blocks

def get_body_font_size(blocks: list[TextBlock]) -> float:
    """
    Determines the most common (body text) font size.
    """
    sizes = [b.font_size for b in blocks if b.block_type == "text" and b.font_size > 0]
    if not sizes:
        return 10.0
    size_counts = {}
    for s in sizes:
        rounded = round(s, 1)
        size_counts[rounded] = size_counts.get(rounded, 0) + 1
    return max(size_counts, key=size_counts.get)

# Each pattern matches a single line of text that should be removed.

_NOISE_LINE_PATTERNS = re.compile(
    r"(?i)"
    r"^copyright\s*©.*|"
    r"^©\s*.+|"
    r"^ISBN\s*.*|"
    r"^ISSN\s*.*|"
    r"^[\d\-\s]{5,20}$|"
    r"^(author|designer|editor|production\s*editor|cover\s*image)\s*[\t ]*.*|"
    r"^an\s+esa\s+production$|"
    r"^[A-Z]{2,4}[-/]\d{2,4}.*|"
    r"^(credit|image|photo|source|spacecraft)\s*:.*|"
    r"^\(?(ESA|NASA|JPL|ATG\s*medialab)[/&,;\s\w()]*\)?\s*$|"
    r"^[\w\s]+\((ESA|NASA|JPL)[/&,;\s\w()]*\)(,\s*[\w\s]+\([/&,;\s\w()]*\))*\s*$|"
    r"^.{3,60}[\t ]{2,}\d{1,3}\s*$"
)

_ROLE_LABEL = re.compile(
    r"(?i)^(author|designer|editor|production\s*editor|cover\s*image)\s*$"
)


def clean_extracted_text(text: str) -> str:
    """
    Remove metadata, image credits, TOC entries, and colophon sections
    from extracted page text via line-level filtering.

    Works in two passes:
      1. Remove any line that directly matches a noise pattern.
      2. Remove orphan "value" lines that followed a role label
         (e.g. "Stuart Clark" after "Author").
    """
    lines = text.splitlines()
    cleaned = []
    skip_next_value = False

    for line in lines:
        stripped = line.strip()

        if not stripped:
            skip_next_value = False
            cleaned.append(line)
            continue

        if _NOISE_LINE_PATTERNS.match(stripped):
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


def extract_page(page: fitz.Page, page_num: int) -> PageResult:
    """
    Extract a single page with colum-aware reading order
    """
    blocks = get_text_blocks(page)

    if not blocks:
        return PageResult(
            page_num=page_num,
            layout="empty",
            text = "",
            quality_score=0.0,
            issues=["empty_page"]
        )

    body_size = get_body_font_size(blocks)
    content_blocks = [b for b in blocks if b.block_type != "header_footer"]
    column_info = detect_columns(content_blocks, page.rect.width, body_size)

    ordered_blocks = assemble_reading_order(column_info)

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
            lines.append(f"\n{prefix}{block.text.strip()}\n")
        else:
            lines.append(block.text.strip())

    raw_text = "\n".join(lines)

    raw_text = clean_extracted_text(raw_text)

    return PageResult(
        page_num=page_num,
        layout=column_info["layout"],
        text=raw_text,
        heading=[h.text for h in headings],
        block_count=len(blocks),
        char_count=len(raw_text),
    )

def extract_pdf(pdf_path: str, verbose: bool = False) -> tuple[str, ExtractionReport]:
    """
    Extract an entire PDF with column aware reading order.
    """
    doc = fitz.open(pdf_path)
    report = ExtractionReport(total_pages=len(doc))

    all_text = []
    all_text.append(f"# Extracted: {Path(pdf_path).name}\n")

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

        if result.text.strip():
            all_text.append(f"\n---\n*Page {result.page_num} [{result.layout}] (quality: {result.quality_score:.2f})*\n")
            all_text.append(result.text)

        if verbose:
            status = "OK" if result.quality_score > 0.7 else "WARN" if result.quality_score > 0.4 else "FAIL"
            print(f" Page {result.page_num:3d}: {result.layout:<12s} "
                  )
    doc.close()

    qualities = [p["quality"] for p in report.page_results]
    report.avg_quality = statistics.mean(qualities) if qualities else 0.0

    full_text= "\n".join(all_text)
    return full_text, report


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

    # Save output
    output_path = args.output or str(Path(pdf_path).with_suffix(".extracted.md"))
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(full_text)
    print(f"Output saved to: {output_path}")

    # Print report
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

        # Flag low-quality pages
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
