"""
Markdown chunker for RAG system
--------------------------------

Splits extracted markdown (from pdf_extractor) into token-bounded chunks
suitable for embedding and retrieval.

Algorithm:
 1. Strip extraction metadata (page separators, file header)
 2. Parse heading hierarchy into sections with section_path
 3. Identify atomic blocks (tables, figure captions, paragraphs, sentences)
 4. Greedy assembly with token-based overlap
 5. Build ChunkResult objects with metadata
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import tiktoken


@dataclass
class ChunkResult:
    content: str
    chunk_index: int
    section_path: str | None
    token_count: int
    metadata: dict = field(default_factory=dict)


@dataclass
class _Section:
    title: str | None
    level: int
    path: str | None
    blocks: list[_Block] = field(default_factory=list)


@dataclass
class _Block:
    text: str
    kind: str


_ENCODER_CACHE: dict[str, tiktoken.Encoding] = {}


def _get_encoder(encoding_name: str = "cl100k_base") -> tiktoken.Encoding:
    if encoding_name not in _ENCODER_CACHE:
        _ENCODER_CACHE[encoding_name] = tiktoken.get_encoding(encoding_name)
    return _ENCODER_CACHE[encoding_name]


def count_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    if not text:
        return 0
    return len(_get_encoder(encoding_name).encode(text))


_PAGE_SEPARATOR_RE = re.compile(r"\n?---\n\*Page \d+.*?\*\n?", re.DOTALL)
_EXTRACTED_HEADER_RE = re.compile(r"^# Extracted:.*\n*")


def _strip_extraction_metadata(markdown: str) -> str:
    text = _EXTRACTED_HEADER_RE.sub("", markdown)
    text = _PAGE_SEPARATOR_RE.sub("\n", text)
    return text.strip()


_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$")


def _parse_sections(text: str) -> list[_Section]:
    lines = text.split("\n")
    sections: list[_Section] = []
    heading_stack: list[tuple[int, str]] = []
    current_lines: list[str] = []

    def _flush(path: str | None, title: str | None, level: int) -> None:
        body = "\n".join(current_lines).strip()
        if body or title:
            section = _Section(title=title, level=level, path=path)
            section.blocks = _extract_blocks(body) if body else []
            sections.append(section)

    current_path: str | None = None
    current_title: str | None = None
    current_level = 0

    for line in lines:
        m = _HEADING_RE.match(line)
        if m:
            _flush(current_path, current_title, current_level)
            current_lines = []

            level = len(m.group(1))
            title = m.group(2).strip()

            heading_stack = [(lvl, ttl) for lvl, ttl in heading_stack if lvl < level]
            heading_stack.append((level, title))

            current_path = " > ".join(t for _, t in heading_stack)
            current_title = title
            current_level = level
        else:
            current_lines.append(line)

    _flush(current_path, current_title, current_level)
    return sections


_TABLE_ROW_RE = re.compile(r"^\|.*\|$")
_FIGURE_RE = re.compile(r"^\[Figure:.*\]$")


def _extract_blocks(text: str) -> list[_Block]:
    lines = text.split("\n")
    blocks: list[_Block] = []
    i = 0

    while i < len(lines):
        line = lines[i]

        if _TABLE_ROW_RE.match(line.strip()):
            table_lines = []
            while i < len(lines) and _TABLE_ROW_RE.match(lines[i].strip()):
                table_lines.append(lines[i])
                i += 1
            blocks.append(_Block(text="\n".join(table_lines), kind="table"))
            continue

        if _FIGURE_RE.match(line.strip()):
            blocks.append(_Block(text=line.strip(), kind="figure"))
            i += 1
            continue

        if line.strip() == "":
            i += 1
            continue

        para_lines = []
        while i < len(lines):
            ln = lines[i]
            if (
                ln.strip() == ""
                or _TABLE_ROW_RE.match(ln.strip())
                or _FIGURE_RE.match(ln.strip())
            ):
                break
            para_lines.append(ln)
            i += 1
        if para_lines:
            blocks.append(_Block(text="\n".join(para_lines), kind="paragraph"))

    return blocks


_ABBREVIATIONS = frozenset(
    {
        "Fig",
        "fig",
        "Figs",
        "figs",
        "e.g",
        "i.e",
        "etc",
        "al",
        "Dr",
        "Mr",
        "Ms",
        "Prof",
        "No",
        "Vol",
        "vs",
        "Eq",
        "eq",
        "approx",
        "cf",
        "Ref",
        "ref",
        "Sect",
        "sect",
        "Ch",
        "ch",
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
        "Inc",
        "Corp",
        "Ltd",
    }
)

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def _split_sentences(text: str) -> list[str]:
    raw_parts = _SENTENCE_SPLIT_RE.split(text)
    if len(raw_parts) <= 1:
        return raw_parts

    merged: list[str] = []
    for part in raw_parts:
        if merged and _ends_with_abbreviation(merged[-1]):
            merged[-1] = merged[-1] + " " + part
        else:
            merged.append(part)
    return merged


def _ends_with_abbreviation(text: str) -> bool:
    m = re.search(r"(\w+)\.\s*$", text)
    if not m:
        return False
    return m.group(1) in _ABBREVIATIONS


def _compute_overlap_text(content: str, overlap_tokens: int, encoding_name: str) -> str:
    if overlap_tokens <= 0:
        return ""
    enc = _get_encoder(encoding_name)
    tokens = enc.encode(content)
    if len(tokens) <= overlap_tokens:
        return content
    overlap_token_ids = tokens[-overlap_tokens:]
    return enc.decode(overlap_token_ids)


def _assemble_chunks(
    sections: list[_Section],
    max_tokens: int,
    overlap_tokens: int,
    encoding_name: str,
    caller_metadata: dict | None,
) -> list[ChunkResult]:
    results: list[ChunkResult] = []
    chunk_index = 0

    for section in sections:
        if not section.blocks:
            continue

        flat_blocks = _flatten_blocks(section.blocks, max_tokens, encoding_name)
        if not flat_blocks:
            continue

        current_parts: list[str] = []
        current_tokens = 0
        has_table = False
        has_figure = False

        def _emit() -> None:
            nonlocal chunk_index, current_parts, current_tokens
            nonlocal has_table, has_figure
            if not current_parts:
                return
            content = "\n\n".join(current_parts).strip()
            if not content:
                return
            token_count = count_tokens(content, encoding_name)
            meta = dict(caller_metadata) if caller_metadata else {}
            if has_table:
                meta["has_table"] = True
            if has_figure:
                meta["has_figure"] = True
            if token_count > max_tokens:
                meta["is_oversized"] = True
            results.append(
                ChunkResult(
                    content=content,
                    chunk_index=chunk_index,
                    section_path=section.path,
                    token_count=token_count,
                    metadata=meta,
                )
            )
            chunk_index += 1

        for block_text, kind in flat_blocks:
            block_tokens = count_tokens(block_text, encoding_name)

            if kind == "table" and block_tokens > max_tokens:
                _emit()
                current_parts = [block_text]
                current_tokens = block_tokens
                has_table = True
                has_figure = False
                _emit()
                current_parts = []
                current_tokens = 0
                has_table = False
                has_figure = False
                continue

            if current_tokens + block_tokens <= max_tokens:
                current_parts.append(block_text)
                current_tokens += block_tokens
                if kind == "table":
                    has_table = True
                if kind == "figure":
                    has_figure = True
            else:
                _emit()
                overlap_text = ""
                if results and overlap_tokens > 0:
                    overlap_text = _compute_overlap_text(
                        results[-1].content, overlap_tokens, encoding_name
                    )
                if overlap_text:
                    current_parts = [overlap_text, block_text]
                    current_tokens = count_tokens(
                        "\n\n".join(current_parts), encoding_name
                    )
                else:
                    current_parts = [block_text]
                    current_tokens = block_tokens
                has_table = kind == "table"
                has_figure = kind == "figure"

        _emit()

    return results


def _flatten_blocks(
    blocks: list[_Block], max_tokens: int, encoding_name: str
) -> list[tuple[str, str]]:
    flat: list[tuple[str, str]] = []
    for block in blocks:
        if block.kind in ("table", "figure"):
            flat.append((block.text, block.kind))
        else:
            block_tokens = count_tokens(block.text, encoding_name)
            if block_tokens <= max_tokens:
                flat.append((block.text, block.kind))
            else:
                sentences = _split_sentences(block.text)
                for sent in sentences:
                    sent = sent.strip()
                    sent_tokens = count_tokens(sent, encoding_name)
                    if sent_tokens <= max_tokens:
                        flat.append((sent, "sentence"))
                    else:
                        for fragment in _split_by_tokens(
                            sent, max_tokens, encoding_name
                        ):
                            flat.append((fragment, "fragment"))
    return flat


def _split_by_tokens(text: str, max_tokens: int, encoding_name: str) -> list[str]:
    enc = _get_encoder(encoding_name)
    tokens = enc.encode(text)
    parts: list[str] = []
    for i in range(0, len(tokens), max_tokens):
        parts.append(enc.decode(tokens[i : i + max_tokens]))
    return parts


def chunk_markdown(
    markdown: str,
    *,
    max_tokens: int = 1024,
    overlap_tokens: int = 128,
    encoding_name: str = "cl100k_base",
    metadata: dict | None = None,
) -> list[ChunkResult]:
    if not markdown or not markdown.strip():
        return []

    cleaned = _strip_extraction_metadata(markdown)
    if not cleaned:
        return []

    sections = _parse_sections(cleaned)
    if not sections:
        return []

    return _assemble_chunks(
        sections, max_tokens, overlap_tokens, encoding_name, metadata
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Markdown chunker for RAG systems")
    parser.add_argument("input_md", help="Path to extracted markdown file")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--overlap", type=int, default=128)
    parser.add_argument("--encoding", default="cl100k_base")
    parser.add_argument("--output-json", "-o", help="Save chunks as JSON")
    args = parser.parse_args()

    md_path = Path(args.input_md)
    if not md_path.exists():
        print(f"Error: File not found: {md_path}")
        return

    markdown = md_path.read_text(encoding="utf-8")
    chunks = chunk_markdown(
        markdown,
        max_tokens=args.max_tokens,
        overlap_tokens=args.overlap,
        encoding_name=args.encoding,
    )

    print(f"Produced {len(chunks)} chunks")
    for c in chunks:
        print(
            f"  [{c.chunk_index}] {c.token_count:>5} tokens"
            f"  path={c.section_path!r}"
            f"  meta={c.metadata}"
        )

    if args.output_json:
        data = [
            {
                "content": c.content,
                "chunk_index": c.chunk_index,
                "section_path": c.section_path,
                "token_count": c.token_count,
                "metadata": c.metadata,
            }
            for c in chunks
        ]
        Path(args.output_json).write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"Saved to {args.output_json}")


if __name__ == "__main__":
    main()
