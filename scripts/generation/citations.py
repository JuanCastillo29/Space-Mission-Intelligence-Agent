from __future__ import annotations

import re

from scripts.retrival.schemas import ContextBlock

from scripts.generation.schemas import Citation

_CITATION_RE = re.compile(r"\[(\d+)]")
_SOURCES_SECTION_RE = re.compile(
    r"\n*##?\s*Sources.*",
    re.DOTALL | re.IGNORECASE,
)


def extract_citation_refs(text: str) -> set[int]:
    return {int(m.group(1)) for m in _CITATION_RE.finditer(text)}


def validate_citations(text: str, blocks: list[ContextBlock]) -> str:
    valid_refs = {b.ref_index for b in blocks}

    def _replace(match: re.Match[str]) -> str:
        ref = int(match.group(1))
        return match.group(0) if ref in valid_refs else ""

    cleaned = _CITATION_RE.sub(_replace, text)
    cleaned = re.sub(r"  +", " ", cleaned)
    return cleaned.strip()


def build_sources_section(
    text: str,
    blocks: list[ContextBlock],
) -> tuple[str, list[Citation]]:
    used_refs = extract_citation_refs(text)
    block_map = {b.ref_index: b for b in blocks}

    citations: list[Citation] = []
    lines: list[str] = ["## Sources"]

    for ref in sorted(used_refs):
        block = block_map.get(ref)
        if block is None:
            continue

        entry = f"- [{ref}] {block.source_title}"
        if block.section_path:
            entry += f" — {block.section_path}"
        lines.append(entry)

        citations.append(
            Citation(
                ref_index=ref,
                source_title=block.source_title,
                section_path=block.section_path,
                document_id=block.document_id,
            )
        )

    sources_text = "\n".join(lines)
    return sources_text, citations


def postprocess_response(
    raw_text: str,
    blocks: list[ContextBlock],
) -> tuple[str, list[Citation], str]:
    body = _SOURCES_SECTION_RE.sub("", raw_text).strip()
    body = validate_citations(body, blocks)
    sources_section, citations = build_sources_section(body, blocks)
    answer = f"{body}\n\n{sources_section}" if citations else body
    return answer, citations, sources_section
