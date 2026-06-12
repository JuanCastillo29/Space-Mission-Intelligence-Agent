from __future__ import annotations

import numpy as np

from scripts.retrival.schemas import ContextBlock, RetrievalResult, ScoredChunk

DEFAULT_TOP_K = 5
MMR_LAMBDA = 0.7


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    va = np.asarray(a, dtype=np.float32)
    vb = np.asarray(b, dtype=np.float32)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    if denom == 0:
        return 0.0
    return float(np.dot(va, vb) / denom)


def mmr_diversity_filter(
    chunks: list[ScoredChunk],
    *,
    top_k: int = DEFAULT_TOP_K,
    lambda_param: float = MMR_LAMBDA,
) -> list[ScoredChunk]:
    if not chunks:
        return []

    candidates = [c for c in chunks if c.embedding is not None]
    no_embedding = [c for c in chunks if c.embedding is None]

    if not candidates:
        return (no_embedding + candidates)[:top_k]

    selected: list[ScoredChunk] = []
    remaining = list(candidates)

    while remaining and len(selected) < top_k:
        best_idx = 0
        best_mmr = float("-inf")

        for i, candidate in enumerate(remaining):
            relevance = candidate.score

            if selected:
                max_sim = max(
                    _cosine_similarity(candidate.embedding, s.embedding)  # type: ignore[arg-type]
                    for s in selected
                )
            else:
                max_sim = 0.0

            mmr = lambda_param * relevance - (1 - lambda_param) * max_sim

            if mmr > best_mmr:
                best_mmr = mmr
                best_idx = i

        selected.append(remaining.pop(best_idx))

    return selected


def assemble_context(
    query: str,
    chunks: list[ScoredChunk],
    source_titles: dict[str, str],
    *,
    semantic_count: int = 0,
    keyword_count: int = 0,
    fused_count: int = 0,
    reranked_count: int = 0,
) -> RetrievalResult:
    blocks: list[ContextBlock] = []
    context_parts: list[str] = []

    for i, chunk in enumerate(chunks, start=1):
        doc_id_str = str(chunk.document_id)
        title = source_titles.get(doc_id_str, "Unknown Source")

        section = chunk.section_path or ""
        header = f"[{i}] {title}"
        if section:
            header += f" — {section}"

        context_parts.append(f"{header}\n{chunk.content}")

        blocks.append(
            ContextBlock(
                ref_index=i,
                content=chunk.content,
                source_title=title,
                section_path=chunk.section_path,
                document_id=chunk.document_id,
                chunk_id=chunk.chunk_id,
                score=chunk.score,
            )
        )

    return RetrievalResult(
        context_text="\n\n".join(context_parts),
        blocks=blocks,
        query=query,
        semantic_count=semantic_count,
        keyword_count=keyword_count,
        fused_count=fused_count,
        reranked_count=reranked_count,
        final_count=len(blocks),
    )
