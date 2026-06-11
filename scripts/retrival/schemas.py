from __future__ import annotations

import uuid
from typing import Any

from pydantic import BaseModel, ConfigDict


class ScoredChunk(BaseModel):
    model_config = ConfigDict(frozen=True)

    chunk_id: uuid.UUID
    document_id: uuid.UUID
    content: str
    score: float
    chunk_index: int
    section_path: str | None = None
    token_count: int = 0
    metadata_: dict[str, Any] = {}
    embedding: list[float] | None = None


class ContextBlock(BaseModel):
    model_config = ConfigDict(frozen=True)

    ref_index: int
    content: str
    source_title: str
    section_path: str | None
    document_id: uuid.UUID
    chunk_id: uuid.UUID
    score: float


class RetrievalResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    context_text: str
    blocks: list[ContextBlock]
    query: str
    semantic_count: int
    keyword_count: int
    fused_count: int
    reranked_count: int
    final_count: int
