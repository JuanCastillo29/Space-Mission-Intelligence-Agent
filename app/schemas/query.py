from __future__ import annotations

import uuid

from pydantic import BaseModel, ConfigDict


class QueryRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    question: str


class CitationOut(BaseModel):
    model_config = ConfigDict(frozen=True)

    ref_index: int
    source_title: str
    section_path: str | None
    document_id: uuid.UUID


class QueryResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    answer: str
    citations: list[CitationOut]
    sources_section: str
    query_type: str
    model_name: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    latency_ms: float | None = None
