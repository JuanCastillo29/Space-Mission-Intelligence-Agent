from __future__ import annotations

import uuid
from enum import Enum

from pydantic import BaseModel, ConfigDict


class QueryType(str, Enum):
    RETRIEVAL = "retrieval"
    STRUCTURED = "structured"
    HYBRID = "hybrid"


class RoutingResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    query: str
    query_type: QueryType
    confidence: float
    reasoning: str


class ChatMessage(BaseModel):
    model_config = ConfigDict(frozen=True)

    role: str
    content: str


class Citation(BaseModel):
    model_config = ConfigDict(frozen=True)

    ref_index: int
    source_title: str
    section_path: str | None
    document_id: uuid.UUID


class GenerationResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    answer: str
    citations: list[Citation]
    sources_section: str
    raw_llm_output: str
    query: str
    query_type: QueryType
    model_name: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    latency_ms: float | None = None
