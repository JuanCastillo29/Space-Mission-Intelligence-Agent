from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class HealthResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    status: str
    database: bool
    embedder_loaded: bool
    reranker_loaded: bool
    version: str
