from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict


class DocumentOut(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: uuid.UUID
    title: str
    source_type: str
    source_url: str | None
    mission_name: str | None
    chunk_count: int
    ingested_at: datetime
    metadata: dict[str, Any] | None = None


class DocumentListResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    documents: list[DocumentOut]
    total: int
