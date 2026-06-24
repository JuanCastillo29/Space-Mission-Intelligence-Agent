from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class IngestRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    file_path: str


class IngestResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    success: bool
    message: str
    document_title: str | None = None
