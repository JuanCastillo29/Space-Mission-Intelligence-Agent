from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class EvaluateResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    status: str
    message: str
