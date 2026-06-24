from __future__ import annotations

from fastapi import APIRouter

from app.schemas.evaluate import EvaluateResponse

router = APIRouter()


@router.post("/evaluate", response_model=EvaluateResponse, status_code=501)
async def run_evaluation() -> EvaluateResponse:
    return EvaluateResponse(
        status="not_implemented",
        message="Evaluation framework not yet integrated.",
    )
