from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request
from app.schemas.evaluate import (
    EvalConfigSummary,
    EvalRequest,
    EvaluateResponse,
)
from scripts.evaluation.config import eval_settings
from scripts.evaluation.results import load_latest_results
from scripts.evaluation.run import run_evaluation
from scripts.evaluation.schemas import EvalRunResult

log = logging.getLogger(__name__)

router = APIRouter()


def _summarise(result: EvalRunResult) -> EvalConfigSummary:
    r = result.aggregate_metrics.retrieval
    g = result.aggregate_metrics.generation
    return EvalConfigSummary(
        config_name=result.config_name,
        run_id=result.run_id,
        precision_at_5=r.precision_at_k.get(5),
        recall_at_5=r.recall_at_k.get(5),
        mrr=r.mrr,
        faithfulness=g.faithfulness,
        answer_relevancy=g.answer_relevancy,
        citation_accuracy=g.citation_accuracy,
        hallucination_rate=g.hallucination_rate,
        avg_latency_ms=result.aggregate_metrics.avg_latency_ms,
        num_queries=len(result.per_query_results),
        num_passed=sum(1 for q in result.per_query_results if q.passed),
    )


@router.post("/evaluate", response_model=EvaluateResponse)
async def run_eval(
    body: EvalRequest,
    request: Request,
) -> EvaluateResponse:
    try:
        results = await run_evaluation(
            body.configs,
            dataset_path=body.dataset_path,
            output_dir=body.output_dir,
            generate=body.generate,
            ragas_enabled=body.ragas_enabled,
            embedder=request.app.state.embedder,
            reranker=getattr(request.app.state, "reranker", None),
            generation_pipeline=request.app.state.generation_pipeline,
        )
    except KeyError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception:
        log.exception("Evaluation run failed")
        raise HTTPException(status_code=500, detail="Evaluation run failed")

    summaries = [_summarise(r) for r in results]
    out_dir = body.output_dir or eval_settings.EVAL_RESULTS_DIR
    return EvaluateResponse(
        status="completed",
        message=f"Ran {len(results)} config(s) over the golden dataset.",
        summaries=summaries,
        results_dir=out_dir,
    )


@router.get("/evaluate/results", response_model=EvaluateResponse)
async def get_results() -> EvaluateResponse:
    results = load_latest_results(eval_settings.EVAL_RESULTS_DIR)
    if not results:
        return EvaluateResponse(
            status="empty",
            message="No evaluation results found. Run POST /evaluate first.",
        )

    summaries = [_summarise(r) for r in results]
    return EvaluateResponse(
        status="ok",
        message=f"Loaded latest results for {len(results)} config(s).",
        summaries=summaries,
        results_dir=eval_settings.EVAL_RESULTS_DIR,
    )
