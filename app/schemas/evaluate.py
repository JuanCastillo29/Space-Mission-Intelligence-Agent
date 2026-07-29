from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class EvalRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    configs: list[str] | None = None
    dataset_path: str | None = None
    output_dir: str | None = None
    generate: bool = True
    ragas_enabled: bool | None = None


class EvalConfigSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    config_name: str
    run_id: str
    precision_at_5: float | None = None
    recall_at_5: float | None = None
    mrr: float
    faithfulness: float | None = None
    answer_relevancy: float | None = None
    citation_accuracy: float
    hallucination_rate: float
    avg_latency_ms: float
    num_queries: int
    num_passed: int


class EvaluateResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    status: str
    message: str
    summaries: list[EvalConfigSummary] = []
    results_dir: str | None = None
