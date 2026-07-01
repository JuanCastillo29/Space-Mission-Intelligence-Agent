from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

Difficulty = Literal["easy", "medium", "hard"]
Category = Literal[
    "single_doc_factual",
    "cross_doc_comparison",
    "multi_source_reasoning",
    "structured_data",
    "unanswerable",
]


class GoldenQAPair(BaseModel):
    model_config = ConfigDict(frozen=True)

    query: str
    expected_answer: str
    ground_truth_chunk_ids: list[str]
    difficulty: Difficulty
    category: Category


class GoldenDataset(BaseModel):
    model_config = ConfigDict(frozen=True)

    version: str
    created_at: str
    pairs: list[GoldenQAPair]


class RetrievalMetrics(BaseModel):
    model_config = ConfigDict(frozen=True)

    precision_at_k: dict[int, float]
    recall_at_k: dict[int, float]
    mrr: float


class GenerationMetrics(BaseModel):
    model_config = ConfigDict(frozen=True)

    faithfulness: float | None = None
    answer_relevancy: float | None = None
    context_precision: float | None = None
    context_recall: float | None = None
    citation_accuracy: float
    hallucination_rate: float
    unanswerable_detection_rate: float | None = None


class SingleEvalResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    query: str
    expected_answer: str
    generated_answer: str

    retrieved_chunk_ids: list[str]

    retrieval_metrics: RetrievalMetrics
    generation_metrics: GenerationMetrics

    latency_ms: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None

    passed: bool
    notes: str | None = None


class AggregateMetrics(BaseModel):
    model_config = ConfigDict(frozen=True)

    retrieval: RetrievalMetrics
    generation: GenerationMetrics

    avg_latency_ms: float
    avg_total_tokens: float


class EvalRunResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    run_id: str
    timestamp: str

    config_name: str
    config_dict: dict[str, Any]

    aggregate_metrics: AggregateMetrics
    per_query_results: list[SingleEvalResult]

    total_latency_ms: float
    total_tokens: int
