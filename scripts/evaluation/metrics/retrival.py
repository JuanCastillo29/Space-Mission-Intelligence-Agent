from __future__ import annotations

from collections import defaultdict
from statistics import mean

from scripts.evaluation.schemas import RetrievalMetrics

def precision_at_k(
        retrieval_chunks_ids: list[str],
        ground_truth_chunks_ids: list[str],
        k: int
) -> float:
    """
    Precision@k = relevant retrieved in top-K / K
    """
    if k <= 0:
        raise ValueError("k must be greater than 0")
    top_k = retrieval_chunks_ids[:k]
    if not top_k:
        return 0.0

    gt = set(ground_truth_chunks_ids)
    relevant = sum(chunk_id in gt for chunk_id in top_k)

    return relevant / k

def recall_at_k(
        retrieval_chunks_ids: list[str],
        ground_truth_chunks_ids: list[str],
        k: int
) -> float:
    """
    Recall@k = relevant retrieved in top-K / total relevant
    """
    gt = set(ground_truth_chunks_ids)

    if not gt:
        return 1.0

    top_k = retrieval_chunks_ids[:k]
    relevant = sum(chunk_id in gt for chunk_id in top_k)
    return relevant / len(gt)

def mean_reciprocal_rank(
        retrieval_chunks_ids: list[str],
        ground_truth_chunks_ids: list[str],
) -> float:
    """
    Reciprocal rank of first relevant document
    """
    gt = set(ground_truth_chunks_ids)

    for rank, chunk_id in enumerate(retrieval_chunks_ids, start=1):
        if chunk_id in gt:
            return 1.0/rank

    return 0.0

def compute_retrieval_metrics(
        retrieval_chunks_ids: list[str],
        ground_truth_chunks_ids: list[str],
        ks: tuple[int, ...] = (3, 5, 10),
) -> RetrievalMetrics:
    """
    Compute retrieval metrics for a single query
    """
    precision = {
        k: precision_at_k(retrieval_chunks_ids, ground_truth_chunks_ids, k) for k in ks
    }

    recall = {
        k: recall_at_k(retrieval_chunks_ids, ground_truth_chunks_ids, k) for k in ks
    }

    mrr = mean_reciprocal_rank(retrieval_chunks_ids, ground_truth_chunks_ids)

    return RetrievalMetrics(
        precision_at_k=precision,
        recall_at_k=recall,
        mrr=mrr,
    )

def aggregate_retrieval_metrics(
    metrics: list[RetrievalMetrics],
) -> RetrievalMetrics:
    """
    Average retrieval metrics across all evaluation examples.
    """

    if not metrics:
        return RetrievalMetrics(
            precision_at_k={},
            recall_at_k={},
            mrr=0.0,
        )

    precision_values: dict[int, list[float]] = defaultdict(list)
    recall_values: dict[int, list[float]] = defaultdict(list)

    for result in metrics:
        for k, value in result.precision_at_k.items():
            precision_values[k].append(value)

        for k, value in result.recall_at_k.items():
            recall_values[k].append(value)

    avg_precision = {
        k: mean(values)
        for k, values in precision_values.items()
    }

    avg_recall = {
        k: mean(values)
        for k, values in recall_values.items()
    }

    avg_mrr = mean(result.mrr for result in metrics)

    return RetrievalMetrics(
        precision_at_k=avg_precision,
        recall_at_k=avg_recall,
        mrr=avg_mrr,
    )