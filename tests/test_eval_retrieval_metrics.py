"""Tests for evaluation retrieval metrics — pure functions, no DB required."""

import pytest

from scripts.evaluation.metrics.retrival import (
    aggregate_retrieval_metrics,
    compute_retrieval_metrics,
    mean_reciprocal_rank,
    precision_at_k,
    recall_at_k,
)
from scripts.evaluation.schemas import RetrievalMetrics


class TestPrecisionAtK:
    def test_all_relevant(self):
        assert precision_at_k(["a", "b", "c"], ["a", "b", "c"], k=3) == 1.0

    def test_none_relevant(self):
        assert precision_at_k(["x", "y", "z"], ["a", "b"], k=3) == 0.0

    def test_partial(self):
        assert precision_at_k(["a", "x", "b"], ["a", "b"], k=3) == pytest.approx(2 / 3)

    def test_k_larger_than_retrieved(self):
        assert precision_at_k(["a"], ["a", "b"], k=5) == pytest.approx(1 / 5)

    def test_empty_retrieved(self):
        assert precision_at_k([], ["a"], k=3) == 0.0

    def test_k_zero_raises(self):
        with pytest.raises(ValueError):
            precision_at_k(["a"], ["a"], k=0)

    def test_only_counts_top_k(self):
        retrieved = ["x", "y", "a"]
        assert precision_at_k(retrieved, ["a"], k=2) == 0.0
        assert precision_at_k(retrieved, ["a"], k=3) == pytest.approx(1 / 3)


class TestRecallAtK:
    def test_all_recalled(self):
        assert recall_at_k(["a", "b"], ["a", "b"], k=2) == 1.0

    def test_none_recalled(self):
        assert recall_at_k(["x", "y"], ["a", "b"], k=2) == 0.0

    def test_partial(self):
        assert recall_at_k(["a", "x"], ["a", "b"], k=2) == 0.5

    def test_empty_ground_truth_returns_one(self):
        assert recall_at_k(["a", "b"], [], k=3) == 1.0

    def test_k_smaller_than_ground_truth(self):
        assert recall_at_k(["a"], ["a", "b", "c"], k=1) == pytest.approx(1 / 3)

    def test_empty_retrieved(self):
        assert recall_at_k([], ["a"], k=3) == 0.0


class TestMRR:
    def test_first_is_relevant(self):
        assert mean_reciprocal_rank(["a", "b", "c"], ["a"]) == 1.0

    def test_second_is_relevant(self):
        assert mean_reciprocal_rank(["x", "a", "b"], ["a"]) == 0.5

    def test_third_is_relevant(self):
        assert mean_reciprocal_rank(["x", "y", "a"], ["a"]) == pytest.approx(1 / 3)

    def test_none_relevant(self):
        assert mean_reciprocal_rank(["x", "y", "z"], ["a"]) == 0.0

    def test_empty_retrieved(self):
        assert mean_reciprocal_rank([], ["a"]) == 0.0

    def test_multiple_relevant_returns_first(self):
        assert mean_reciprocal_rank(["x", "a", "b"], ["a", "b"]) == 0.5


class TestComputeRetrievalMetrics:
    def test_basic(self):
        m = compute_retrieval_metrics(["a", "x", "b"], ["a", "b"], ks=(2, 3))
        assert m.precision_at_k[2] == 0.5
        assert m.precision_at_k[3] == pytest.approx(2 / 3)
        assert m.recall_at_k[2] == 0.5
        assert m.recall_at_k[3] == 1.0
        assert m.mrr == 1.0

    def test_frozen(self):
        m = compute_retrieval_metrics(["a"], ["a"], ks=(3,))
        with pytest.raises(Exception):
            m.mrr = 0.5  # type: ignore[misc]


class TestAggregateRetrievalMetrics:
    def test_average(self):
        m1 = RetrievalMetrics(precision_at_k={5: 1.0}, recall_at_k={5: 1.0}, mrr=1.0)
        m2 = RetrievalMetrics(precision_at_k={5: 0.0}, recall_at_k={5: 0.0}, mrr=0.0)
        agg = aggregate_retrieval_metrics([m1, m2])
        assert agg.precision_at_k[5] == 0.5
        assert agg.recall_at_k[5] == 0.5
        assert agg.mrr == 0.5

    def test_empty_list(self):
        agg = aggregate_retrieval_metrics([])
        assert agg.mrr == 0.0
        assert agg.precision_at_k == {}

    def test_single(self):
        m = RetrievalMetrics(
            precision_at_k={3: 0.8, 5: 0.6}, recall_at_k={3: 0.5, 5: 0.9}, mrr=0.75
        )
        agg = aggregate_retrieval_metrics([m])
        assert agg.mrr == 0.75
        assert agg.precision_at_k[3] == 0.8
