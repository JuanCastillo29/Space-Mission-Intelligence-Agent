"""Tests for evaluation generation metrics.

Requires rapidfuzz and nltk (available in the Docker dev environment).
"""

import pytest
from rapidfuzz import fuzz  # noqa: F401 – validates dep is present

from scripts.evaluation.metrics.generation import (
    _is_refusal,
    aggregate_generation_metrics,
    citation_accuracy,
    compute_generation_metrics,
    hallucination_rate,
    unanswerable_rate,
)
from scripts.evaluation.schemas import GenerationMetrics


# ── citation_accuracy ────────────────────────────────────────────────────────


class TestCitationAccuracy:
    def test_no_citations(self):
        assert citation_accuracy("No refs here.", 5) == 1.0

    def test_all_valid(self):
        assert citation_accuracy("See [1] and [2].", 3) == 1.0

    def test_all_invalid(self):
        assert citation_accuracy("See [10].", 3) == 0.0

    def test_mixed(self):
        assert citation_accuracy("See [1] and [10].", 3) == 0.5

    def test_zero_context_blocks(self):
        assert citation_accuracy("See [1].", 0) == 0.0


# ── hallucination_rate ───────────────────────────────────────────────────────


class TestHallucinationRate:
    def test_fully_supported(self):
        context = "Gaia was launched in 2013."
        answer = "Gaia was launched in 2013."
        rate = hallucination_rate(answer, context, "")
        assert rate == 0.0

    def test_fully_hallucinated(self):
        context = "Gaia was launched in 2013."
        answer = "Jupiter has 95 known moons orbiting it."
        rate = hallucination_rate(answer, context, "")
        assert rate == 1.0

    def test_empty_answer(self):
        assert hallucination_rate("", "some context", "some truth") == 0.0

    def test_ground_truth_supports(self):
        answer = "The mission cost 700 million euros."
        rate = hallucination_rate(
            answer, "unrelated context", "The mission cost 700 million euros."
        )
        assert rate == 0.0


# ── _is_refusal / unanswerable_rate ──────────────────────────────────────────


class TestIsRefusal:
    def test_refusal_phrases(self):
        assert _is_refusal("I don't know the answer.")
        assert _is_refusal("I cannot answer this question.")
        assert _is_refusal("This is not available in the provided documents.")

    def test_non_refusal(self):
        assert not _is_refusal("Gaia was launched in 2013.")


class TestUnanswerableRate:
    def test_all_refuse(self):
        results = [(True, "I don't know."), (True, "I cannot answer this.")]
        assert unanswerable_rate(results) == 1.0

    def test_none_refuse(self):
        results = [(True, "It was blue."), (True, "42.")]
        assert unanswerable_rate(results) == 0.0

    def test_mixed(self):
        results = [(True, "I don't know."), (True, "Blue.")]
        assert unanswerable_rate(results) == 0.5

    def test_no_unanswerable_queries(self):
        results = [(False, "Some answer.")]
        assert unanswerable_rate(results) == 1.0

    def test_ignores_answerable(self):
        results = [(False, "Gaia."), (True, "I don't know.")]
        assert unanswerable_rate(results) == 1.0


# ── compute_generation_metrics ───────────────────────────────────────────────


class TestComputeGenerationMetrics:
    def test_answerable_query(self):
        m = compute_generation_metrics(
            generated_answer="Gaia launched in 2013 [1].",
            expected_answer="Gaia launched in 2013.",
            retrieved_context=["Gaia launched in 2013."],
            retrieved_chunk_ids=["chunk-1"],
        )
        assert m.citation_accuracy == 1.0
        assert m.hallucination_rate == 0.0
        assert m.unanswerable_detection_rate is None

    def test_unanswerable_correct_refusal(self):
        m = compute_generation_metrics(
            generated_answer="This is not available in the provided documents.",
            expected_answer="N/A",
            retrieved_context=[],
            retrieved_chunk_ids=[],
            is_unanswerable=True,
        )
        assert m.unanswerable_detection_rate == 1.0

    def test_unanswerable_wrong_answer(self):
        m = compute_generation_metrics(
            generated_answer="The pen was blue.",
            expected_answer="N/A",
            retrieved_context=[],
            retrieved_chunk_ids=[],
            is_unanswerable=True,
        )
        assert m.unanswerable_detection_rate == 0.0

    def test_ragas_passthrough(self):
        m = compute_generation_metrics(
            generated_answer="answer [1]",
            expected_answer="answer",
            retrieved_context=["answer"],
            retrieved_chunk_ids=["c1"],
            ragas_metrics={"faithfulness": 0.95, "answer_relevancy": 0.88},
        )
        assert m.faithfulness == 0.95
        assert m.answer_relevancy == 0.88


# ── aggregate_generation_metrics ─────────────────────────────────────────────


class TestAggregateGenerationMetrics:
    def test_average(self):
        m1 = GenerationMetrics(
            citation_accuracy=1.0, hallucination_rate=0.0, faithfulness=0.9
        )
        m2 = GenerationMetrics(
            citation_accuracy=0.5, hallucination_rate=0.4, faithfulness=0.7
        )
        agg = aggregate_generation_metrics([m1, m2])
        assert agg.citation_accuracy == pytest.approx(0.75)
        assert agg.hallucination_rate == pytest.approx(0.2)
        assert agg.faithfulness == pytest.approx(0.8)

    def test_none_fields_excluded_from_average(self):
        m1 = GenerationMetrics(citation_accuracy=1.0, hallucination_rate=0.0)
        m2 = GenerationMetrics(
            citation_accuracy=0.5, hallucination_rate=0.2, faithfulness=0.9
        )
        agg = aggregate_generation_metrics([m1, m2])
        assert agg.faithfulness == 0.9

    def test_empty_list(self):
        agg = aggregate_generation_metrics([])
        assert agg.citation_accuracy == 0.0
        assert agg.hallucination_rate == 0.0

    def test_unanswerable_rate_averaged(self):
        m1 = GenerationMetrics(
            citation_accuracy=1.0,
            hallucination_rate=0.0,
            unanswerable_detection_rate=1.0,
        )
        m2 = GenerationMetrics(
            citation_accuracy=1.0,
            hallucination_rate=0.0,
            unanswerable_detection_rate=0.0,
        )
        agg = aggregate_generation_metrics([m1, m2])
        assert agg.unanswerable_detection_rate == 0.5
