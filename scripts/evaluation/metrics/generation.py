from __future__ import annotations

from statistics import mean

from nltk.tokenize import sent_tokenize
from rapidfuzz import fuzz

from scripts.evaluation.schemas import GenerationMetrics
from scripts.generation.citations import extract_citation_refs

SIMILARITY_THRESHOLD = 60

REFUSAL_PHRASES = (
    "don't know",
    "cannot answer",
    "not enough information",
    "insufficient information",
    "not provided",
    "not available in the provided documents",
)


def compute_ragas_metrics(
    questions: list[str],
    answers: list[str],
    contexts: list[list[str]],
    ground_truths: list[str],
) -> dict[str, float]:
    """
    Runs RAGAS over a batch and returns average metric values.

    Column names follow the ragas>=0.2 `evaluate()` dataset convention;
    verify against the installed ragas version if the API has moved on.
    """

    from datasets import Dataset
    from ragas import evaluate as ragas_evaluate
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    dataset = Dataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        }
    )

    results = ragas_evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )
    scores = results.to_pandas()

    return {
        "faithfulness": float(scores["faithfulness"].mean()),
        "answer_relevancy": float(scores["answer_relevancy"].mean()),
        "context_precision": float(scores["context_precision"].mean()),
        "context_recall": float(scores["context_recall"].mean()),
    }


def citation_accuracy(answer: str, num_context_blocks: int) -> float:
    """
    Fraction of cited refs (e.g. "[2]") that point to an actually retrieved
    context block. Ref numbers are 1-indexed positions, matching the
    ordering `assemble_context` uses to build citations.
    """

    refs = extract_citation_refs(answer)
    if not refs:
        return 1.0

    valid = sum(1 for ref in refs if 1 <= ref <= num_context_blocks)
    return valid / len(refs)


def hallucination_rate(
    answer: str,
    context: str,
    ground_truth: str,
) -> float:
    """
    Fraction of answer sentences unsupported by either context or reference answer.
    """

    sentences = sent_tokenize(answer)

    if not sentences:
        return 0.0

    support = context + "\n" + ground_truth

    hallucinated = 0.0

    for sentence in sentences:
        score = fuzz.partial_ratio(sentence, support)

        if score < SIMILARITY_THRESHOLD:
            hallucinated += 1

    return hallucinated / len(sentences)


def _is_refusal(answer: str) -> bool:
    lowered = answer.lower()
    return any(phrase in lowered for phrase in REFUSAL_PHRASES)


def unanswerable_rate(
    results: list[tuple[bool, str]],
) -> float:
    """
    results = [(is_unanswerable, generated_answer), ...]
    Fraction of unanswerable-category queries where the model correctly refused.
    """

    relevant = [r for r in results if r[0]]

    if not relevant:
        return 1.0

    refusals = sum(1 for _, answer in relevant if _is_refusal(answer))

    return refusals / len(relevant)


def compute_generation_metrics(
    *,
    generated_answer: str,
    expected_answer: str,
    retrieved_context: list[str],
    retrieved_chunk_ids: list[str],
    ragas_metrics: dict[str, float] | None = None,
    is_unanswerable: bool = False,
) -> GenerationMetrics:
    """
    Compute generation metrics for a single evaluation example.
    """

    ragas_metrics = ragas_metrics or {}

    context_text = "\n".join(retrieved_context)

    return GenerationMetrics(
        faithfulness=ragas_metrics.get("faithfulness"),
        answer_relevancy=ragas_metrics.get("answer_relevancy"),
        context_precision=ragas_metrics.get("context_precision"),
        context_recall=ragas_metrics.get("context_recall"),
        citation_accuracy=citation_accuracy(
            generated_answer,
            len(retrieved_chunk_ids),
        ),
        hallucination_rate=hallucination_rate(
            generated_answer,
            context_text,
            expected_answer,
        ),
        unanswerable_detection_rate=(
            float(_is_refusal(generated_answer)) if is_unanswerable else None
        ),
    )


def aggregate_generation_metrics(
    metrics: list[GenerationMetrics],
) -> GenerationMetrics:
    """
    Macro-average generation metrics across evaluation examples.
    """

    if not metrics:
        return GenerationMetrics(
            citation_accuracy=0.0,
            hallucination_rate=0.0,
        )

    def avg(values: list[float | None]) -> float | None:
        vals = [v for v in values if v is not None]
        return mean(vals) if vals else None

    return GenerationMetrics(
        faithfulness=avg([m.faithfulness for m in metrics]),
        answer_relevancy=avg([m.answer_relevancy for m in metrics]),
        context_precision=avg([m.context_precision for m in metrics]),
        context_recall=avg([m.context_recall for m in metrics]),
        citation_accuracy=mean(m.citation_accuracy for m in metrics),
        hallucination_rate=mean(m.hallucination_rate for m in metrics),
        unanswerable_detection_rate=avg(
            [m.unanswerable_detection_rate for m in metrics]
        ),
    )
