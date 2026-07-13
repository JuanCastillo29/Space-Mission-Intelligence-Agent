"""Evaluation and ablation CLI.

Runs the golden QA dataset through one or more ablation configurations,
computing retrieval metrics (Precision@k, Recall@k, MRR), custom generation
metrics (citation accuracy, hallucination rate, unanswerable detection), and
optional batched RAGAS metrics, then writes timestamped JSON results.

Usage:
    python -m scripts.evaluation.run                       # all configs
    python -m scripts.evaluation.run --configs baseline    # one config
    python -m scripts.evaluation.run --no-llm              # retrieval only
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from datetime import datetime, timezone

from sqlalchemy.ext.asyncio import AsyncSession

from db import async_session_factory
from ingestion.embedder import Embedder, SentenceTransformerEmbedder
from scripts.evaluation.ablation.configs import AblationConfig, get_configs
from scripts.evaluation.ablation.runner import AblationRunner
from scripts.evaluation.config import eval_settings
from scripts.evaluation.dataset import load_dataset, resolve_ground_truth_chunk_ids
from scripts.evaluation.metrics.generation import (
    aggregate_generation_metrics,
    compute_generation_metrics,
    compute_ragas_metrics,
)
from scripts.evaluation.metrics.retrival import (
    aggregate_retrieval_metrics,
    compute_retrieval_metrics,
)
from scripts.evaluation.results import compare_runs, save_result
from scripts.evaluation.schemas import (
    AggregateMetrics,
    EvalRunResult,
    GenerationMetrics,
    GoldenDataset,
    GoldenQAPair,
    RetrievalMetrics,
    SingleEvalResult,
)
from scripts.generation.pipeline import GenerationPipeline
from scripts.retrival.reranker import BGEReranker

log = logging.getLogger(__name__)

_SKIPPED_ANSWER = "(generation skipped: --no-llm)"


def _did_pass(
    pair: GoldenQAPair,
    retrieval: RetrievalMetrics,
    generation: GenerationMetrics,
    has_ground_truth: bool,
    k_values: list[int],
) -> bool:
    """Heuristic pass/fail for a single query.

    - unanswerable: the model must refuse.
    - answerable: low hallucination and, when ground-truth chunks are known,
      at least one of them retrieved within the largest k.
    """
    if pair.category == "unanswerable":
        return generation.unanswerable_detection_rate == 1.0

    ok = generation.hallucination_rate <= 0.5
    if has_ground_truth:
        max_k = max(k_values)
        ok = ok and retrieval.recall_at_k.get(max_k, 0.0) > 0.0
    return ok


async def run_config(
    config: AblationConfig,
    dataset: GoldenDataset,
    ground_truth: dict[str, list],
    session: AsyncSession,
    *,
    embedder: Embedder | None,
    reranker: BGEReranker | None,
    generation_pipeline: GenerationPipeline | None,
    k_values: list[int],
    ragas_enabled: bool,
    generate: bool,
) -> EvalRunResult:
    runner = AblationRunner(
        config,
        embedder=embedder,
        reranker=reranker,
        generation_pipeline=generation_pipeline,
    )

    timestamp = datetime.now(timezone.utc)
    run_id = f"{config.name}_{timestamp.strftime('%Y%m%d_%H%M%S')}"
    log.info("Running config %r (%d pairs)", config.name, len(dataset.pairs))

    per_query_results: list[SingleEvalResult] = []
    per_query_generation: list[GenerationMetrics] = []
    retrieval_for_agg: list[RetrievalMetrics] = []

    ragas_questions: list[str] = []
    ragas_answers: list[str] = []
    ragas_contexts: list[list[str]] = []
    ragas_ground_truths: list[str] = []

    total_latency = 0.0
    total_tokens = 0

    for pair in dataset.pairs:
        is_unanswerable = pair.category == "unanswerable"
        gt_ids = [str(u) for u in ground_truth.get(pair.query, [])]
        has_gt = bool(gt_ids)

        execution = await runner.execute_query(
            pair.query, session, generate=generate
        )
        retrieved_ids = execution.retrieved_chunk_ids
        contexts = execution.context_texts

        retrieval_metrics = compute_retrieval_metrics(
            retrieved_ids, gt_ids, tuple(k_values)
        )
        if has_gt:
            retrieval_for_agg.append(retrieval_metrics)

        gen_result = execution.generation_result
        answer = gen_result.answer if gen_result else _SKIPPED_ANSWER

        if gen_result is not None:
            generation_metrics = compute_generation_metrics(
                generated_answer=answer,
                expected_answer=pair.expected_answer,
                retrieved_context=contexts,
                retrieved_chunk_ids=retrieved_ids,
                ragas_metrics=None,
                is_unanswerable=is_unanswerable,
            )
            latency = gen_result.latency_ms or 0.0
            prompt_tokens = gen_result.prompt_tokens
            completion_tokens = gen_result.completion_tokens
            query_tokens = (prompt_tokens or 0) + (completion_tokens or 0)

            if ragas_enabled and not is_unanswerable and contexts:
                ragas_questions.append(pair.query)
                ragas_answers.append(answer)
                ragas_contexts.append(contexts)
                ragas_ground_truths.append(pair.expected_answer)
        else:
            # Retrieval-only run: neutral generation metrics.
            generation_metrics = GenerationMetrics(
                citation_accuracy=1.0, hallucination_rate=0.0
            )
            latency = 0.0
            prompt_tokens = completion_tokens = None
            query_tokens = 0

        total_latency += latency
        total_tokens += query_tokens
        per_query_generation.append(generation_metrics)

        per_query_results.append(
            SingleEvalResult(
                query=pair.query,
                expected_answer=pair.expected_answer,
                generated_answer=answer,
                retrieved_chunk_ids=retrieved_ids,
                retrieval_metrics=retrieval_metrics,
                generation_metrics=generation_metrics,
                latency_ms=latency,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=query_tokens or None,
                passed=_did_pass(
                    pair, retrieval_metrics, generation_metrics, has_gt, k_values
                ),
            )
        )

    aggregate_retrieval = aggregate_retrieval_metrics(retrieval_for_agg)
    aggregate_generation = aggregate_generation_metrics(per_query_generation)

    if ragas_enabled and ragas_questions:
        log.info("Computing RAGAS over %d answerable queries", len(ragas_questions))
        try:
            ragas_scores = compute_ragas_metrics(
                ragas_questions,
                ragas_answers,
                ragas_contexts,
                ragas_ground_truths,
            )
            aggregate_generation = aggregate_generation.model_copy(update=ragas_scores)
        except Exception as exc:  # RAGAS/LLM failures shouldn't sink the run
            log.warning("RAGAS computation failed: %s", exc)

    n = len(per_query_results) or 1
    aggregate = AggregateMetrics(
        retrieval=aggregate_retrieval,
        generation=aggregate_generation,
        avg_latency_ms=total_latency / n,
        avg_total_tokens=total_tokens / n,
    )

    return EvalRunResult(
        run_id=run_id,
        timestamp=timestamp.isoformat(),
        config_name=config.name,
        config_dict=config.to_dict(),
        aggregate_metrics=aggregate,
        per_query_results=per_query_results,
        total_latency_ms=total_latency,
        total_tokens=total_tokens,
    )


def _print_summary(results: list[EvalRunResult]) -> None:
    table = compare_runs(results)
    columns = [
        "precision@5",
        "recall@5",
        "mrr",
        "faithfulness",
        "citation_accuracy",
        "hallucination_rate",
    ]

    header = f"{'config':20}" + "".join(f"{c:>20}" for c in columns)
    print("\n" + header)
    print("-" * len(header))
    for config_name, metrics in table.items():
        row = f"{config_name:20}"
        for col in columns:
            value = metrics.get(col)
            row += f"{'—':>20}" if value is None else f"{value:>20.3f}"
        print(row)
    print()


async def run_evaluation(
    config_names: list[str] | None,
    *,
    dataset_path: str | None = None,
    output_dir: str | None = None,
    generate: bool = True,
    ragas_enabled: bool | None = None,
) -> list[EvalRunResult]:
    dataset = load_dataset(dataset_path or eval_settings.EVAL_DATASET_PATH)
    configs = get_configs(config_names)
    out_dir = output_dir or eval_settings.EVAL_RESULTS_DIR
    ragas = eval_settings.EVAL_RAGAS_ENABLED if ragas_enabled is None else ragas_enabled
    ragas = ragas and generate

    # Shared across configs: embedder, reranker, generation pipeline are
    # expensive to construct and stateless, so build them once.
    embedder: Embedder = SentenceTransformerEmbedder()
    reranker = BGEReranker()
    generation_pipeline = GenerationPipeline() if generate else None

    results: list[EvalRunResult] = []
    async with async_session_factory() as session:
        ground_truth = await resolve_ground_truth_chunk_ids(dataset.pairs, session)

        for config in configs:
            result = await run_config(
                config,
                dataset,
                ground_truth,
                session,
                embedder=embedder,
                reranker=reranker,
                generation_pipeline=generation_pipeline,
                k_values=eval_settings.EVAL_RETRIEVAL_K_VALUES,
                ragas_enabled=ragas,
                generate=generate,
            )
            path = save_result(result, out_dir)
            log.info("Saved %s", path)
            results.append(result)

    return results


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    parser = argparse.ArgumentParser(description="Run RAG evaluation and ablations.")
    parser.add_argument(
        "--configs",
        nargs="*",
        default=None,
        help="Ablation config names to run (default: all).",
    )
    parser.add_argument("--dataset", default=None, help="Path to golden QA JSON.")
    parser.add_argument("--output-dir", default=None, help="Where to write results.")
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Skip generation and RAGAS; retrieval metrics only.",
    )
    parser.add_argument(
        "--no-ragas",
        action="store_true",
        help="Run generation but skip RAGAS metrics.",
    )
    args = parser.parse_args()

    results = asyncio.run(
        run_evaluation(
            args.configs,
            dataset_path=args.dataset,
            output_dir=args.output_dir,
            generate=not args.no_llm,
            ragas_enabled=False if args.no_ragas else None,
        )
    )

    _print_summary(results)


if __name__ == "__main__":
    main()
