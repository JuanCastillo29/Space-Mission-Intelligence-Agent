from __future__ import annotations

import json
from pathlib import Path

from scripts.evaluation.schemas import EvalRunResult


def save_result(result: EvalRunResult, output_dir: str | Path) -> Path:
    """Persist a run result as ``{config_name}_{timestamp}.json``.

    The timestamp is derived from ``result.run_id`` so the on-disk name lines
    up with the run identifier.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    path = out / f"{result.run_id}.json"
    path.write_text(
        json.dumps(result.model_dump(mode="json"), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def load_result(path: str | Path) -> EvalRunResult:
    raw = Path(path).read_text(encoding="utf-8")
    return EvalRunResult.model_validate(json.loads(raw))


def load_latest_results(output_dir: str | Path) -> list[EvalRunResult]:
    """Load the most recent result file for each config name.

    Files are named ``{config_name}_{YYYYMMDD_HHMMSS}.json``; the lexicographic
    max per config is also the chronological latest.
    """
    out = Path(output_dir)
    if not out.exists():
        return []

    latest_by_config: dict[str, Path] = {}
    for path in out.glob("*.json"):
        result = load_result(path)
        current = latest_by_config.get(result.config_name)
        if current is None or path.name > current.name:
            latest_by_config[result.config_name] = path

    return [load_result(p) for p in latest_by_config.values()]


def compare_runs(results: list[EvalRunResult]) -> dict[str, dict[str, float | None]]:
    """Build a ``{config_name: {metric: value}}`` table for side-by-side
    comparison of ablation configs.
    """
    table: dict[str, dict[str, float | None]] = {}

    for result in results:
        retrieval = result.aggregate_metrics.retrieval
        generation = result.aggregate_metrics.generation

        table[result.config_name] = {
            "precision@5": retrieval.precision_at_k.get(5),
            "recall@5": retrieval.recall_at_k.get(5),
            "mrr": retrieval.mrr,
            "faithfulness": generation.faithfulness,
            "answer_relevancy": generation.answer_relevancy,
            "citation_accuracy": generation.citation_accuracy,
            "hallucination_rate": generation.hallucination_rate,
            "avg_latency_ms": result.aggregate_metrics.avg_latency_ms,
        }

    return table
