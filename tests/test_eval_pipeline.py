"""Tests for evaluation schemas, ablation configs, dataset loading, and results.

Pure Python — no DB, no ML models.
"""

import json

import pytest

from scripts.evaluation.ablation.configs import (
    ABLATION_CONFIGS,
    AblationConfig,
    get_configs,
)
from scripts.evaluation.dataset import load_dataset
from scripts.evaluation.results import (
    compare_runs,
    load_latest_results,
    load_result,
    save_result,
)
from scripts.evaluation.schemas import (
    AggregateMetrics,
    EvalRunResult,
    GenerationMetrics,
    GoldenDataset,
    GoldenQAPair,
    RetrievalMetrics,
    SingleEvalResult,
)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _retrieval() -> RetrievalMetrics:
    return RetrievalMetrics(
        precision_at_k={3: 0.667, 5: 0.6},
        recall_at_k={3: 0.5, 5: 0.75},
        mrr=0.833,
    )


def _generation() -> GenerationMetrics:
    return GenerationMetrics(
        faithfulness=0.92,
        citation_accuracy=0.85,
        hallucination_rate=0.08,
    )


def _eval_run(config_name: str = "baseline", ts: str = "120000") -> EvalRunResult:
    return EvalRunResult(
        run_id=f"{config_name}_20260713_{ts}",
        timestamp="2026-07-13T12:00:00+00:00",
        config_name=config_name,
        config_dict={"name": config_name},
        aggregate_metrics=AggregateMetrics(
            retrieval=_retrieval(),
            generation=_generation(),
            avg_latency_ms=250.0,
            avg_total_tokens=400.0,
        ),
        per_query_results=[
            SingleEvalResult(
                query="What is Gaia?",
                expected_answer="A space observatory.",
                generated_answer="Gaia is an ESA space observatory.",
                retrieved_chunk_ids=["abc"],
                retrieval_metrics=_retrieval(),
                generation_metrics=_generation(),
                latency_ms=250.0,
                passed=True,
            ),
        ],
        total_latency_ms=250.0,
        total_tokens=400,
    )


# ── Schemas ──────────────────────────────────────────────────────────────────


class TestSchemas:
    def test_golden_qa_pair_frozen(self):
        pair = GoldenQAPair(
            query="q",
            expected_answer="a",
            ground_truth_chunk_ids=["c1"],
            difficulty="easy",
            category="single_doc_factual",
        )
        with pytest.raises(Exception):
            pair.query = "modified"  # type: ignore[misc]

    def test_retrieval_metrics_frozen(self):
        m = _retrieval()
        with pytest.raises(Exception):
            m.mrr = 0.0  # type: ignore[misc]

    def test_generation_metrics_defaults(self):
        m = GenerationMetrics(citation_accuracy=1.0, hallucination_rate=0.0)
        assert m.faithfulness is None
        assert m.unanswerable_detection_rate is None

    def test_eval_run_result_json_roundtrip(self):
        original = _eval_run()
        data = json.loads(original.model_dump_json())
        restored = EvalRunResult.model_validate(data)
        assert restored.run_id == original.run_id
        assert (
            restored.aggregate_metrics.retrieval.mrr
            == original.aggregate_metrics.retrieval.mrr
        )
        assert len(restored.per_query_results) == 1


# ── Ablation Configs ─────────────────────────────────────────────────────────


class TestAblationConfigs:
    def test_seven_predefined(self):
        assert len(ABLATION_CONFIGS) == 7

    def test_baseline_defaults(self):
        cfg = get_configs(["baseline"])[0]
        assert cfg.search_mode == "hybrid"
        assert cfg.reranking_enabled is True
        assert cfg.chunk_size_filter is None

    def test_get_all(self):
        all_configs = get_configs(None)
        assert len(all_configs) == 7

    def test_get_specific(self):
        configs = get_configs(["semantic_only", "no_reranking"])
        assert len(configs) == 2
        assert configs[0].search_mode == "semantic_only"
        assert configs[1].reranking_enabled is False

    def test_unknown_name_raises(self):
        with pytest.raises(KeyError, match="nonexistent"):
            get_configs(["nonexistent"])

    def test_to_dict(self):
        cfg = AblationConfig(name="test")
        d = cfg.to_dict()
        assert d["name"] == "test"
        assert "search_mode" in d

    def test_frozen(self):
        cfg = AblationConfig(name="test")
        with pytest.raises(Exception):
            cfg.name = "other"  # type: ignore[misc]


# ── Dataset Loading ──────────────────────────────────────────────────────────


class TestDatasetLoading:
    def test_load_golden_qa(self):
        dataset = load_dataset("scripts/evaluation/data/golden_qa.json")
        assert isinstance(dataset, GoldenDataset)
        assert dataset.version == "0.3.0"
        assert len(dataset.pairs) == 12
        assert len(dataset.source_documents) == 3

    def test_categories(self):
        dataset = load_dataset("scripts/evaluation/data/golden_qa.json")
        categories = {p.category for p in dataset.pairs}
        assert "single_doc_factual" in categories
        assert "unanswerable" in categories
        assert "cross_doc_comparison" in categories
        assert "multi_source_reasoning" in categories

    def test_unanswerable_have_empty_ground_truth(self):
        dataset = load_dataset("scripts/evaluation/data/golden_qa.json")
        for pair in dataset.pairs:
            if pair.category == "unanswerable":
                assert pair.ground_truth_chunk_ids == []

    def test_invalid_path_raises(self):
        with pytest.raises(FileNotFoundError):
            load_dataset("nonexistent/path.json")


# ── Results ──────────────────────────────────────────────────────────────────


class TestResults:
    def test_save_and_load(self, tmp_path):
        result = _eval_run()
        path = save_result(result, tmp_path)
        assert path.exists()
        assert path.suffix == ".json"

        loaded = load_result(path)
        assert loaded.run_id == result.run_id
        assert loaded.config_name == result.config_name

    def test_load_latest_picks_newest(self, tmp_path):
        old = _eval_run("baseline", "100000")
        new = _eval_run("baseline", "120000")
        save_result(old, tmp_path)
        save_result(new, tmp_path)

        latest = load_latest_results(tmp_path)
        assert len(latest) == 1
        assert latest[0].run_id == new.run_id

    def test_load_latest_per_config(self, tmp_path):
        save_result(_eval_run("baseline"), tmp_path)
        save_result(_eval_run("semantic_only"), tmp_path)

        latest = load_latest_results(tmp_path)
        assert len(latest) == 2
        names = {r.config_name for r in latest}
        assert names == {"baseline", "semantic_only"}

    def test_load_latest_empty_dir(self, tmp_path):
        assert load_latest_results(tmp_path) == []

    def test_load_latest_nonexistent_dir(self):
        assert load_latest_results("nonexistent_dir_xyz") == []

    def test_compare_runs(self):
        results = [_eval_run("baseline"), _eval_run("semantic_only")]
        table = compare_runs(results)
        assert "baseline" in table
        assert "semantic_only" in table
        assert table["baseline"]["mrr"] == 0.833
        assert table["baseline"]["precision@5"] == 0.6
        assert table["baseline"]["citation_accuracy"] == 0.85

    def test_creates_output_dir(self, tmp_path):
        out = tmp_path / "nested" / "results"
        save_result(_eval_run(), out)
        assert out.exists()
