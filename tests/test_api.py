"""Tests for the FastAPI API layer — mocked pipelines, no real ML models or DB."""

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import get_embedder, get_generation_pipeline, get_session
from app.main import create_app
from scripts.generation.schemas import Citation, GenerationResult, QueryType


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_generation_result(query: str = "test query") -> GenerationResult:
    doc_id = uuid.uuid4()
    return GenerationResult(
        answer="The Huygens probe landed on Titan [1].",
        citations=[
            Citation(
                ref_index=1,
                source_title="Cassini-Huygens Report",
                section_path="Chapter 3",
                document_id=doc_id,
            )
        ],
        sources_section="## Sources\n- [1] Cassini-Huygens Report — Chapter 3",
        raw_llm_output="The Huygens probe landed on Titan [1].",
        query=query,
        query_type=QueryType.RETRIEVAL,
        model_name="llama-3.1-70b-versatile",
        prompt_tokens=120,
        completion_tokens=45,
        latency_ms=350.0,
    )


def _mock_document_row(
    title: str = "Test Document",
    chunk_count: int = 10,
) -> tuple[MagicMock, int]:
    doc = MagicMock()
    doc.id = uuid.uuid4()
    doc.title = title
    doc.source_type = MagicMock(value="PDF")
    doc.source_url = None
    doc.mission_name = "Cassini"
    doc.ingested_at = datetime(2025, 1, 1, tzinfo=timezone.utc)
    doc.metadata_ = {"pages": 42}
    return doc, chunk_count


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture()
def mock_session():
    return AsyncMock(spec=AsyncSession)


@pytest.fixture()
def mock_pipeline():
    pipeline = AsyncMock()
    pipeline.run.return_value = _make_generation_result()
    return pipeline


@pytest.fixture()
def mock_embedder():
    embedder = MagicMock()
    embedder.dim = 1024
    return embedder


@pytest.fixture()
def client(mock_session, mock_pipeline, mock_embedder):
    app = create_app()

    async def _override_session():
        yield mock_session

    app.dependency_overrides[get_session] = _override_session
    app.dependency_overrides[get_generation_pipeline] = lambda: mock_pipeline
    app.dependency_overrides[get_embedder] = lambda: mock_embedder

    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


# ── Health ───────────────────────────────────────────────────────────────────


class TestHealth:
    def test_healthy(self, client, mock_session):
        mock_session.execute.return_value = None
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["database"] is True
        assert body["version"] == "v1"

    def test_db_down(self, client, mock_session):
        mock_session.execute.side_effect = Exception("connection refused")
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["database"] is False


# ── Query ────────────────────────────────────────────────────────────────────


class TestQuery:
    def test_success(self, client, mock_pipeline):
        resp = client.post(
            "/api/v1/query",
            json={"question": "What is the Huygens probe?"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "Huygens" in body["answer"]
        assert len(body["citations"]) == 1
        assert body["query_type"] == "retrieval"
        assert body["model_name"] == "llama-3.1-70b-versatile"
        mock_pipeline.run.assert_called_once()

    def test_pipeline_error(self, client, mock_pipeline):
        mock_pipeline.run.side_effect = RuntimeError("LLM timeout")
        resp = client.post(
            "/api/v1/query",
            json={"question": "failing query"},
        )
        assert resp.status_code == 502
        assert "failed" in resp.json()["detail"].lower()

    def test_missing_question(self, client):
        resp = client.post("/api/v1/query", json={})
        assert resp.status_code == 422


# ── Ingest ───────────────────────────────────────────────────────────────────


class TestIngest:
    def test_file_not_found(self, client):
        resp = client.post(
            "/api/v1/ingest",
            json={"file_path": "/nonexistent/file.pdf"},
        )
        assert resp.status_code == 400
        assert "does not exist" in resp.json()["detail"]

    def test_non_pdf(self, client, tmp_path):
        txt = tmp_path / "notes.txt"
        txt.write_text("hello")
        resp = client.post(
            "/api/v1/ingest",
            json={"file_path": str(txt)},
        )
        assert resp.status_code == 400
        assert "PDF" in resp.json()["detail"]

    @patch("app.routes.ingest.ingest_pdf", new_callable=AsyncMock)
    def test_success(self, mock_ingest, client, tmp_path):
        pdf = tmp_path / "report.pdf"
        pdf.write_bytes(b"%PDF-1.4 fake")
        mock_ingest.return_value = True

        resp = client.post(
            "/api/v1/ingest",
            json={"file_path": str(pdf)},
        )
        assert resp.status_code == 201
        body = resp.json()
        assert body["success"] is True
        assert body["message"] == "Ingested successfully"

    @patch("app.routes.ingest.ingest_pdf", new_callable=AsyncMock)
    def test_skip_duplicate(self, mock_ingest, client, tmp_path):
        pdf = tmp_path / "report.pdf"
        pdf.write_bytes(b"%PDF-1.4 fake")
        mock_ingest.return_value = False

        resp = client.post(
            "/api/v1/ingest",
            json={"file_path": str(pdf)},
        )
        assert resp.status_code == 201
        assert "Skipped" in resp.json()["message"]


# ── Documents ────────────────────────────────────────────────────────────────


class TestDocuments:
    def test_empty(self, client, mock_session):
        mock_session.scalar.return_value = 0
        mock_result = MagicMock()
        mock_result.all.return_value = []
        mock_session.execute.return_value = mock_result

        resp = client.get("/api/v1/documents")
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 0
        assert body["documents"] == []

    def test_with_documents(self, client, mock_session):
        mock_session.scalar.return_value = 1
        row = _mock_document_row("Mission Report", 25)
        mock_result = MagicMock()
        mock_result.all.return_value = [row]
        mock_session.execute.return_value = mock_result

        resp = client.get("/api/v1/documents")
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 1
        assert body["documents"][0]["title"] == "Mission Report"
        assert body["documents"][0]["chunk_count"] == 25

    def test_pagination_params(self, client, mock_session):
        mock_session.scalar.return_value = 0
        mock_result = MagicMock()
        mock_result.all.return_value = []
        mock_session.execute.return_value = mock_result

        resp = client.get("/api/v1/documents?limit=10&offset=5")
        assert resp.status_code == 200


# ── Evaluate ─────────────────────────────────────────────────────────────────


def _make_eval_run_result(config_name: str = "baseline"):
    from scripts.evaluation.schemas import (
        AggregateMetrics,
        EvalRunResult,
        GenerationMetrics,
        RetrievalMetrics,
        SingleEvalResult,
    )

    retrieval = RetrievalMetrics(
        precision_at_k={3: 0.667, 5: 0.6, 10: 0.5},
        recall_at_k={3: 0.5, 5: 0.75, 10: 1.0},
        mrr=0.833,
    )
    generation = GenerationMetrics(
        faithfulness=0.92,
        answer_relevancy=0.88,
        citation_accuracy=0.85,
        hallucination_rate=0.08,
    )
    return EvalRunResult(
        run_id=f"{config_name}_20260713_120000",
        timestamp="2026-07-13T12:00:00+00:00",
        config_name=config_name,
        config_dict={"name": config_name},
        aggregate_metrics=AggregateMetrics(
            retrieval=retrieval,
            generation=generation,
            avg_latency_ms=250.0,
            avg_total_tokens=400.0,
        ),
        per_query_results=[
            SingleEvalResult(
                query="What is Gaia?",
                expected_answer="A space observatory.",
                generated_answer="Gaia is an ESA space observatory.",
                retrieved_chunk_ids=["abc"],
                retrieval_metrics=retrieval,
                generation_metrics=generation,
                latency_ms=250.0,
                passed=True,
            ),
        ],
        total_latency_ms=250.0,
        total_tokens=400,
    )


class TestEvaluate:
    @patch(
        "app.routes.evaluate.run_evaluation",
        new_callable=AsyncMock,
    )
    def test_post_evaluate(self, mock_run, client):
        mock_run.return_value = [_make_eval_run_result()]
        resp = client.post(
            "/api/v1/evaluate",
            json={"configs": ["baseline"]},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "completed"
        assert len(body["summaries"]) == 1
        assert body["summaries"][0]["config_name"] == "baseline"
        assert body["summaries"][0]["mrr"] == 0.833
        assert body["summaries"][0]["num_passed"] == 1

    @patch(
        "app.routes.evaluate.run_evaluation",
        new_callable=AsyncMock,
    )
    def test_post_invalid_config(self, mock_run, client):
        mock_run.side_effect = KeyError("Unknown ablation config 'bad'")
        resp = client.post(
            "/api/v1/evaluate",
            json={"configs": ["bad"]},
        )
        assert resp.status_code == 422

    @patch("app.routes.evaluate.load_latest_results")
    def test_get_results(self, mock_load, client):
        mock_load.return_value = [_make_eval_run_result()]
        resp = client.get("/api/v1/evaluate/results")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert len(body["summaries"]) == 1

    @patch("app.routes.evaluate.load_latest_results")
    def test_get_results_empty(self, mock_load, client):
        mock_load.return_value = []
        resp = client.get("/api/v1/evaluate/results")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "empty"
        assert body["summaries"] == []
