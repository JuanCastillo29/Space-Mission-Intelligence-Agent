"""Tests for query routing — keyword fast path and LLM classification."""

import json
from unittest.mock import AsyncMock

import pytest

from scripts.generation.router import QueryRouter
from scripts.generation.schemas import QueryType


def _mock_client(response_text: str) -> AsyncMock:
    client = AsyncMock()
    client.complete.return_value = (
        response_text,
        {"prompt_tokens": 5, "completion_tokens": 10},
    )
    return client


class TestKeywordFastPath:
    @pytest.mark.asyncio
    async def test_norad_id(self):
        router = QueryRouter(client=_mock_client(""))
        result = await router.classify("What is the NORAD ID of ISS?")
        assert result.query_type == QueryType.STRUCTURED
        assert result.confidence == 0.95

    @pytest.mark.asyncio
    async def test_tle(self):
        router = QueryRouter(client=_mock_client(""))
        result = await router.classify("Show me the TLE for Sentinel-1")
        assert result.query_type == QueryType.STRUCTURED

    @pytest.mark.asyncio
    async def test_inclination(self):
        router = QueryRouter(client=_mock_client(""))
        result = await router.classify("What is the inclination of the ISS orbit?")
        assert result.query_type == QueryType.STRUCTURED

    @pytest.mark.asyncio
    async def test_satellite_catalog(self):
        router = QueryRouter(client=_mock_client(""))
        result = await router.classify("Search the satellite catalog for Copernicus")
        assert result.query_type == QueryType.STRUCTURED

    @pytest.mark.asyncio
    async def test_skips_llm_call(self):
        client = _mock_client("")
        router = QueryRouter(client=client)
        await router.classify("What is the NORAD ID of ISS?")
        client.complete.assert_not_called()


class TestLLMClassification:
    @pytest.mark.asyncio
    async def test_retrieval_query(self):
        response = json.dumps(
            {
                "query_type": "retrieval",
                "confidence": 0.9,
                "reasoning": "document lookup",
            }
        )
        router = QueryRouter(client=_mock_client(response))
        result = await router.classify("What were the goals of Rosetta?")
        assert result.query_type == QueryType.RETRIEVAL
        assert result.confidence == 0.9

    @pytest.mark.asyncio
    async def test_structured_query(self):
        response = json.dumps(
            {
                "query_type": "structured",
                "confidence": 0.85,
                "reasoning": "needs database",
            }
        )
        router = QueryRouter(client=_mock_client(response))
        result = await router.classify("How many active satellites are there?")
        assert result.query_type == QueryType.STRUCTURED

    @pytest.mark.asyncio
    async def test_hybrid_query(self):
        response = json.dumps(
            {
                "query_type": "hybrid",
                "confidence": 0.8,
                "reasoning": "needs both",
            }
        )
        router = QueryRouter(client=_mock_client(response))
        result = await router.classify("Compare Rosetta orbit with current debris")
        assert result.query_type == QueryType.HYBRID


class TestFallbackBehavior:
    @pytest.mark.asyncio
    async def test_defaults_to_retrieval_on_invalid_json(self):
        router = QueryRouter(client=_mock_client("not json"))
        result = await router.classify("some query")
        assert result.query_type == QueryType.RETRIEVAL
        assert result.confidence == 0.5

    @pytest.mark.asyncio
    async def test_defaults_to_retrieval_on_missing_key(self):
        router = QueryRouter(client=_mock_client('{"wrong_key": "value"}'))
        result = await router.classify("some query")
        assert result.query_type == QueryType.RETRIEVAL

    @pytest.mark.asyncio
    async def test_defaults_to_retrieval_on_invalid_query_type(self):
        response = json.dumps({"query_type": "unknown", "confidence": 0.9})
        router = QueryRouter(client=_mock_client(response))
        result = await router.classify("some query")
        assert result.query_type == QueryType.RETRIEVAL

    @pytest.mark.asyncio
    async def test_no_client_raises(self):
        router = QueryRouter()
        with pytest.raises(RuntimeError):
            await router.classify("query")
