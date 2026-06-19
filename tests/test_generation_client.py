"""Tests for LLM client implementations and fallback logic."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from groq import APIConnectionError, RateLimitError

from scripts.generation.client import FallbackClient, GroqClient, MistralClient
from scripts.generation.schemas import ChatMessage

MESSAGES = [ChatMessage(role="user", content="hello")]
USAGE = {"prompt_tokens": 10, "completion_tokens": 20}


def _mock_groq_response(content: str = "response"):
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = content
    response.usage = MagicMock()
    response.usage.prompt_tokens = 10
    response.usage.completion_tokens = 20
    return response


def _mock_mistral_response(content: str = "response"):
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = content
    response.usage = MagicMock()
    response.usage.prompt_tokens = 10
    response.usage.completion_tokens = 20
    return response


class TestGroqClient:
    @pytest.mark.asyncio
    async def test_complete_returns_content_and_usage(self):
        with patch("scripts.generation.client.AsyncGroq") as mock_cls:
            mock_instance = MagicMock()
            mock_instance.chat.completions.create = AsyncMock(
                return_value=_mock_groq_response("test answer")
            )
            mock_cls.return_value = mock_instance

            client = GroqClient(api_key="key", model="llama-3.1-70b-versatile")
            content, usage = await client.complete(MESSAGES)

        assert content == "test answer"
        assert usage["prompt_tokens"] == 10
        assert usage["completion_tokens"] == 20

    @pytest.mark.asyncio
    async def test_passes_model_and_params(self):
        with patch("scripts.generation.client.AsyncGroq") as mock_cls:
            mock_instance = MagicMock()
            mock_create = AsyncMock(return_value=_mock_groq_response())
            mock_instance.chat.completions.create = mock_create
            mock_cls.return_value = mock_instance

            client = GroqClient(
                api_key="key", model="my-model", temperature=0.5, max_tokens=1024
            )
            await client.complete(MESSAGES)

            call_kwargs = mock_create.call_args[1]
            assert call_kwargs["model"] == "my-model"
            assert call_kwargs["temperature"] == 0.5
            assert call_kwargs["max_tokens"] == 1024


class TestMistralClient:
    @pytest.mark.asyncio
    async def test_complete_returns_content_and_usage(self):
        with patch("scripts.generation.client.Mistral") as mock_cls:
            mock_instance = MagicMock()
            mock_instance.chat.complete_async = AsyncMock(
                return_value=_mock_mistral_response("mistral answer")
            )
            mock_cls.return_value = mock_instance

            client = MistralClient(api_key="key", model="mistral-large-latest")
            content, usage = await client.complete(MESSAGES)

        assert content == "mistral answer"
        assert usage["prompt_tokens"] == 10
        assert usage["completion_tokens"] == 20


class TestFallbackClient:
    @pytest.mark.asyncio
    async def test_uses_primary_when_healthy(self):
        primary = AsyncMock()
        primary.complete.return_value = ("primary answer", USAGE)
        fallback = AsyncMock()

        client = FallbackClient(primary, fallback)
        content, _ = await client.complete(MESSAGES)

        assert content == "primary answer"
        fallback.complete.assert_not_called()

    @pytest.mark.asyncio
    async def test_falls_back_on_rate_limit(self):
        primary = AsyncMock()
        primary.complete.side_effect = RateLimitError(
            message="rate limited",
            response=MagicMock(status_code=429),
            body=None,
        )
        fallback = AsyncMock()
        fallback.complete.return_value = ("fallback answer", USAGE)

        client = FallbackClient(primary, fallback)
        content, _ = await client.complete(MESSAGES)

        assert content == "fallback answer"

    @pytest.mark.asyncio
    async def test_falls_back_on_connection_error(self):
        primary = AsyncMock()
        primary.complete.side_effect = APIConnectionError(request=MagicMock())
        fallback = AsyncMock()
        fallback.complete.return_value = ("fallback answer", USAGE)

        client = FallbackClient(primary, fallback)
        content, _ = await client.complete(MESSAGES)

        assert content == "fallback answer"

    @pytest.mark.asyncio
    async def test_falls_back_on_timeout(self):
        primary = AsyncMock()
        primary.complete.side_effect = TimeoutError()
        fallback = AsyncMock()
        fallback.complete.return_value = ("fallback answer", USAGE)

        client = FallbackClient(primary, fallback)
        content, _ = await client.complete(MESSAGES)

        assert content == "fallback answer"

    @pytest.mark.asyncio
    async def test_both_fail_propagates(self):
        primary = AsyncMock()
        primary.complete.side_effect = TimeoutError()
        fallback = AsyncMock()
        fallback.complete.side_effect = TimeoutError()

        client = FallbackClient(primary, fallback)
        with pytest.raises(TimeoutError):
            await client.complete(MESSAGES)
