from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

from groq import (
    APIConnectionError,
    APIStatusError,
    AsyncGroq,
    RateLimitError,
)
from mistralai.client import Mistral

from scripts.generation.schemas import ChatMessage

logger = logging.getLogger(__name__)


@runtime_checkable
class ChatClient(Protocol):
    async def complete(
        self,
        messages: list[ChatMessage],
    ) -> tuple[str, dict[str, int]]: ...


class GroqClient:
    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> None:
        self._client = AsyncGroq(api_key=api_key)
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens

    async def complete(
        self,
        messages: list[ChatMessage],
    ) -> tuple[str, dict[str, int]]:
        response = await self._client.chat.completions.create(
            model=self._model,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            messages=[{"role": m.role, "content": m.content} for m in messages],
        )

        content = response.choices[0].message.content or ""

        usage = response.usage
        token_usage = {
            "prompt_tokens": usage.prompt_tokens if usage else 0,
            "completion_tokens": usage.completion_tokens if usage else 0,
        }

        return content, token_usage


class MistralClient:
    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> None:
        self._client = Mistral(api_key=api_key)
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens

    async def complete(
        self,
        messages: list[ChatMessage],
    ) -> tuple[str, dict[str, int]]:
        response = await self._client.chat.complete_async(
            model=self._model,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            messages=[{"role": m.role, "content": m.content} for m in messages],
        )

        content = response.choices[0].message.content or ""

        usage = response.usage
        token_usage = {
            "prompt_tokens": usage.prompt_tokens if usage else 0,
            "completion_tokens": usage.completion_tokens if usage else 0,
        }

        return content, token_usage


class FallbackClient:
    """
    Tries the primary provider first and falls back on transient
    provider failures (rate limits, API errors, connection issues).
    """

    def __init__(
        self,
        primary: ChatClient,
        fallback: ChatClient,
    ) -> None:
        self._primary = primary
        self._fallback = fallback

    async def complete(
        self,
        messages: list[ChatMessage],
    ) -> tuple[str, dict[str, int]]:
        try:
            return await self._primary.complete(messages)
        except (
            RateLimitError,
            APIStatusError,
            APIConnectionError,
            TimeoutError,
        ) as exc:
            logger.warning("Primary LLM failed (%s), falling back", exc)
            return await self._fallback.complete(messages)
