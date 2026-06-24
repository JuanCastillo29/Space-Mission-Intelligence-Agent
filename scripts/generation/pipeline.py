from __future__ import annotations

import os
import time

from sqlalchemy.ext.asyncio import AsyncSession

from scripts.generation.citations import postprocess_response
from scripts.generation.client import (
    ChatClient,
    FallbackClient,
    GroqClient,
    MistralClient,
)
from scripts.generation.config import GenerationSettings
from scripts.generation.prompt import build_system_prompt, build_user_prompt
from scripts.generation.router import QueryRouter
from scripts.generation.schemas import ChatMessage, GenerationResult, QueryType
from scripts.retrival.pipeline import RetrievalPipeline
from scripts.retrival.schemas import RetrievalResult


def _build_default_client(settings: GenerationSettings) -> ChatClient:
    primary = GroqClient(
        api_key=os.environ["GROQ_API_KEY"],
        model=settings.LLM_MODEL,
        temperature=settings.LLM_TEMPERATURE,
        max_tokens=settings.LLM_MAX_TOKENS,
    )
    mistral_key = os.environ.get("MISTRAL_API_KEY")
    if mistral_key:
        fallback = MistralClient(
            api_key=mistral_key,
            model=settings.LLM_FALLBACK_MODEL,
            temperature=settings.LLM_TEMPERATURE,
            max_tokens=settings.LLM_MAX_TOKENS,
        )
        return FallbackClient(primary, fallback)
    return primary


class GenerationPipeline:
    def __init__(
        self,
        client: ChatClient | None = None,
        router: QueryRouter | None = None,
        settings: GenerationSettings | None = None,
    ) -> None:
        self._client = client
        self._router = router
        self._settings = settings or GenerationSettings()

    @property
    def client(self) -> ChatClient:
        if self._client is None:
            self._client = _build_default_client(self._settings)
        return self._client

    @property
    def router(self) -> QueryRouter:
        if self._router is None:
            self._router = QueryRouter(client=self.client)
        return self._router

    async def generate(
        self,
        retrieval_result: RetrievalResult,
        *,
        temperature: float | None = None,
    ) -> GenerationResult:
        system_msg = ChatMessage(role="system", content=build_system_prompt())
        user_msg = ChatMessage(
            role="user",
            content=build_user_prompt(
                retrieval_result.query,
                retrieval_result.context_text,
            ),
        )

        start = time.perf_counter()
        raw_content, usage = await self.client.complete([system_msg, user_msg])
        latency_ms = (time.perf_counter() - start) * 1000

        answer, citations, sources_section = postprocess_response(
            raw_content, retrieval_result.blocks
        )

        return GenerationResult(
            answer=answer,
            citations=citations,
            sources_section=sources_section,
            raw_llm_output=raw_content,
            query=retrieval_result.query,
            query_type=QueryType.RETRIEVAL,
            model_name=self._settings.LLM_MODEL,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            latency_ms=latency_ms,
        )

    async def run(
        self,
        query: str,
        session: AsyncSession,
        *,
        retrieval_pipeline: RetrievalPipeline | None = None,
    ) -> GenerationResult:
        routing = await self.router.classify(query)

        pipeline = retrieval_pipeline or RetrievalPipeline()
        retrieval_result = await pipeline.run(query, session)

        result = await self.generate(retrieval_result)

        return GenerationResult(
            answer=result.answer,
            citations=result.citations,
            sources_section=result.sources_section,
            raw_llm_output=result.raw_llm_output,
            query=result.query,
            query_type=routing.query_type,
            model_name=result.model_name,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            latency_ms=result.latency_ms,
        )
