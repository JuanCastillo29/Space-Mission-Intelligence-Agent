from __future__ import annotations

import json
import logging
import re

from scripts.generation.client import ChatClient
from scripts.generation.schemas import ChatMessage, QueryType, RoutingResult
from scripts.generation.prompt import build_routing_prompt

logger = logging.getLogger(__name__)

_STRUCTURED_KEYWORDS = re.compile(
    r"\b(NORAD\s*ID|TLE|inclination|apoapsis|periapsis|"
    r"eccentricity|orbital\s+period|semi.?major\s+axis|"
    r"how\s+many\s+satellites|list\s+all\s+satellites|"
    r"satellite\s+catalog)\b",
    re.IGNORECASE,
)


class QueryRouter:
    def __init__(self, client: ChatClient | None = None) -> None:
        self._client = client

    @property
    def client(self) -> ChatClient:
        if self._client is None:
            raise RuntimeError("QueryRouter requires a ChatClient")
        return self._client

    async def classify(self, query: str) -> RoutingResult:
        if _STRUCTURED_KEYWORDS.search(query):
            return RoutingResult(
                query=query,
                query_type=QueryType.STRUCTURED,
                confidence=0.95,
                reasoning="Query contains structured-data keywords",
            )

        prompt = build_routing_prompt(query)
        messages = [
            ChatMessage(role="system", content="You are a query classifier."),
            ChatMessage(role="user", content=prompt),
        ]

        try:
            raw, _ = await self.client.complete(messages)
            parsed = json.loads(raw)
            query_type = QueryType(parsed["query_type"])
            return RoutingResult(
                query=query,
                query_type=query_type,
                confidence=float(parsed.get("confidence", 0.8)),
                reasoning=parsed.get("reasoning", ""),
            )
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            logger.warning("Routing parse failed (%s), defaulting to retrieval", exc)
            return RoutingResult(
                query=query,
                query_type=QueryType.RETRIEVAL,
                confidence=0.5,
                reasoning="Fallback: could not parse router response",
            )
