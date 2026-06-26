from __future__ import annotations

import httpx

from config import API_BASE_URL

_client = httpx.Client(base_url=API_BASE_URL, timeout=120.0)


def check_health() -> dict:
    try:
        resp = _client.get("/health")
        resp.raise_for_status()
        return resp.json()
    except (httpx.ConnectError, httpx.HTTPStatusError, httpx.TimeoutException):
        return {
            "status": "unreachable",
            "database": False,
            "embedder_loaded": False,
            "reranker_loaded": False,
            "version": "unknown",
        }


def submit_query(question: str) -> dict:
    resp = _client.post("/query", json={"question": question})
    resp.raise_for_status()
    return resp.json()


def list_documents(limit: int = 50, offset: int = 0) -> dict:
    resp = _client.get("/documents", params={"limit": limit, "offset": offset})
    resp.raise_for_status()
    return resp.json()
