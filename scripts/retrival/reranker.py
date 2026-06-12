from __future__ import annotations

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from scripts.retrival.schemas import ScoredChunk

MODEL_NAME = "BAAI/bge-reranker-v2-m3"

DEFAULT_TOP_K = 5


class BGEReranker:
    def __init__(self, model_name: str = MODEL_NAME, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name
        ).to(self.device)

        self.model.eval()

    @torch.inference_mode()
    def score_pairs(
        self, query: str, documents: list[str], batch_size: int = 16
    ) -> list[float]:
        scores: list[float] = []

        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i : i + batch_size]
            pairs = [[query, doc] for doc in batch_docs]

            inputs = self.tokenizer(
                pairs,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self.device)

            logits = self.model(**inputs).logits.view(-1).float()
            scores.extend(logits.tolist())

        return scores

    def rerank(
        self,
        query: str,
        chunks: list[ScoredChunk],
        *,
        top_k: int = DEFAULT_TOP_K,
        batch_size: int = 16,
    ) -> list[ScoredChunk]:
        if not chunks:
            return []

        documents = [c.content for c in chunks]
        scores = self.score_pairs(query, documents, batch_size=batch_size)

        scored = sorted(
            zip(chunks, scores), key=lambda pair: pair[1], reverse=True
        )

        return [
            chunk.model_copy(update={"score": score})
            for chunk, score in scored[:top_k]
        ]
