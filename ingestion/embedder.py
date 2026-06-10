from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Embedder(Protocol):
    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class SentenceTransformerEmbedder:
    MODEL_NAME = "BAAI/bge-large-en-v1.5"

    def __init__(self) -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(self.MODEL_NAME)
        self.dim: int = self.model.get_sentence_embedding_dimension()

    def embed(self, texts: list[str]) -> list[list[float]]:
        embeddings = self.model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        )
        return embeddings.tolist()
