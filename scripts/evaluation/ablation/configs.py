from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

SearchMode = Literal["semantic_only", "keyword_only", "hybrid"]


@dataclass(frozen=True)
class AblationConfig:
    """A single ablation configuration.

    Each dimension maps to one lever in the retrieval pipeline:

    - ``search_mode``      -> which search function is used
    - ``reranking_enabled``-> real BGE reranker vs. a pass-through no-op
    - ``chunk_size_filter``-> restrict search to chunks tagged with this
      ``eval_chunk_size`` (produced by ``rechunk.py``); ``None`` uses the
      corpus as ingested
    - ``metadata_filtering``-> whether a per-query ``metadata_filter`` on the
      chunk JSONB is honoured (when disabled, the filter is ignored)
    """

    name: str
    search_mode: SearchMode = "hybrid"
    reranking_enabled: bool = True
    chunk_size_filter: int | None = None
    metadata_filtering: bool = True
    mmr_lambda: float = 0.7
    search_top_k: int = 20
    rerank_top_k: int = 10
    final_top_k: int = 5

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# The ablation matrix described in the README. ``baseline`` is the reference
# configuration; every other entry changes exactly one lever relative to it.
ABLATION_CONFIGS: list[AblationConfig] = [
    AblationConfig(name="baseline"),
    AblationConfig(name="semantic_only", search_mode="semantic_only"),
    AblationConfig(name="keyword_only", search_mode="keyword_only"),
    AblationConfig(name="no_reranking", reranking_enabled=False),
    AblationConfig(name="chunk_512", chunk_size_filter=512),
    AblationConfig(name="chunk_2048", chunk_size_filter=2048),
    AblationConfig(name="no_metadata_filter", metadata_filtering=False),
]

CONFIGS_BY_NAME: dict[str, AblationConfig] = {c.name: c for c in ABLATION_CONFIGS}


def get_configs(names: list[str] | None = None) -> list[AblationConfig]:
    """Resolve config names to ``AblationConfig`` objects.

    ``None`` returns the full matrix. Unknown names raise ``KeyError`` with the
    list of valid names to make CLI typos obvious.
    """
    if names is None:
        return list(ABLATION_CONFIGS)

    resolved: list[AblationConfig] = []
    for name in names:
        if name not in CONFIGS_BY_NAME:
            valid = ", ".join(CONFIGS_BY_NAME)
            raise KeyError(f"Unknown ablation config {name!r}. Valid names: {valid}")
        resolved.append(CONFIGS_BY_NAME[name])
    return resolved
