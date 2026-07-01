from __future__ import annotations

import json
from pathlib import Path
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from db import Chunk
from scripts.evaluation.schemas import GoldenDataset, GoldenQAPair


def load_dataset(path: str | Path) -> GoldenDataset:
    raw = Path(path).read_text(encoding="utf-8")
    return GoldenDataset.model_validate(json.loads(raw))


async def resolve_ground_truth_chunk_ids(
    pairs: list[GoldenQAPair],
    session: AsyncSession,
) -> dict[str, list[UUID]]:
    mapping: dict[str, list[UUID]] = {}

    for pair in pairs:
        resolved: list[UUID] = []
        for chunk_ref in pair.ground_truth_chunk_ids:
            try:
                resolved.append(UUID(chunk_ref))
            except ValueError:
                stmt = select(Chunk.id).where(
                    Chunk.content.contains(chunk_ref)
                )
                rows = await session.execute(stmt)
                resolved.extend(row[0] for row in rows.all())

        mapping[pair.query] = resolved

    return mapping
