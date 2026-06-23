"""Data-quality analysis for the Space Mission Intelligence database.

Connects to PostgreSQL and produces a report covering completeness,
consistency, and validity checks across the documents, chunks, and
satellites tables.  Pass ``--llm`` to get an AI-generated interpretation
of the findings via the project's Groq / Mistral clients.

Usage (from the dev container or with POSTGRES_HOST=localhost):
    python -m scripts.data_quality [--localhost] [--llm]
"""

from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass, field
from enum import Enum

from sqlalchemy import create_engine, func, select, text
from sqlalchemy.orm import Session

from db.config import db_settings
from db.models import Chunk, Document, Satellite

# ── Report types ─────────────────────────────────────────────────────────────


class Severity(str, Enum):
    OK = "OK"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


@dataclass
class Finding:
    check: str
    severity: Severity
    detail: str
    value: str | int | float | None = None


@dataclass
class QualityReport:
    findings: list[Finding] = field(default_factory=list)

    def add(
        self,
        check: str,
        severity: Severity,
        detail: str,
        value: str | int | float | None = None,
    ) -> None:
        self.findings.append(Finding(check, severity, detail, value))

    def to_text(self) -> str:
        lines: list[str] = []
        for f in self.findings:
            val = f" ({f.value})" if f.value is not None else ""
            lines.append(f"[{f.severity.value}] {f.check}{val}: {f.detail}")
        counts = {s: 0 for s in Severity}
        for f in self.findings:
            counts[f.severity] += 1
        summary = " | ".join(f"{s.value}: {c}" for s, c in counts.items())
        lines.append(f"\nTotals: {summary}")
        return "\n".join(lines)

    def print(self) -> None:
        header = f"{'Check':<45} {'Severity':<10} {'Value':<12} Detail"
        print("\n" + "=" * 120)
        print("  SPACE MISSION INTELLIGENCE — DATA QUALITY REPORT")
        print("=" * 120)
        print(header)
        print("-" * 120)
        for f in self.findings:
            val = str(f.value) if f.value is not None else ""
            sev = f.severity.value
            print(f"  {f.check:<43} {sev:<10} {val:<12} {f.detail}")
        print("-" * 120)
        counts = {s: 0 for s in Severity}
        for f in self.findings:
            counts[f.severity] += 1
        summary = " | ".join(f"{s.value}: {c}" for s, c in counts.items())
        print(f"  Summary: {summary}")
        print("=" * 120 + "\n")


# ── Checks ───────────────────────────────────────────────────────────────────


def check_table_counts(session: Session, report: QualityReport) -> None:
    doc_count = session.scalar(select(func.count()).select_from(Document))
    chunk_count = session.scalar(select(func.count()).select_from(Chunk))
    sat_count = session.scalar(select(func.count()).select_from(Satellite))

    report.add("documents.row_count", Severity.OK, "Total documents", doc_count)
    report.add("chunks.row_count", Severity.OK, "Total chunks", chunk_count)

    if sat_count == 0:
        report.add(
            "satellites.row_count",
            Severity.WARNING,
            "No satellites loaded — TLE data missing",
            0,
        )
    else:
        report.add("satellites.row_count", Severity.OK, "Total satellites", sat_count)

    if doc_count and chunk_count:
        ratio = round(chunk_count / doc_count, 1)
        report.add(
            "chunks_per_document.avg",
            Severity.OK,
            "Average chunks/document",
            ratio,
        )


def check_document_completeness(session: Session, report: QualityReport) -> None:
    total = session.scalar(select(func.count()).select_from(Document))
    null_url = session.scalar(select(func.count()).where(Document.source_url.is_(None)))
    null_mission = session.scalar(
        select(func.count()).where(Document.mission_name.is_(None))
    )

    if null_url == total:
        report.add(
            "documents.source_url",
            Severity.WARNING,
            "All documents have NULL source_url — provenance is untraceable",
            null_url,
        )
    elif null_url and null_url > 0:
        report.add(
            "documents.source_url",
            Severity.INFO,
            f"{null_url}/{total} documents missing source_url",
            null_url,
        )
    else:
        report.add("documents.source_url", Severity.OK, "All populated", total)

    if null_mission == total:
        report.add(
            "documents.mission_name",
            Severity.WARNING,
            "All documents have NULL mission_name — no mission tagging",
            null_mission,
        )
    elif null_mission and null_mission > 0:
        report.add(
            "documents.mission_name",
            Severity.INFO,
            f"{null_mission}/{total} documents missing mission_name",
            null_mission,
        )
    else:
        report.add("documents.mission_name", Severity.OK, "All populated", total)

    source_types = session.execute(
        select(Document.source_type, func.count()).group_by(Document.source_type)
    ).all()
    types_str = ", ".join(f"{st.value}={n}" for st, n in source_types)
    if len(source_types) == 1:
        report.add(
            "documents.source_type_diversity",
            Severity.INFO,
            f"Only one source type present: {types_str}",
            1,
        )
    else:
        report.add(
            "documents.source_type_diversity",
            Severity.OK,
            f"Source types: {types_str}",
            len(source_types),
        )


def check_document_metadata(session: Session, report: QualityReport) -> None:
    rows = session.execute(
        text("""
            SELECT
                avg((metadata->>'avg_quality')::float) as avg_q,
                min((metadata->>'avg_quality')::float) as min_q,
                count(*) filter (
                    WHERE (metadata->>'avg_quality')::float < 0.7
                ) as low_quality_docs
            FROM documents
            WHERE metadata IS NOT NULL AND metadata::text != '{}'
        """)
    ).one()

    if rows.avg_q is not None:
        report.add(
            "documents.extraction_quality.avg",
            Severity.OK,
            "Mean PDF extraction quality score",
            round(rows.avg_q, 3),
        )
        if rows.min_q < 0.7:
            report.add(
                "documents.extraction_quality.min",
                Severity.WARNING,
                "Lowest extraction quality is below 0.7 threshold",
                round(rows.min_q, 3),
            )
        else:
            report.add(
                "documents.extraction_quality.min",
                Severity.OK,
                "Minimum extraction quality",
                round(rows.min_q, 3),
            )
        if rows.low_quality_docs > 0:
            report.add(
                "documents.low_quality_count",
                Severity.WARNING,
                "Documents with extraction quality < 0.7",
                rows.low_quality_docs,
            )


def check_chunk_completeness(session: Session, report: QualityReport) -> None:
    total = session.scalar(select(func.count()).select_from(Chunk))
    null_emb = session.scalar(select(func.count()).where(Chunk.embedding.is_(None)))
    null_section = session.scalar(
        select(func.count()).where(Chunk.section_path.is_(None))
    )
    null_search = session.scalar(
        select(func.count()).where(Chunk.search_vector.is_(None))
    )

    if null_emb and null_emb > 0:
        report.add(
            "chunks.null_embeddings",
            Severity.ERROR,
            "Chunks missing embeddings — vector search broken for these",
            null_emb,
        )
    else:
        report.add(
            "chunks.null_embeddings",
            Severity.OK,
            "All chunks have embeddings",
            0,
        )

    if null_search and null_search > 0:
        report.add(
            "chunks.null_search_vector",
            Severity.ERROR,
            "Chunks missing tsvector — full-text search broken for these",
            null_search,
        )
    else:
        report.add(
            "chunks.null_search_vector",
            Severity.OK,
            "All chunks have search_vector populated",
            0,
        )

    if null_section and null_section > 0:
        pct = round(100 * null_section / total, 1) if total else 0
        report.add(
            "chunks.null_section_path",
            Severity.INFO,
            f"{null_section}/{total} chunks ({pct}%) missing section_path",
            null_section,
        )


def check_chunk_token_distribution(session: Session, report: QualityReport) -> None:
    stats = session.execute(
        select(
            func.min(Chunk.token_count),
            func.max(Chunk.token_count),
            func.avg(Chunk.token_count),
        )
    ).one()
    min_t, max_t, avg_t = stats

    report.add(
        "chunks.token_count.range",
        Severity.OK,
        f"min={min_t}, avg={round(avg_t)}, max={max_t}",
    )

    micro_count = session.scalar(select(func.count()).where(Chunk.token_count <= 5))
    if micro_count and micro_count > 0:
        total = session.scalar(select(func.count()).select_from(Chunk))
        pct = round(100 * micro_count / total, 1) if total else 0
        sev = Severity.WARNING if pct > 10 else Severity.INFO
        report.add(
            "chunks.micro_chunks",
            sev,
            f"Chunks with <= 5 tokens ({pct}%) — likely noise from PDF artifacts",
            micro_count,
        )

    oversized = session.scalar(select(func.count()).where(Chunk.token_count > 800))
    if oversized and oversized > 0:
        total = session.scalar(select(func.count()).select_from(Chunk))
        pct = round(100 * oversized / total, 1) if total else 0
        sev = Severity.WARNING if pct > 15 else Severity.INFO
        report.add(
            "chunks.oversized_chunks",
            sev,
            f"Chunks with > 800 tokens ({pct}%) — may exceed context limits",
            oversized,
        )


def check_chunk_metadata(session: Session, report: QualityReport) -> None:
    result = session.execute(
        text("""
            SELECT
                count(*) as total,
                count(*) filter (
                    WHERE metadata IS NULL OR metadata::text = '{}'
                ) as empty_metadata
            FROM chunks
        """)
    ).one()

    if result.empty_metadata > 0:
        pct = round(100 * result.empty_metadata / result.total, 1)
        sev = Severity.WARNING if pct > 50 else Severity.INFO
        report.add(
            "chunks.empty_metadata",
            sev,
            f"{result.empty_metadata}/{result.total} chunks ({pct}%) have no metadata",
            result.empty_metadata,
        )
    else:
        report.add(
            "chunks.empty_metadata",
            Severity.OK,
            "All chunks have populated metadata",
            0,
        )


def check_referential_integrity(session: Session, report: QualityReport) -> None:
    orphaned = session.scalar(
        text("""
            SELECT count(*) FROM chunks c
            LEFT JOIN documents d ON d.id = c.document_id
            WHERE d.id IS NULL
        """)
    )

    if orphaned and orphaned > 0:
        report.add(
            "integrity.orphaned_chunks",
            Severity.ERROR,
            "Chunks referencing non-existent documents",
            orphaned,
        )
    else:
        report.add(
            "integrity.orphaned_chunks",
            Severity.OK,
            "No orphaned chunks — all reference valid documents",
            0,
        )

    empty_docs = session.scalar(
        text("""
            SELECT count(*) FROM documents d
            LEFT JOIN chunks c ON c.document_id = d.id
            WHERE c.id IS NULL
        """)
    )

    if empty_docs and empty_docs > 0:
        report.add(
            "integrity.empty_documents",
            Severity.WARNING,
            "Documents with zero chunks — ingestion may have failed",
            empty_docs,
        )
    else:
        report.add(
            "integrity.empty_documents",
            Severity.OK,
            "All documents have at least one chunk",
            0,
        )


def check_chunk_index_continuity(session: Session, report: QualityReport) -> None:
    gaps = session.execute(
        text("""
            SELECT d.title
            FROM documents d
            JOIN chunks c ON c.document_id = d.id
            GROUP BY d.id, d.title
            HAVING max(c.chunk_index) - min(c.chunk_index) + 1 != count(c.id)
        """)
    ).all()

    if gaps:
        titles = ", ".join(row[0] for row in gaps[:5])
        report.add(
            "integrity.chunk_index_gaps",
            Severity.WARNING,
            f"Documents with non-contiguous chunk indices: {titles}",
            len(gaps),
        )
    else:
        report.add(
            "integrity.chunk_index_gaps",
            Severity.OK,
            "All documents have contiguous chunk indices (0..N)",
            0,
        )


def check_duplicates(session: Session, report: QualityReport) -> None:
    dup_checksums = session.execute(
        text("""
            SELECT checksum, count(*) as n
            FROM documents GROUP BY checksum HAVING count(*) > 1
        """)
    ).all()

    if dup_checksums:
        report.add(
            "integrity.duplicate_checksums",
            Severity.ERROR,
            "Duplicate document checksums found",
            len(dup_checksums),
        )
    else:
        report.add(
            "integrity.duplicate_checksums",
            Severity.OK,
            "All document checksums are unique",
            0,
        )

    dup_chunks = session.scalar(
        text("""
            SELECT count(*) FROM (
                SELECT document_id, chunk_index, count(*) as n
                FROM chunks
                GROUP BY document_id, chunk_index
                HAVING count(*) > 1
            ) sub
        """)
    )

    if dup_chunks and dup_chunks > 0:
        report.add(
            "integrity.duplicate_chunk_indices",
            Severity.ERROR,
            "Duplicate (document_id, chunk_index) pairs",
            dup_chunks,
        )
    else:
        report.add(
            "integrity.duplicate_chunk_indices",
            Severity.OK,
            "No duplicate chunk indices within any document",
            0,
        )


def check_satellite_data(session: Session, report: QualityReport) -> None:
    count = session.scalar(select(func.count()).select_from(Satellite))
    if not count or count == 0:
        report.add(
            "satellites.empty",
            Severity.WARNING,
            "No satellite data — TLE ingestion not yet run",
            0,
        )
        return

    null_launch = session.scalar(
        select(func.count()).where(Satellite.launch_date.is_(None))
    )
    null_period = session.scalar(
        select(func.count()).where(Satellite.period_minutes.is_(None))
    )
    null_type = session.scalar(
        select(func.count()).where(Satellite.object_type.is_(None))
    )

    for col, null_n, label in [
        ("launch_date", null_launch, "satellites missing launch_date"),
        ("period_minutes", null_period, "satellites missing period_minutes"),
        ("object_type", null_type, "satellites missing object_type"),
    ]:
        if null_n and null_n > 0:
            pct = round(100 * null_n / count, 1)
            sev = Severity.INFO if pct < 20 else Severity.WARNING
            report.add(f"satellites.null_{col}", sev, label, null_n)

    bad_ecc = session.scalar(
        select(func.count()).where(
            (Satellite.eccentricity < 0) | (Satellite.eccentricity >= 1)
        )
    )
    if bad_ecc and bad_ecc > 0:
        report.add(
            "satellites.invalid_eccentricity",
            Severity.ERROR,
            "Satellites with eccentricity outside [0, 1)",
            bad_ecc,
        )

    bad_incl = session.scalar(
        select(func.count()).where(
            (Satellite.inclination < 0) | (Satellite.inclination > 180)
        )
    )
    if bad_incl and bad_incl > 0:
        report.add(
            "satellites.invalid_inclination",
            Severity.ERROR,
            "Satellites with inclination outside [0, 180]",
            bad_incl,
        )


# ── Main ─────────────────────────────────────────────────────────────────────


def run_quality_report(session: Session) -> QualityReport:
    report = QualityReport()

    check_table_counts(session, report)
    check_document_completeness(session, report)
    check_document_metadata(session, report)
    check_chunk_completeness(session, report)
    check_chunk_token_distribution(session, report)
    check_chunk_metadata(session, report)
    check_referential_integrity(session, report)
    check_chunk_index_continuity(session, report)
    check_duplicates(session, report)
    check_satellite_data(session, report)

    return report


# ── LLM analysis ─────────────────────────────────────────────────────────────

_ANALYSIS_SYSTEM_PROMPT = """\
You are a data-quality engineer specialised in RAG (Retrieval-Augmented \
Generation) pipelines and space-mission datasets.

You will receive the output of an automated data-quality scan run against a \
PostgreSQL database that stores:
  • documents  – ingested PDFs / HTML / JSON / TLE about space missions
  • chunks     – text chunks with vector embeddings and full-text search vectors
  • satellites – orbital elements derived from TLE data

Analyse the findings and produce a structured report with:
1. **Executive Summary** – 2-3 sentences on overall health.
2. **Critical Issues** – anything that will degrade retrieval or generation \
quality right now (ERROR and high-impact WARNING items).
3. **Improvement Opportunities** – lower-priority items that would raise \
data quality (INFO and minor WARNING items).
4. **Recommended Actions** – a prioritised, actionable checklist the team \
can follow, with concrete steps (e.g. SQL queries, pipeline changes, \
ingestion tweaks).
5. **Impact on RAG Quality** – explain how each issue specifically affects \
retrieval precision, recall, or answer generation quality.

Be specific and quantitative — reference the actual numbers from the report. \
Keep the tone professional but direct.\
"""

_CHUNK_ANALYSIS_SYSTEM_PROMPT = """\
You are a data-quality engineer specialised in RAG (Retrieval-Augmented \
Generation) pipelines for space-mission intelligence.

You will receive a **sampled set of text chunks** from a PostgreSQL database \
used for retrieval-augmented generation about space missions.  The chunks are \
grouped into categories so you can compare quality across segments.

Analyse the actual text content and produce a structured report with:

1. **Content Quality Summary** – overall assessment of the text quality, \
readability, and information density across the sample.

2. **Micro-Chunks Analysis** (≤ 5 tokens) – what are these fragments?  \
Are they PDF artefacts (page numbers, headers, footers), meaningful \
abbreviations, or something else?  Should they be filtered, merged with \
neighbours, or kept?

3. **Low-Quality Document Chunks** – assess the chunks from documents with \
low extraction scores.  Are they garbled, missing content, or containing OCR \
errors?  How badly would they pollute retrieval results?

4. **Oversized Chunks** (> 800 tokens) – do these contain coherent, \
self-contained passages or are they runaway extractions?  Would splitting \
them improve retrieval precision?

5. **Normal Chunks Baseline** – are the "healthy" chunks well-formed?  \
Do they contain meaningful, self-contained passages suitable for RAG?

6. **Recommended Actions** – prioritised, concrete steps to improve chunk \
quality (filtering rules, merging heuristics, re-chunking strategies, \
content cleaning regexes).

7. **Estimated Impact** – which fixes would yield the biggest improvement \
in retrieval precision and recall?

Be specific — quote actual chunk content to illustrate your points.  \
Keep the tone professional but direct.\
"""

MICRO_CHUNK_LIMIT = 30
OVERSIZED_SAMPLE = 5
LOW_QUALITY_SAMPLE = 10
NORMAL_SAMPLE = 8


def _sample_chunks(session: Session) -> str:
    sections: list[str] = []

    # 1. Micro-chunks (≤ 5 tokens) — they're tiny so we can grab many
    micro = session.execute(
        text("""
            SELECT c.chunk_index, c.token_count, c.content,
                   c.section_path, d.title
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            WHERE c.token_count <= 5
            ORDER BY c.token_count, d.title
            LIMIT :lim
        """),
        {"lim": MICRO_CHUNK_LIMIT},
    ).all()
    if micro:
        lines = [
            f"  [{r.title} | idx={r.chunk_index} | {r.token_count}tok "
            f"| section={r.section_path}]\n  {r.content!r}"
            for r in micro
        ]
        sections.append(
            f"## MICRO-CHUNKS (≤5 tokens) — {len(micro)} samples\n" + "\n\n".join(lines)
        )

    # 2. Chunks from low-quality documents (extraction quality < 0.7)
    low_q = session.execute(
        text("""
            SELECT c.chunk_index, c.token_count, c.content,
                   c.section_path, d.title,
                   (d.metadata->>'avg_quality')::float as doc_quality
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            WHERE (d.metadata->>'avg_quality')::float < 0.7
            ORDER BY c.chunk_index
            LIMIT :lim
        """),
        {"lim": LOW_QUALITY_SAMPLE},
    ).all()
    if low_q:
        lines = [
            f"  [{r.title} | idx={r.chunk_index} | {r.token_count}tok "
            f"| doc_quality={r.doc_quality} | section={r.section_path}]\n  {r.content!r}"
            for r in low_q
        ]
        sections.append(
            f"## LOW-QUALITY DOCUMENT CHUNKS — {len(low_q)} samples\n"
            + "\n\n".join(lines)
        )

    # 3. Oversized chunks (> 800 tokens) — random sample
    oversized = session.execute(
        text("""
            SELECT c.chunk_index, c.token_count, c.content,
                   c.section_path, d.title
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            WHERE c.token_count > 800
            ORDER BY random()
            LIMIT :lim
        """),
        {"lim": OVERSIZED_SAMPLE},
    ).all()
    if oversized:
        lines = [
            f"  [{r.title} | idx={r.chunk_index} | {r.token_count}tok "
            f"| section={r.section_path}]\n  {r.content[:500]!r}... [truncated]"
            for r in oversized
        ]
        sections.append(
            f"## OVERSIZED CHUNKS (>800 tokens) — {len(oversized)} samples\n"
            + "\n\n".join(lines)
        )

    # 4. Normal chunks — random baseline sample (50-500 tokens)
    normal = session.execute(
        text("""
            SELECT c.chunk_index, c.token_count, c.content,
                   c.section_path, d.title
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            WHERE c.token_count BETWEEN 50 AND 500
            ORDER BY random()
            LIMIT :lim
        """),
        {"lim": NORMAL_SAMPLE},
    ).all()
    if normal:
        lines = [
            f"  [{r.title} | idx={r.chunk_index} | {r.token_count}tok "
            f"| section={r.section_path}]\n  {r.content[:500]!r}"
            for r in normal
        ]
        sections.append(
            f"## NORMAL CHUNKS (50-500 tokens) — {len(normal)} baseline samples\n"
            + "\n\n".join(lines)
        )

    return "\n\n" + "\n\n".join(sections)


def _build_llm_client():  # noqa: ANN201 — returns ChatClient (import is local)
    import os

    from scripts.generation.client import (
        ChatClient,
        FallbackClient,
        GroqClient,
        MistralClient,
    )
    from scripts.generation.config import GenerationSettings

    settings = GenerationSettings()

    primary: ChatClient = GroqClient(
        api_key=os.environ["GROQ_API_KEY"],
        model=settings.LLM_MODEL,
        temperature=0.3,
        max_tokens=4096,
    )

    mistral_key = os.environ.get("MISTRAL_API_KEY")
    if mistral_key:
        fallback = MistralClient(
            api_key=mistral_key,
            model=settings.LLM_FALLBACK_MODEL,
            temperature=0.3,
            max_tokens=4096,
        )
        client: ChatClient = FallbackClient(primary, fallback)
    else:
        client = primary

    return client


async def llm_analyse(report: QualityReport) -> str:
    from scripts.generation.schemas import ChatMessage

    client = _build_llm_client()

    messages = [
        ChatMessage(role="system", content=_ANALYSIS_SYSTEM_PROMPT),
        ChatMessage(
            role="user",
            content=(
                "Here is the automated data-quality report:\n\n"
                f"{report.to_text()}\n\n"
                "Please analyse these findings."
            ),
        ),
    ]

    content, usage = await client.complete(messages)
    tokens = usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)
    print(f"  (LLM analysis used {tokens} tokens)\n")
    return content


async def llm_analyse_chunks(chunk_sample: str) -> str:
    from scripts.generation.schemas import ChatMessage

    client = _build_llm_client()

    messages = [
        ChatMessage(role="system", content=_CHUNK_ANALYSIS_SYSTEM_PROMPT),
        ChatMessage(
            role="user",
            content=(
                "Here are sampled chunks from the database, grouped by "
                "category:\n\n"
                f"{chunk_sample}\n\n"
                "Please analyse the content quality of these chunks."
            ),
        ),
    ]

    content, usage = await client.complete(messages)
    tokens = usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)
    print(f"  (LLM chunk analysis used {tokens} tokens)\n")
    return content


def main() -> None:
    url = db_settings.sync_url
    if "--localhost" in sys.argv:
        url = url.replace("@db:", "@localhost:")

    engine = create_engine(url)

    with Session(engine) as session:
        report = run_quality_report(session)
        report.print()

        if "--llm" in sys.argv:
            print("=" * 120)
            print("  LLM ANALYSIS (metadata)")
            print("=" * 120)
            analysis = asyncio.run(llm_analyse(report))
            print(analysis)
            print("=" * 120 + "\n")

        if "--llm-chunks" in sys.argv:
            print("  Sampling chunks from database ...")
            chunk_sample = _sample_chunks(session)
            char_count = len(chunk_sample)
            print(f"  Sample size: {char_count:,} characters\n")
            print("=" * 120)
            print("  LLM ANALYSIS (chunk content)")
            print("=" * 120)
            analysis = asyncio.run(llm_analyse_chunks(chunk_sample))
            print(analysis)
            print("=" * 120 + "\n")


if __name__ == "__main__":
    main()
