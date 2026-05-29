import enum
import uuid
from datetime import date, datetime
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Date,
    DateTime,
    Enum,
    Float,
    Index,
    Integer,
    String,
    Text,
    ForeignKey,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

EMBEDDING_DIM = 1024


class Base(DeclarativeBase):
    pass


class SourceType(str, enum.Enum):
    PDF = "pdf"
    HTML = "html"
    JSON = "json"
    TLE = "tle"
    CSV = "csv"


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    title: Mapped[str] = mapped_column(Text, nullable=False)
    source_url: Mapped[str | None] = mapped_column(Text)
    source_type: Mapped[SourceType] = mapped_column(
        Enum(
            SourceType,
            name="source_type",
            create_constraint=True,
            values_callable=lambda e: [member.value for member in e],
        ),
        nullable=False,
    )
    mission_name: Mapped[str | None] = mapped_column(String(256))
    checksum: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    metadata_: Mapped[dict | None] = mapped_column("metadata", JSONB, default=dict)
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    chunks: Mapped[list["Chunk"]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Document {self.title!r} ({self.source_type.value})>"


class Chunk(Base):
    __tablename__ = "chunks"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBEDDING_DIM))
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    section_path: Mapped[str | None] = mapped_column(Text)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False)
    metadata_: Mapped[dict | None] = mapped_column("metadata", JSONB, default=dict)
    search_vector: Mapped[Any] = mapped_column(TSVECTOR, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    document: Mapped["Document"] = relationship(back_populates="chunks")

    __table_args__ = (
        Index("ix_chunks_document_id", "document_id"),
        Index("ix_chunks_search_vector", "search_vector", postgresql_using="gin"),
        Index(
            "ix_chunks_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 200},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )

    def __repr__(self) -> str:
        return f"<Chunk {self.id} doc={self.document_id} idx={self.chunk_index}>"


class Satellite(Base):
    __tablename__ = "satellites"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    norad_id: Mapped[int] = mapped_column(Integer, unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String(256), nullable=False)
    international_designator: Mapped[str | None] = mapped_column(String(32))
    epoch: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    mean_motion: Mapped[float] = mapped_column(Float, nullable=False)
    eccentricity: Mapped[float] = mapped_column(Float, nullable=False)
    inclination: Mapped[float] = mapped_column(Float, nullable=False)
    ra_of_asc_node: Mapped[float] = mapped_column(Float, nullable=False)
    arg_of_pericenter: Mapped[float] = mapped_column(Float, nullable=False)
    mean_anomaly: Mapped[float] = mapped_column(Float, nullable=False)
    rev_number: Mapped[int | None] = mapped_column(Integer)
    classification: Mapped[str | None] = mapped_column(String(16))
    object_type: Mapped[str | None] = mapped_column(String(32))
    launch_date: Mapped[date | None] = mapped_column(Date)
    decay_date: Mapped[date | None] = mapped_column(Date)
    period_minutes: Mapped[float | None] = mapped_column(Float)
    apoapsis_km: Mapped[float | None] = mapped_column(Float)
    periapsis_km: Mapped[float | None] = mapped_column(Float)
    rcs_meters2: Mapped[float | None] = mapped_column(Float)
    source: Mapped[str] = mapped_column(String(64), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    __table_args__ = (
        Index("ix_satellites_name", "name"),
        Index("ix_satellites_object_type", "object_type"),
    )

    def __repr__(self) -> str:
        return f"<Satellite {self.name!r} (NORAD {self.norad_id})>"
