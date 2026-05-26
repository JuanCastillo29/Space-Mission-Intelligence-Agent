"""Initial schema: documents, chunks, satellites.

Revision ID: 001
Revises:
Create Date: 2025-05-26
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR, UUID

revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

EMBEDDING_DIM = 1024


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    source_type = sa.Enum("pdf", "html", "json", "tle", "csv", name="source_type")
    source_type.create(op.get_bind())

    op.create_table(
        "documents",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("title", sa.Text, nullable=False),
        sa.Column("source_url", sa.Text),
        sa.Column(
            "source_type",
            sa.Enum("pdf", "html", "json", "tle", "csv", name="source_type", create_constraint=False),
            nullable=False,
        ),
        sa.Column("mission_name", sa.String(256)),
        sa.Column("checksum", sa.String(64), unique=True, nullable=False),
        sa.Column("metadata", JSONB, server_default="{}"),
        sa.Column(
            "ingested_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
        ),
    )

    op.create_table(
        "chunks",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "document_id",
            UUID(as_uuid=True),
            sa.ForeignKey("documents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("embedding", Vector(EMBEDDING_DIM)),
        sa.Column("chunk_index", sa.Integer, nullable=False),
        sa.Column("section_path", sa.Text),
        sa.Column("token_count", sa.Integer, nullable=False),
        sa.Column("metadata", JSONB, server_default="{}"),
        sa.Column("search_vector", TSVECTOR),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
        ),
    )

    op.create_index("ix_chunks_document_id", "chunks", ["document_id"])
    op.create_index(
        "ix_chunks_search_vector",
        "chunks",
        ["search_vector"],
        postgresql_using="gin",
    )
    op.create_index(
        "ix_chunks_embedding_hnsw",
        "chunks",
        ["embedding"],
        postgresql_using="hnsw",
        postgresql_with={"m": 16, "ef_construction": 200},
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )

    op.execute(
        """
        CREATE FUNCTION chunks_search_vector_update() RETURNS trigger AS $$
        BEGIN
            NEW.search_vector := to_tsvector('english', NEW.content);
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
        """
    )
    op.execute(
        """
        CREATE TRIGGER chunks_search_vector_trigger
            BEFORE INSERT OR UPDATE OF content ON chunks
            FOR EACH ROW
            EXECUTE FUNCTION chunks_search_vector_update();
        """
    )

    op.create_table(
        "satellites",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("norad_id", sa.Integer, unique=True, nullable=False),
        sa.Column("name", sa.String(256), nullable=False),
        sa.Column("international_designator", sa.String(32)),
        sa.Column("epoch", sa.DateTime(timezone=True), nullable=False),
        sa.Column("mean_motion", sa.Float, nullable=False),
        sa.Column("eccentricity", sa.Float, nullable=False),
        sa.Column("inclination", sa.Float, nullable=False),
        sa.Column("ra_of_asc_node", sa.Float, nullable=False),
        sa.Column("arg_of_pericenter", sa.Float, nullable=False),
        sa.Column("mean_anomaly", sa.Float, nullable=False),
        sa.Column("rev_number", sa.Integer),
        sa.Column("classification", sa.String(16)),
        sa.Column("object_type", sa.String(32)),
        sa.Column("launch_date", sa.Date),
        sa.Column("decay_date", sa.Date),
        sa.Column("period_minutes", sa.Float),
        sa.Column("apoapsis_km", sa.Float),
        sa.Column("periapsis_km", sa.Float),
        sa.Column("rcs_meters2", sa.Float),
        sa.Column("source", sa.String(64), nullable=False),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
        ),
    )

    op.create_index("ix_satellites_name", "satellites", ["name"])
    op.create_index("ix_satellites_object_type", "satellites", ["object_type"])


def downgrade() -> None:
    op.drop_table("satellites")
    op.execute("DROP TRIGGER IF EXISTS chunks_search_vector_trigger ON chunks")
    op.execute("DROP FUNCTION IF EXISTS chunks_search_vector_update()")
    op.drop_table("chunks")
    op.drop_table("documents")
    sa.Enum(name="source_type").drop(op.get_bind())
    op.execute("DROP EXTENSION IF EXISTS vector")
