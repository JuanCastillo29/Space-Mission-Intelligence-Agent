import uuid
from datetime import datetime, timezone

from sqlalchemy import select

from db.models import Chunk, Document, Satellite, SourceType, EMBEDDING_DIM


class TestDocument:
    def test_insert_and_read(self, session):
        doc = Document(
            title="Rosetta Mission Report",
            source_type=SourceType.PDF,
            checksum="abc123",
        )
        session.add(doc)
        session.flush()

        result = session.scalar(select(Document).where(Document.checksum == "abc123"))
        assert result is not None
        assert result.title == "Rosetta Mission Report"
        assert result.source_type == SourceType.PDF
        assert isinstance(result.id, uuid.UUID)

    def test_checksum_unique(self, session):
        doc1 = Document(title="Doc A", source_type=SourceType.PDF, checksum="dup")
        doc2 = Document(title="Doc B", source_type=SourceType.HTML, checksum="dup")
        session.add(doc1)
        session.flush()
        session.add(doc2)

        from sqlalchemy.exc import IntegrityError
        import pytest

        with pytest.raises(IntegrityError):
            session.flush()

    def test_ingested_at_default(self, session):
        doc = Document(
            title="Auto-timestamp",
            source_type=SourceType.JSON,
            checksum="ts123",
        )
        session.add(doc)
        session.flush()

        assert doc.ingested_at is not None

    def test_metadata_default(self, session):
        doc = Document(
            title="Meta test",
            source_type=SourceType.CSV,
            checksum="meta1",
        )
        session.add(doc)
        session.flush()

        result = session.scalar(select(Document).where(Document.checksum == "meta1"))
        assert result.metadata_ is not None


class TestChunk:
    def _make_doc(self, session, checksum):
        doc = Document(title="Parent", source_type=SourceType.PDF, checksum=checksum)
        session.add(doc)
        session.flush()
        return doc

    def test_insert_with_embedding(self, session):
        doc = self._make_doc(session, "emb1")
        embedding = [0.1] * EMBEDDING_DIM

        chunk = Chunk(
            document_id=doc.id,
            content="Thermal analysis results from Rosetta.",
            embedding=embedding,
            chunk_index=0,
            token_count=7,
        )
        session.add(chunk)
        session.flush()

        result = session.scalar(select(Chunk).where(Chunk.document_id == doc.id))
        assert result is not None
        assert result.content == "Thermal analysis results from Rosetta."
        assert len(result.embedding) == EMBEDDING_DIM

    def test_search_vector_trigger(self, session):
        doc = self._make_doc(session, "tsvec1")

        chunk = Chunk(
            document_id=doc.id,
            content="Solar wind measurements during perihelion approach",
            chunk_index=0,
            token_count=6,
        )
        session.add(chunk)
        session.flush()

        # Force PostgreSQL to return the trigger-populated value
        session.expire(chunk)
        assert chunk.search_vector is not None

    def test_fulltext_search(self, session):
        doc = self._make_doc(session, "fts1")

        chunk = Chunk(
            document_id=doc.id,
            content="The ECSS thermal standard defines requirements for spacecraft thermal control",
            chunk_index=0,
            token_count=10,
        )
        session.add(chunk)
        session.flush()

        result = session.execute(
            select(Chunk).where(Chunk.search_vector.match("thermal & spacecraft"))
        ).scalar_one_or_none()

        assert result is not None
        assert result.id == chunk.id

    def test_cascade_delete(self, session):
        doc = self._make_doc(session, "cascade1")
        chunk = Chunk(
            document_id=doc.id,
            content="Will be deleted",
            chunk_index=0,
            token_count=3,
        )
        session.add(chunk)
        session.flush()
        chunk_id = chunk.id

        session.delete(doc)
        session.flush()

        assert session.get(Chunk, chunk_id) is None

    def test_document_relationship(self, session):
        doc = self._make_doc(session, "rel1")
        chunk = Chunk(
            document_id=doc.id,
            content="Linked to parent",
            chunk_index=0,
            token_count=3,
        )
        session.add(chunk)
        session.flush()

        assert chunk.document.title == "Parent"
        assert len(doc.chunks) == 1


class TestSatellite:
    def test_insert_and_read(self, session):
        sat = Satellite(
            norad_id=25544,
            name="ISS (ZARYA)",
            epoch=datetime(2024, 1, 1, tzinfo=timezone.utc),
            mean_motion=15.489,
            eccentricity=0.0006,
            inclination=51.6435,
            ra_of_asc_node=290.0,
            arg_of_pericenter=120.0,
            mean_anomaly=240.0,
            object_type="PAYLOAD",
            source="celestrak",
        )
        session.add(sat)
        session.flush()

        result = session.scalar(select(Satellite).where(Satellite.norad_id == 25544))
        assert result is not None
        assert result.name == "ISS (ZARYA)"
        assert result.object_type == "PAYLOAD"

    def test_norad_id_unique(self, session):
        sat1 = Satellite(
            norad_id=99999,
            name="SAT-A",
            epoch=datetime(2024, 1, 1, tzinfo=timezone.utc),
            mean_motion=15.0,
            eccentricity=0.001,
            inclination=51.0,
            ra_of_asc_node=0.0,
            arg_of_pericenter=0.0,
            mean_anomaly=0.0,
            source="celestrak",
        )
        sat2 = Satellite(
            norad_id=99999,
            name="SAT-B",
            epoch=datetime(2024, 1, 1, tzinfo=timezone.utc),
            mean_motion=14.0,
            eccentricity=0.002,
            inclination=52.0,
            ra_of_asc_node=0.0,
            arg_of_pericenter=0.0,
            mean_anomaly=0.0,
            source="celestrak",
        )
        session.add(sat1)
        session.flush()
        session.add(sat2)

        from sqlalchemy.exc import IntegrityError
        import pytest

        with pytest.raises(IntegrityError):
            session.flush()
