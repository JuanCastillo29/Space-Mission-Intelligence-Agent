import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from db.config import db_settings
from db.models import Base

TSVECTOR_TRIGGER_SQL = """
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_proc WHERE proname = 'chunks_search_vector_update'
    ) THEN
        CREATE FUNCTION chunks_search_vector_update() RETURNS trigger AS $fn$
        BEGIN
            NEW.search_vector := to_tsvector('english', NEW.content);
            RETURN NEW;
        END;
        $fn$ LANGUAGE plpgsql;
    END IF;
END $$;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_trigger WHERE tgname = 'chunks_search_vector_trigger'
    ) THEN
        CREATE TRIGGER chunks_search_vector_trigger
            BEFORE INSERT OR UPDATE OF content ON chunks
            FOR EACH ROW
            EXECUTE FUNCTION chunks_search_vector_update();
    END IF;
END $$;
"""


@pytest.fixture(scope="session")
def engine():
    eng = create_engine(db_settings.sync_url)
    yield eng
    eng.dispose()


@pytest.fixture(scope="session")
def tables(engine):
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()
    Base.metadata.create_all(engine)
    with engine.connect() as conn:
        conn.execute(text(TSVECTOR_TRIGGER_SQL))
        conn.commit()
    yield
    Base.metadata.drop_all(engine)


@pytest.fixture()
def session(engine, tables):
    """Yield a session wrapped in a transaction that rolls back after each test."""
    connection = engine.connect()
    transaction = connection.begin()
    sess = Session(bind=connection)
    yield sess
    sess.close()
    if transaction.is_active:
        transaction.rollback()
    connection.close()
