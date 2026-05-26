"""Download seed data from Hugging Face and load it into the database.

Runs once after migrations. Skips if the database already contains data
(idempotent — safe to re-run).
"""

import json
import logging
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import create_engine, func, select, text
from sqlalchemy.orm import Session

from db.config import db_settings
from db.models import Base, Chunk, Document, Satellite

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# ── Configuration ────────────────────────────────────────────────────────────
# TODO: Replace with your Hugging Face dataset repo
HF_DATASET_REPO = "juanc/space-mission-intelligence-data"
HF_FILES = {
    "documents": "seed/documents.json",
    "chunks": "seed/chunks.json",
    "satellites": "seed/satellites.json",
}


def download_seed_files(dest_dir: Path) -> dict[str, Path]:
    """Download seed files from Hugging Face Hub.

    Returns a mapping of table name → local file path.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        logger.error("huggingface_hub is not installed. Run: pip install huggingface_hub")
        sys.exit(1)

    paths: dict[str, Path] = {}
    for table_name, hf_path in HF_FILES.items():
        logger.info("Downloading %s from %s ...", hf_path, HF_DATASET_REPO)
        local = hf_hub_download(
            repo_id=HF_DATASET_REPO,
            filename=hf_path,
            repo_type="dataset",
            local_dir=str(dest_dir),
        )
        paths[table_name] = Path(local)
    return paths


def is_db_populated(session: Session) -> bool:
    """Check if the database already has data."""
    count = session.scalar(select(func.count()).select_from(Document))
    return count is not None and count > 0


def load_documents(session: Session, path: Path) -> dict[str, uuid.UUID]:
    """Load documents and return a mapping of checksum → generated UUID."""
    with open(path) as f:
        rows = json.load(f)

    checksum_to_id: dict[str, uuid.UUID] = {}
    for row in rows:
        doc = Document(
            title=row["title"],
            source_url=row.get("source_url"),
            source_type=row["source_type"],
            mission_name=row.get("mission_name"),
            checksum=row["checksum"],
            metadata_=row.get("metadata", {}),
        )
        session.add(doc)
        session.flush()
        checksum_to_id[row["checksum"]] = doc.id

    logger.info("Loaded %d documents.", len(rows))
    return checksum_to_id


def load_chunks(session: Session, path: Path, checksum_to_id: dict[str, uuid.UUID]) -> None:
    """Load chunks, resolving document references by checksum."""
    with open(path) as f:
        rows = json.load(f)

    for row in rows:
        doc_id = checksum_to_id.get(row["document_checksum"])
        if doc_id is None:
            logger.warning("Skipping chunk — unknown document checksum: %s", row["document_checksum"])
            continue

        chunk = Chunk(
            document_id=doc_id,
            content=row["content"],
            embedding=row.get("embedding"),
            chunk_index=row["chunk_index"],
            section_path=row.get("section_path"),
            token_count=row["token_count"],
            metadata_=row.get("metadata", {}),
        )
        session.add(chunk)

    logger.info("Loaded %d chunks.", len(rows))


def load_satellites(session: Session, path: Path) -> None:
    """Load satellite orbital data."""
    with open(path) as f:
        rows = json.load(f)

    for row in rows:
        sat = Satellite(
            norad_id=row["norad_id"],
            name=row["name"],
            international_designator=row.get("international_designator"),
            epoch=datetime.fromisoformat(row["epoch"]),
            mean_motion=row["mean_motion"],
            eccentricity=row["eccentricity"],
            inclination=row["inclination"],
            ra_of_asc_node=row["ra_of_asc_node"],
            arg_of_pericenter=row["arg_of_pericenter"],
            mean_anomaly=row["mean_anomaly"],
            rev_number=row.get("rev_number"),
            classification=row.get("classification"),
            object_type=row.get("object_type"),
            launch_date=row.get("launch_date"),
            decay_date=row.get("decay_date"),
            period_minutes=row.get("period_minutes"),
            apoapsis_km=row.get("apoapsis_km"),
            periapsis_km=row.get("periapsis_km"),
            rcs_meters2=row.get("rcs_meters2"),
            source=row["source"],
        )
        session.add(sat)

    logger.info("Loaded %d satellites.", len(rows))


def main() -> None:
    engine = create_engine(db_settings.sync_url)

    with Session(engine) as session:
        if is_db_populated(session):
            logger.info("Database already contains data — skipping seed.")
            return

        logger.info("Database is empty. Downloading seed data ...")
        dest = Path("/tmp/seed_data")
        dest.mkdir(parents=True, exist_ok=True)

        try:
            paths = download_seed_files(dest)
        except Exception as exc:
            logger.warning("Could not download seed data: %s", exc)
            logger.info("Skipping seed — the database is empty but functional.")
            logger.info("Run 'make ingest' later to populate it.")
            return

        checksum_to_id = load_documents(session, paths["documents"])
        load_chunks(session, paths["chunks"], checksum_to_id)
        load_satellites(session, paths["satellites"])

        session.commit()
        logger.info("Seed complete.")


if __name__ == "__main__":
    main()
