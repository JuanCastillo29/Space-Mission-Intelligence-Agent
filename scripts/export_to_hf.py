"""Export database tables to JSON and upload to Hugging Face as a dataset."""

import json
import logging
import sys
from datetime import date, datetime
from pathlib import Path
from uuid import UUID

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from db.config import db_settings
from db.models import Chunk, Document, Satellite

logging.basicConfig(level=logging.INFO, format="%(levelname)-5s  %(message)s")
log = logging.getLogger(__name__)

DEST_DIR = Path("/tmp/hf_seed_export/seed")
HF_REPO = "JuanCastillo29/space-mission-intelligence-data"


def _serialize(obj: object) -> str | float | None:
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, date):
        return obj.isoformat()
    if isinstance(obj, UUID):
        return str(obj)
    raise TypeError(f"Cannot serialize {type(obj)}")


def export_documents(session: Session, dest: Path) -> None:
    rows = session.execute(select(Document)).scalars().all()
    data = []
    for doc in rows:
        data.append(
            {
                "title": doc.title,
                "source_url": doc.source_url,
                "source_type": doc.source_type.value,
                "mission_name": doc.mission_name,
                "checksum": doc.checksum,
                "metadata": doc.metadata_,
            }
        )
    path = dest / "documents.json"
    path.write_text(json.dumps(data, indent=2, default=_serialize))
    log.info("Exported %d documents to %s", len(data), path)


def export_chunks(session: Session, dest: Path) -> None:
    docs = {
        d.id: d.checksum
        for d in session.execute(select(Document)).scalars().all()
    }
    rows = session.execute(select(Chunk)).scalars().all()
    data = []
    for chunk in rows:
        data.append(
            {
                "document_checksum": docs[chunk.document_id],
                "content": chunk.content,
                "embedding": chunk.embedding.tolist()
                if chunk.embedding is not None
                else None,
                "chunk_index": chunk.chunk_index,
                "section_path": chunk.section_path,
                "token_count": chunk.token_count,
                "metadata": chunk.metadata_,
            }
        )
    path = dest / "chunks.json"
    path.write_text(json.dumps(data, indent=2, default=_serialize))
    log.info("Exported %d chunks to %s", len(data), path)


def export_satellites(session: Session, dest: Path) -> None:
    rows = session.execute(select(Satellite)).scalars().all()
    data = []
    for sat in rows:
        data.append(
            {
                "norad_id": sat.norad_id,
                "name": sat.name,
                "international_designator": sat.international_designator,
                "epoch": sat.epoch.isoformat(),
                "mean_motion": sat.mean_motion,
                "eccentricity": sat.eccentricity,
                "inclination": sat.inclination,
                "ra_of_asc_node": sat.ra_of_asc_node,
                "arg_of_pericenter": sat.arg_of_pericenter,
                "mean_anomaly": sat.mean_anomaly,
                "rev_number": sat.rev_number,
                "classification": sat.classification,
                "object_type": sat.object_type,
                "launch_date": sat.launch_date.isoformat()
                if sat.launch_date
                else None,
                "decay_date": sat.decay_date.isoformat()
                if sat.decay_date
                else None,
                "period_minutes": sat.period_minutes,
                "apoapsis_km": sat.apoapsis_km,
                "periapsis_km": sat.periapsis_km,
                "rcs_meters2": sat.rcs_meters2,
                "source": sat.source,
            }
        )
    path = dest / "satellites.json"
    path.write_text(json.dumps(data, indent=2, default=_serialize))
    log.info("Exported %d satellites to %s", len(data), path)


def upload_to_hf(dest: Path) -> None:
    from huggingface_hub import HfApi

    api = HfApi()

    api.create_repo(
        repo_id=HF_REPO,
        repo_type="dataset",
        exist_ok=True,
    )
    log.info("Repo %s ready.", HF_REPO)

    api.upload_folder(
        folder_path=str(dest.parent),
        repo_id=HF_REPO,
        repo_type="dataset",
    )
    log.info("Upload complete: https://huggingface.co/datasets/%s", HF_REPO)


def main() -> None:
    DEST_DIR.mkdir(parents=True, exist_ok=True)

    url = db_settings.sync_url
    if "--localhost" in sys.argv:
        url = url.replace("@db:", "@localhost:")

    engine = create_engine(url)

    with Session(engine) as session:
        export_documents(session, DEST_DIR)
        export_chunks(session, DEST_DIR)
        export_satellites(session, DEST_DIR)

    upload_to_hf(DEST_DIR)


if __name__ == "__main__":
    main()
