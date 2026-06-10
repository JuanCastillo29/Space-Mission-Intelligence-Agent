from __future__ import annotations

import hashlib
from pathlib import Path


def compute_file_checksum(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()
