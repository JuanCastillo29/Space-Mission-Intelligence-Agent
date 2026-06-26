from __future__ import annotations

import base64
from functools import lru_cache
from pathlib import Path

_ASSETS = Path(__file__).resolve().parent / "assets"


@lru_cache(maxsize=16)
def svg_to_img_tag(name: str, width: int = 48, height: int = 48) -> str:
    raw = (_ASSETS / f"{name}.svg").read_text()
    b64 = base64.b64encode(raw.encode()).decode()
    return f'<img src="data:image/svg+xml;base64,{b64}" width="{width}" height="{height}" />'
