"""
Shared on-disk layout: ``<path>/meta.json`` + payload files.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def model_dir(path: str | Path) -> Path:
    d = Path(path)
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_meta(d: Path, payload: dict[str, Any]) -> None:
    with open(d / "meta.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def read_meta(d: Path) -> dict[str, Any]:
    with open(d / "meta.json", encoding="utf-8") as f:
        return json.load(f)
