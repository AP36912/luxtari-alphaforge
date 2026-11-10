"""
Artifact storage (e.g. Parquet) plus lightweight manifests.

Manifests should record provider, symbol set, date range, schema version, and
content hash so backtests can be tied to exact inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class DatasetManifest:
    """Metadata describing a persisted dataset slice."""

    path: Path
    provider: str
    schema_version: str


def write_placeholder_manifest(manifest: DatasetManifest) -> None:
    """Reserved for manifest serialization in Phase 2."""
    _ = manifest
