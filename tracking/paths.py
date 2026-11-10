"""Resolve tracking DB path relative to project root."""

from __future__ import annotations

from pathlib import Path

from config.experiment_schema import ExperimentConfig
from config.settings import get_project_paths


def resolve_tracking_db_path(cfg: ExperimentConfig) -> Path:
    p = Path(cfg.tracking.db_path)
    return p if p.is_absolute() else get_project_paths().root / p
