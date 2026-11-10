"""Shared deterministic helpers for agent orchestration."""

from __future__ import annotations

from pathlib import Path

from config.experiment_schema import ExperimentConfig, ValidationConfig
from config.settings import get_project_paths


def resolve_data_path(cfg: ExperimentConfig) -> Path:
    """Resolve ``cfg.data.path`` relative to project root (same rule as experiment pipeline)."""
    paths = get_project_paths()
    p = Path(cfg.data.path)
    if not str(cfg.data.path).strip():
        raise ValueError("data.path is required")
    return p if p.is_absolute() else paths.root / p


def compute_holdout_split_index(
    n_samples: int,
    *,
    horizon: int,
    validation: ValidationConfig,
) -> int:
    """
    Chronological train | holdout split index (first index of holdout), matching pipeline logic.
    """
    split = int(validation.train_fraction * n_samples)
    split = max(split, 2 * horizon + validation.n_splits + 5)
    split = min(split, n_samples - max(20, horizon + 5))
    if split >= n_samples - 5:
        raise ValueError("train_fraction leaves too few holdout rows")
    return split


def seed_model_params(extra: dict[str, object], seed: int) -> dict[str, object]:
    """Ensure ``random_state`` for reproducible sklearn-style models."""
    out = dict(extra)
    out["random_state"] = int(seed)
    return out
