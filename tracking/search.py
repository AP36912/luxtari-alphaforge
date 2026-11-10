"""
Controlled Cartesian grid search over explicit parameter lists.

Each trial runs the full :func:`config.pipeline.run_experiment` (purged K-fold + chronological
holdout + backtest) — **no** random shuffles or test leakage beyond what the pipeline already prevents.
"""

from __future__ import annotations

from dataclasses import replace
from itertools import product
from pathlib import Path
from typing import Any

from config.experiment_schema import ExperimentConfig, TrackingConfig
from config.pipeline import run_experiment
from tracking.apply_params import apply_param_path
from tracking.paths import resolve_tracking_db_path


def run_parameter_search(
    base: ExperimentConfig,
    grid: dict[str, tuple[Any, ...]],
    *,
    batch_id: str,
    db_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """
    Cartesian product of ``grid`` values; each trial gets a unique ``run_id`` and is logged if tracking enabled.

    Args:
        base: Starting experiment config (data path, seed base, etc.).
        grid: Map **dotted path** → tuple of values (see :mod:`tracking.apply_params`).
        batch_id: Stored as ``search_batch_id`` for leaderboard / reporting filters.
        db_path: Override SQLite path; default uses ``base.tracking.db_path`` resolved to project root.

    Returns:
        List of pipeline result dicts with ``trial_index``, ``param_values`` attached.
    """
    if not grid:
        raise ValueError("grid must be non-empty")
    keys = list(grid.keys())
    for k, vals in grid.items():
        if not isinstance(vals, tuple):
            raise TypeError(f"grid[{k!r}] must be a tuple of values")

    db = Path(db_path) if db_path is not None else resolve_tracking_db_path(base)
    db_str = str(db)

    results: list[dict[str, Any]] = []
    for trial_index, values in enumerate(product(*(grid[k] for k in keys))):
        cfg = base
        param_values = dict(zip(keys, values, strict=True))
        for path, val in param_values.items():
            cfg = apply_param_path(cfg, path, val)

        tc = TrackingConfig(
            enabled=True,
            db_path=db_str,
            search_batch_id=batch_id,
            search_trial_index=trial_index,
        )
        run_id = f"{batch_id}_t{trial_index:04d}"
        cfg = replace(
            cfg,
            tracking=tc,
            output=replace(cfg.output, run_id=run_id),
        )
        out = run_experiment(cfg)
        out = dict(out)
        out["trial_index"] = trial_index
        out["param_values"] = param_values
        out["search_batch_id"] = batch_id
        results.append(out)
    return results
